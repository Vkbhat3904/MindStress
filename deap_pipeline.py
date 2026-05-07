import os
import pickle
import numpy as np
import scipy.signal as signal
from scipy.interpolate import interp1d
from typing import Tuple, List, Dict

class DEAPPipeline:
    """
    Pipeline to load DEAP dataset files, extract EEG/peripheral signals,
    generate binary stress labels, and create segmented time-series windows.
    """
    
    def __init__(self, data_dir: str, window_size_sec: float = 30.0, overlap_sec: float = 15.0, 
                 sample_rate: int = 128, remove_baseline: bool = True,
                 apply_eeg_filter: bool = True, apply_normalization: bool = True,
                 extract_hrv: bool = True, extract_eeg_features: bool = True):
        """
        Args:
            data_dir (str): Directory containing the DEAP .dat files (e.g., s01.dat, s02.dat).
            window_size_sec (float): Size of each sliding window in seconds.
            overlap_sec (float): Overlap between sliding windows in seconds.
            sample_rate (int): Sampling rate of the DEAP dataset (default is 128Hz).
            remove_baseline (bool): Whether to remove the 3-second baseline (first 3*128 samples).
        """
        self.data_dir = data_dir
        self.sample_rate = sample_rate
        self.window_size_pts = int(window_size_sec * sample_rate)
        self.overlap_pts = int(overlap_sec * sample_rate)
        self.step_size_pts = self.window_size_pts - self.overlap_pts
        
        self.apply_eeg_filter = apply_eeg_filter
        self.apply_normalization = apply_normalization
        self.extract_hrv = extract_hrv
        self.extract_eeg_features = extract_eeg_features
        
        # Buffer for filter design
        if self.apply_eeg_filter:
            nyq = 0.5 * self.sample_rate
            self.filter_b, self.filter_a = signal.butter(4, [4.0 / nyq, 45.0 / nyq], btype='bandpass')
        
        # DEAP trials are 63 seconds long. First 3 seconds are baseline.
        self.baseline_pts = 3 * sample_rate if remove_baseline else 0
        
        if self.step_size_pts <= 0:
            raise ValueError("Overlap must be strictly less than the window size.")

    def load_participant_data(self, file_path: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Loads a single participant's .dat file.
        Returns:
            data: shape (40, 40, 8064) -> (videos, channels, samples)
            labels: shape (40, 4) -> (videos, labels[valence, arousal, dominance, liking])
        """
        with open(file_path, 'rb') as f:
            # DEAP files are pickled as python 2 objects, latin1 encoding is required for python 3
            content = pickle.load(f, encoding='latin1')
            
        data = content['data']
        labels = content['labels']
        return data, labels

    def apply_bandpass_filter(self, eeg_data: np.ndarray) -> np.ndarray:
        """
        Applies a 4–45 Hz bandpass filter to the EEG data.
        eeg_data shape: (channels, samples)
        """
        filtered_data = signal.filtfilt(self.filter_b, self.filter_a, eeg_data, axis=-1)
        return filtered_data
        
    def normalize_signal(self, data: np.ndarray) -> np.ndarray:
        """
        Applies Z-score normalization along the time axis.
        data shape: (channels, samples)
        """
        mean = np.mean(data, axis=-1, keepdims=True)
        std = np.std(data, axis=-1, keepdims=True)
        # Avoid division by zero
        std[std == 0] = 1.0
        return (data - mean) / std

    def extract_eeg_features_from_window(self, eeg_window: np.ndarray) -> np.ndarray:
        """
        Extracts Alpha, Beta, Theta band power, and Frontal Alpha Asymmetry.
        eeg_window shape: (32, window_pts)
        Returns: 1D array of features. (32 * 3 + 1 = 97 features)
        """
        nperseg = self.sample_rate * 2  # 2-second windows for Welch's
        if eeg_window.shape[-1] < nperseg:
            nperseg = eeg_window.shape[-1]
            
        freqs, psd = signal.welch(eeg_window, fs=self.sample_rate, nperseg=nperseg, axis=-1)
        df = freqs[1] - freqs[0] if len(freqs) > 1 else 1.0
        
        # Helper to compute band power
        def bandpower(band):
            idx = np.logical_and(freqs >= band[0], freqs <= band[1])
            return np.sum(psd[:, idx], axis=-1) * df
            
        theta_power = bandpower((4, 8))
        alpha_power = bandpower((8, 13))
        beta_power = bandpower((13, 30))
        
        # Frontal Alpha Asymmetry (FAA)
        # F3 is channel index 2, F4 is channel index 19 (0-indexed standard DEAP 32 channel setup)
        alpha_f3 = max(alpha_power[2], 1e-10)
        alpha_f4 = max(alpha_power[19], 1e-10)
        
        faa = np.log(alpha_f4) - np.log(alpha_f3)
        
        return np.concatenate([theta_power, alpha_power, beta_power, [faa]])

    def extract_hrv_features(self, ecg_signal: np.ndarray) -> np.ndarray:
        """
        Extracts HRV features from the ECG/BVP signal.
        ecg_signal shape: (samples,)
        Returns [mean_hr, sdnn, rmssd, lf_hf_ratio]
        """
        # Find peaks in the BVP signal
        # Distance ensures a max plausible heart rate of ~220 BPM (128Hz / (220/60) ~ 35 samples distance)
        distance = int(self.sample_rate / (220/60))
        peaks, _ = signal.find_peaks(ecg_signal, distance=distance)
        
        if len(peaks) < 2:
            # Handle cases with zero or one peak gracefully by returning a full 4-feature HRV vector.
            return np.array([0.0, 0.0, 0.0, 0.0])
            
        # Calculate RR intervals in seconds
        rr_intervals = np.diff(peaks) / self.sample_rate
        
        mean_hr = 60.0 / np.mean(rr_intervals)
        sdnn = np.std(rr_intervals)
        
        # RMSSD
        if len(rr_intervals) > 1:
            rmssd = np.sqrt(np.mean(np.square(np.diff(rr_intervals))))
        else:
            rmssd = 0.0
            
        # LF / HF Ratio
        lf_hf_ratio = 0.0
        if len(peaks) > 10:
            times = peaks[1:] / self.sample_rate
            resample_freq = 4.0
            t_interp = np.arange(times[0], times[-1], 1.0 / resample_freq)
            
            if len(t_interp) > 10:
                f_interp = interp1d(times, rr_intervals, kind='cubic', fill_value='extrapolate')
                rr_interp = f_interp(t_interp)
                
                nperseg = min(256, len(rr_interp))
                freqs, psd = signal.welch(rr_interp, fs=resample_freq, nperseg=nperseg)
                df = freqs[1] - freqs[0] if len(freqs) > 1 else 1.0
                
                lf_idx = np.logical_and(freqs >= 0.04, freqs <= 0.15)
                hf_idx = np.logical_and(freqs >= 0.15, freqs <= 0.4)
                
                lf_power = np.sum(psd[lf_idx]) * df
                hf_power = np.sum(psd[hf_idx]) * df
                
                if hf_power > 0:
                    lf_hf_ratio = lf_power / hf_power

        return np.array([mean_hr, sdnn, rmssd, lf_hf_ratio])

    def extract_stress_labels(self, labels: np.ndarray) -> np.ndarray:
        """
        Converts DEAP labels to binary stress labels.
        Rule: stress = arousal > 5 AND valence < 5
        Args:
            labels: shape (40, 4) where idx 0 is valence, idx 1 is arousal
        Returns:
            stress_labels: shape (40,) binary array (1 for stress, 0 for non-stress)
        """
        valence = labels[:, 0]
        arousal = labels[:, 1]
        
        # Binary condition for stress
        stress_condition = (arousal > 5) & (valence < 5)
        stress_labels = stress_condition.astype(int)
        
        return stress_labels

    def extract_stress_labels_5class(self, labels: np.ndarray) -> np.ndarray:
        """
        Converts DEAP labels to 5-class ordinal stress labels using a combined stress score.

        Stress Score = arousal - valence  (range: -8 to +8)
        High arousal + Low valence = High stress (high score)
        Low arousal  + High valence = Low stress (low score)

        Class mapping:
            0 = Very Low  (score <= -2)
            1 = Low       (score -1 to 1)
            2 = Moderate  (score 2 to 4)
            3 = High      (score 5 to 6)
            4 = Very High (score >= 7)

        Args:
            labels: shape (40, 4) where col 0 = valence, col 1 = arousal (both 1-9)
        Returns:
            stress_labels: shape (40,) integer array with values 0-4
        """
        valence = labels[:, 0]
        arousal = labels[:, 1]

        # Combined stress score: high arousal + low valence = high stress
        stress_score = arousal - valence  # range: -8 to +8

        # Use dataset percentiles to split into 5 balanced classes.
        # This reduces extreme imbalance in the original DEAP label mapping.
        thresholds = np.percentile(stress_score, [20, 40, 60, 80])
        stress_labels = np.digitize(stress_score, thresholds, right=True)

        # Ensure the resulting labels are in the 0-4 range.
        stress_labels = np.clip(stress_labels, 0, 4).astype(int)

        return stress_labels

    def create_segments(self, participant_data: np.ndarray, participant_labels: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Segments the continuous time-series data into discrete windows.
        
        Args:
            participant_data: (40, 40, 8064)
            participant_labels: (40,) stress labels
            
        Returns:
            X: segmented data of shape (num_windows, channels, window_size)
            y: labels for each window of shape (num_windows,)
            hrv: HRV features for each window of shape (num_windows, 4)
            features: Multimodal features combined (EEG powers + FAA + HRV) (num_windows, 101)
        """
        X_list = []
        y_list = []
        hrv_list = []
        multi_features_list = []
        
        num_videos = participant_data.shape[0]
        num_channels = participant_data.shape[1]
        total_samples = participant_data.shape[2]
        
        for video_idx in range(num_videos):
            # Extract data for the current video, optionally skipping the baseline
            video_data = participant_data[video_idx, :, self.baseline_pts:]
            video_label = participant_labels[video_idx]
            
            # Apply continuous preprocessing before segmenting
            if self.apply_eeg_filter:
                video_data[:32, :] = self.apply_bandpass_filter(video_data[:32, :])
            
            if self.apply_normalization:
                video_data = self.normalize_signal(video_data)
            
            # Slide window over the video's time series
            for start_idx in range(0, video_data.shape[1] - self.window_size_pts + 1, self.step_size_pts):
                end_idx = start_idx + self.window_size_pts
                
                # Shape: (40, window_size_pts) -> 32 EEG + 8 Peripheral
                window = video_data[:, start_idx:end_idx]
                
                hrv = None
                if self.extract_hrv:
                    # BVP is typically channel 39, index 38 in DEAP 40-channel config
                    hrv = self.extract_hrv_features(window[38, :])
                    hrv_list.append(hrv)
                
                if getattr(self, 'extract_eeg_features', False):
                    eeg_feats = self.extract_eeg_features_from_window(window[:32, :])
                    # Combine EEG and HRV if available
                    if hrv is not None:
                        combined = np.concatenate([eeg_feats, hrv])
                    else:
                        combined = eeg_feats
                    multi_features_list.append(combined)

                X_list.append(window)
                y_list.append(video_label)
                
        return np.array(X_list), np.array(y_list), np.array(hrv_list), np.array(multi_features_list)

    def process_all(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Processes all .dat files in the data directory.
        Returns combined segments and labels across all processed participants.
        """
        global_X = []
        global_y = []
        global_hrv = []
        global_features = []
        
        files = [f for f in os.listdir(self.data_dir) if f.endswith('.dat')]
        
        if not files:
            print(f"No .dat files found in {self.data_dir}")
            return np.array([]), np.array([]), np.array([])
            
        print(f"Found {len(files)} .dat files. Starting processing...")
        
        for file_name in files:
            file_path = os.path.join(self.data_dir, file_name)
            print(f"Processing {file_name}...")
            
            data, original_labels = self.load_participant_data(file_path)
            stress_labels = self.extract_stress_labels(original_labels)
            
            X_windows, y_windows, hrv_windows, feature_windows = self.create_segments(data, stress_labels)
            
            global_X.append(X_windows)
            global_y.append(y_windows)
            if self.extract_hrv:
                global_hrv.append(hrv_windows)
            if getattr(self, 'extract_eeg_features', False):
                global_features.append(feature_windows)
            
        final_X = np.concatenate(global_X, axis=0)
        final_y = np.concatenate(global_y, axis=0)
        final_hrv = np.concatenate(global_hrv, axis=0) if self.extract_hrv and global_hrv else np.array([])
        final_features = np.concatenate(global_features, axis=0) if getattr(self, 'extract_eeg_features', False) and global_features else np.array([])
        
        print("\nProcessing complete!")
        print(f"Total windowed samples (X structure): {final_X.shape}")
        print(f"Total labels (y structure): {final_y.shape}")
        if self.extract_hrv:
            print(f"Total HRV features: {final_hrv.shape}")
        if getattr(self, 'extract_eeg_features', False):
            print(f"Total Multimodal Features: {final_features.shape}")
            
        # Example shape output: (Total_Segments, 40_channels, Window_Size_Pts)
        return final_X, final_y, final_hrv, final_features

if __name__ == '__main__':
    # ==========================================
    # Example Usage:
    # ==========================================
    # Replace 'path_to_deap_dataset' with your actual dataset folder.
    # Typical DEAP .dat files: s01.dat, s02.dat, ..., s32.dat
    #
    # Parameters:
    # window_size_sec = 1.0 (creates 128 data point windows)
    # overlap_sec = 0.5 (50% overlap between windows for data augmentation)
    
    DATA_DIRECTORY = r"c:\Users\sambh\Downloads\archive (9)\deap-dataset\data_preprocessed_python"
    
    if os.path.exists(DATA_DIRECTORY):
        pipeline = DEAPPipeline(
            data_dir=DATA_DIRECTORY,
            window_size_sec=30.0,
            overlap_sec=15.0,
            remove_baseline=True, # Skips the 3-second baseline
            extract_eeg_features=True
        )
        
        # X: (N_windows, 40, window_pts), y: (N_windows,)
        X, y, hrv, features = pipeline.process_all()
        
        # Save the preprocessed data for fast loading later
        print("Saving preprocessed data...")
        np.save('X_preprocessed.npy', X)
        np.save('y_preprocessed.npy', y)
        if pipeline.extract_hrv:
            np.save('hrv_preprocessed.npy', hrv)
        if pipeline.extract_eeg_features:
            np.save('multimodal_features.npy', features)
            
        print("Saved output arrays safely.")
    else:
        print(f"Directory {DATA_DIRECTORY} does not exist. Please update DATA_DIRECTORY.")
