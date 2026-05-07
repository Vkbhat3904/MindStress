"""
Generate synthetic DEAP-format .dat files for testing the NeuroStress pipeline.

Creates 5 .dat files in c:\Stress\synthetic_data\ — one per stress profile:
  - synthetic_very_low.dat   (Very Low stress:  high valence, low arousal)
  - synthetic_low.dat        (Low stress)
  - synthetic_moderate.dat   (Moderate stress)
  - synthetic_high.dat       (High stress)
  - synthetic_very_high.dat  (Very High stress: low valence, high arousal)

Each file is a pickle dict matching the real DEAP format:
  { 'data': np.ndarray (N_trials, 40, 8064),
    'labels': np.ndarray (N_trials, 4) }

Channels 0-31:  EEG  (bandpass-realistic sinusoids + pink noise)
Channels 32-39: Peripheral  (EMG × 4, GSR, Resp, Temp, BVP/ECG)
Labels columns: [valence, arousal, dominance, liking]  (all 1-9 scale)

Usage:
    python generate_synthetic_dat.py
"""

import os
import pickle
import numpy as np

# ── Output directory ──────────────────────────────────────────
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "synthetic_data")

# ── DEAP constants ────────────────────────────────────────────
SAMPLE_RATE   = 128          # Hz
TRIAL_SAMPLES = 8064         # 63 seconds × 128 Hz (3 s baseline + 60 s trial)
N_CHANNELS    = 40           # 32 EEG + 8 peripheral
N_EEG         = 32
TRIALS_PER_FILE = 5          # small files for quick testing


# ── Stress profile definitions ────────────────────────────────
PROFILES = {
    "very_low": {
        "valence":  (7.0, 9.0),   # high valence
        "arousal":  (1.0, 3.0),   # low arousal
        "alpha_amp":  1.5,        # strong alpha → relaxed
        "beta_amp":   0.3,        # low beta
        "theta_amp":  0.6,
        "noise_std":  0.15,
        "heart_rate": 62,         # BPM
        "hr_var":     0.02,       # very regular
        "gsr_level":  0.3,        # low galvanic skin response
    },
    "low": {
        "valence":  (6.0, 8.0),
        "arousal":  (2.0, 4.0),
        "alpha_amp":  1.2,
        "beta_amp":   0.5,
        "theta_amp":  0.7,
        "noise_std":  0.20,
        "heart_rate": 70,
        "hr_var":     0.04,
        "gsr_level":  0.4,
    },
    "moderate": {
        "valence":  (4.0, 6.0),
        "arousal":  (4.0, 6.0),
        "alpha_amp":  0.8,
        "beta_amp":   0.9,
        "theta_amp":  0.8,
        "noise_std":  0.30,
        "heart_rate": 82,
        "hr_var":     0.06,
        "gsr_level":  0.6,
    },
    "high": {
        "valence":  (2.0, 4.0),
        "arousal":  (6.0, 8.0),
        "alpha_amp":  0.4,        # suppressed alpha
        "beta_amp":   1.6,        # strong beta → anxious
        "theta_amp":  0.9,
        "noise_std":  0.40,
        "heart_rate": 96,
        "hr_var":     0.10,
        "gsr_level":  0.8,
    },
    "very_high": {
        "valence":  (1.0, 3.0),
        "arousal":  (7.0, 9.0),
        "alpha_amp":  0.2,        # minimal alpha
        "beta_amp":   2.0,        # dominant beta/gamma
        "theta_amp":  1.2,        # theta bursts (stress marker)
        "noise_std":  0.55,
        "heart_rate": 112,
        "hr_var":     0.15,       # irregular
        "gsr_level":  1.0,
    },
}


# ── Helpers ───────────────────────────────────────────────────

def pink_noise(n_samples: int, amplitude: float = 1.0) -> np.ndarray:
    """Generate 1/f (pink) noise via spectral shaping."""
    white = np.random.randn(n_samples)
    fft = np.fft.rfft(white)
    freqs = np.fft.rfftfreq(n_samples, d=1.0 / SAMPLE_RATE)
    freqs[0] = 1.0  # avoid div-by-zero
    fft /= np.sqrt(freqs)
    return np.fft.irfft(fft, n=n_samples) * amplitude


def generate_eeg_channel(n_samples: int, alpha_amp: float, beta_amp: float,
                         theta_amp: float, noise_std: float,
                         channel_idx: int) -> np.ndarray:
    """
    Generate a single EEG channel with realistic band-specific oscillations.

    Each channel gets a slightly different frequency and phase offset based on
    its index, mimicking spatial variation across the scalp.
    """
    t = np.arange(n_samples) / SAMPLE_RATE
    rng = np.random.default_rng(seed=channel_idx * 137 + int(alpha_amp * 1000))

    # Per-channel frequency offsets for natural variation
    offset = rng.uniform(-0.5, 0.5)

    # Theta band (4–8 Hz)
    theta = theta_amp * np.sin(2 * np.pi * (6.0 + offset * 0.5) * t + rng.uniform(0, 2 * np.pi))

    # Alpha band (8–13 Hz)
    alpha = alpha_amp * np.sin(2 * np.pi * (10.0 + offset) * t + rng.uniform(0, 2 * np.pi))

    # Beta band (13–30 Hz)
    beta = beta_amp * np.sin(2 * np.pi * (20.0 + offset * 2) * t + rng.uniform(0, 2 * np.pi))

    # Add harmonics for realism
    alpha += 0.3 * alpha_amp * np.sin(2 * np.pi * (11.5 + offset) * t + rng.uniform(0, 2 * np.pi))
    beta  += 0.4 * beta_amp  * np.sin(2 * np.pi * (25.0 + offset) * t + rng.uniform(0, 2 * np.pi))

    # Pink noise floor
    noise = pink_noise(n_samples, amplitude=noise_std)

    signal = theta + alpha + beta + noise

    # Realistic EEG amplitude scaling (microvolts-like range, normalized)
    signal *= (0.8 + 0.4 * rng.random())

    return signal


def generate_ecg_channel(n_samples: int, heart_rate: float, hr_var: float) -> np.ndarray:
    """
    Generate a synthetic BVP/ECG-like signal with QRS-like peaks at the target heart rate.
    """
    t = np.arange(n_samples) / SAMPLE_RATE
    rng = np.random.default_rng(seed=int(heart_rate * 100))

    ibi = 60.0 / heart_rate  # inter-beat interval in seconds

    # Build heartbeat times with variability
    beat_times = []
    current_t = 0.1  # first beat offset
    while current_t < t[-1]:
        beat_times.append(current_t)
        current_t += ibi + rng.normal(0, hr_var)
    beat_times = np.array(beat_times)

    # Build signal from Gaussian QRS templates
    signal = np.zeros(n_samples)
    for bt in beat_times:
        # QRS-like peak (narrow Gaussian)
        qrs = np.exp(-0.5 * ((t - bt) / 0.02) ** 2) * 1.5
        # T-wave (broader)
        t_wave = np.exp(-0.5 * ((t - bt - 0.2) / 0.08) ** 2) * 0.4
        signal += qrs + t_wave

    # Add baseline wander + noise
    signal += 0.1 * np.sin(2 * np.pi * 0.15 * t)
    signal += rng.normal(0, 0.05, n_samples)

    return signal


def generate_peripheral_channels(n_samples: int, profile: dict) -> np.ndarray:
    """
    Generate 8 peripheral channels (indices 32–39):
      32-35: hEOG, vEOG, zEMG, tEMG   (eye/muscle artifacts)
      36:    GSR / Skin Conductance
      37:    Respiration
      38:    Plethysmograph (BVP/ECG) ← this is what HRV features use
      39:    Temperature
    """
    rng = np.random.default_rng(seed=42)
    t = np.arange(n_samples) / SAMPLE_RATE
    channels = np.zeros((8, n_samples))

    hr = profile["heart_rate"]
    gsr = profile["gsr_level"]

    # Channels 0-3 (indices 32-35): EOG + EMG — low-freq artifacts + noise
    for i in range(4):
        channels[i] = (0.5 * np.sin(2 * np.pi * (0.3 + 0.1 * i) * t) +
                        rng.normal(0, 0.2, n_samples))

    # Channel 4 (index 36): GSR — slow drift
    channels[4] = gsr + 0.2 * np.sin(2 * np.pi * 0.05 * t) + rng.normal(0, 0.03, n_samples)

    # Channel 5 (index 37): Respiration — sinusoid at ~0.25 Hz (15 breaths/min)
    resp_rate = 0.2 + 0.1 * (hr / 100)
    channels[5] = 0.8 * np.sin(2 * np.pi * resp_rate * t) + rng.normal(0, 0.05, n_samples)

    # Channel 6 (index 38): BVP / ECG — the critical one for HRV extraction
    channels[6] = generate_ecg_channel(n_samples, hr, profile["hr_var"])

    # Channel 7 (index 39): Temperature — near-constant with drift
    channels[7] = 36.5 + 0.1 * np.sin(2 * np.pi * 0.01 * t) + rng.normal(0, 0.02, n_samples)

    return channels


def generate_trial(profile: dict, trial_idx: int) -> np.ndarray:
    """Generate one trial of shape (40, 8064)."""
    data = np.zeros((N_CHANNELS, TRIAL_SAMPLES))

    # EEG channels 0–31
    for ch in range(N_EEG):
        data[ch] = generate_eeg_channel(
            TRIAL_SAMPLES,
            alpha_amp=profile["alpha_amp"],
            beta_amp=profile["beta_amp"],
            theta_amp=profile["theta_amp"],
            noise_std=profile["noise_std"],
            channel_idx=ch + trial_idx * N_EEG,
        )

    # Peripheral channels 32–39
    data[N_EEG:] = generate_peripheral_channels(TRIAL_SAMPLES, profile)

    return data


def generate_labels(profile: dict, n_trials: int) -> np.ndarray:
    """
    Generate (n_trials, 4) labels: [valence, arousal, dominance, liking].
    Valence and arousal come from the profile ranges; dominance and liking
    are randomized in a realistic range.
    """
    rng = np.random.default_rng()
    labels = np.zeros((n_trials, 4))

    v_lo, v_hi = profile["valence"]
    a_lo, a_hi = profile["arousal"]

    labels[:, 0] = rng.uniform(v_lo, v_hi, n_trials)  # valence
    labels[:, 1] = rng.uniform(a_lo, a_hi, n_trials)  # arousal
    labels[:, 2] = rng.uniform(3.0, 7.0, n_trials)    # dominance
    labels[:, 3] = rng.uniform(3.0, 7.0, n_trials)    # liking

    return labels


def generate_dat_file(profile_name: str, profile: dict, out_dir: str):
    """Generate and save a single synthetic .dat file."""
    print(f"  Generating {TRIALS_PER_FILE} trials for '{profile_name}' profile...")

    data = np.zeros((TRIALS_PER_FILE, N_CHANNELS, TRIAL_SAMPLES))
    for t in range(TRIALS_PER_FILE):
        data[t] = generate_trial(profile, trial_idx=t)

    labels = generate_labels(profile, TRIALS_PER_FILE)

    content = {"data": data, "labels": labels}

    filename = f"synthetic_{profile_name}.dat"
    filepath = os.path.join(out_dir, filename)

    with open(filepath, "wb") as f:
        pickle.dump(content, f, protocol=pickle.HIGHEST_PROTOCOL)

    file_size_mb = os.path.getsize(filepath) / (1024 * 1024)
    print(f"  ✅ Saved: {filename}  ({file_size_mb:.1f} MB)")
    print(f"     data shape:   {data.shape}")
    print(f"     labels shape: {labels.shape}")
    print(f"     valence range: [{labels[:,0].min():.1f}, {labels[:,0].max():.1f}]")
    print(f"     arousal range: [{labels[:,1].min():.1f}, {labels[:,1].max():.1f}]")
    print()


# ── Main ──────────────────────────────────────────────────────

def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    print("=" * 60)
    print("  Synthetic DEAP .dat File Generator")
    print("=" * 60)
    print(f"  Output directory: {OUT_DIR}")
    print(f"  Trials per file:  {TRIALS_PER_FILE}")
    print(f"  Channels:         {N_CHANNELS} (32 EEG + 8 peripheral)")
    print(f"  Samples/trial:    {TRIAL_SAMPLES} ({TRIAL_SAMPLES/SAMPLE_RATE:.0f}s @ {SAMPLE_RATE}Hz)")
    print("=" * 60)
    print()

    for name, profile in PROFILES.items():
        generate_dat_file(name, profile, OUT_DIR)

    print("=" * 60)
    print("  All synthetic .dat files generated successfully!")
    print(f"  Files are in: {OUT_DIR}")
    print()
    print("  You can now:")
    print("  1. Upload any .dat file via the NeuroStress web UI")
    print("  2. Use them as input to the DEAP pipeline")
    print("  3. Test the /predict_dat API endpoint")
    print("=" * 60)


if __name__ == "__main__":
    main()
