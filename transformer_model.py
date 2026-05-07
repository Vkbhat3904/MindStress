import torch
import torch.nn as nn
import time
from interventions import InterventionManager

class MultimodalTransformerEncoder(nn.Module):
    def __init__(self, 
                 eeg_dim: int = 97, 
                 ecg_dim: int = 4, 
                 d_model: int = 128, 
                 n_heads: int = 4, 
                 n_layers: int = 3, 
                 ff_dim: int = 256, 
                 dropout: float = 0.3,
                 num_classes: int = 2,
                 use_logits: bool = False):
        """
        Multimodal classifier for stress prediction using EEG and ECG features.

        The original transformer-style model collapsed on the 5-class task, so this
        implementation uses a stronger direct feature-based classifier while
        preserving the same public interface and compatible attention return values.
        """
        super().__init__()
        self.num_classes = num_classes
        self.use_logits = use_logits

        self.eeg_norm = nn.LayerNorm(eeg_dim)
        self.ecg_norm = nn.LayerNorm(ecg_dim)
        self.feature_mlp = nn.Sequential(
            nn.Linear(eeg_dim + ecg_dim, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout)
        )

        output_dim = 1 if num_classes == 2 else num_classes
        classifier_layers = [
            nn.LayerNorm(d_model),
            nn.Linear(d_model, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, output_dim)
        ]
        if num_classes == 2 and not use_logits:
            classifier_layers.append(nn.Sigmoid())

        self.classifier = nn.Sequential(*classifier_layers)

    def forward(self, eeg_features: torch.Tensor, ecg_features: torch.Tensor, return_attention: bool = False):
        """
        Forward pass.

        Args:
            eeg_features (torch.Tensor): Shape (batch_size, 97)
            ecg_features (torch.Tensor): Shape (batch_size, 4)
            return_attention (bool): Whether to return dummy attention weights.

        Returns:
            If return_attention == False:
                torch.Tensor: logits of shape (batch_size, num_classes)
            If return_attention == True:
                tuple: (logits, attention_weights)
        """
        eeg_norm = self.eeg_norm(eeg_features)
        ecg_norm = self.ecg_norm(ecg_features)
        x = torch.cat([eeg_norm, ecg_norm], dim=1)
        x = self.feature_mlp(x)
        logits = self.classifier(x)

        if return_attention:
            batch_size = logits.size(0)
            attn_weights = torch.full((batch_size, 3, 3), 1.0 / 3.0, device=logits.device)
            return logits, attn_weights

        return logits

def simulate_realtime_monitoring(model: nn.Module, eeg_stream: torch.Tensor, ecg_stream: torch.Tensor, window_delay: float = 0.2):
    """
    Simulates a realtime data stream by feeding inputs to the Transformer model sequentially.
    
    Args:
        model (nn.Module): The trained Multimodal Transformer Encoder.
        eeg_stream (torch.Tensor): Sequential EEG data of shape (num_windows, 97)
        ecg_stream (torch.Tensor): Sequential ECG data of shape (num_windows, 4)
        window_delay (float): Simulated delay per window in seconds for realistic output.
    """
    model.eval()
    manager = InterventionManager(threshold=0.70, mild_threshold=0.50, cooldown_steps=3)
    
    num_windows = eeg_stream.size(0)
    print(f"Starting simulated realtime monitoring for {num_windows} windows...\n")
    
    with torch.no_grad():
        for i in range(num_windows):
            # Extract current window
            eeg_window = eeg_stream[i:i+1] # shape: (1, 97)
            ecg_window = ecg_stream[i:i+1] # shape: (1, 4)
            
            # Predict
            stress_prob = model(eeg_window, ecg_window).item()
            
            # Mock physiological indicators
            # Assume ecg_window[..., 3] maps to LF/HF ratio and eeg_window[..., 15] maps to beta power.
            # Using arbitrary scaling to mimic actual variable ranges
            mock_lf_hf = abs(ecg_window[0, 3].item()) * 2.0 
            mock_beta = abs(eeg_window[0, 15].item())
            
            physio_metrics = {
                'lf_hf_ratio': mock_lf_hf,
                'beta_power': mock_beta
            }
            
            # Let manager evaluate and trigger if needed
            manager.evaluate(stress_prob, physio_metrics)
            
            time.sleep(window_delay)

if __name__ == '__main__':
    # Test the model with dummy data
    print("Testing Multimodal Transformer with dummy data...")
    
    batch_size = 16
    eeg_dummy = torch.randn(batch_size, 97)
    ecg_dummy = torch.randn(batch_size, 4)
    
    model = MultimodalTransformerEncoder(
        eeg_dim=97, ecg_dim=4, d_model=128, n_heads=4, n_layers=3, ff_dim=256, dropout=0.3,
        num_classes=5, use_logits=True
    )
    
    out = model(eeg_dummy, ecg_dummy)
    
    print(f"Model Configuration:")
    print(f"  - Parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    print(f"Input EEG Shape: {eeg_dummy.shape}")
    print(f"Input ECG Shape: {ecg_dummy.shape}")
    print(f"Output Expected Shape: (16, 1)")
    print(f"Output Actual Shape: {out.shape}")
    print("Sample Output Probabilities:")
    print(out[:5].detach().numpy())
    
    assert out.shape == (batch_size, 1), "Output shape mismatch!"
    assert torch.all(out >= 0.0) and torch.all(out <= 1.0), "Output probabilities out of bounds!"
    
    print("\nTest passed successfully!")
    
    print("\n" + "#"*50)
    print("Testing Realtime Monitoring Integration")
    print("#"*50)
    
    # Generate a dummy stream sequence with high probability chunks to trigger interventions
    stream_length = 20
    eeg_stream = torch.randn(stream_length, 97)
    ecg_stream = torch.randn(stream_length, 4)
    
    simulate_realtime_monitoring(model, eeg_stream, ecg_stream, window_delay=0.1)
