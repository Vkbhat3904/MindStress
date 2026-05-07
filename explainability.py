import torch
import torch.nn as nn
import numpy as np
import shap
import matplotlib.pyplot as plt
import os
from transformer_model import MultimodalTransformerEncoder

# ---------------------------------------------------------
# SHAP Wrapper
# ---------------------------------------------------------
class SHAPModelWrapper(nn.Module):
    """
    Wraps the MultimodalTransformerEncoder to accept a single concatenated 
    tensor of shape (batch, 101) so SHAP DeepExplainer can compute gradients.
    """
    def __init__(self, model: MultimodalTransformerEncoder):
        super().__init__()
        self.model = model
        self.model.eval()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch_size, 101)
        # Split back into EEG (97) and ECG (4)
        eeg_features = x[:, :97]
        ecg_features = x[:, 97:]
        
        # Predict
        return self.model(eeg_features, ecg_features)


# ---------------------------------------------------------
# XAI Utilities
# ---------------------------------------------------------
def generate_feature_names():
    """Generates the 101 feature names for plotting readability."""
    eeg_names = []
    bands = ['Theta', 'Alpha', 'Beta']
    for band in bands:
        for ch in range(32):
            eeg_names.append(f"EEG_Ch{ch}_{band}")
    eeg_names.append("Frontal_Alpha_Asymmetry")
    
    ecg_names = ["HeartRate_Mean", "SDNN", "RMSSD", "LF_HF_Ratio"]
    
    return eeg_names + ecg_names

def explain_predictions_shap(model: MultimodalTransformerEncoder, background_data: tuple, test_data: tuple, out_dir: str = "xai_plots"):
    """
    Computes SHAP values and generates Summary and Force plots.
    
    Args:
        model: Trained MultimodalTransformerEncoder
        background_data: Tuple of (eeg_bg, ecg_bg) tensors to act as SHAP baselines
        test_data: Tuple of (eeg_test, ecg_test) tensors to evaluate
    """
    os.makedirs(out_dir, exist_ok=True)
    
    eeg_bg, ecg_bg = background_data
    eeg_test, ecg_test = test_data
    
    bg_concat = torch.cat([eeg_bg, ecg_bg], dim=1)
    test_concat = torch.cat([eeg_test, ecg_test], dim=1)
    
    wrapper = SHAPModelWrapper(model)
    
    # Using DeepExplainer for PyTorch models
    explainer = shap.DeepExplainer(wrapper, bg_concat)
    shap_values_tensor = explainer.shap_values(test_concat)
    
    # shap_values could be a list for multi-output, but our model is single output (1 unit)
    if isinstance(shap_values_tensor, list):
        shap_values = shap_values_tensor[0]
    else:
        shap_values = shap_values_tensor
        
    feature_names = generate_feature_names()
    
    # 1. Summary Plot (Global Importance)
    plt.figure()
    shap.summary_plot(shap_values, test_concat.numpy(), feature_names=feature_names, show=False)
    plt.savefig(os.path.join(out_dir, "shap_summary_plot.png"), bbox_inches='tight')
    plt.close()
    
    # 2. Force Plot / Waterfall logic for First Sample (Local Importance)
    # Re-initialize JS in a notebook env, but here we just render matplotlib figures if possible
    expected_value = explainer.expected_value
    if isinstance(expected_value, np.ndarray) and len(expected_value) == 1:
        expected_value = expected_value[0]
        
    sample_idx = 0
    
    # For waterfall plot
    # SHAP expects 1D arrays for a single sample waterfall plot
    values_1d = shap_values[sample_idx]
    if len(values_1d.shape) > 1:
        values_1d = values_1d.flatten()
        
    shap.waterfall_plot(shap.Explanation(
        values=values_1d, 
        base_values=expected_value, 
        data=test_concat[sample_idx].numpy(), 
        feature_names=feature_names
    ), show=False)
    plt.savefig(os.path.join(out_dir, f"shap_waterfall_sample_{sample_idx}.png"), bbox_inches='tight')
    plt.close()
    
    # 3. Aggregate EEG Channel Importance Map (F3 vs F4 testing)
    # SHAP values shape: (batch, 101). Indices 0-95 are band powers.
    # Group mean absolute SHAP values by channel
    mean_abs_shap = np.mean(np.abs(shap_values), axis=0)
    
    channel_importance = np.zeros(32)
    for ch in range(32):
        # average importance of Theta, Alpha, and Beta for this channel
        channel_importance[ch] = (mean_abs_shap[ch] + mean_abs_shap[32+ch] + mean_abs_shap[64+ch]) / 3.0
        
    plt.figure(figsize=(10,6))
    channels = np.arange(32)
    plt.bar(channels, channel_importance)
    plt.xticks(channels, [f"Ch{i}" for i in channels], rotation=90)
    # Emphasize F3 (Ch 2) and F4 (Ch 19)
    plt.bar([2, 19], [channel_importance[2], channel_importance[19]], color='red', label='F3 / F4')
    plt.title("EEG Channel-Level Global Importance")
    plt.xlabel("Channel Index")
    plt.ylabel("Mean |SHAP Value|")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "eeg_channel_importance.png"))
    plt.close()
    
    print(f"SHAP plots successfully saved to {out_dir}/")


def extract_attention_weights(model: MultimodalTransformerEncoder, eeg_test: torch.Tensor, ecg_test: torch.Tensor, out_dir: str = "xai_plots"):
    """
    Extracts and plots the attention weights of the [CLS] token over [EEG] and [ECG] modalities.
    """
    os.makedirs(out_dir, exist_ok=True)
    model.eval()
    
    with torch.no_grad():
        _, attn_weights = model(eeg_test, ecg_test, return_attention=True)
    
    # attn_weights shape from PyTorch MultiheadAttention (batch_size, seq_len, seq_len)
    # Seq is: [CLS, EEG, ECG]
    # We want how CLS (index 0) attends to everything else
    cls_attention = attn_weights[:, 0, :]  # Shape: (batch_size, 3)
    
    # Average across the batch
    avg_cls_attention = torch.mean(cls_attention, dim=0).numpy()
    
    labels = ['self [CLS]', 'Brain [EEG]', 'Heart [ECG]']
    
    plt.figure(figsize=(6,4))
    plt.bar(labels, avg_cls_attention, color=['gray', 'blue', 'red'])
    plt.title("Average Cross-Modal Attention Weights")
    plt.ylabel("Attention Weight (Sum = 1.0)")
    for i, v in enumerate(avg_cls_attention):
        plt.text(i, v + 0.01, f"{v:.3f}", ha='center', va='bottom')
    plt.ylim(0, 1.1)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "attention_weights.png"))
    plt.close()
    
    print(f"Attention plot successfully saved to {out_dir}/")


if __name__ == '__main__':
    print("Testing Explainability Module...\n")
    
    # Dummy Model
    model = MultimodalTransformerEncoder(d_model=128, n_layers=2)
    
    # Dummy Data: 100 background samples, 5 test samples
    eeg_bg = torch.randn(100, 97)
    ecg_bg = torch.randn(100, 4)
    
    eeg_test = torch.randn(5, 97)
    ecg_test = torch.randn(5, 4)
    
    print("1. Running SHAP Explainer...")
    explain_predictions_shap(model, (eeg_bg, ecg_bg), (eeg_test, ecg_test), out_dir="c:/Stress/xai_plots")
    
    print("\n2. Extracting Attention...")
    extract_attention_weights(model, eeg_test, ecg_test, out_dir="c:/Stress/xai_plots")
    
    print("\nXAI Test Complete. Check the created plots in c:/Stress/xai_plots/")
