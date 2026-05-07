import torch
import numpy as np
import json
from typing import Dict, List, Any, Tuple, Optional
from transformer_model import MultimodalTransformerEncoder
from context_builder import build_context
from stress_management_planner import StressManagementPlanner
from explainability import generate_feature_names

class IntegratedStressPipeline:
    """
    Integrated pipeline connecting Multimodal Transformer → Context Builder → RAG → LLM Planner
    """

    def __init__(self, model_path: Optional[str] = None, device: str = 'cpu'):
        """
        Initialize the integrated pipeline.

        Args:
            model_path: Path to trained transformer model weights (optional)
            device: Device to run model on ('cpu' or 'cuda')
        """
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')

        # Initialize transformer model
        self.model = MultimodalTransformerEncoder(
            eeg_dim=97, ecg_dim=4, d_model=128, n_heads=4, n_layers=3,
            ff_dim=256, dropout=0.3, num_classes=5, use_logits=True
        ).to(self.device)

        # Load trained weights if provided
        if model_path:
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            print(f"Loaded model weights from {model_path}")

        self.model.eval()

        # Initialize RAG system
        self.planner = StressManagementPlanner()

        # Stress level mapping (assuming 5-class classification)
        self.stress_levels = ['very_low', 'low', 'moderate', 'high', 'very_high']
        self.feature_names = generate_feature_names()

    def predict_stress(self, eeg_features: torch.Tensor, ecg_features: torch.Tensor) -> Dict[str, Any]:
        """
        Predict stress level and extract features from transformer model.

        Args:
            eeg_features: EEG features tensor of shape (batch_size, 97)
            ecg_features: ECG features tensor of shape (batch_size, 4)

        Returns:
            Dictionary containing stress predictions and extracted features
        """
        with torch.no_grad():
            # Get model predictions
            logits = self.model(eeg_features, ecg_features)  # Shape: (batch_size, 5)

            # Convert logits to probabilities
            probabilities = torch.softmax(logits, dim=-1)

            # Get predicted class and probability
            pred_class_idx = torch.argmax(probabilities, dim=-1).item()
            stress_probability = probabilities[0, pred_class_idx].item()
            stress_level = self.stress_levels[pred_class_idx]

            # Extract physiological metrics (mock for now - in real implementation,
            # these would be derived from the actual feature values)
            physiological_metrics = self._extract_physiological_metrics(eeg_features, ecg_features)

            # Extract top SHAP features (simplified - in practice would use actual SHAP)
            top_shap_features = self._extract_top_features(eeg_features, ecg_features)

            # Get all features as dict
            all_features = self._tensor_to_dict(eeg_features, ecg_features)

            return {
                'stress_level': stress_level,
                'stress_probability': stress_probability,
                'physiological_metrics': physiological_metrics,
                'top_shap_features': top_shap_features,
                'all_features': all_features,
                'raw_logits': logits.cpu().numpy().tolist(),
                'probabilities': probabilities.cpu().numpy().tolist()
            }

    def _extract_physiological_metrics(self, eeg_features: torch.Tensor, ecg_features: torch.Tensor) -> Dict[str, float]:
        """Extract key physiological metrics from features."""
        # In a real implementation, these would map to actual physiological indicators
        # For now, using simplified mappings
        metrics = {}

        # ECG features (assuming standard order: HR, SDNN, RMSSD, LF/HF)
        if ecg_features.shape[-1] >= 4:
            metrics['lf_hf_ratio'] = abs(ecg_features[0, 3].item()) * 2.0  # LF/HF ratio
            metrics['heart_rate'] = abs(ecg_features[0, 0].item()) * 80 + 60  # Mock HR
            metrics['sdnn'] = abs(ecg_features[0, 1].item()) * 50 + 20  # Mock SDNN
            metrics['rmssd'] = abs(ecg_features[0, 2].item()) * 30 + 15  # Mock RMSSD

        # EEG features - extract key bands
        if eeg_features.shape[-1] >= 97:
            # Assuming standard EEG band order, extract representative values
            metrics['beta_power'] = abs(eeg_features[0, 15].item()) * 20  # Mock beta power
            metrics['alpha_power'] = abs(eeg_features[0, 5].item()) * 15  # Mock alpha power
            metrics['theta_power'] = abs(eeg_features[0, 25].item()) * 10  # Mock theta power
            metrics['faa'] = (eeg_features[0, 45].item() - eeg_features[0, 35].item()) * 0.5  # Mock FAA

        return metrics

    def _extract_top_features(self, eeg_features: torch.Tensor, ecg_features: torch.Tensor) -> List[str]:
        """Extract top contributing features (simplified SHAP simulation)."""
        # In practice, this would use actual SHAP values
        # For now, return some representative features based on physiological importance
        top_features = []

        # Always include key physiological indicators
        top_features.extend(['lf_hf_ratio', 'beta_power', 'heart_rate'])

        # Add some EEG features that might be important
        if eeg_features.shape[-1] > 10:
            top_features.extend(['eeg_alpha_frontal', 'eeg_beta_central', 'eeg_theta_parietal'])

        return top_features[:5]  # Return top 5

    def _tensor_to_dict(self, eeg_features: torch.Tensor, ecg_features: torch.Tensor) -> Dict[str, Any]:
        """Convert tensor features to dictionary format."""
        features = {}
        flat = torch.cat([eeg_features[0], ecg_features[0]], dim=0).tolist()
        for name, value in zip(self.feature_names, flat):
            features[name] = float(value)
        return features

    def process_sample(self, eeg_features: torch.Tensor, ecg_features: torch.Tensor) -> Dict[str, Any]:
        """
        Process a single sample through the complete pipeline.

        Args:
            eeg_features: EEG features tensor of shape (1, 97) or (97,)
            ecg_features: ECG features tensor of shape (1, 4) or (4,)

        Returns:
            Complete stress management plan with structured output
        """
        # Ensure tensors are on correct device and have batch dimension
        if eeg_features.dim() == 1:
            eeg_features = eeg_features.unsqueeze(0)
        if ecg_features.dim() == 1:
            ecg_features = ecg_features.unsqueeze(0)

        eeg_features = eeg_features.to(self.device)
        ecg_features = ecg_features.to(self.device)

        # Step 1: Get model predictions and features
        print("Step 1: Analyzing physiological data with transformer model...")
        model_output = self.predict_stress(eeg_features, ecg_features)

        # Step 2: Build context using Groq
        print("Step 2: Building stress context with AI analysis...")
        context_string = build_context(
            stress_level=model_output['stress_level'],
            stress_probability=model_output['stress_probability'],
            top_shap_features=model_output['top_shap_features'],
            physiological_metrics=model_output['physiological_metrics'],
            all_features=model_output['all_features']
        )

        # Step 3: Retrieve relevant techniques from RAG
        print("Step 3: Retrieving relevant stress management techniques...")
        retrieved = self.planner.retrieve_techniques(context_string, n_results=5)

        # Step 4: Generate structured management plan using Groq
        print("Step 4: Generating personalized management plan...")
        plan_result = self.planner.generate_management_plan(context_string, retrieved)

        # Step 5: Format final output
        final_output = {
            "model_analysis": model_output,
            "generated_context": context_string,
            "retrieved_techniques": {
                "count": len(retrieved['metadatas'][0]),
                "techniques": [
                    {
                        "name": meta['technique_name'],
                        "description": meta['description'][:100] + "...",
                        "applicable_conditions": meta['applicable_conditions'],
                        "duration": meta['duration']
                    }
                    for meta in retrieved['metadatas'][0]
                ]
            },
            "management_plan": plan_result.get('plan', plan_result.get('error', 'Plan generation failed'))
        }

        return final_output

    def process_batch(self, eeg_batch: torch.Tensor, ecg_batch: torch.Tensor) -> List[Dict[str, Any]]:
        """
        Process a batch of samples through the pipeline.

        Args:
            eeg_batch: Batch of EEG features (batch_size, 97)
            ecg_batch: Batch of ECG features (batch_size, 4)

        Returns:
            List of results for each sample in the batch
        """
        results = []
        batch_size = eeg_batch.shape[0]

        for i in range(batch_size):
            eeg_sample = eeg_batch[i:i+1]  # Shape: (1, 97)
            ecg_sample = ecg_batch[i:i+1]  # Shape: (1, 4)

            result = self.process_sample(eeg_sample, ecg_sample)
            results.append(result)

            print(f"Processed sample {i+1}/{batch_size}")

        return results


# Example usage and testing
if __name__ == "__main__":
    print("=" * 80)
    print("INTEGRATED STRESS MANAGEMENT PIPELINE TEST")
    print("=" * 80)

    # Initialize pipeline
    pipeline = IntegratedStressPipeline()

    # Create sample physiological data
    eeg_sample = torch.randn(1, 97)  # Mock EEG features
    ecg_sample = torch.randn(1, 4)   # Mock ECG features

    print("\nProcessing sample through complete pipeline...")
    print("(This will take a few moments due to LLM API calls)")
    print("-" * 60)

    # Process through pipeline
    result = pipeline.process_sample(eeg_sample, ecg_sample)

    # Display results
    print("\n" + "=" * 80)
    print("PIPELINE RESULTS")
    print("=" * 80)

    print(f"Stress Level: {result['model_analysis']['stress_level']}")
    print(f"Confidence: {result['model_analysis']['stress_probability']:.3f}")
    print(f"Top Features: {result['model_analysis']['top_shap_features']}")

    print(f"\nRetrieved {result['retrieved_techniques']['count']} techniques:")
    for tech in result['retrieved_techniques']['techniques']:
        print(f"  - {tech['name']} ({tech['duration']})")

    print("\nGenerated Context Preview:")
    print(f"  {result['generated_context'][:150]}...")

    if 'stress_level' in result['management_plan']:
        print("\nStructured Management Plan:")
        plan = result['management_plan']
        print(f"  Stress Level: {plan['stress_level']}")
        print(f"  Cause: {plan['cause']}")
        print(f"  Physiological: {plan['physiological_interpretation']}")
        print(f"  Recommended Actions: {len(plan['recommended_actions'])} techniques")
        for i, action in enumerate(plan['recommended_actions'][:2], 1):
            print(f"    {i}. {action['technique']} ({action['duration']})")
    else:
        print(f"\nPlan Generation Issue: {result['management_plan']}")

    print("\n" + "=" * 80)
    print("PIPELINE EXECUTION COMPLETE")
    print("=" * 80)