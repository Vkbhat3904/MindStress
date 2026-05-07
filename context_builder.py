import os
import json
from dotenv import load_dotenv
from groq import Groq
from explainability import generate_feature_names

# Load environment variables
load_dotenv()

def _safe_groq_client():
    """Create Groq client only when API key is available."""
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        return None
    try:
        return Groq(api_key=api_key)
    except Exception:
        return None

def build_context(stress_level, stress_probability, top_shap_features, physiological_metrics, all_features):
    """
    Converts model outputs into a textual context for RAG system using Groq.

    Args:
        stress_level (str): One of ['very_low', 'low', 'moderate', 'high', 'very_high']
        stress_probability (float): Probability score for the stress level
        top_shap_features (list or dict): Top contributing features from SHAP analysis
        physiological_metrics (dict): Key metrics like {'lf_hf_ratio': value, 'beta_power': value, 'faa': value}
        all_features (dict or list): All 101 features as a dictionary or list

    Returns:
        str: Textual context describing the stress condition, causes, and management suggestions
    """

    # Ensure all 101 features are explicitly represented with canonical names.
    canonical_names = generate_feature_names()
    features_payload = all_features
    if isinstance(all_features, list):
        if len(all_features) == len(canonical_names):
            features_payload = {
                name: float(value) for name, value in zip(canonical_names, all_features)
            }
        else:
            features_payload = {f"feature_{i}": float(v) for i, v in enumerate(all_features)}
    elif isinstance(all_features, dict):
        # If caller sent generic keys (feature_0, eeg_feature_0, ecg_feature_0),
        # convert to canonical feature names so the LLM sees true signal semantics.
        mapped = {}
        for key, value in all_features.items():
            idx = None
            if key.startswith("feature_"):
                try:
                    idx = int(key.split("_")[-1])
                except ValueError:
                    idx = None
            elif key.startswith("eeg_feature_"):
                try:
                    idx = int(key.split("_")[-1])
                except ValueError:
                    idx = None
            elif key.startswith("ecg_feature_"):
                try:
                    idx = 97 + int(key.split("_")[-1])
                except ValueError:
                    idx = None

            if idx is not None and 0 <= idx < len(canonical_names):
                mapped[canonical_names[idx]] = float(value)
            else:
                mapped[key] = value
        features_payload = mapped
    else:
        features_payload = {"raw_features": str(all_features)}

    # Prepare the prompt
    prompt = f"""
    Based on the following stress analysis data, generate a comprehensive textual description of the individual's stress condition.
    Include:

    STRESS ANALYSIS DATA:
    - Stress Level: {stress_level}
    - Stress Probability: {stress_probability:.3f}
    - Top Contributing Features: {', '.join(top_shap_features) if isinstance(top_shap_features, list) else str(top_shap_features)}
    - Key Physiological Metrics: {json.dumps(physiological_metrics, indent=2)}
    - Full Feature Vector (all 101 features): {json.dumps(features_payload, indent=2)}

    Please provide a detailed analysis covering:
    1. Current stress condition interpretation
    2. Physiological indicators and their significance
    3. Potential causes of the stress level
    4. Immediate concerns or observations
    5. General recommendations for stress management
    6. How the full feature vector supports the assessment

    Keep the response focused and evidence-based. Structure it as a coherent paragraph suitable for use in a RAG system.
    """

    try:
        client = _safe_groq_client()
        if client is None:
            return build_context_fallback(
                stress_level,
                stress_probability,
                top_shap_features,
                physiological_metrics,
                all_features,
            )

        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",  # Use latest stable Groq model
            messages=[
                {"role": "system", "content": "You are an expert psychologist and physiologist analyzing stress biomarkers from EEG and ECG data."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=1000,
            temperature=0.3
        )

        context = response.choices[0].message.content.strip()
        return context

    except Exception as e:
        return f"Error generating context with Groq: {str(e)}. Please check your GROQ_API_KEY and connection."


def build_context_fallback(stress_level, stress_probability, top_shap_features, physiological_metrics, all_features):
    """
    Fallback context builder when API is unavailable - generates basic context from data.
    """
    total_features = len(all_features) if isinstance(all_features, (list, dict)) else 0
    context = f"""
    Current stress assessment indicates {stress_level} stress level with {stress_probability:.1%} confidence.
    Key physiological indicators include: {', '.join([f'{k}: {v:.2f}' for k, v in physiological_metrics.items()])}.
    Top contributing factors are: {', '.join(top_shap_features) if isinstance(top_shap_features, list) else str(top_shap_features)}.
    Full multimodal input included {total_features} features for this assessment.
    This suggests {'high sympathetic activation' if physiological_metrics.get('lf_hf_ratio', 1) > 2 else 'moderate autonomic response'}.
    """

    return context.strip()

# Example usage (for testing)
if __name__ == "__main__":
    # Sample inputs
    sample_stress_level = "high"
    sample_probability = 0.85
    sample_shap = ["lf_hf_ratio", "beta_power", "heart_rate_variability"]
    sample_metrics = {"lf_hf_ratio": 2.5, "beta_power": 15.3, "faa": -0.2}
    sample_features = {"feature_1": 0.5, "feature_2": 1.2, "feature_3": -0.8}  # Placeholder for 101 features

    context = build_context(
        stress_level=sample_stress_level,
        stress_probability=sample_probability,
        top_shap_features=sample_shap,
        physiological_metrics=sample_metrics,
        all_features=sample_features
    )

    print("Generated Context:")
    print(context)