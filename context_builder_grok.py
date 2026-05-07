import os
import json
from groq import Groq

# Initialize Grok client
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

def build_context(stress_level, stress_probability, top_shap_features, physiological_metrics, all_features):
    """
    Converts model outputs into a textual context for RAG system using Grok.

    Args:
        stress_level (str): One of ['very_low', 'low', 'moderate', 'high', 'very_high']
        stress_probability (float): Probability score for the stress level
        top_shap_features (list or dict): Top contributing features from SHAP analysis
        physiological_metrics (dict): Key metrics like {'lf_hf_ratio': value, 'beta_power': value, 'faa': value}
        all_features (dict or list): All 101 features as a dictionary or list

    Returns:
        str: Textual context describing the stress condition, causes, and management suggestions
    """

    # Prepare the prompt
    prompt = f"""
    Based on the following stress analysis data, generate a comprehensive textual description of the individual's stress condition.
    Include:

    STRESS ANALYSIS DATA:
    - Stress Level: {stress_level}
    - Stress Probability: {stress_probability:.3f}
    - Top Contributing Features: {', '.join(top_shap_features) if isinstance(top_shap_features, list) else str(top_shap_features)}
    - Key Physiological Metrics: {json.dumps(physiological_metrics, indent=2)}

    Please provide a detailed analysis covering:
    1. Current stress condition interpretation
    2. Physiological indicators and their significance
    3. Potential causes of the stress level
    4. Immediate concerns or observations
    5. General recommendations for stress management

    Keep the response focused and evidence-based. Structure it as a coherent paragraph suitable for use in a RAG system.
    """

    try:
        response = client.chat.completions.create(
            model="grok-beta",  # or "grok-2-1212" for latest
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
        return f"Error generating context with Grok: {str(e)}. Please check your GROQ_API_KEY and connection."


def build_context_fallback(stress_level, stress_probability, top_shap_features, physiological_metrics, all_features):
    """
    Fallback context builder when API is unavailable - generates basic context from data.
    """
    context = f"""
    Current stress assessment indicates {stress_level} stress level with {stress_probability:.1%} confidence.
    Key physiological indicators include: {', '.join([f'{k}: {v:.2f}' for k, v in physiological_metrics.items()])}.
    Top contributing factors are: {', '.join(top_shap_features) if isinstance(top_shap_features, list) else str(top_shap_features)}.
    This suggests {'high sympathetic activation' if physiological_metrics.get('lf_hf_ratio', 1) > 2 else 'moderate autonomic response'}.
    """

    return context.strip()