import os
import json
from dotenv import load_dotenv
from groq import Groq
from rag_system import StressRAGSystem

# Load environment variables
load_dotenv()

def _safe_groq_client():
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        return None
    try:
        return Groq(api_key=api_key)
    except Exception:
        return None

class StressManagementPlanner:
    def __init__(self, rag_system=None):
        """
        Initialize the planner with RAG system for knowledge base retrieval.
        
        Args:
            rag_system (StressRAGSystem): Instance of RAG system. If None, creates a new one.
        """
        self.rag_system = rag_system or StressRAGSystem()

    def retrieve_techniques(self, context_string, n_results=5):
        """
        Retrieve top N most relevant stress management techniques based on context.
        
        Args:
            context_string (str): Textual description of stress condition
            n_results (int): Number of techniques to retrieve (default: 3)
        
        Returns:
            dict: Retrieved techniques with metadata
        """
        results = self.rag_system.search_similar(context_string, n_results=n_results)
        return results

    def generate_management_plan(self, stress_context, retrieved_techniques):
        """
        Generate a comprehensive stress management plan using Groq LLM.
        
        Args:
            stress_context (str): Context describing the stress condition (from context_builder)
            retrieved_techniques (dict): Retrieved techniques from ChromaDB
        
        Returns:
            dict: Structured management plan with explanations and instructions
        """
        
        # Format retrieved techniques for the prompt
        techniques_text = self._format_techniques(retrieved_techniques)
        
        prompt = f"""You are an AI Stress Management Assistant based on clinical and evidence-based guidelines (CBT, MBSR, WHO, NIH) and a ChromaDB-backed retrieval system.

Your role is to analyze stress predictions derived from EEG/ECG physiological signals and all 101 engineered features, then generate a personalized and context-aware stress management plan.

You will be provided with:
- stress level (very_low, low, moderate, high, very_high)
- stress probability (0 to 1)
- physiological indicators and full 101-feature context
- top explainable AI features (SHAP-style output)
- retrieved evidence-based knowledge from WHO/CBT/MBSR/NIH via ChromaDB

Your task:
1) Interpret the stress condition and explain what the stress level means.
2) Determine whether stress appears primarily physiological, cognitive, emotional, or mixed.
3) Analyze physiological signals (e.g., LF/HF ratio, beta power, EEG asymmetry) with short clinical-style interpretation.
4) Select only the most suitable interventions from retrieved knowledge (do not list everything).
5) Generate personalized, practical, time-bound, step-by-step actions.
6) Use supportive, human-friendly language (not robotic).
7) Keep guidance safe and general-purpose; do not provide medical diagnosis or harmful advice.

Do NOT:
- blindly repeat retrieved passages
- output raw database entries
- provide generic advice like "just relax"

Based on the following stress analysis and retrieved knowledge, generate a comprehensive personalized plan grounded in retrieved evidence and tailored to the individual's stress profile.

STRESS CONTEXT:
{stress_context}

RETRIEVED RELEVANT TECHNIQUES:
{techniques_text}

Please structure your response as a JSON object with the following EXACT fields:
{{
  "stress_level": "very_low|low|moderate|high|very_high",
  "cause": "Brief explanation of what is causing the stress based on physiological signals and context",
  "physiological_interpretation": "How the physiological metrics relate to the stress state",
  "recommended_actions": [
    {{
      "technique": "Name of the technique",
      "steps": "Step-by-step instructions for performing the technique",
      "duration": "Expected duration for the technique"
    }}
  ]
}}

Ensure the response is valid JSON only, no additional text or formatting. Focus on the most relevant 3-5 techniques from retrieved knowledge, prioritizing a practical mix that matches stress severity and SHAP-highlighted drivers."""

        try:
            client = _safe_groq_client()
            if client is None:
                return {
                    "success": True,
                    "source": "fallback",
                    "plan": self._fallback_plan(stress_context, retrieved_techniques),
                }

            response = client.chat.completions.create(
                model="llama-3.3-70b-versatile",  # Use latest stable Groq model
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are an AI Stress Management Assistant. "
                            "Use CBT/MBSR/WHO/NIH-aligned reasoning with retrieved ChromaDB evidence. "
                            "Produce supportive, safe, personalized, actionable guidance. "
                            "Do not diagnose. Do not provide harmful/extreme advice. "
                            "Return valid JSON only in the requested schema."
                        ),
                    },
                    {"role": "user", "content": prompt}
                ],
                max_tokens=1500,
                temperature=0.3
            )
            
            # Parse the response as JSON
            response_text = response.choices[0].message.content.strip()
            
            # Extract JSON from the response (handle cases where LLM includes markdown formatting)
            if response_text.startswith('```json'):
                response_text = response_text[7:]
            if response_text.startswith('```'):
                response_text = response_text[3:]
            if response_text.endswith('```'):
                response_text = response_text[:-3]
            
            response_text = response_text.strip()
            plan = json.loads(response_text)
            
            return {
                "success": True,
                "source": "groq",
                "plan": plan
            }
        
        except json.JSONDecodeError as e:
            return {
                "success": False,
                "source": "groq",
                "error": f"Failed to parse LLM response as JSON: {str(e)}",
                "raw_response": response_text if 'response_text' in locals() else "N/A"
            }
        except Exception as e:
            return {
                "success": True,
                "source": "fallback",
                "plan": self._fallback_plan(stress_context, retrieved_techniques),
                "warning": f"LLM unavailable, used deterministic fallback: {str(e)}"
            }

    def _format_techniques(self, retrieved_techniques):
        """Format retrieved techniques for display in prompt."""
        formatted = []
        
        if 'metadatas' in retrieved_techniques and retrieved_techniques['metadatas']:
            for i, metadata in enumerate(retrieved_techniques['metadatas'][0], 1):
                src = metadata.get('source', '')
                src_note = f" (source: {src})" if src else ""
                ac = metadata.get('applicable_conditions', [])
                if isinstance(ac, str):
                    ac = [ac]
                sl = metadata.get('stress_level_mapping', [])
                if isinstance(sl, str):
                    sl = [sl]
                steps = metadata.get('steps_to_perform', [])
                if isinstance(steps, str):
                    steps = [steps]
                technique_text = f"""
Technique {i}: {metadata.get('technique_name', 'Unknown')}{src_note}
- Description: {metadata.get('description', '')}
- Applicable Conditions: {', '.join(ac)}
- Stress Levels: {', '.join(sl)}
- Duration: {metadata.get('duration', '')}
- Steps: {'; '.join(steps)}
"""
                formatted.append(technique_text)
        
        return "\n".join(formatted)

    def _fallback_plan(self, stress_context, retrieved_techniques):
        """Generate a deterministic plan when LLM is unavailable."""
        level = "moderate"
        for candidate in ["very_high", "high", "moderate", "low", "very_low"]:
            if candidate in str(stress_context).lower():
                level = candidate
                break

        actions = []
        if 'metadatas' in retrieved_techniques and retrieved_techniques.get('metadatas'):
            for metadata in retrieved_techniques['metadatas'][0][:5]:
                steps = metadata.get("steps_to_perform", [])
                if isinstance(steps, list):
                    steps_text = " ".join(str(s) for s in steps[:4])
                else:
                    steps_text = str(steps)
                actions.append(
                    {
                        "technique": metadata.get("technique_name", "Stress management technique"),
                        "steps": steps_text or metadata.get("description", "Follow guided practice carefully."),
                        "duration": metadata.get("duration", "10-15 minutes"),
                    }
                )

        if not actions:
            actions = [
                {
                    "technique": "Slow-paced breathing",
                    "steps": "Inhale for 4 seconds, exhale for 6 seconds, repeat for 10 cycles.",
                    "duration": "5-10 minutes",
                },
                {
                    "technique": "Body relaxation scan",
                    "steps": "Relax each muscle group from feet to head while breathing slowly.",
                    "duration": "10 minutes",
                },
                {
                    "technique": "Structured journaling",
                    "steps": "Write key stress triggers, controllable actions, and one immediate next step.",
                    "duration": "10-15 minutes",
                },
            ]

        return {
            "stress_level": level,
            "cause": "Stress profile suggests autonomic and cognitive load imbalance based on provided context.",
            "physiological_interpretation": "Patterns in physiological and explainability signals indicate stress-system activation and reduced recovery balance.",
            "recommended_actions": actions,
        }

    def generate_full_plan(self, stress_level, stress_probability, top_shap_features, 
                          physiological_metrics, all_features, context_string):
        """
        End-to-end function to generate a complete stress management plan.
        
        Args:
            stress_level (str): One of ['very_low', 'low', 'moderate', 'high', 'very_high']
            stress_probability (float): Probability score
            top_shap_features (list): Top SHAP features
            physiological_metrics (dict): Physiological metrics
            all_features (dict/list): All 101 features
            context_string (str): Pre-generated context from context_builder
        
        Returns:
            dict: Complete comprehensive plan
        """
        
        # Retrieve relevant techniques
        retrieved = self.retrieve_techniques(context_string, n_results=5)
        
        # Generate management plan
        plan = self.generate_management_plan(context_string, retrieved)
        
        return {
            "stress_profile": {
                "level": stress_level,
                "probability": stress_probability,
                "top_features": top_shap_features,
                "physiological_metrics": physiological_metrics
            },
            "context": context_string,
            "retrieved_techniques": retrieved,
            "management_plan": plan
        }


# Example usage
if __name__ == "__main__":
    planner = StressManagementPlanner()
    
    # Sample context
    sample_context = """
    High stress with elevated LF/HF ratio (2.5) indicating sympathetic activation. 
    Increased beta power (15.3) suggests cognitive overload. 
    Reduced heart rate variability indicates reduced parasympathetic tone. 
    Individual is experiencing work-related stress with poor sleep quality.
    """
    
    # Retrieve techniques
    print("=" * 80)
    print("RETRIEVING RELEVANT TECHNIQUES...")
    print("=" * 80)
    retrieved = planner.retrieve_techniques(sample_context)
    print("\nRetrieved Techniques:")
    for i, meta in enumerate(retrieved['metadatas'][0], 1):
        print(f"{i}. {meta['technique_name']}")
    
    # Generate management plan
    print("\n" + "=" * 80)
    print("GENERATING STRESS MANAGEMENT PLAN...")
    print("=" * 80)
    plan_result = planner.generate_management_plan(sample_context, retrieved)
    
    if plan_result['success']:
        print("\nGenerated Plan:")
        print(json.dumps(plan_result['plan'], indent=2))
    else:
        print(f"\nError: {plan_result['error']}")
        print(f"Raw response: {plan_result.get('raw_response', 'N/A')}")
