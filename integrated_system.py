import os
import json
from dotenv import load_dotenv
from context_builder import build_context
from stress_management_planner import StressManagementPlanner

# Load environment variables
load_dotenv()

class StressManagementSystem:
    """
    End-to-end system integrating context building, RAG retrieval, and LLM-based planning.
    """
    
    def __init__(self):
        self.planner = StressManagementPlanner()
    
    def generate_complete_plan(self, stress_level, stress_probability, top_shap_features,
                               physiological_metrics, all_features):
        """
        Generate a complete stress management plan from raw model outputs.
        
        Args:
            stress_level (str): One of ['very_low', 'low', 'moderate', 'high', 'very_high']
            stress_probability (float): Probability score (0-1)
            top_shap_features (list): Top contributing features
            physiological_metrics (dict): Key metrics (lf_hf_ratio, beta_power, faa, etc.)
            all_features (dict/list): All 101 features
        
        Returns:
            dict: Comprehensive stress management plan
        """
        
        # Step 1: Build context using Groq
        print("Step 1: Building stress context from model outputs...")
        context_string = build_context(
            stress_level=stress_level,
            stress_probability=stress_probability,
            top_shap_features=top_shap_features,
            physiological_metrics=physiological_metrics,
            all_features=all_features
        )
        print(f"Context generated: {context_string[:100]}...")
        
        # Step 2: Retrieve relevant techniques from RAG
        print("\nStep 2: Retrieving relevant techniques from knowledge base...")
        retrieved = self.planner.retrieve_techniques(context_string, n_results=5)
        print(f"Retrieved {len(retrieved['metadatas'][0])} techniques")
        
        # Step 3: Generate management plan using Groq
        print("\nStep 3: Generating personalized management plan...")
        plan_result = self.planner.generate_management_plan(context_string, retrieved)
        
        # Compile final output
        complete_plan = {
            "status": "success" if plan_result['success'] else "error",
            "plan_source": plan_result.get("source", "unknown"),
            "input_data": {
                "stress_level": stress_level,
                "stress_probability": stress_probability,
                "top_shap_features": top_shap_features,
                "physiological_metrics": physiological_metrics
            },
            "generated_context": context_string,
            "retrieved_techniques": {
                "count": len(retrieved['metadatas'][0]),
                "techniques": [
                    {
                        "name": meta['technique_name'],
                        "description": meta['description'],
                        "applicable_conditions": meta['applicable_conditions'],
                        "stress_levels": meta['stress_level_mapping'],
                        "duration": meta['duration']
                    }
                    for meta in retrieved['metadatas'][0]
                ]
            },
            "management_plan": plan_result.get('plan', plan_result.get('error', 'Unknown error'))
        }
        
        return complete_plan
    
    def save_plan(self, plan, output_file="stress_management_plan.json"):
        """Save the generated plan to a JSON file."""
        with open(output_file, 'w') as f:
            json.dump(plan, f, indent=2)
        print(f"\nPlan saved to {output_file}")


# Example usage
if __name__ == "__main__":
    system = StressManagementSystem()
    
    # Sample inputs (simulating model outputs)
    sample_input = {
        "stress_level": "high",
        "stress_probability": 0.85,
        "top_shap_features": ["lf_hf_ratio", "beta_power", "heart_rate", "cortisol", "sleep_quality"],
        "physiological_metrics": {
            "lf_hf_ratio": 2.5,
            "beta_power": 15.3,
            "faa": -0.2,
            "heart_rate": 95,
            "cortisol": 450,
            "sleep_quality": 3.5
        },
        "all_features": {
            "feature_1": 0.5,
            "feature_2": 1.2,
            "feature_3": -0.8,
            # ... (in real scenario, this would contain all 101 features)
        }
    }
    
    print("=" * 80)
    print("STRESS MANAGEMENT SYSTEM - END-TO-END EXAMPLE")
    print("=" * 80)
    
    # Generate complete plan
    plan = system.generate_complete_plan(**sample_input)
    
    # Save and display
    system.save_plan(plan)
    print("\n" + "=" * 80)
    print("COMPLETE PLAN SUMMARY")
    print("=" * 80)
    print(json.dumps(plan, indent=2))
