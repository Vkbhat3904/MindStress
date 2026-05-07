# Stress Management System - Implementation Guide

## Overview

This system integrates multiple components to generate personalized stress management plans:

1. **Context Builder** (`context_builder.py`) - Converts model outputs into textual context
2. **RAG System** (`rag_system.py`) - Retrieves relevant techniques from knowledge base
3. **Management Planner** (`stress_management_planner.py`) - Generates structured plans using Groq
4. **Integrated System** (`integrated_system.py`) - End-to-end orchestration

## Quick Start

### 1. Setup

```bash
# Set your Groq API key in .env
GROQ_API_KEY=your_groq_api_key_here
```

### 2. Basic Usage

```python
from integrated_system import StressManagementSystem

system = StressManagementSystem()

# Prepare your model outputs
stress_data = {
    "stress_level": "high",
    "stress_probability": 0.85,
    "top_shap_features": ["lf_hf_ratio", "beta_power", "heart_rate"],
    "physiological_metrics": {
        "lf_hf_ratio": 2.5,
        "beta_power": 15.3,
        "faa": -0.2,
        "heart_rate": 95
    },
    "all_features": {...}  # All 101 features
}

# Generate complete plan
plan = system.generate_complete_plan(**stress_data)

# Save to file
system.save_plan(plan, "my_stress_plan.json")
```

## Component Details

### Context Builder (`context_builder.py`)

**Function:** `build_context()`

Converts raw model outputs into a comprehensive textual description using Groq.

**Input:**
- `stress_level`: Classification (very_low, low, moderate, high, very_high)
- `stress_probability`: Score 0-1
- `top_shap_features`: List of important features
- `physiological_metrics`: Dict of key metrics
- `all_features`: All 101 features from model

**Output:**
- Textual context string describing stress condition and causes

**Example Output:**
```
"High stress with elevated LF/HF ratio (2.5) indicating sympathetic activation. 
Increased beta power (15.3) suggests cognitive overload. Reduced heart rate 
variability indicates reduced parasympathetic tone..."
```

### RAG System (`rag_system.py`)

**Class:** `StressRAGSystem`

**Key Methods:**
- `load_and_embed_knowledge_base()` - Loads knowledge base into ChromaDB
- `search_similar(query, n_results=3)` - Retrieves top N similar techniques

**How it works:**
1. Encodes context string using SentenceTransformer
2. Compares with embedded knowledge base entries
3. Returns top matching techniques with metadata

### Management Planner (`stress_management_planner.py`)

**Class:** `StressManagementPlanner`

**Key Methods:**

#### `retrieve_techniques(context_string, n_results=3)`
- Takes: Context description
- Returns: Top N most relevant techniques with metadata

#### `generate_management_plan(stress_context, retrieved_techniques)`
- Takes: Stress context + retrieved techniques
- Returns: Structured JSON plan with:

**Plan Structure:**
```json
{
  "stress_cause_explanation": "Why the stress is happening",
  "physiological_reasoning": "How metrics relate to stress",
  "recommended_interventions": ["Technique 1", "Technique 2", ...],
  "primary_technique": "Most important intervention",
  "action_plan": [
    "1. First step",
    "2. Second step",
    "3. Third step",
    ...
  ],
  "implementation_timeline": "When to do each intervention",
  "expected_outcomes": "What to expect",
  "contraindications": "When NOT to use these",
  "progress_monitoring": "How to track improvement"
}
```

### Integrated System (`integrated_system.py`)

**Class:** `StressManagementSystem`

**Main Method:** `generate_complete_plan()`

Orchestrates the entire pipeline:
1. → Context Builder (model outputs → textual context)
2. → RAG System (context → retrieve techniques)
3. → Management Planner (context + techniques → structured plan)

**Output:** Complete JSON with all components

## Data Flow

```
Model Outputs (stress_level, features, metrics)
    ↓
[Context Builder + Groq]
    ↓
Textual Context Description
    ↓
[RAG System + ChromaDB]
    ↓
Top 3 Techniques + Metadata
    ↓
[Management Planner + Groq]
    ↓
Structured Stress Management Plan
    ↓
JSON Output + Actionable Instructions
```

## Example Output Structure

```json
{
  "status": "success",
  "input_data": {
    "stress_level": "high",
    "stress_probability": 0.85,
    "top_shap_features": ["lf_hf_ratio", "beta_power"],
    "physiological_metrics": {...}
  },
  "generated_context": "High stress with elevated LF/HF ratio...",
  "retrieved_techniques": {
    "count": 3,
    "techniques": [
      {
        "name": "Deep Breathing Exercise",
        "description": "...",
        "applicable_conditions": ["high LF/HF ratio"],
        "stress_levels": ["moderate", "high", "very_high"],
        "duration": "5-10 minutes"
      },
      ...
    ]
  },
  "management_plan": {
    "stress_cause_explanation": "...",
    "physiological_reasoning": "...",
    "recommended_interventions": [...],
    "action_plan": [
      "1. First step",
      "2. Second step",
      ...
    ],
    ...
  }
}
```

## Integration with Your Model

### With Transformer Model

```python
from transformer_model import StressTransformer
from integrated_system import StressManagementSystem

# Get model predictions
model = StressTransformer()
predictions = model.predict(eeg_data)

# Extract outputs
stress_level = predictions['stress_level']
stress_probability = predictions['probability']
shap_values = predictions['shap_features']
physiological_metrics = predictions['metrics']
all_features = predictions['all_101_features']

# Generate plan
system = StressManagementSystem()
plan = system.generate_complete_plan(
    stress_level=stress_level,
    stress_probability=stress_probability,
    top_shap_features=shap_values,
    physiological_metrics=physiological_metrics,
    all_features=all_features
)
```

### With API

```python
from api import app
from integrated_system import StressManagementSystem

@app.route('/generate-plan', methods=['POST'])
def generate_plan():
    data = request.json
    system = StressManagementSystem()
    plan = system.generate_complete_plan(**data)
    return jsonify(plan)
```

## Customization

### Modify Number of Retrieved Techniques
```python
planner.retrieve_techniques(context, n_results=5)  # Get top 5 instead of 3
```

### Use Different Embedding Model
Edit `rag_system.py`:
```python
self.model = SentenceTransformer('your-model-name')
```

### Change Groq Model
Edit `stress_management_planner.py` or `context_builder.py`:
```python
model='llama-3.3-70b-versatile'  # Use latest stable Groq model
```

## Troubleshooting

### "API Key Error"
- Verify `.env` file has correct key
- Check key is active in Groq console

### "ChromaDB Connection Error"
- Ensure `./chroma_db` directory exists and is writable
- Check disk space availability

### "Incomplete Knowledge Base"
- Run `rag_system.py` to reinitialize ChromaDB
- Verify `stress_knowledge_base.json` exists and is valid

## Performance Notes

- First run initializes embeddings (~30-60 seconds)
- Subsequent searches are fast (<1 second)
- Groq API calls are generally very fast (~1-3 seconds)
- Total end-to-end time: ~10-15 seconds

## Output Files

- `stress_management_plan.json` - Generated plan
- `./chroma_db/` - ChromaDB vector database
- `./test_output.txt` - Test logs (if running tests)
