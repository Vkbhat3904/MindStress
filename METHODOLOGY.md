# Methodology

## 1. Study Objective

This project presents an end-to-end multimodal stress assessment and intervention framework that integrates:

- physiological stress prediction from EEG and ECG-derived features,
- model explainability for transparent interpretation,
- retrieval-augmented generation (RAG) using an evidence-based stress knowledge base,
- and large language model (LLM)-driven personalized stress management planning.

The target output is a clinically informed, actionable, and personalized stress management plan conditioned on predicted stress severity, physiological indicators, and retrieved intervention evidence.

## 2. System Overview

The proposed pipeline is implemented as a sequential architecture:

1. **Feature input (101-dimensional):** 97 EEG features + 4 ECG/HRV features  
2. **Multimodal Transformer inference:** 5-class stress prediction (`very_low`, `low`, `moderate`, `high`, `very_high`)  
3. **Explainability layer:** attention weights + permutation-based SHAP-style feature attribution  
4. **Context synthesis:** physiological and explainability signals transformed into structured stress context  
5. **RAG retrieval:** top relevant stress-management evidence retrieved from ChromaDB  
6. **LLM planning (Groq):** personalized intervention plan generation constrained to evidence-based guidance  
7. **Safety fallback:** deterministic plan path when LLM is unavailable

## 3. Data Representation and Features

### 3.1 Feature Space

The model operates on a fixed-length vector of 101 engineered features:

- **EEG (97 features):**
  - 32 channels x 3 spectral bands (Theta, Alpha, Beta) = 96 features
  - 1 frontal alpha asymmetry feature
- **ECG/HRV (4 features):**
  - mean heart rate,
  - SDNN,
  - RMSSD,
  - LF/HF ratio.

### 3.2 Feature Naming and Semantics

All features are mapped to canonical names for interpretability and LLM personalization (e.g., `EEG_Ch0_Theta`, `Frontal_Alpha_Asymmetry`, `LF_HF_Ratio`).  
This naming is consistently propagated through prediction, explainability, context building, and management planning.

## 4. Multimodal Stress Classification Model

### 4.1 Model Architecture

A multimodal transformer encoder is used with separate EEG and ECG branches merged through tokenized fusion:

- EEG feature embedding,
- ECG feature embedding,
- learned `[CLS]` token for global representation,
- multi-head self-attention encoder layers,
- classification head producing logits for 5 stress classes.

### 4.2 Training/Inference Setting

The system loads pre-trained weights (`best_model.pth`) for inference and computes softmax probabilities from model logits.  
Predicted class probability is reported as stress confidence.

## 5. Explainability Strategy

Two complementary explainability signals are generated:

1. **Cross-modal attention analysis:**  
   attention from the `[CLS]` token to EEG and ECG tokens indicates relative modality contribution.

2. **Permutation-based SHAP-style attribution:**  
   each input feature is ablated (set to zero) to estimate contribution via class-probability drop; normalized absolute effects provide per-feature importance scores.

These outputs serve both interpretability and downstream personalization inputs.

## 6. Context Construction for Personalized Planning

A context-builder module aggregates:

- stress level + probability,
- top explainability features,
- physiological summary metrics,
- full named 101-feature vector.

This context is provided to the LLM as a structured narrative input to support individualized intervention synthesis.

## 7. Retrieval-Augmented Knowledge Grounding

### 7.1 Knowledge Sources

The intervention knowledge base includes curated and WHO-derived stress management content aligned with CBT, MBSR, and public-health guidance.

### 7.2 Vector Store and Retrieval

- embeddings are created using `all-MiniLM-L6-v2`,
- content is stored in persistent ChromaDB,
- semantically similar techniques are retrieved (`top-k`) using context query embeddings.

Retrieved metadata includes technique descriptions, applicable conditions, stress-level mappings, durations, and structured steps.

## 8. LLM-Based Plan Generation

Groq-hosted LLM inference is used to generate structured personalized plans under strict JSON schema constraints.  
Prompt constraints enforce:

- evidence grounding in retrieved knowledge,
- contextual selection (not exhaustive listing),
- stress severity adaptation,
- practical time-bound steps,
- supportive language and safety boundaries.

### Output Schema

The generated plan includes:

- `stress_level`,
- `cause`,
- `physiological_interpretation`,
- `recommended_actions` (technique, steps, duration).

## 9. Reliability and Safety Controls

To maintain service continuity:

- when Groq is unavailable, deterministic fallback context/plan logic is triggered,
- response metadata records plan origin (`groq` or `fallback`),
- retrieval and intervention outputs remain bounded to safe, non-diagnostic, general-purpose support guidance.

## 10. API and Deployment Workflow

The framework is exposed through a FastAPI interface:

- `/predict`: stress class and probabilities,
- `/explain`: prediction + attention + feature attributions,
- `/stress-management`: RAG-grounded personalized plan generation.

This enables reproducible integration with frontend clinical dashboards or research evaluation scripts.

## 11. Reproducibility Notes for Manuscript

For publication, report:

- model dimensions and class schema (5-class stress taxonomy),
- exact feature composition (97 EEG + 4 ECG),
- explainability protocol (attention + permutation ablation),
- embedding model and vector store configuration,
- retrieval depth (`k`) and prompt constraints,
- fallback policy and response source tracking.

These details ensure methodological transparency and support reproducibility across independent validation settings.

