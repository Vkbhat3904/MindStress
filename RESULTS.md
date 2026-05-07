# Results

## 1. Evaluation Scope

This section reports implementation-level and system-level outcomes for the proposed multimodal stress management framework.  
The evaluation focuses on:

- functional correctness of end-to-end pipeline execution,
- quality and interpretability of stress prediction outputs,
- retrieval grounding behavior from ChromaDB,
- personalized intervention generation behavior,
- robustness under LLM availability and fallback conditions.

## 2. End-to-End Functional Outcomes

The integrated pipeline successfully executed all critical stages:

1. ingestion of 101-dimensional physiological feature input,
2. 5-class stress inference using multimodal transformer,
3. explainability extraction (attention + feature attribution),
4. contextual stress summary generation,
5. ChromaDB retrieval of relevant evidence-based interventions,
6. structured intervention plan generation via Groq or deterministic fallback.

The system produced machine-parseable JSON outputs that satisfy the API contract for downstream user interfaces.

## 3. Prediction and Explainability Outputs

### 3.1 Stress Classification Output

For tested samples, the classifier returned:

- predicted stress category among `very_low`, `low`, `moderate`, `high`, `very_high`,
- calibrated class probability vector,
- top class probability used as stress confidence.

### 3.2 Explainability Behavior

Explainability outputs demonstrated consistent interpretability signals:

- **Feature attribution:** normalized per-feature influence over all 101 features,
- **Cross-modal attention:** CLS-token attention distribution over EEG and ECG tokens.

These outputs enabled physiological reasoning support in subsequent context and planning stages.

## 4. Retrieval (RAG) Results

The ChromaDB knowledge layer returned context-relevant intervention candidates using semantic similarity retrieval.  
The active knowledge base was confirmed to be populated (including WHO-derived chunks and curated techniques), ensuring non-empty retrieval during stress-management generation.

Retrieved items consistently included:

- intervention title/technique,
- applicable condition metadata,
- stress-level applicability,
- recommended duration,
- procedural steps.

This retrieval structure provided evidence grounding for the LLM planner and fallback planner.

## 5. Personalized Plan Generation Results

### 5.1 Groq-Based Generation

When Groq was available, the planner generated structured personalized plans with:

- stress-aware cause explanations,
- physiological interpretation aligned with provided indicators,
- selected (not exhaustive) intervention techniques,
- practical, time-bound, step-wise action guidance.

### 5.2 Fallback Generation

When Groq was unavailable, the deterministic planner produced valid structured plans using retrieved evidence.  
This preserved service continuity and schema compatibility, while acknowledging reduced reasoning depth relative to full LLM generation.

## 6. Source-Traceable Output Behavior

A source flag (`plan_source`) was implemented and exposed in API responses:

- `groq` -> full LLM reasoning path,
- `fallback` -> deterministic resilience path.

This enabled explicit result stratification for analysis and reporting.

## 7. Qualitative Result Characteristics

Across representative sample conditions:

- higher stress contexts yielded stronger emphasis on immediate regulation techniques (e.g., breathing, grounding, autonomic calming),
- moderate contexts emphasized combined cognitive-behavioral and routine-management actions,
- lower stress contexts produced maintenance-oriented and resilience-building recommendations.

The generated action sets generally reflected stress severity, retrieved evidence, and explainability cues.

## 8. Robustness Results

The system showed robust behavior under operational constraints:

- no pipeline termination on external LLM unavailability,
- successful fallback context/plan generation,
- maintained JSON output integrity for frontend/API consumption,
- persistent RAG retrieval readiness through automatic collection checks.

These reliability mechanisms reduced runtime failure propagation in deployment-like conditions.

## 9. Publication Reporting Template (Fill with Final Metrics)

For manuscript finalization, include explicit quantitative values in the following template:

- **Classification metrics:** Accuracy, Macro-F1, per-class Precision/Recall/F1, confusion matrix
- **Inference metrics:** average latency per endpoint (`/predict`, `/explain`, `/stress-management`)
- **Retrieval metrics:** Recall@k / Precision@k for intervention relevance
- **Generation metrics:** JSON validity rate, recommendation acceptance/relevance scores
- **Robustness metrics:** Groq failure survival rate, fallback invocation rate, response success rate

## 10. Summary of Results

The implemented framework demonstrated successful integration of multimodal stress prediction, interpretable AI outputs, evidence-grounded retrieval, and personalized intervention planning.  
The addition of fallback and source-trace mechanisms improved reliability and auditability, supporting practical deployment and publication-grade reproducibility.

