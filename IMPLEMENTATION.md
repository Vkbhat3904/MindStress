# Implementation

## 1. Implementation Environment

The system is implemented in Python with a service-oriented backend architecture.  
Core runtime components include:

- **PyTorch** for multimodal transformer inference,
- **FastAPI** for deployment and endpoint orchestration,
- **SentenceTransformers** for semantic embeddings,
- **ChromaDB** for persistent vector retrieval,
- **Groq LLM API** for context reasoning and intervention synthesis.

The implementation supports CPU execution and optionally utilizes CUDA when available.

## 2. Module-Level Implementation

The codebase is organized into modular components to separate modeling, explainability, retrieval, and plan generation.

### 2.1 Classification and Signal Modeling

- `transformer_model.py` implements the multimodal transformer encoder.
- `train_transformer.py` handles model training and checkpoint generation.
- `deap_pipeline.py` processes physiological inputs and constructs model-ready feature tensors.

At inference, the model consumes a `101`-dimensional feature vector and outputs logits for 5 stress classes. Softmax conversion yields per-class probabilities and top predicted stress severity.

### 2.2 Explainability Layer

- `explainability.py` provides:
  - canonical feature naming for all 101 signals,
  - permutation-based feature attribution (SHAP-style),
  - cross-modal attention extraction and plotting.

This implementation provides both local and modality-level interpretation used by downstream planning.

### 2.3 Context Construction

- `context_builder.py` transforms numeric outputs into a semantically rich stress context.
- Inputs include:
  - predicted stress level,
  - stress probability,
  - top explainability features,
  - physiological metrics,
  - full named 101-feature vector.

Named feature normalization ensures LLM prompts contain interpretable signal semantics rather than index-based keys.

### 2.4 Retrieval-Augmented Knowledge Layer

- `rag_system.py` implements ChromaDB-backed retrieval.
- Embeddings are generated using `all-MiniLM-L6-v2`.
- Knowledge is ingested from curated JSON and WHO-derived chunked corpora.
- Metadata is normalized to satisfy Chroma type constraints and denormalized at query time.

The retrieval engine returns top relevant intervention candidates with structured metadata (`technique_name`, `description`, `conditions`, `duration`, `steps`).

### 2.5 Stress Management Planning

- `stress_management_planner.py` performs final plan generation.
- Retrieved interventions and stress context are injected into constrained LLM prompts.
- Output is parsed into strict JSON schema:
  - `stress_level`,
  - `cause`,
  - `physiological_interpretation`,
  - `recommended_actions`.

The planner now tags output origin (`groq` vs `fallback`) for traceability.

### 2.6 End-to-End Integration

- `integrated_system.py` orchestrates:
  1. context building,
  2. RAG retrieval,
  3. intervention plan generation.

- `integrated_pipeline.py` provides direct tensor-level end-to-end execution from EEG/ECG inputs to final plan object.

## 3. API Implementation

The service is exposed via `api.py` with FastAPI and includes:

- `POST /predict`  
  returns stress class, label, and probability distribution.

- `POST /explain`  
  returns prediction with attention weights and feature importance.

- `POST /predict_dat`  
  accepts DEAP `.dat` files and performs full preprocessing plus prediction.

- `POST /stress-management`  
  generates personalized intervention output and includes `plan_source` (`groq` or `fallback`) in response.

The API enforces input schema validation using Pydantic models and robust error handling using HTTP exceptions.

## 4. Reliability and Fault-Tolerant Implementation

To avoid pipeline interruption during external LLM failure:

1. **Safe Groq initialization** checks API key/runtime availability.
2. **Context fallback** produces deterministic context text when LLM context synthesis fails.
3. **Plan fallback** produces deterministic structured interventions from retrieved evidence when final LLM generation fails.
4. **Source tagging** (`plan_source`) enables transparent downstream evaluation and logging.

This ensures uninterrupted output generation while preserving best-effort personalization.

## 5. Knowledge Ingestion Implementation

- `pdf_knowledge_ingest.py` performs:
  - PDF extraction,
  - cleaning and segmentation,
  - heuristic technique/condition tagging,
  - chunk serialization,
  - optional direct Chroma ingestion.

The ingestion process supports multiple source corpora through configurable prefixes, enabling incremental updates without destructive replacement.

## 6. Prompt Engineering Implementation

The Groq prompts are implemented with explicit constraints for:

- evidence grounding in retrieved passages,
- selective intervention recommendation (not exhaustive listing),
- severity-aware personalization,
- step-by-step actionable instructions,
- safe non-diagnostic language.

Prompt design is optimized for machine-parseable JSON output to maintain API contract stability.

## 7. Runtime Flow (Operational Sequence)

For a complete stress-management request, runtime execution follows:

1. Validate incoming stress/explainability payload,
2. Build context from prediction + physiological + 101-feature data,
3. Retrieve semantically relevant techniques from ChromaDB,
4. Generate personalized plan using Groq LLM,
5. If Groq fails, execute deterministic fallback planner,
6. Return structured response with source attribution.

## 8. Implementation Artifacts and Outputs

Key implementation outputs include:

- trained model checkpoints (`best_model.pth`),
- Chroma persistent database (`chroma_db/`),
- explainability artifacts (`xai_plots/`),
- generated JSON plans and pipeline test outputs.

These artifacts support reproducibility, auditability, and manuscript-level reporting.

## 9. Reproducibility and Reporting Notes

For publication-grade reproducibility, implementation reporting should include:

- dependency versions and runtime environment,
- model checkpoint version used for inference,
- Chroma collection name and embedding model,
- retrieval depth (`top-k`) and prompt schema,
- fallback trigger policy and source-flag distribution,
- endpoint-level validation strategy.

This implementation design enables transparent replication and evaluation across external cohorts and deployment settings.

