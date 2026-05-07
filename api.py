import os
import tempfile
import pickle
import torch
import numpy as np
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from typing import List, Dict

from deap_pipeline import DEAPPipeline

from explainability import generate_feature_names
from integrated_system import StressManagementSystem
from transformer_model import MultimodalTransformerEncoder

NUM_CLASSES  = 5
CLASS_NAMES  = ["Very Low", "Low", "Moderate", "High", "Very High"]

app = FastAPI(
    title="NeuroStress API",
    description="5-Class Stress Predictor — Multimodal Transformer (EEG + ECG)",
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve test-sample JSON files at /test_samples/<file>
_samples_dir = os.path.join(os.path.dirname(__file__), "test_samples")
if os.path.isdir(_samples_dir):
    app.mount("/test_samples", StaticFiles(directory=_samples_dir), name="test_samples")

@app.get("/", include_in_schema=False)
def serve_frontend():
    index_path = os.path.join(os.path.dirname(__file__), "index.html")
    return FileResponse(index_path, media_type="text/html")

# ── Pydantic Models ──────────────────────────────────────────
class PredictRequest(BaseModel):
    features: List[float] = Field(
        ..., min_length=101, max_length=101,
        description="101 engineered features: 97 EEG (theta/alpha/beta/FAA) + 4 ECG/HRV."
    )

class PredictResponse(BaseModel):
    stress_class: int            # 0-4
    stress_level: str            # "Very Low" … "Very High"
    stress_probability: float    # probability of the predicted class
    class_probabilities: List[float]  # all 5 softmax probs

class ExplainResponse(BaseModel):
    stress_class: int
    stress_level: str
    stress_probability: float
    class_probabilities: List[float]
    attention_weights: Dict[str, float]
    shap_values: Dict[str, float]

class StressManagementRequest(BaseModel):
    stress_level: str = Field(..., description="Stress level: very_low, low, moderate, high, very_high")
    stress_probability: float = Field(..., ge=0.0, le=1.0, description="Probability of the stress level")
    shap_features: List[str] = Field(..., description="List of top contributing SHAP features")
    physiological_metrics: Dict[str, float] = Field(..., description="Dictionary of physiological metrics (LF/HF, beta power, etc.)")
    all_features: Dict[str, float] = Field(..., description="All 101 features as key-value pairs")

class RecommendedAction(BaseModel):
    technique: str
    steps: str
    duration: str

class StressManagementResponse(BaseModel):
    plan_source: str = Field(..., description="Source of plan generation: groq or fallback")
    stress_level: str
    cause: str
    physiological_interpretation: str
    recommended_actions: List[RecommendedAction]

# ── Globals ──────────────────────────────────────────────────
model        = None
stress_system = None
feature_names = generate_feature_names()
device        = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ── Startup ──────────────────────────────────────────────────
@app.on_event("startup")
def load_model():
    global model, stress_system
    model_path = "best_model.pth"

    model = MultimodalTransformerEncoder(
        eeg_dim=97, ecg_dim=4, d_model=128, n_heads=4, n_layers=3,
        ff_dim=256, dropout=0.3, num_classes=NUM_CLASSES, use_logits=True
    )

    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"Loaded 5-class weights from {model_path}.")
    else:
        print(f"WARNING: {model_path} not found — using random weights!")

    model.to(device)
    model.eval()

    # Initialize stress management system
    stress_system = StressManagementSystem()
    print("Initialized stress management system with RAG and LLM.")

# ── Helpers ──────────────────────────────────────────────────
def predict_from_features(feature_array: torch.Tensor):
    """Run forward pass; return (stress_class, stress_level, top_prob, all_probs_list)."""
    eeg = feature_array[:, :97].to(device)
    ecg = feature_array[:, 97:].to(device)

    with torch.no_grad():
        logits = model(eeg, ecg)               # (1, 5)
        probs  = torch.softmax(logits, dim=1)  # (1, 5)
        cls    = int(torch.argmax(probs, dim=1).item())

    prob_list = [round(float(p), 4) for p in probs[0]]
    return cls, CLASS_NAMES[cls], prob_list[cls], prob_list

# ── Endpoints ────────────────────────────────────────────────
@app.post("/predict", response_model=PredictResponse)
def predict_stress(request: PredictRequest):
    if model is None:
        raise HTTPException(status_code=503, detail="Model unavailable.")
    try:
        feat = torch.tensor([request.features], dtype=torch.float32)
        cls, level, top_prob, all_probs = predict_from_features(feat)
        return PredictResponse(
            stress_class=cls,
            stress_level=level,
            stress_probability=top_prob,
            class_probabilities=all_probs
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/explain", response_model=ExplainResponse)
def explain_stress(request: PredictRequest):
    if model is None:
        raise HTTPException(status_code=503, detail="Model unavailable.")
    try:
        feat = torch.tensor([request.features], dtype=torch.float32).to(device)
        eeg  = feat[:, :97]
        ecg  = feat[:, 97:]

        # Forward with attention
        with torch.no_grad():
            logits, attn_weights = model(eeg, ecg, return_attention=True)
            probs = torch.softmax(logits, dim=1)
            cls   = int(torch.argmax(probs, dim=1).item())

        prob_list = [round(float(p), 4) for p in probs[0]]
        top_prob  = prob_list[cls]

        # Attention dict
        cls_attn = attn_weights[0, 0, :].cpu().numpy()
        attention_dict = {
            "self_cls":      round(float(cls_attn[0]), 4),
            "eeg_attention": round(float(cls_attn[1]), 4),
            "ecg_attention": round(float(cls_attn[2]), 4),
        }

        # Permutation-based feature importance (ablation)
        base_cls_prob = top_prob
        importances = []
        with torch.no_grad():
            for i in range(101):
                ablated = feat.clone()
                ablated[0, i] = 0.0
                logits_ab = model(ablated[:, :97], ablated[:, 97:])
                probs_ab  = torch.softmax(logits_ab, dim=1)
                ablated_prob = float(probs_ab[0, cls].item())
                importances.append(base_cls_prob - ablated_prob)

        abs_imp   = [abs(x) for x in importances]
        total_imp = sum(abs_imp) + 1e-9
        shap_dict = {
            name: round(abs(val) / total_imp, 6)
            for name, val in zip(feature_names, importances)
        }

        return ExplainResponse(
            stress_class=cls,
            stress_level=CLASS_NAMES[cls],
            stress_probability=top_prob,
            class_probabilities=prob_list,
            attention_weights=attention_dict,
            shap_values=shap_dict
        )
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/predict_dat", response_model=PredictResponse)
async def predict_from_dat(file: UploadFile = File(...)):
    """
    Upload a DEAP-format .dat file and get a stress prediction.
    The file is processed through the full pipeline (bandpass filter,
    normalization, EEG band power extraction, HRV features) and the
    averaged 101-feature vector is passed to the model.
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model unavailable.")

    if not file.filename.endswith('.dat'):
        raise HTTPException(status_code=400, detail="Only .dat files are accepted.")

    tmp_path = None
    try:
        # Save uploaded file to a temporary location
        content = await file.read()
        with tempfile.NamedTemporaryFile(delete=False, suffix='.dat') as tmp:
            tmp.write(content)
            tmp_path = tmp.name

        # Load using the same pipeline as training
        pipeline = DEAPPipeline(
            data_dir=os.path.dirname(tmp_path),
            window_size_sec=30.0,
            overlap_sec=15.0,
            remove_baseline=True,
            extract_eeg_features=True
        )

        data, labels = pipeline.load_participant_data(tmp_path)
        stress_labels = pipeline.extract_stress_labels_5class(labels)
        X_windows, y_windows, hrv_windows, feature_windows = pipeline.create_segments(data, stress_labels)

        if feature_windows.shape[0] == 0:
            raise HTTPException(
                status_code=422,
                detail="Could not extract features from the .dat file. File may be too short or corrupted."
            )

        # Average across all windows to get a single 101-dim feature vector
        avg_features = feature_windows.mean(axis=0).tolist()

        feat_tensor = torch.tensor([avg_features], dtype=torch.float32)
        cls, level, top_prob, all_probs = predict_from_features(feat_tensor)

        return PredictResponse(
            stress_class=cls,
            stress_level=level,
            stress_probability=top_prob,
            class_probabilities=all_probs
        )

    except HTTPException:
        raise
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=400, detail=f"Error processing .dat file: {str(e)}")
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)


@app.post("/stress-management", response_model=StressManagementResponse)
def get_stress_management_plan(request: StressManagementRequest):
    """
    Generate a personalized stress management plan using RAG and LLM.

    Takes stress analysis results and returns structured intervention recommendations
    based on evidence-based techniques from CBT, MBSR, WHO, and NIH guidelines.
    """
    if stress_system is None:
        raise HTTPException(status_code=503, detail="Stress management system unavailable.")

    try:
        # Generate complete plan
        plan = stress_system.generate_complete_plan(
            stress_level=request.stress_level,
            stress_probability=request.stress_probability,
            top_shap_features=request.shap_features,
            physiological_metrics=request.physiological_metrics,
            all_features=request.all_features
        )

        # Check if plan generation was successful
        if plan["status"] != "success":
            error_msg = plan.get("management_plan", "Unknown error occurred")
            raise HTTPException(status_code=500, detail=f"Plan generation failed: {str(error_msg)}")

        # Extract the structured management plan
        management_plan = plan.get("management_plan")
        
        if not management_plan:
            raise HTTPException(status_code=500, detail="No management plan generated")
        
        if isinstance(management_plan, str):
            # If it's a string error, return it as detail
            raise HTTPException(status_code=500, detail=f"Management plan error: {management_plan}")

        # Validate required fields in management plan
        required_fields = ["stress_level", "cause", "physiological_interpretation", "recommended_actions"]
        for field in required_fields:
            if field not in management_plan:
                raise HTTPException(status_code=500, detail=f"Missing required field in plan: {field}")

        # Convert recommended actions to response format
        recommended_actions = []
        for action in management_plan.get("recommended_actions", []):
            recommended_actions.append(
                RecommendedAction(
                    technique=action.get("technique", "Unknown"),
                    steps=action.get("steps", "No steps provided"),
                    duration=action.get("duration", "Not specified")
                )
            )

        return StressManagementResponse(
            plan_source=plan.get("plan_source", "unknown"),
            stress_level=management_plan.get("stress_level", request.stress_level),
            cause=management_plan.get("cause", "Stress detected based on physiological signals"),
            physiological_interpretation=management_plan.get("physiological_interpretation", "Analysis in progress"),
            recommended_actions=recommended_actions
        )

    except HTTPException:
        raise
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error generating stress management plan: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    # Use an alternative port if 8000 is already in use
    try:
        uvicorn.run(app, host="127.0.0.1", port=8000)
    except OSError as e:
        if "address already in use" in str(e):
            print("Port 8000 is busy, switching to 8001")
            uvicorn.run(app, host="127.0.0.1", port=8001)
        else:
            raise
