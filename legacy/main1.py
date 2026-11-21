"""
Machining Stability Prediction API
Uses SVM model trained on stable/unstable RPM + DOC data.
Accepts input from n8n HTTP node.
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator
import pickle
import numpy as np
import os

app = FastAPI(
    title="Machining Stability Predictor",
    description="SVM-based API to predict machining stability from RPM and Depth of Cut (DOC).",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Load model on startup ──────────────────────────────────────────────────────
MODEL_PATH = os.path.join(os.path.dirname(__file__), "svm_model.pkl")

try:
    with open(MODEL_PATH, "rb") as f:
        bundle = pickle.load(f)
    MODEL = bundle["model"]
    SCALER = bundle["scaler"]
    print("✅ SVM model loaded successfully.")
except FileNotFoundError:
    raise RuntimeError(
        f"Model file not found at {MODEL_PATH}. "
        "Run train_model.py first to generate svm_model.pkl."
    )


# ── Schemas ────────────────────────────────────────────────────────────────────
class PredictRequest(BaseModel):
    rpm: float = Field(..., gt=0, description="Spindle speed in RPM (e.g. 5000)")
    doc: float = Field(..., gt=0, description="Depth of cut (e.g. 0.002)")

    @validator("rpm")
    def rpm_range(cls, v):
        if v < 100 or v > 50000:
            raise ValueError("RPM must be between 100 and 50,000")
        return v

    @validator("doc")
    def doc_range(cls, v):
        if v <= 0 or v > 1.0:
            raise ValueError("DOC must be between 0 and 1.0")
        return v


class OptimizeRequest(BaseModel):
    rpm: float = Field(..., gt=0, description="Spindle speed in RPM")
    doc: float = Field(..., gt=0, description="Current depth of cut (assumed unstable)")
    min_stability_probability: float = Field(
        0.5, ge=0.1, le=0.99,
        description="Minimum required stability probability (default 0.5)"
    )

    @validator("rpm")
    def rpm_range(cls, v):
        if v < 100 or v > 50000:
            raise ValueError("RPM must be between 100 and 50,000")
        return v

    @validator("doc")
    def doc_range(cls, v):
        if v <= 0 or v > 1.0:
            raise ValueError("DOC must be between 0 and 1.0")
        return v


class PredictResponse(BaseModel):
    rpm: float
    doc: float
    prediction: str
    stable_probability: float
    unstable_probability: float
    confidence: float
    mrr: float
    recommendation: str


class OptimizeResponse(BaseModel):
    rpm: float
    input_doc: float
    current_mrr: float
    is_input_unstable: bool
    current_stable_probability: float
    max_stable_doc: float | None
    stability_probability_at_max_doc: float | None
    max_mrr: float | None
    mrr_change: float | None
    doc_reduction_needed: float | None
    recommendation: str


# ── Helper: find max stable DOC for a given RPM ────────────────────────────────
def find_max_stable_doc(
    rpm: float,
    min_prob: float = 0.5,
    doc_min: float = 0.0001,
    doc_max: float = 0.030,
    n_points: int = 10000,
):
    """
    Scans DOC values and returns the highest DOC where
    stable_probability >= min_prob.
    Returns (max_doc, stable_prob_at_max_doc) or (None, None).
    """
    doc_values = np.linspace(doc_min, doc_max, n_points)
    X = np.column_stack([np.full(n_points, rpm), doc_values])
    X_scaled = SCALER.transform(X)
    proba = MODEL.predict_proba(X_scaled)
    stable_probs = proba[:, 1]

    stable_mask = stable_probs >= min_prob
    if not stable_mask.any():
        return None, None

    # Find the first index where stability drops below threshold
    transitions = np.where(np.diff(stable_mask.astype(int)) < 0)[0]
    if len(transitions) > 0:
        max_idx = transitions[0]
    else:
        max_idx = np.where(stable_mask)[0][-1]

    return float(doc_values[max_idx]), float(stable_probs[max_idx])


# ── Endpoints ──────────────────────────────────────────────────────────────────
@app.get("/", tags=["Health"])
def root():
    return {"status": "ok", "message": "Machining Stability API v2 is running."}


@app.get("/health", tags=["Health"])
def health():
    return {"status": "healthy", "model": "SVM (RBF kernel, balanced class weights)"}


@app.post("/predict", response_model=PredictResponse, tags=["Prediction"])
def predict(req: PredictRequest):
    """
    Predict machining stability for given RPM + DOC.
    Also returns MRR = RPM x DOC.
    """
    X = np.array([[req.rpm, req.doc]])
    X_scaled = SCALER.transform(X)

    proba = MODEL.predict_proba(X_scaled)[0]
    pred_label = MODEL.predict(X_scaled)[0]

    stable_prob = float(proba[1])
    unstable_prob = float(proba[0])
    confidence = float(max(proba))
    prediction = "Stable" if pred_label == 1 else "Unstable"
    mrr = round(req.rpm * req.doc, 6)

    if prediction == "Stable":
        recommendation = (
            f"STABLE at RPM={req.rpm}, DOC={req.doc:.5f}. "
            f"MRR = {mrr:.4f}. Confidence: {confidence:.1%}."
        )
    else:
        recommendation = (
            f"UNSTABLE at RPM={req.rpm}, DOC={req.doc:.5f}. "
            f"Use /optimize to find the maximum safe DOC for this RPM."
        )

    return PredictResponse(
        rpm=req.rpm,
        doc=req.doc,
        prediction=prediction,
        stable_probability=round(stable_prob, 4),
        unstable_probability=round(unstable_prob, 4),
        confidence=round(confidence, 4),
        mrr=mrr,
        recommendation=recommendation,
    )


@app.post("/optimize", response_model=OptimizeResponse, tags=["Optimization"])
def optimize(req: OptimizeRequest):
    """
    Given RPM + DOC (assumed unstable zone), this endpoint:

    1. Confirms whether the input is actually unstable
    2. Scans all DOC values for this RPM to find the maximum
       DOC that keeps machining stable (prob >= min_stability_probability)
    3. Returns:
       - max_stable_doc: highest safe depth of cut
       - stability_probability_at_max_doc: how confident the model is
       - max_mrr: best material removal rate achievable (RPM x max_stable_doc)
       - doc_reduction_needed: how much to reduce from current DOC
       - mrr_change: gain or loss vs current MRR
    """
    # Step 1: Check current input stability
    X = np.array([[req.rpm, req.doc]])
    X_scaled = SCALER.transform(X)
    proba = MODEL.predict_proba(X_scaled)[0]
    pred_label = MODEL.predict(X_scaled)[0]

    is_unstable = pred_label == 0
    current_stable_prob = float(proba[1])
    current_mrr = round(req.rpm * req.doc, 6)

    # Step 2: Find max stable DOC
    max_doc, sp_at_max = find_max_stable_doc(
        rpm=req.rpm,
        min_prob=req.min_stability_probability
    )

    if max_doc is None:
        return OptimizeResponse(
            rpm=req.rpm,
            input_doc=req.doc,
            current_mrr=current_mrr,
            is_input_unstable=is_unstable,
            current_stable_probability=round(current_stable_prob, 4),
            max_stable_doc=None,
            stability_probability_at_max_doc=None,
            max_mrr=None,
            mrr_change=None,
            doc_reduction_needed=None,
            recommendation=(
                f"No stable DOC found for RPM={req.rpm} with min stability "
                f">= {req.min_stability_probability:.0%}. "
                "Consider changing the RPM."
            ),
        )

    max_mrr = round(req.rpm * max_doc, 6)
    mrr_change = round(max_mrr - current_mrr, 6)
    doc_reduction = round(req.doc - max_doc, 6)

    if is_unstable:
        recommendation = (
            f"Input is UNSTABLE (stability: {current_stable_prob:.1%}). "
            f"Reduce DOC from {req.doc:.5f} to {max_doc:.5f} "
            f"(reduce by {doc_reduction:.5f}). "
            f"MRR: {current_mrr:.4f} to {max_mrr:.4f} "
            f"(change: {mrr_change:+.4f})."
        )
    else:
        recommendation = (
            f"Input is already STABLE (stability: {current_stable_prob:.1%}). "
            f"Maximum safe DOC for RPM={req.rpm} is {max_doc:.5f}. "
            f"Max achievable MRR = {max_mrr:.4f}."
        )

    return OptimizeResponse(
        rpm=req.rpm,
        input_doc=req.doc,
        current_mrr=current_mrr,
        is_input_unstable=is_unstable,
        current_stable_probability=round(current_stable_prob, 4),
        max_stable_doc=round(max_doc, 6),
        stability_probability_at_max_doc=round(sp_at_max, 4),
        max_mrr=max_mrr,
        mrr_change=mrr_change,
        doc_reduction_needed=doc_reduction,
        recommendation=recommendation,
    )


@app.post("/predict/batch", tags=["Prediction"])
def predict_batch(requests: list[PredictRequest]):
    """Predict stability for multiple RPM/DOC combinations."""
    if len(requests) > 500:
        raise HTTPException(status_code=400, detail="Batch size must be ≤ 500.")

    results = []
    for req in requests:
        X = np.array([[req.rpm, req.doc]])
        X_scaled = SCALER.transform(X)
        proba = MODEL.predict_proba(X_scaled)[0]
        pred_label = MODEL.predict(X_scaled)[0]
        mrr = req.rpm * req.doc

        results.append({
            "rpm": req.rpm,
            "doc": req.doc,
            "prediction": "Stable" if pred_label == 1 else "Unstable",
            "stable_probability": round(float(proba[1]), 4),
            "unstable_probability": round(float(proba[0]), 4),
            "confidence": round(float(max(proba)), 4),
            "mrr": round(mrr, 6),
        })

    return {"count": len(results), "results": results}
