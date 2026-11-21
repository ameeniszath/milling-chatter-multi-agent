
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator
import pickle
import numpy as np
import os

app = FastAPI(
    title="Machining Stability Predictor",
    description="SVM-based API to predict machining stability from RPM and Depth of Cut (DOC).",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
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
class PredictRequest(BaseModel):
    rpm: float = Field(..., gt=0, description="Spindle speed in RPM (e.g. 5000)")
    doc: float = Field(..., gt=0, description="Depth of cut in mm or meters (e.g. 0.002)")

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
    prediction: str          # "Stable" or "Unstable"
    stable_probability: float
    unstable_probability: float
    confidence: float        # max probability as confidence score
    recommendation: str
@app.get("/", tags=["Health"])
def root():
    return {"status": "ok", "message": "Machining Stability API is running."}
@app.get("/health", tags=["Health"])
def health():
    return {"status": "healthy", "model": "SVM (RBF kernel, balanced class weights)"}
@app.post("/predict", response_model=PredictResponse, tags=["Prediction"])
def predict(req: PredictRequest):
    X = np.array([[req.rpm, req.doc]])
    X_scaled = SCALER.transform(X)

    proba = MODEL.predict_proba(X_scaled)[0]   # [P(unstable), P(stable)]
    pred_label = MODEL.predict(X_scaled)[0]    # 0=unstable, 1=stable

    stable_prob = float(proba[1])
    unstable_prob = float(proba[0])
    confidence = float(max(proba))
    prediction = "Stable" if pred_label == 1 else "Unstable"

    if prediction == "Stable":
        recommendation = (
            f"Machining conditions are STABLE (confidence: {confidence:.1%}). "
            "Proceed with current parameters."
        )
    else:
        recommendation = (
            f"Machining conditions are UNSTABLE (confidence: {confidence:.1%}). "
            "Consider reducing DOC or adjusting RPM to avoid chatter."
        )

    return PredictResponse(
        rpm=req.rpm,
        doc=req.doc,
        prediction=prediction,
        stable_probability=round(stable_prob, 4),
        unstable_probability=round(unstable_prob, 4),
        confidence=round(confidence, 4),
        recommendation=recommendation,
    )


@app.post("/predict/batch", tags=["Prediction"])
def predict_batch(requests: list[PredictRequest]):
    """
    Predict stability for multiple RPM/DOC combinations in one call.
    Useful for scanning a parameter grid from n8n.
    """
    if len(requests) > 500:
        raise HTTPException(status_code=400, detail="Batch size must be ≤ 500.")

    results = []
    for req in requests:
        X = np.array([[req.rpm, req.doc]])
        X_scaled = SCALER.transform(X)
        proba = MODEL.predict_proba(X_scaled)[0]
        pred_label = MODEL.predict(X_scaled)[0]

        results.append({
            "rpm": req.rpm,
            "doc": req.doc,
            "prediction": "Stable" if pred_label == 1 else "Unstable",
            "stable_probability": round(float(proba[1]), 4),
            "unstable_probability": round(float(proba[0]), 4),
            "confidence": round(float(max(proba)), 4),
        })

    return {"count": len(results), "results": results}
