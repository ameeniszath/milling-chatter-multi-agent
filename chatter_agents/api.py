"""REST service exposing the four agents - what the n8n workflow's HTTP nodes call.

    uvicorn chatter_agents.api:app --port 8000      then open http://localhost:8000/docs
"""
from __future__ import annotations

import io

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from fastapi import FastAPI, File, HTTPException, UploadFile  # noqa: E402
from fastapi.responses import StreamingResponse  # noqa: E402
from pydantic import BaseModel, Field  # noqa: E402

from .agent2_signal import extract_features  # noqa: E402
from .agent4_decision import DecisionAgent  # noqa: E402
from .data import Recording, load_mat  # noqa: E402
from .orchestrator import load_trained  # noqa: E402

app = FastAPI(title="Milling chatter multi-agent system",
              description="Agent 1 acquisition - Agent 2 FFT features - Agent 3 chatter detection & "
                          "stability lobe diagram - Agent 4 safe, high-output cutting advice.",
              version="2.0.0")

DETECTOR, LOBE, HISTORY = load_trained()
DECISION = DecisionAgent(LOBE, history=HISTORY)


class SignalIn(BaseModel):
    rpm: float = Field(..., gt=0, examples=[7500])
    doc_mm: float = Field(..., gt=0, examples=[8.0])
    time: list[float]
    acc: list[float]


class CutRequest(BaseModel):
    rpm: float = Field(..., ge=500, le=40000, examples=[9500])
    doc_mm: float = Field(..., gt=0, le=50, examples=[3.0])
    rpm_min: float | None = Field(None, description="Lowest speed the machine/tool allows")
    rpm_max: float | None = Field(None, description="Highest speed the machine/tool allows")


def _detect(feats: dict) -> dict:
    p = float(DETECTOR.stable_probability([feats])[0])
    return {"detected": "stable" if p >= 0.5 else "chatter", "p_stable": round(p, 4)}


def _clean(d: dict) -> dict:
    """JSON-safe: round floats, and send NaN (signal not supplied, e.g. no displacement) as null."""
    out = {}
    for k, v in d.items():
        if isinstance(v, (float, np.floating)):
            out[k] = None if not np.isfinite(v) else round(float(v), 6)
        else:
            out[k] = v
    return out


@app.get("/health")
def health():
    return {"status": "ok", "detector": DETECTOR.model_name, "lobe_model": LOBE.model_name,
            "test_points": int(len(HISTORY))}


# ---------------------------------------------------------------- Agent 1
@app.post("/agent1/extract")
async def agent1_extract(file: UploadFile = File(...), include_signal: bool = False):
    try:
        rec = load_mat(await file.read())
    except Exception as e:  # noqa: BLE001
        raise HTTPException(400, f"Could not read .mat file: {e}")
    out = {"rpm": rec.rpm, "doc_mm": rec.doc_mm, "samples": int(len(rec.acc)), "fs_hz": round(rec.fs, 1)}
    if include_signal:
        out.update({"time": rec.time.tolist(), "acc": rec.acc.tolist()})
    return out


# ---------------------------------------------------------------- Agent 2
@app.post("/agent2/features")
def agent2_features(sig: SignalIn):
    if len(sig.time) != len(sig.acc) or len(sig.acc) < 256:
        raise HTTPException(422, "time and acc must have the same length (>= 256 samples)")
    rec = Recording(rpm=sig.rpm, doc_mm=sig.doc_mm, time=np.asarray(sig.time), acc=np.asarray(sig.acc))
    return _clean(extract_features(rec))


# ---------------------------------------------------------------- Agent 3
@app.post("/agent3/detect")
def agent3_detect(features: dict):
    try:
        return _detect(features)
    except KeyError as e:
        raise HTTPException(422, f"missing feature {e}")


@app.get("/agent3/lobe")
def agent3_lobe(step_rpm: float = 250.0):
    rv = np.arange(LOBE.rpm_range[0], LOBE.rpm_range[1] + 1e-9, step_rpm)
    return {"rpm": rv.tolist(), "doc_limit_mm": np.round(LOBE.limit_depth(rv), 3).tolist()}


@app.get("/agent3/lobe.png")
def agent3_lobe_png():
    rv = np.arange(LOBE.rpm_range[0], LOBE.rpm_range[1] + 1e-9, 25.0)
    fig, ax = plt.subplots(figsize=(9, 4.5), dpi=120)
    ax.plot(rv, LOBE.limit_depth(rv), color="#2a78d6", lw=2, label="Learned stability limit")
    for lab, col, mk, name in [(1, "#2a78d6", "o", "stable"), (0, "#eb6834", "X", "chatter")]:
        h = HISTORY[HISTORY.label == lab]
        ax.scatter(h.rpm, h.doc_mm, c=col, marker=mk, s=30, label=f"test cut: {name}")
    ax.set_xlabel("Spindle speed (rpm)")
    ax.set_ylabel("Axial depth of cut (mm)")
    ax.set_ylim(0, 16)
    ax.grid(alpha=0.3)
    ax.legend(loc="upper left", frameon=False)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return StreamingResponse(buf, media_type="image/png")


# ---------------------------------------------------------------- Agent 4
@app.post("/agent4/advise")
def agent4_advise(req: CutRequest):
    return DECISION.advise(req.rpm, req.doc_mm, rpm_min=req.rpm_min, rpm_max=req.rpm_max).to_dict()


# ---------------------------------------------------------------- whole chain
@app.post("/pipeline/analyse")
async def pipeline_analyse(file: UploadFile = File(...)):
    """Upload one test-cut recording: Agent 1 -> 2 -> 3 -> 4 in one call."""
    try:
        rec = load_mat(await file.read())
    except Exception as e:  # noqa: BLE001
        raise HTTPException(400, f"Could not read .mat file: {e}")
    feats = extract_features(rec)
    return {"recording": {"rpm": rec.rpm, "doc_mm": rec.doc_mm, "samples": int(len(rec.acc))},
            "features": _clean(feats), "detection": _detect(feats),
            "advice": DECISION.advise(rec.rpm, rec.doc_mm).to_dict()}
