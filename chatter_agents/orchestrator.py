"""Coordinator: passes JSON messages between the four agents and keeps the shared test history.

Flow for every new test cut:
  Agent 1 (acquire) -> Agent 2 (features) -> Agent 3 (stable / chatter) -> history
  -> Agent 3 re-learns the lobe diagram -> Agent 4 advice for that cutting condition.
Every hand-over is logged as a message, the same JSON contract the n8n HTTP nodes carry.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from .agent1_acquisition import AcquisitionAgent
from .agent2_signal import SignalAgent
from .agent3_stability import ChatterDetector, LobeModel
from .agent4_decision import DecisionAgent

REPO = Path(__file__).resolve().parent.parent
MODEL_PATH = REPO / "results" / "models.joblib"


class Coordinator:
    def __init__(self, detector: ChatterDetector, history: pd.DataFrame, lobe_name: str = "SVM",
                 alert_hook=None):
        self.signal = SignalAgent()
        self.detector = detector
        self.history = history[["rpm", "doc_mm", "label"]].copy().reset_index(drop=True)
        self.lobe_name = lobe_name
        self.lobe = LobeModel(lobe_name).fit(self.history.rpm, self.history.doc_mm, self.history.label)
        self.decision = DecisionAgent(self.lobe, history=self.history)
        self.alert_hook = alert_hook
        self.log: list[dict] = []

    # ------------------------------------------------------------------ messaging
    def _msg(self, sender, receiver, kind, payload):
        m = {"t": round(time.time(), 3), "from": sender, "to": receiver, "type": kind, "payload": payload}
        self.log.append(m)
        return m

    # ------------------------------------------------------------------ one test cut
    def process(self, rec) -> dict:
        self._msg("agent1_acquisition", "agent2_signal", "recording",
                  {"source": rec.source, "rpm": rec.rpm, "doc_mm": rec.doc_mm, "samples": len(rec.acc)})
        feats = self.signal.run(rec)
        self._msg("agent2_signal", "agent3_stability", "features",
                  {k: round(v, 4) if isinstance(v, float) else v for k, v in feats.items()})
        p_stable = float(self.detector.stable_probability([feats])[0])
        label = int(p_stable >= 0.5)
        verdict = {"source": rec.source, "rpm": rec.rpm, "doc_mm": rec.doc_mm,
                   "detected": "stable" if label else "chatter", "p_stable": round(p_stable, 4)}
        self._msg("agent3_stability", "coordinator", "detection", verdict)
        if not label and self.alert_hook:
            self.alert_hook(verdict)                       # e-mail / Slack / n8n webhook
        self.history.loc[len(self.history)] = [rec.rpm, rec.doc_mm, label]
        self.lobe = LobeModel(self.lobe_name).fit(self.history.rpm, self.history.doc_mm, self.history.label)
        self.decision = DecisionAgent(self.lobe, history=self.history)
        self._msg("agent3_stability", "agent4_decision", "lobe_updated", {"test_points": len(self.history)})
        advice = self.decision.advise(rec.rpm, rec.doc_mm).to_dict()
        self._msg("agent4_decision", "operator", "advice", advice)
        return {"detection": verdict, "advice": advice}

    def run_folder(self, folder) -> list[dict]:
        agent1 = AcquisitionAgent(folder)
        return [self.process(r) for r in agent1.poll()]

    def save_log(self, path):
        Path(path).write_text("\n".join(json.dumps(m) for m in self.log) + "\n", encoding="utf-8")


def train_default():
    """Re-create the deployed models from the shipped tables (used when the saved models were written by a
    different scikit-learn version): detector on all noise levels, lobe model on the detections."""
    feats = pd.read_csv(REPO / "data" / "features.csv")
    detector = ChatterDetector("SVM").fit(feats.to_dict("records"), feats.label)
    det = pd.read_csv(REPO / "results" / "detections_20db.csv")
    history = det[["rpm", "doc_mm"]].assign(label=det.detected)
    lobe = LobeModel("SVM").fit(history.rpm, history.doc_mm, history.label)
    return {"detector": detector, "lobe": lobe, "history": history}


def load_trained(model_path=MODEL_PATH):
    """(detector, lobe_model, history) saved by scripts/evaluate.py; retrained if unreadable here."""
    import sklearn

    try:
        bundle = joblib.load(model_path)
        if bundle.get("sklearn") != sklearn.__version__:
            raise ValueError("saved with another scikit-learn version")
    except Exception:  # noqa: BLE001
        bundle = train_default()
    return bundle["detector"], bundle["lobe"], bundle["history"]


def default_coordinator(alert_hook=None) -> Coordinator:
    detector, _, history = load_trained()
    return Coordinator(detector, history, alert_hook=alert_hook)


def as_float_dict(d: dict) -> dict:
    return {k: (float(v) if isinstance(v, (np.floating, np.integer)) else v) for k, v in d.items()}
