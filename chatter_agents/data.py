"""Data contracts shared by all agents.

A *recording* is one milling test cut: spindle speed, axial depth of cut and the
vibration signals measured on the tool. The JSON form of a recording is what
Agent 1 hands to Agent 2 (the same contract the n8n HTTP nodes use).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

# Cutter used in the dataset: 4 flutes. Inferred from the spectra of stable cuts,
# whose energy sits exactly on multiples of 4 x spindle frequency
# (e.g. 7 500 rpm -> 500, 1 000, 1 500 Hz).
N_TEETH = 4


@dataclass
class Recording:
    rpm: float
    doc_mm: float
    time: np.ndarray
    acc: np.ndarray                    # tool acceleration, feed (x) direction
    disp: np.ndarray | None = None     # tool displacement (m), x
    force: np.ndarray | None = None    # cutting force (N), x
    source: str = ""
    meta: dict = field(default_factory=dict)

    @property
    def fs(self) -> float:
        return 1.0 / float(np.mean(np.diff(self.time)))

    def to_json(self) -> dict:
        out = {"rpm": self.rpm, "doc_mm": self.doc_mm, "time": self.time.tolist(),
               "acc": self.acc.tolist(), "source": self.source}
        if self.disp is not None:
            out["disp"] = self.disp.tolist()
        if self.force is not None:
            out["force"] = self.force.tolist()
        return out

    @classmethod
    def from_json(cls, d: dict) -> "Recording":
        arr = lambda k: np.asarray(d[k], dtype=float) if d.get(k) is not None else None  # noqa: E731
        return cls(rpm=float(d["rpm"]), doc_mm=float(d["doc_mm"]), time=arr("time"), acc=arr("acc"),
                   disp=arr("disp"), force=arr("force"), source=d.get("source", ""))


def load_mat(path_or_bytes) -> Recording:
    """Read one test cut saved by the time-domain milling simulation (.mat, v5).

    Keys: omega (rpm), b (axial depth, m), time, xtacc / xtpos (tool acc / position, x),
    Forcex (cutting force, x).
    """
    from scipy.io import loadmat
    import io

    src = ""
    if isinstance(path_or_bytes, (bytes, bytearray)):
        m = loadmat(io.BytesIO(path_or_bytes))
    else:
        src = Path(path_or_bytes).name
        m = loadmat(path_or_bytes)
    flat = lambda k: np.asarray(m[k], dtype=float).ravel() if k in m else None  # noqa: E731
    return Recording(rpm=float(flat("omega")[0]), doc_mm=float(flat("b")[0]) * 1000.0,
                     time=flat("time"), acc=flat("xtacc"), disp=flat("xtpos"), force=flat("Forcex"),
                     source=src)


def load_boundary(csv_path) -> tuple[np.ndarray, np.ndarray]:
    """Analytical stability boundary (theory): limiting axial depth (mm) vs spindle speed (rpm)."""
    import pandas as pd

    df = pd.read_csv(csv_path)
    return df["rpm"].to_numpy(float), df["doc_limit_mm"].to_numpy(float)


def boundary_label(rpm, doc_mm, b_rpm, b_lim) -> np.ndarray:
    """1 = stable (below the theoretical limit), 0 = chatter."""
    lim = np.interp(np.asarray(rpm, float), b_rpm, b_lim)
    return (np.asarray(doc_mm, float) < lim).astype(int)
