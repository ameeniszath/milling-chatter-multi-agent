"""Agent 4 - decision and advice.

Given the cutting condition an operator wants (spindle speed + depth of cut), Agent 4:
  1. predicts whether it will chatter (from Agent 3's learned lobe diagram),
  2. if needed, proposes the deepest safe cut at the same speed, and
  3. searches the allowed speed range for a "lobe pocket" - a speed where the machine can cut
     much deeper without chatter - and picks the option with the highest metal removal rate.

Metal removal rate (MRR) for milling:  MRR = ae x ap x vf,  vf = fz x z x n
(ae radial width, ap axial depth, fz feed per tooth, z teeth, n spindle speed).
With ae, fz, z fixed, MRR is proportional to n x ap.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from .data import N_TEETH


@dataclass
class CutSettings:
    radial_width_mm: float = 5.0       # ae (assumed, used only to express MRR in cm^3/min)
    feed_per_tooth_mm: float = 0.05    # fz
    teeth: int = N_TEETH


def mrr_cm3_min(rpm: float, doc_mm: float, s: CutSettings = CutSettings()) -> float:
    vf = s.feed_per_tooth_mm * s.teeth * rpm          # mm/min
    return s.radial_width_mm * doc_mm * vf / 1000.0    # cm^3/min


@dataclass
class Advice:
    rpm: float
    doc_mm: float
    stable_probability: float
    status: str
    mrr_cm3_min: float
    option_same_speed: dict | None
    option_best_speed: dict | None
    recommendation: str

    def to_dict(self) -> dict:
        return asdict(self)


class DecisionAgent:
    name = "agent4_decision"

    def __init__(self, lobe_model, rpm_range=(2000.0, 20000.0), safety_margin: float = 0.20,
                 settings: CutSettings = CutSettings(), history=None, same_speed_tol_rpm: float = 50.0):
        self.lobe = lobe_model
        self.rpm_range = rpm_range
        self.safety_margin = safety_margin
        self.settings = settings
        self.tol = same_speed_tol_rpm
        self._cache: dict[float, float] = {}
        # deepest cut already PROVEN stable at each tested speed (rpm -> mm)
        self.verified = {}
        if history is not None:
            for s, grp in history.groupby("rpm"):
                st = grp[grp.label == 1].doc_mm
                if len(st):
                    self.verified[float(s)] = float(st.max())

    def _verified_cap(self, rpm: float):
        near = [d for s, d in self.verified.items() if abs(s - rpm) <= self.tol]
        return min(near) if near else None

    def safe_depth(self, rpm_values) -> np.ndarray:
        """Depth the agent is willing to recommend: learned limit minus the safety margin, and never
        deeper than the deepest cut already proven stable when that speed has been tested
        (cached per speed - the lobe diagram does not change between requests)."""
        rpm_values = np.atleast_1d(np.asarray(rpm_values, float))
        todo = [r for r in rpm_values if float(r) not in self._cache]
        if todo:
            for r, lim in zip(todo, self.lobe.limit_depth(np.array(todo))):
                d = (1.0 - self.safety_margin) * float(lim)
                cap = self._verified_cap(float(r))
                self._cache[float(r)] = min(d, cap) if cap is not None else d
        return np.array([self._cache[float(r)] for r in rpm_values])

    def advise(self, rpm: float, doc_mm: float, rpm_min: float | None = None,
               rpm_max: float | None = None, rpm_step: float = 50.0) -> Advice:
        rpm_min = self.rpm_range[0] if rpm_min is None else max(rpm_min, self.rpm_range[0])
        rpm_max = self.rpm_range[1] if rpm_max is None else min(rpm_max, self.rpm_range[1])
        p = float(self.lobe.stable_probability(np.array([rpm]), np.array([doc_mm]))[0])
        status = "stable" if p >= 0.5 else "chatter"
        mrr_now = mrr_cm3_min(rpm, doc_mm, self.settings)

        same_doc = float(self.safe_depth([rpm])[0])
        same = {"rpm": rpm, "doc_mm": round(same_doc, 3),
                "mrr_cm3_min": round(mrr_cm3_min(rpm, same_doc, self.settings), 3)}

        speeds = np.arange(rpm_min, rpm_max + 1e-9, rpm_step)
        depths = self.safe_depth(speeds)
        mrr = np.array([mrr_cm3_min(s, d, self.settings) for s, d in zip(speeds, depths)])
        k = int(np.argmax(mrr))
        best = {"rpm": float(speeds[k]), "doc_mm": round(float(depths[k]), 3),
                "mrr_cm3_min": round(float(mrr[k]), 3)}

        if status == "chatter":
            base = same["mrr_cm3_min"]
            msg = (f"Chatter predicted at {rpm:.0f} rpm / {doc_mm:.2f} mm (P(stable) = {p:.0%}). "
                   f"Safe at this speed: {same['doc_mm']:.2f} mm. ")
        else:
            base = mrr_now
            msg = f"Stable at {rpm:.0f} rpm / {doc_mm:.2f} mm (P(stable) = {p:.0%}). "
        if best["mrr_cm3_min"] > base * 1.05:
            ratio = best["mrr_cm3_min"] / base if base > 0 else float("inf")
            gain = f"{ratio:.1f}x the" if ratio >= 2 else f"+{ratio - 1:.0%}"
            msg += (f"Higher output: {best['rpm']:.0f} rpm at {best['doc_mm']:.2f} mm depth "
                    f"({gain} metal removal rate, {self.safety_margin:.0%} safety margin).")
        else:
            msg += "Current speed is already near the best available."
        return Advice(rpm=rpm, doc_mm=doc_mm, stable_probability=round(p, 4), status=status,
                      mrr_cm3_min=round(mrr_now, 3), option_same_speed=same, option_best_speed=best,
                      recommendation=msg)

    def run(self, request: dict) -> dict:
        return self.advise(**request).to_dict()
