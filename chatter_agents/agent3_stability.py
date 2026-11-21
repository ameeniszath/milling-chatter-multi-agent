"""Agent 3 - stability classification and stability-lobe-diagram (SLD) learning.

Two jobs:
1. ChatterDetector: decides "stable" or "chatter" for one test cut from Agent 2's vibration
   features only (speed and depth are hidden from it). SVM, Random Forest and Naive Bayes are
   compared, as in the original course project.
2. LobeModel: learns the stability lobe diagram - the depth-of-cut limit at every spindle
   speed - from labelled (rpm, depth) test points, and can be compared with theory.
"""
from __future__ import annotations

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from .agent2_signal import ML_FEATURES


def chatter_models(seed: int = 0) -> dict:
    return {
        "SVM": make_pipeline(StandardScaler(), SVC(C=10, gamma="scale", class_weight="balanced",
                                                   probability=True, random_state=seed)),
        "Random Forest": RandomForestClassifier(n_estimators=300, class_weight="balanced", random_state=seed),
        "Naive Bayes": make_pipeline(StandardScaler(), GaussianNB()),
    }


class ChatterDetector:
    """Stable (1) / chatter (0) from vibration features."""

    def __init__(self, model_name: str = "SVM", seed: int = 0):
        self.model_name = model_name
        self.model = chatter_models(seed)[model_name]

    @staticmethod
    def matrix(feature_rows) -> np.ndarray:
        return np.array([[float(r[f]) for f in ML_FEATURES] for r in feature_rows])

    def fit(self, feature_rows, labels):
        self.model.fit(self.matrix(feature_rows), np.asarray(labels, int))
        return self

    def predict(self, feature_rows) -> np.ndarray:
        return self.model.predict(self.matrix(feature_rows))

    def stable_probability(self, feature_rows) -> np.ndarray:
        proba = self.model.predict_proba(self.matrix(feature_rows))
        return proba[:, list(self.model.classes_).index(1)]


# ---------------------------------------------------------------- lobe diagram
def monotonic_augment(rpm, doc_mm, labels, doc_max_mm: float, step_mm: float = 0.5):
    """Physics prior: at a fixed speed, cutting deeper than a depth that chattered also chatters,
    and cutting shallower than a stable depth is also stable. Testing stops once chatter is found,
    so without this the classifier would guess "stable" in the untested deep region."""
    R, Dp, L = [list(a) for a in (rpm, doc_mm, labels)]
    for s in np.unique(rpm):
        at = rpm == s
        unstable = doc_mm[at & (labels == 0)]
        stable = doc_mm[at & (labels == 1)]
        if len(unstable):
            for d in np.arange(unstable.min() + step_mm, doc_max_mm + 1e-9, step_mm):
                R.append(s); Dp.append(d); L.append(0)
        if len(stable):
            for d in np.arange(0.1, stable.max(), step_mm):
                R.append(s); Dp.append(d); L.append(1)
    return np.array(R), np.array(Dp), np.array(L)


def lobe_models(seed: int = 0) -> dict:
    # rpm is scaled to 0-1 over the tested range so that one RBF width suits both axes
    return {
        "SVM": SVC(C=300, gamma=40.0, class_weight="balanced", probability=True, random_state=seed),
        "Random Forest": RandomForestClassifier(n_estimators=400, min_samples_leaf=1, random_state=seed),
        "Naive Bayes": GaussianNB(),
    }


class LobeModel:
    """Learns P(stable | rpm, depth) and the limiting depth at each speed."""

    def __init__(self, model_name: str = "SVM", rpm_range=(2000.0, 20000.0), doc_max_mm: float = 16.0,
                 seed: int = 0):
        self.model_name = model_name
        self.model = lobe_models(seed)[model_name]
        self.rpm_range = tuple(map(float, rpm_range))
        self.doc_max_mm = float(doc_max_mm)

    def _x(self, rpm, doc_mm) -> np.ndarray:
        r0, r1 = self.rpm_range
        rpm = (np.asarray(rpm, float) - r0) / (r1 - r0)
        doc = np.asarray(doc_mm, float) / self.doc_max_mm
        return np.column_stack([np.ravel(rpm), np.ravel(doc)])

    def fit(self, rpm, doc_mm, labels, monotonic: bool = True):
        rpm, doc_mm, labels = (np.asarray(a, float) for a in (rpm, doc_mm, labels))
        if monotonic:
            rpm, doc_mm, labels = monotonic_augment(rpm, doc_mm, labels, self.doc_max_mm)
        self.model.fit(self._x(rpm, doc_mm), labels.astype(int))
        return self

    def stable_probability(self, rpm, doc_mm) -> np.ndarray:
        proba = self.model.predict_proba(self._x(rpm, doc_mm))
        return proba[:, list(self.model.classes_).index(1)].reshape(np.shape(rpm))

    def limit_depth(self, rpm_values, doc_grid=None, p_min: float = 0.5, min_doc_mm: float = 0.5) -> np.ndarray:
        """Largest depth (mm) that is still predicted stable at each speed, scanning up from zero:
        the first depth where P(stable) drops below p_min ends the stable zone. Depths below
        min_doc_mm (lighter than any test cut) are taken as stable - very light cuts do not chatter."""
        rpm_values = np.atleast_1d(np.asarray(rpm_values, float))
        doc_grid = np.linspace(0.05, self.doc_max_mm, 320) if doc_grid is None else np.asarray(doc_grid, float)
        R, D = np.meshgrid(rpm_values, doc_grid)          # rows = depth
        P = self.stable_probability(R, D)
        ok = (P >= p_min) | (D < min_doc_mm)
        lim = np.zeros(len(rpm_values))
        for j in range(len(rpm_values)):
            bad = np.where(~ok[:, j])[0]
            first_bad = bad[0] if len(bad) else len(doc_grid)
            lim[j] = doc_grid[first_bad - 1] if first_bad > 0 else 0.0
        return lim


def compare_with_theory(model: LobeModel, b_rpm, b_lim, rpm_values=None, doc_max: float = 15.0,
                        n_doc: int = 300) -> dict:
    """How well a learned SLD matches the analytical one, over the tested speed range."""
    rpm_values = np.arange(2000, 20001, 50, dtype=float) if rpm_values is None else rpm_values
    true_lim = np.interp(rpm_values, b_rpm, b_lim)
    learned = model.limit_depth(rpm_values)
    docs = np.linspace(doc_max / n_doc, doc_max, n_doc)
    R, D = np.meshgrid(rpm_values, docs)
    truth = D < np.interp(R, b_rpm, b_lim)
    pred = model.stable_probability(R, D) >= 0.5
    err = np.abs(np.minimum(learned, doc_max) - np.minimum(true_lim, doc_max))
    return {
        "map_agreement_pct": float((truth == pred).mean() * 100),
        "limit_mae_mm": float(err.mean()),
        "limit_median_err_mm": float(np.median(err)),
        "unsafe_area_pct": float((pred & ~truth).mean() * 100),   # predicted stable but really chatter
    }
