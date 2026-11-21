"""Train and evaluate every agent, write results/metrics.json, the figures and the deployed models.

    python scripts/evaluate.py            (uses data/features.csv - no raw dataset needed)
"""
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
from chatter_agents.agent2_signal import ML_FEATURES  # noqa: E402
from chatter_agents.agent3_stability import (ChatterDetector, LobeModel, chatter_models,  # noqa: E402
                                             compare_with_theory)
from chatter_agents.agent4_decision import DecisionAgent, mrr_cm3_min  # noqa: E402
from chatter_agents.data import load_boundary  # noqa: E402

RES = REPO / "results"
FIG = RES / "figures"
DEPLOYED = "SVM"
REPEATS = 10
SAFETY = 0.20


def cv_accuracy(model, X, y, groups=None, repeats=REPEATS):
    accs, recalls = [], []
    oof_votes = np.zeros(len(y))
    for seed in range(repeats):
        if groups is None:
            splits = StratifiedKFold(5, shuffle=True, random_state=seed).split(X, y)
        else:
            splits = StratifiedGroupKFold(5, shuffle=True, random_state=seed).split(X, y, groups)
        pred = np.zeros(len(y), int)
        for tr, te in splits:
            m = clone(model).fit(X[tr], y[tr])
            pred[te] = m.predict(X[te])
        accs.append((pred == y).mean())
        recalls.append(((pred == 0) & (y == 0)).sum() / (y == 0).sum())
        oof_votes += pred
    return float(np.mean(accs)), float(np.std(accs)), float(np.mean(recalls)), (oof_votes / repeats >= 0.5).astype(int)


def bracket_limits(rpm, doc, label):
    """Non-ML baseline: per tested speed, midpoint between deepest stable and shallowest chatter cut."""
    out = []
    for s in np.unique(rpm):
        st, un = doc[(rpm == s) & (label == 1)], doc[(rpm == s) & (label == 0)]
        lo = st.max() if len(st) else 0.0
        out.append((s, (lo + un.min()) / 2 if len(un) else lo))
    return np.array(out)


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    feats = pd.read_csv(REPO / "data" / "features.csv")
    b_rpm, b_lim = load_boundary(REPO / "data" / "stability_boundary.csv")
    clean = feats[feats.snr_db == "clean"].reset_index(drop=True)
    metrics = {"dataset": {"test_cuts": int(len(clean)), "stable": int(clean.label.sum()),
                           "chatter": int((clean.label == 0).sum()),
                           "speeds_rpm": [float(clean.rpm.min()), float(clean.rpm.max())],
                           "depths_mm": [float(clean.doc_mm.min()), float(clean.doc_mm.max())]}}

    # ---------------------------------------------------------- 1. chatter detection (Agents 2+3)
    det = {}
    for snr in ["clean", "30", "20", "10"]:
        sub = feats[feats.snr_db.astype(str) == snr]
        X, y = sub[ML_FEATURES].to_numpy(), sub.label.to_numpy()
        det[snr] = {}
        for name, model in chatter_models().items():
            acc, sd, rec, _ = cv_accuracy(model, X, y)
            det[snr][name] = {"accuracy_pct": round(acc * 100, 1), "std_pct": round(sd * 100, 1),
                              "chatter_recall_pct": round(rec * 100, 1)}
    metrics["chatter_detection_cv"] = det

    # deployed detector: trained on all noise levels together, tested on unseen recordings
    X_all, y_all = feats[ML_FEATURES].to_numpy(), feats.label.to_numpy()
    groups = feats.file.to_numpy()
    acc, sd, rec, _ = cv_accuracy(chatter_models()[DEPLOYED], X_all, y_all, groups)
    metrics["deployed_detector"] = {"model": DEPLOYED, "trained_on": "clean + 30/20/10 dB noise",
                                    "accuracy_pct": round(acc * 100, 1), "std_pct": round(sd * 100, 1),
                                    "chatter_recall_pct": round(rec * 100, 1)}

    # labels the system would have produced itself: out-of-fold detections on 20 dB data
    noisy = feats[feats.snr_db.astype(str) == "20"].reset_index(drop=True)
    *_, oof = cv_accuracy(chatter_models()[DEPLOYED], noisy[ML_FEATURES].to_numpy(), noisy.label.to_numpy())
    clean["detected"] = oof
    cm = confusion_matrix(clean.label, clean.detected, labels=[1, 0])
    metrics["detection_20db_confusion"] = {"rows_true_cols_pred": "[stable, chatter]", "matrix": cm.tolist()}

    # ---------------------------------------------------------- 2. lobe diagram (Agent 3)
    sld = {}
    for name in ["SVM", "Random Forest", "Naive Bayes"]:
        m_true = LobeModel(name).fit(clean.rpm, clean.doc_mm, clean.label)
        m_det = LobeModel(name).fit(clean.rpm, clean.doc_mm, clean.detected)
        sld[name] = {"from_true_labels": {k: round(v, 3) for k, v in compare_with_theory(m_true, b_rpm, b_lim).items()},
                     "from_detected_labels": {k: round(v, 3) for k, v in compare_with_theory(m_det, b_rpm, b_lim).items()}}
    br = bracket_limits(clean.rpm.to_numpy(), clean.doc_mm.to_numpy(), clean.detected.to_numpy())
    rv = np.arange(2000, 20001, 50.0)
    sld["Bracket + interpolation (non-ML baseline)"] = {"from_detected_labels": {"limit_mae_mm": round(float(np.abs(
        np.minimum(np.interp(rv, br[:, 0], br[:, 1]), 15) - np.minimum(np.interp(rv, b_rpm, b_lim), 15)).mean()), 3)}}
    metrics["stability_lobe_diagram"] = sld
    lobe = LobeModel("SVM").fit(clean.rpm, clean.doc_mm, clean.detected)

    # ---------------------------------------------------------- 3. decisions (Agent 4)
    true_lim = lambda r: np.interp(r, b_rpm, b_lim)  # noqa: E731
    sens = {}
    for margin in [0.10, 0.15, 0.20, 0.25]:
        ag = DecisionAgent(lobe, safety_margin=margin, history=clean[["rpm", "doc_mm"]].assign(label=clean.detected))
        ok_same = [ag.advise(r, d).option_same_speed for r, d in zip(clean.rpm, clean.doc_mm)]
        ok_same = np.mean([o["doc_mm"] < true_lim(o["rpm"]) for o in ok_same])
        sp = np.arange(2000, 20001, 50.0)
        sd = ag.safe_depth(sp)
        safe_all = np.mean(sd < true_lim(sp))
        sens[f"{int(margin * 100)}%"] = {"same_speed_advice_truly_stable_pct": round(100 * ok_same, 1),
                                         "all_speeds_safe_depth_truly_stable_pct": round(100 * safe_all, 1)}
    metrics["safety_margin_study"] = sens
    agent4 = DecisionAgent(lobe, safety_margin=SAFETY, history=clean[["rpm", "doc_mm"]].assign(label=clean.detected))
    safe_same, safe_best, gains = [], [], []
    for r, d in zip(clean.rpm, clean.doc_mm):
        a = agent4.advise(r, d)
        safe_same.append(a.option_same_speed["doc_mm"] < true_lim(a.option_same_speed["rpm"]))
        safe_best.append(a.option_best_speed["doc_mm"] < true_lim(a.option_best_speed["rpm"]))
    speeds = np.arange(2000, 20001, 50.0)
    safe_depth = agent4.safe_depth(speeds)
    mrr = np.array([mrr_cm3_min(s, dd) for s, dd in zip(speeds, safe_depth)])
    k = int(np.argmax(mrr))
    blim_min = float(b_lim.min())
    conservative = mrr_cm3_min(20000, (1 - SAFETY) * blim_min)          # no lobe diagram: stay below the global minimum
    # chatter requests: best option vs simply reducing depth at the same speed
    for r, d, lab in zip(clean.rpm, clean.doc_mm, clean.label):
        if lab == 0:
            a = agent4.advise(r, d)
            gains.append(a.option_best_speed["mrr_cm3_min"] / max(a.option_same_speed["mrr_cm3_min"], 1e-9))
    metrics["decision_agent"] = {
        "safety_margin_pct": int(SAFETY * 100),
        "requests_evaluated": int(len(clean)),
        "same_speed_advice_truly_stable_pct": round(100 * np.mean(safe_same), 1),
        "best_speed_advice_truly_stable_pct": round(100 * np.mean(safe_best), 1),
        "best_pocket": {"rpm": float(speeds[k]), "doc_mm": round(float(safe_depth[k]), 2),
                        "mrr_cm3_min": round(float(mrr[k]), 1),
                        "theory_limit_mm": round(float(true_lim(speeds[k])), 2)},
        "conservative_rule": {"rpm": 20000, "doc_mm": round((1 - SAFETY) * blim_min, 2), "mrr_cm3_min": round(conservative, 1),
                              "note": "depth kept below the lowest chatter limit at any speed (no lobe diagram)"},
        "mrr_gain_vs_conservative_x": round(float(mrr[k] / conservative), 1),
        "median_mrr_gain_for_chatter_requests_x": round(float(np.median(gains)), 1),
    }

    # ---------------------------------------------------------- save
    deployed = ChatterDetector(DEPLOYED).fit(feats.to_dict("records"), feats.label)
    history = clean[["rpm", "doc_mm"]].assign(label=clean.detected)
    import sklearn
    joblib.dump({"detector": deployed, "lobe": lobe, "history": history, "sklearn": sklearn.__version__},
                RES / "models.joblib")
    (RES / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    clean.to_csv(RES / "detections_20db.csv", index=False, float_format="%.6g")
    print(json.dumps(metrics, indent=2))

    import make_figures
    make_figures.main(clean, feats, metrics, lobe, agent4, b_rpm, b_lim)


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).parent))
    main()
