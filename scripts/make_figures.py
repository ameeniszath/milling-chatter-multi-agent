"""Figures for the README (called by evaluate.py)."""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
from chatter_agents.agent2_signal import spectrum  # noqa: E402
from chatter_agents.data import N_TEETH, load_mat  # noqa: E402
from chatter_agents.agent4_decision import mrr_cm3_min  # noqa: E402

FIG = REPO / "results" / "figures"
# reference palette (categorical slots 1-3 in fixed order) + neutral inks
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, MUTED, GRID, BASE, SURF = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"
STABLE, CHATTER = BLUE, ORANGE

plt.rcParams.update({
    "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
    "axes.edgecolor": BASE, "axes.labelcolor": INK2, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.spines.top": False,
    "axes.spines.right": False, "font.size": 10, "axes.titlesize": 11, "axes.titleweight": "bold",
    "axes.titlecolor": INK, "legend.frameon": False, "lines.linewidth": 2,
})


def fig_spectra():
    cases = [("time3_43.mat", "Stable: 7 500 rpm, 2 mm", STABLE),
             ("time3_49.mat", "Chatter: 7 500 rpm, 8 mm", CHATTER),
             ("time3_133.mat", "Stable but resonant: 15 000 rpm, 8 mm", STABLE)]
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6), sharey=False)
    for ax, (f, title, col) in zip(axes, cases):
        r = load_mat(REPO / "data" / "sample_recordings" / f)
        n = len(r.acc)
        fr, amp = spectrum(r.acc[n // 2:], r.fs)
        f_tp = r.rpm / 60 * N_TEETH
        for k in range(1, int(3000 // f_tp) + 1):
            ax.axvline(k * f_tp, color=MUTED, lw=0.8, ls=":", zorder=1)
        sel = fr <= 3000
        ax.plot(fr[sel], amp[sel] / amp[sel].max(), color=col, lw=1.4, zorder=3)
        ax.set_title(title, loc="left")
        ax.set_xlabel("Frequency (Hz)")
        ax.set_ylim(0, 1.08)
    axes[0].set_ylabel("Tool vibration (normalised)")
    axes[1].annotate("chatter peak between\ntooth-passing harmonics", xy=(1129, 1.0), xytext=(1500, 0.75),
                     color=INK2, fontsize=9, arrowprops=dict(arrowstyle="-", color=MUTED))
    fig.text(0.01, -0.02, "Dotted lines: tooth-passing frequency and its harmonics (rpm / 60 x 4 teeth).",
             color=MUTED, fontsize=9)
    fig.tight_layout()
    fig.savefig(FIG / "1_vibration_spectra.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def fig_feature(feats):
    fig, axes = plt.subplots(1, 2, figsize=(12, 3.8), sharex=True)
    for ax, snr, title in [(axes[0], "clean", "Clean signal"), (axes[1], "20", "With sensor noise (SNR 20 dB)")]:
        sub = feats[feats.snr_db.astype(str) == snr]
        for lab, col, mk, name in [(1, STABLE, "o", "Stable"), (0, CHATTER, "X", "Chatter")]:
            s = sub[sub.label == lab]
            ax.scatter(s.rpm, s.chatter_ratio * 100, s=34, c=col, marker=mk, edgecolors=SURF, linewidths=0.8,
                       label=name, zorder=3)
        ax.set_title(title, loc="left")
        ax.set_xlabel("Spindle speed (rpm)")
        ax.set_yscale("symlog", linthresh=0.1)
        ax.set_ylim(-0.02, 110)
    axes[0].set_ylabel("Vibration energy off the\ntooth harmonics (%)")
    axes[0].legend(loc="center right")
    fig.tight_layout()
    fig.savefig(FIG / "2_chatter_indicator.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def fig_detection(metrics):
    det = metrics["chatter_detection_cv"]
    levels = ["clean", "30", "20", "10"]
    labels = ["Clean", "30 dB", "20 dB", "10 dB"]
    models = ["SVM", "Random Forest", "Naive Bayes"]
    fig, ax = plt.subplots(figsize=(8.5, 3.8))
    w = 0.26
    x = np.arange(len(levels))
    for i, (m, col) in enumerate(zip(models, [BLUE, ORANGE, AQUA])):
        acc = [det[lv][m]["accuracy_pct"] for lv in levels]
        sd = [det[lv][m]["std_pct"] for lv in levels]
        bars = ax.bar(x + (i - 1) * w, acc, w - 0.03, yerr=sd, color=col, label=m, capsize=2,
                      error_kw=dict(ecolor=INK2, lw=0.8), zorder=3)
        if m == metrics["deployed_detector"]["model"]:
            for b, a in zip(bars, acc):
                ax.text(b.get_x() + b.get_width() / 2, a + 0.6, f"{a:.1f}", ha="center", va="bottom",
                        fontsize=8, color=INK2)
    ax.set_xticks(x, labels)
    ax.set_xlabel("Accelerometer signal-to-noise ratio")
    ax.set_ylabel("Cross-validated accuracy (%)")
    ax.set_ylim(80, 101)
    ax.legend(ncol=3, loc="lower left")
    ax.set_title("Chatter detection from vibration only (190 test cuts, 10 x 5-fold CV)", loc="left")
    fig.tight_layout()
    fig.savefig(FIG / "3_detection_accuracy.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def fig_sld(clean, lobe, agent4, b_rpm, b_lim, metrics):
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(b_rpm, b_lim, color=INK, lw=1.6, label="Theory (analytical lobe diagram)", zorder=2)
    rv = np.arange(2000, 20001, 25.0)
    ax.plot(rv, lobe.limit_depth(rv), color=BLUE, lw=2, label="Learned by Agent 3 (SVM)", zorder=3)
    for lab, col, mk, name in [(1, STABLE, "o", "Test cut detected stable"), (0, CHATTER, "X", "Test cut detected chatter")]:
        s = clean[clean.detected == lab]
        ax.scatter(s.rpm, s.doc_mm, s=40, c=col, marker=mk, edgecolors=SURF, linewidths=0.8, label=name, zorder=4)
    wrong = clean[clean.detected != clean.label]
    ax.scatter(wrong.rpm, wrong.doc_mm, s=120, facecolors="none", edgecolors=INK, linewidths=1.2,
               label=f"Detection disagrees with theory ({len(wrong)})", zorder=5)
    bp = metrics["decision_agent"]["best_pocket"]
    ax.scatter([bp["rpm"]], [bp["doc_mm"]], s=260, marker="*", c=AQUA, edgecolors=INK, linewidths=0.8,
               label="Agent 4: highest-output safe setting", zorder=6)
    ax.set_xlabel("Spindle speed (rpm)")
    ax.set_ylabel("Axial depth of cut (mm)")
    ax.set_ylim(0, 16)
    ax.set_xlim(2000, 20000)
    ax.legend(loc="upper left", fontsize=9)
    ax.set_title("Stability lobe diagram: below the curve = no chatter", loc="left")
    fig.tight_layout()
    fig.savefig(FIG / "4_stability_lobe_diagram.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def fig_mrr(agent4, b_rpm, b_lim, metrics):
    rv = np.arange(2000, 20001, 25.0)
    safe = agent4.safe_depth(rv)
    mrr = [mrr_cm3_min(r, d) for r, d in zip(rv, safe)]
    keep = 1 - metrics["decision_agent"]["safety_margin_pct"] / 100
    theory = [mrr_cm3_min(r, keep * d) for r, d in zip(rv, np.interp(rv, b_rpm, b_lim))]
    cons = metrics["decision_agent"]["conservative_rule"]
    fig, ax = plt.subplots(figsize=(11, 3.8))
    ax.plot(rv, theory, color=MUTED, lw=1.2, ls="--", label="Same rule applied to the theory curve")
    ax.plot(rv, mrr, color=BLUE, lw=2, label=f"Agent 4 safe setting (learned limit - {metrics['decision_agent']['safety_margin_pct']} %)")
    cons_line = [mrr_cm3_min(r, cons["doc_mm"]) for r in rv]
    ax.plot(rv, cons_line, color=ORANGE, lw=2, label="No lobe diagram: depth below the lowest limit")
    bp = metrics["decision_agent"]["best_pocket"]
    ax.scatter([bp["rpm"]], [bp["mrr_cm3_min"]], s=220, marker="*", c=AQUA, edgecolors=INK, linewidths=0.8, zorder=5)
    ax.annotate(f"{metrics['decision_agent']['mrr_gain_vs_conservative_x']}x the conservative rate\n"
                f"{bp['rpm']:.0f} rpm, {bp['doc_mm']:.1f} mm", xy=(bp["rpm"], bp["mrr_cm3_min"]),
                xytext=(bp["rpm"] + 900, bp["mrr_cm3_min"] * 0.72), color=INK2, fontsize=9,
                arrowprops=dict(arrowstyle="-", color=MUTED))
    ax.set_xlabel("Spindle speed (rpm)")
    ax.set_ylabel("Metal removal rate\n(cm³/min)")
    ax.set_xlim(2000, 20000)
    ax.legend(loc="upper left", fontsize=9)
    ax.set_title("Choosing the speed with the lobe diagram multiplies output without chatter", loc="left")
    fig.tight_layout()
    fig.savefig(FIG / "5_metal_removal_rate.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def main(clean, feats, metrics, lobe, agent4, b_rpm, b_lim):
    FIG.mkdir(parents=True, exist_ok=True)
    fig_spectra()
    fig_feature(feats)
    fig_detection(metrics)
    fig_sld(clean, lobe, agent4, b_rpm, b_lim, metrics)
    fig_mrr(agent4, b_rpm, b_lim, metrics)
    print("figures ->", FIG)
