"""Write the numbers from results/metrics.json into README.md (placeholders __LIKE_THIS__)."""
import json
import re
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
m = json.loads((REPO / "results" / "metrics.json").read_text(encoding="utf-8"))
readme = (REPO / "README.md").read_text(encoding="utf-8")

det = m["chatter_detection_cv"]
rows = ["| Model | Clean | 30 dB | 20 dB | 10 dB |", "|---|---|---|---|---|"]
for name in ["SVM", "Random Forest", "Naive Bayes"]:
    rows.append(f"| {name} | " + " | ".join(f"{det[s][name]['accuracy_pct']:.1f} %" for s in ["clean", "30", "20", "10"]) + " |")
detection = "\n".join(rows)

sld = m["stability_lobe_diagram"]
rows = ["| Lobe model (trained on detected labels) | Map agreement | Mean limit error | Area wrongly called safe |",
        "|---|---|---|---|"]
for name in ["SVM", "Random Forest", "Naive Bayes"]:
    r = sld[name]["from_detected_labels"]
    rows.append(f"| {name} | {r['map_agreement_pct']:.1f} % | {r['limit_mae_mm']:.2f} mm | {r['unsafe_area_pct']:.1f} % |")
b = sld["Bracket + interpolation (non-ML baseline)"]["from_detected_labels"]
rows.append(f"| Baseline: midpoint between tested depths, interpolated | n/a | {b['limit_mae_mm']:.2f} mm | n/a |")
sld_table = "\n".join(rows)

d = m["decision_agent"]
rows = ["| Safety margin | Same-speed advice truly stable |", "|---|---|"]
for k, v in m["safety_margin_study"].items():
    rows.append(f"| {k} | {v['same_speed_advice_truly_stable_pct']:.1f} % |")
bp, cons = d["best_pocket"], d["conservative_rule"]
safety = "\n".join(rows) + (
    f"\n\nWith the default 20 % margin, the best-speed advice was chatter-free in "
    f"**{d['best_speed_advice_truly_stable_pct']:.0f} %** of requests. The highest-output safe setting found is "
    f"**{bp['rpm']:.0f} rpm at {bp['doc_mm']:.1f} mm** (theory limit {bp['theory_limit_mm']:.1f} mm), giving "
    f"{bp['mrr_cm3_min']:.0f} cm³/min against {cons['mrr_cm3_min']:.0f} cm³/min when the depth is kept below the lowest "
    f"chatter limit at any speed (**{d['mrr_gain_vs_conservative_x']}x**). For requests that would chatter, the "
    f"best-speed option gives a median **{d['median_mrr_gain_for_chatter_requests_x']}x** the MRR of simply reducing the depth.")

dep = m["deployed_detector"]
for key, val in {"__DETECTION_TABLE__": detection, "__SLD_TABLE__": sld_table, "__SAFETY_TABLE__": safety,
                 "__DEP_ACC__": f"{dep['accuracy_pct']:.1f}", "__DEP_REC__": f"{dep['chatter_recall_pct']:.0f}",
                 "__GAIN__": f"{d['mrr_gain_vs_conservative_x']}"}.items():
    readme = readme.replace(key, val)
left = re.findall(r"__[A-Z_]+__", readme)
(REPO / "README.md").write_text(readme, encoding="utf-8")
print("README filled; placeholders left:", left)
