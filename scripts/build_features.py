"""Run Agent 2 over every recording of the full dataset and save a compact feature table.

The raw dataset (190 simulated milling test cuts, ~170 MB of .mat files) came with the ME5308 course
and is not redistributed here; the feature table it produces is (data/features.csv), together with
the analytical stability boundary and six sample recordings.

    python scripts/build_features.py --data-dir "path/to/Dataset 3 mat"
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.io import loadmat, savemat

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
from chatter_agents.agent1_acquisition import AcquisitionAgent  # noqa: E402
from chatter_agents.agent2_signal import add_sensor_noise, extract_features  # noqa: E402

SAMPLES = {  # file -> why it is a useful example
    "time3_43.mat": "7500 rpm, 2 mm - stable",
    "time3_49.mat": "7500 rpm, 8 mm - chatter",
    "time3_133.mat": "15000 rpm, 8 mm - stable, but large forced vibration (resonance)",
    "time3_157.mat": "16000 rpm, 7 mm - chatter",
    "time3_7.mat": "2500 rpm, 4 mm - chatter",
    "time3_175.mat": "18000 rpm, 3 mm - chatter just above the limit",
}
KEEP_KEYS = ["omega", "b", "time", "xtacc", "xtpos", "Forcex"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    args = ap.parse_args()
    d = Path(args.data_dir)

    sb = loadmat(d / "stability_boundary3.mat")
    b_rpm = sb["omega_vector"].ravel().astype(float)
    b_lim = sb["blim_vector"].ravel() * 1000.0
    pd.DataFrame({"rpm": b_rpm[::10], "doc_limit_mm": np.round(b_lim[::10], 5)}).to_csv(
        REPO / "data" / "stability_boundary.csv", index=False)

    recs = AcquisitionAgent(d, pattern="time3_*.mat").poll()
    rng = np.random.default_rng(42)
    rows = []
    for snr in [None, 30, 20, 10]:
        for r in recs:
            f = extract_features(r if snr is None else add_sensor_noise(r, snr, rng))
            f["file"] = r.source
            f["snr_db"] = "clean" if snr is None else snr
            f["doc_limit_mm"] = float(np.interp(r.rpm, b_rpm, b_lim))
            f["label"] = int(r.doc_mm < f["doc_limit_mm"])          # 1 stable, 0 chatter (theory)
            rows.append(f)
    df = pd.DataFrame(rows)
    df.to_csv(REPO / "data" / "features.csv", index=False, float_format="%.6g")
    print(f"{len(recs)} recordings x 4 noise levels -> data/features.csv "
          f"({(df.snr_db == 'clean').sum()} clean rows, {int(df[df.snr_db == 'clean'].label.sum())} stable)")

    out = REPO / "data" / "sample_recordings"
    out.mkdir(parents=True, exist_ok=True)
    for name in SAMPLES:
        m = loadmat(d / name)
        savemat(out / name, {k: m[k] for k in KEEP_KEYS}, do_compression=True)
    print("samples:", ", ".join(SAMPLES))


if __name__ == "__main__":
    main()
