"""
FastAPI - Stability Lobe Diagram Generator (Random Forest)
Uses raw multipart parsing — works regardless of field names or stream format.
First file = stable, Second file = unstable.

POST http://host.docker.internal:8002/generate_sld_rf
"""

import io
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import StreamingResponse

app = FastAPI(title="Stability Lobe Diagram API - Random Forest")


@app.get("/")
async def root():
    return {"status": "ok", "message": "RF SLD API running. POST to /generate_sld_rf"}

@app.get("/health")
async def health():
    return {"status": "ok"}


def _read_excel(data: bytes) -> pd.DataFrame:
    df = pd.read_excel(io.BytesIO(data))
    df.columns = [c.strip().lower() for c in df.columns]
    return df


def _extract_features(df: pd.DataFrame):
    speed_kw = ["spindle", "speed", "rpm", "n"]
    doc_kw   = ["depth", "doc", "axial", "ap", "mm", "cut"]

    def find_col(keywords):
        for kw in keywords:
            for col in df.columns:
                if kw in col:
                    return col
        return None

    speed_col = find_col(speed_kw) or df.columns[0]
    doc_col   = find_col(doc_kw)   or df.columns[1]
    return df[[speed_col, doc_col]].values.astype(float), speed_col, doc_col


def _parse_multipart(raw_body: bytes, boundary: str):
    sep = ("--" + boundary).encode()
    end = ("--" + boundary + "--").encode()
    parts = []
    for chunk in raw_body.split(sep):
        if not chunk or chunk.strip() in (b"", b"--") or chunk.strip().startswith(b"--"):
            continue
        if end in chunk:
            chunk = chunk[:chunk.index(end)]
        if b"\r\n\r\n" in chunk:
            _, file_data = chunk.split(b"\r\n\r\n", 1)
            file_data = file_data.rstrip(b"\r\n")
            if file_data:
                parts.append(file_data)
    return parts


@app.post("/generate_sld_rf")
async def generate_sld_rf(request: Request):
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline

    # ── Parse multipart ───────────────────────────────────────────────────────
    content_type = request.headers.get("content-type", "")
    if "multipart/form-data" not in content_type:
        raise HTTPException(400, "Expected multipart/form-data")

    boundary = None
    for part in content_type.split(";"):
        part = part.strip()
        if part.startswith("boundary="):
            boundary = part[len("boundary="):].strip()
            break

    if not boundary:
        raise HTTPException(400, "No boundary in content-type")

    raw_body = await request.body()
    parts    = _parse_multipart(raw_body, boundary)

    if len(parts) < 2:
        raise HTTPException(422,
            f"Expected 2 files, found {len(parts)}. "
            "Send stable file first, unstable file second.")

    try:
        df_s = _read_excel(parts[0])
        df_u = _read_excel(parts[1])
    except Exception as e:
        raise HTTPException(400, f"Could not parse Excel: {e}")

    X_s, speed_col, doc_col = _extract_features(df_s)
    X_u, _, _               = _extract_features(df_u)

    X = np.vstack([X_s, X_u])
    y = np.array([0]*len(X_s) + [1]*len(X_u))

    if len(np.unique(y)) < 2:
        raise HTTPException(422, "Need both stable and unstable samples.")

    # ── Train Random Forest ───────────────────────────────────────────────────
    clf = Pipeline([
        ("scaler", StandardScaler()),
        ("rf", RandomForestClassifier(
            n_estimators=300,
            max_depth=None,
            min_samples_split=2,
            min_samples_leaf=1,
            max_features="sqrt",
            bootstrap=True,
            random_state=42,
            n_jobs=-1,
        )),
    ])
    clf.fit(X, y)

    # Feature importances for annotation
    rf_model    = clf.named_steps["rf"]
    importances = rf_model.feature_importances_
    feat_names  = [speed_col, doc_col]

    # ── Decision grid ─────────────────────────────────────────────────────────
    s0, s1 = X[:,0].min(), X[:,0].max()
    d0, d1 = X[:,1].min(), X[:,1].max()
    ms, md = (s1-s0)*0.12, (d1-d0)*0.12

    gs = np.linspace(s0-ms, s1+ms, 600)
    gd = np.linspace(d0-md, d1+md, 600)
    GS, GD = np.meshgrid(gs, gd)
    Z = clf.predict_proba(np.c_[GS.ravel(), GD.ravel()])[:,1].reshape(GS.shape)

    # ── Plot ──────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), dpi=130,
                             gridspec_kw={"width_ratios": [3, 1]})
    ax, ax_imp = axes

    # -- Main SLD plot --
    cf = ax.contourf(GS, GD, Z, levels=np.linspace(0, 1, 101),
                     cmap="RdYlGn_r", alpha=0.60)
    fig.colorbar(cf, ax=ax, label="Instability probability", pad=0.02)

    # Confidence contours at 25%, 75%
    ax.contour(GS, GD, Z, levels=[0.25, 0.75],
               colors=["#27ae60", "#c0392b"], linewidths=1.2,
               linestyles="--", alpha=0.7)

    # Decision boundary
    ax.contour(GS, GD, Z, levels=[0.5],
               colors="black", linewidths=2.5, linestyles="-")

    ax.scatter(X_s[:,0], X_s[:,1], c="#2ecc71", edgecolors="k",
               s=65, lw=0.7, zorder=5, label="Stable")
    ax.scatter(X_u[:,0], X_u[:,1], c="#e74c3c", edgecolors="k",
               s=65, lw=0.7, zorder=5, label="Unstable")

    ax.set_xlabel(f"Spindle Speed  ({speed_col})", fontsize=13)
    ax.set_ylabel(f"Axial Depth of Cut  ({doc_col})", fontsize=13)
    ax.set_title("Stability Lobe Diagram\n(Random Forest — Decision Boundary)",
                 fontsize=13, fontweight="bold", pad=10)
    ax.legend(handles=[
        mpatches.Patch(color="#2ecc71", label="Stable"),
        mpatches.Patch(color="#e74c3c", label="Unstable"),
        plt.Line2D([0],[0], color="black",   lw=2.5, label="SLD boundary (p=0.50)"),
        plt.Line2D([0],[0], color="#27ae60", lw=1.2, ls="--", label="p=0.25"),
        plt.Line2D([0],[0], color="#c0392b", lw=1.2, ls="--", label="p=0.75"),
    ], loc="upper right", fontsize=9)
    ax.grid(True, linestyle="--", alpha=0.35)

    # -- Feature importance bar chart --
    colors_imp = ["#3498db", "#e67e22"]
    bars = ax_imp.barh(feat_names, importances, color=colors_imp,
                       edgecolor="k", linewidth=0.6, height=0.45)
    for bar, val in zip(bars, importances):
        ax_imp.text(val + 0.01, bar.get_y() + bar.get_height()/2,
                    f"{val:.3f}", va="center", ha="left", fontsize=10, fontweight="bold")
    ax_imp.set_xlim(0, 1.0)
    ax_imp.set_xlabel("Importance", fontsize=11)
    ax_imp.set_title("Feature\nImportance", fontsize=11, fontweight="bold")
    ax_imp.grid(True, axis="x", linestyle="--", alpha=0.4)
    ax_imp.spines[["top","right"]].set_visible(False)

    # Global title
    n_stable   = len(X_s)
    n_unstable = len(X_u)
    n_trees    = rf_model.n_estimators
    fig.suptitle(
        f"Random Forest SLD  |  {n_trees} trees  |  "
        f"Stable: {n_stable} pts  ·  Unstable: {n_unstable} pts",
        fontsize=11, color="#555555", y=1.01
    )

    fig.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)

    return StreamingResponse(buf, media_type="image/png",
        headers={"Content-Disposition":
                 'attachment; filename="sld_random_forest.png"'})


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("stability_lobe_api_rf:app", host="0.0.0.0", port=8002, reload=True)
