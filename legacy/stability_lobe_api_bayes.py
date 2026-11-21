"""
FastAPI - Stability Lobe Diagram Generator (Gaussian Naive Bayes)
Uses raw multipart parsing — works regardless of field names or stream format.
First file = stable, Second file = unstable.

POST http://host.docker.internal:8002/generate_sld_bayes
"""

import io
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec

from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import StreamingResponse

app = FastAPI(title="Stability Lobe Diagram API - Gaussian Naive Bayes")


@app.get("/")
async def root():
    return {"status": "ok", "message": "Bayes SLD API running. POST to /generate_sld_bayes"}

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


@app.post("/generate_sld_bayes")
async def generate_sld_bayes(request: Request):
    from sklearn.naive_bayes import GaussianNB
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.metrics import accuracy_score, confusion_matrix

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

    # ── Train Gaussian Naive Bayes ────────────────────────────────────────────
    # Isotonic calibration gives smooth, reliable posterior probabilities
    gnb = GaussianNB()
    clf = Pipeline([
        ("scaler", StandardScaler()),
        ("gnb",    CalibratedClassifierCV(gnb, method="isotonic", cv=min(5, len(y)//2))),
    ])
    clf.fit(X, y)

    y_pred   = clf.predict(X)
    acc      = accuracy_score(y, y_pred)
    cm       = confusion_matrix(y, y_pred)

    # ── Class statistics (mean / std per class) ───────────────────────────────
    scaler      = clf.named_steps["scaler"]
    X_scaled    = scaler.transform(X)
    means_s     = X_s.mean(axis=0)
    means_u     = X_u.mean(axis=0)
    stds_s      = X_s.std(axis=0)
    stds_u      = X_u.std(axis=0)

    # ── Decision grid ─────────────────────────────────────────────────────────
    s0, s1 = X[:,0].min(), X[:,0].max()
    d0, d1 = X[:,1].min(), X[:,1].max()
    ms, md = (s1-s0)*0.14, (d1-d0)*0.14

    gs = np.linspace(s0-ms, s1+ms, 600)
    gd = np.linspace(d0-md, d1+md, 600)
    GS, GD = np.meshgrid(gs, gd)
    Z = clf.predict_proba(np.c_[GS.ravel(), GD.ravel()])[:,1].reshape(GS.shape)

    # ── Plot layout: SLD + Confusion Matrix + Stats ───────────────────────────
    fig = plt.figure(figsize=(15, 6), dpi=130)
    gs_layout = GridSpec(2, 3, figure=fig,
                         width_ratios=[3, 1, 1],
                         hspace=0.45, wspace=0.38)

    ax      = fig.add_subplot(gs_layout[:, 0])   # main SLD (spans both rows)
    ax_cm   = fig.add_subplot(gs_layout[0, 1])   # confusion matrix
    ax_stat = fig.add_subplot(gs_layout[1, 1])   # class stats
    ax_pdf  = fig.add_subplot(gs_layout[:, 2])   # 1-D marginal PDFs

    # ── (1) Main SLD ──────────────────────────────────────────────────────────
    cf = ax.contourf(GS, GD, Z, levels=np.linspace(0,1,101),
                     cmap="RdYlGn_r", alpha=0.60)
    fig.colorbar(cf, ax=ax, label="P(unstable | x)", pad=0.02)

    # Uncertainty band: 0.35 – 0.65
    ax.contourf(GS, GD, Z, levels=[0.35, 0.65],
                colors=["#f39c12"], alpha=0.18)

    # Confidence contours
    ax.contour(GS, GD, Z, levels=[0.1, 0.9],
               colors=["#27ae60","#c0392b"], linewidths=1.1,
               linestyles=":", alpha=0.8)
    ax.contour(GS, GD, Z, levels=[0.5],
               colors="black", linewidths=2.5)

    # Gaussian ellipses (1σ) for each class
    for mean, std, color in [(means_s, stds_s, "#27ae60"),
                              (means_u, stds_u, "#c0392b")]:
        ellipse = mpatches.Ellipse(mean, 2*std[0], 2*std[1],
                                   edgecolor=color, facecolor="none",
                                   linewidth=1.8, linestyle="--", zorder=4)
        ax.add_patch(ellipse)

    ax.scatter(X_s[:,0], X_s[:,1], c="#2ecc71", edgecolors="k",
               s=65, lw=0.7, zorder=5)
    ax.scatter(X_u[:,0], X_u[:,1], c="#e74c3c", edgecolors="k",
               s=65, lw=0.7, zorder=5)

    ax.set_xlabel(f"Spindle Speed  ({speed_col})", fontsize=12)
    ax.set_ylabel(f"Axial Depth of Cut  ({doc_col})", fontsize=12)
    ax.set_title("Stability Lobe Diagram\n(Gaussian Naive Bayes)",
                 fontsize=13, fontweight="bold")
    ax.legend(handles=[
        mpatches.Patch(color="#2ecc71", label="Stable"),
        mpatches.Patch(color="#e74c3c", label="Unstable"),
        plt.Line2D([0],[0], color="black",   lw=2.5,      label="Decision boundary (p=0.5)"),
        plt.Line2D([0],[0], color="#27ae60", lw=1.1, ls=":", label="p=0.10"),
        plt.Line2D([0],[0], color="#c0392b", lw=1.1, ls=":", label="p=0.90"),
        mpatches.Patch(color="#f39c12", alpha=0.4,          label="Uncertainty zone"),
        plt.Line2D([0],[0], color="#27ae60", lw=1.8, ls="--", label="1σ ellipse (stable)"),
        plt.Line2D([0],[0], color="#c0392b", lw=1.8, ls="--", label="1σ ellipse (unstable)"),
    ], loc="upper right", fontsize=7.5, framealpha=0.85)
    ax.grid(True, linestyle="--", alpha=0.3)

    # ── (2) Confusion Matrix ──────────────────────────────────────────────────
    im = ax_cm.imshow(cm, cmap="Blues", interpolation="nearest")
    ax_cm.set_xticks([0,1]); ax_cm.set_yticks([0,1])
    ax_cm.set_xticklabels(["Stable","Unstable"], fontsize=8)
    ax_cm.set_yticklabels(["Stable","Unstable"], fontsize=8, rotation=90, va="center")
    ax_cm.set_xlabel("Predicted", fontsize=9); ax_cm.set_ylabel("Actual", fontsize=9)
    ax_cm.set_title(f"Confusion Matrix\nAcc: {acc*100:.1f}%", fontsize=9, fontweight="bold")
    for i in range(2):
        for j in range(2):
            ax_cm.text(j, i, str(cm[i,j]), ha="center", va="center",
                       fontsize=13, fontweight="bold",
                       color="white" if cm[i,j] > cm.max()/2 else "black")

    # ── (3) Class statistics table ────────────────────────────────────────────
    ax_stat.axis("off")
    tdata = [
        ["", "Stable", "Unstable"],
        [f"μ {speed_col}", f"{means_s[0]:.0f}", f"{means_u[0]:.0f}"],
        [f"σ {speed_col}", f"{stds_s[0]:.0f}",  f"{stds_u[0]:.0f}"],
        [f"μ {doc_col}",   f"{means_s[1]:.3f}", f"{means_u[1]:.3f}"],
        [f"σ {doc_col}",   f"{stds_s[1]:.3f}",  f"{stds_u[1]:.3f}"],
        ["N samples",      str(len(X_s)),        str(len(X_u))],
    ]
    tbl = ax_stat.table(cellText=tdata[1:], colLabels=tdata[0],
                        loc="center", cellLoc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(7.5)
    tbl.scale(1, 1.35)
    for (r, c), cell in tbl.get_celld().items():
        if r == 0:
            cell.set_facecolor("#2c3e50")
            cell.set_text_props(color="white", fontweight="bold")
        elif c == 1:
            cell.set_facecolor("#eafaf1")
        elif c == 2:
            cell.set_facecolor("#fdf2f2")
    ax_stat.set_title("Class Statistics", fontsize=9, fontweight="bold", pad=4)

    # ── (4) Marginal 1-D PDFs (spindle speed) ────────────────────────────────
    from scipy.stats import gaussian_kde
    for data, color, label in [(X_s[:,0], "#2ecc71", "Stable"),
                                (X_u[:,0], "#e74c3c", "Unstable")]:
        kde  = gaussian_kde(data, bw_method="scott")
        xs   = np.linspace(data.min()*0.85, data.max()*1.15, 400)
        ax_pdf.fill_between(xs, kde(xs), alpha=0.30, color=color)
        ax_pdf.plot(xs, kde(xs), color=color, lw=2, label=label)
        ax_pdf.axvline(data.mean(), color=color, lw=1.2, linestyle="--", alpha=0.8)

    ax_pdf.set_xlabel(f"Spindle Speed ({speed_col})", fontsize=9)
    ax_pdf.set_ylabel("Density", fontsize=9)
    ax_pdf.set_title("Marginal PDF\n(Spindle Speed)", fontsize=9, fontweight="bold")
    ax_pdf.legend(fontsize=8)
    ax_pdf.grid(True, linestyle="--", alpha=0.3)
    ax_pdf.spines[["top","right"]].set_visible(False)

    # Global title
    fig.suptitle(
        f"Gaussian Naive Bayes SLD  |  "
        f"Stable: {len(X_s)} pts  ·  Unstable: {len(X_u)} pts  ·  Accuracy: {acc*100:.1f}%",
        fontsize=11, color="#444444", y=1.02
    )

    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)

    return StreamingResponse(buf, media_type="image/png",
        headers={"Content-Disposition":
                 'attachment; filename="sld_bayes.png"'})


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("stability_lobe_api_bayes:app", host="0.0.0.0", port=8002, reload=True)
