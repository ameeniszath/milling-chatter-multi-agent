"""
FastAPI - Stability Lobe Diagram Generator
Uses raw multipart parsing — works regardless of field names or stream format.
First file = stable, Second file = unstable.

POST http://host.docker.internal:8002/generate_sld
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

app = FastAPI(title="Stability Lobe Diagram API")


@app.get("/")
async def root():
    return {"status": "ok", "message": "SLD API running. POST to /generate_sld"}

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


@app.post("/generate_sld")
async def generate_sld(request: Request):
    from sklearn.svm import SVC
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline

    # ── Read raw body and extract all file bytes from multipart ──────────────
    content_type = request.headers.get("content-type", "")
    if "multipart/form-data" not in content_type:
        raise HTTPException(400, "Expected multipart/form-data")

    # Extract boundary
    boundary = None
    for part in content_type.split(";"):
        part = part.strip()
        if part.startswith("boundary="):
            boundary = part[len("boundary="):].strip()
            break

    if not boundary:
        raise HTTPException(400, "No boundary found in content-type")

    raw_body = await request.body()

    # ── Manual multipart split ────────────────────────────────────────────────
    sep = ("--" + boundary).encode()
    end = ("--" + boundary + "--").encode()

    parts = []
    for chunk in raw_body.split(sep):
        if not chunk or chunk.strip() == b"--" or chunk.strip() == b"" or chunk.strip().startswith(b"--"):
            continue
        if end in chunk:
            chunk = chunk[:chunk.index(end)]
        # Split headers from body at \r\n\r\n
        if b"\r\n\r\n" in chunk:
            _, file_data = chunk.split(b"\r\n\r\n", 1)
            # Strip trailing \r\n
            file_data = file_data.rstrip(b"\r\n")
            if file_data:
                parts.append(file_data)

    if len(parts) < 2:
        raise HTTPException(422,
            f"Expected 2 files in multipart body, found {len(parts)}. "
            "Ensure both stable and unstable Excel files are attached.")

    # First part = stable, second part = unstable
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

    # ── Train SVM ─────────────────────────────────────────────────────────────
    clf = Pipeline([
        ("scaler", StandardScaler()),
        ("svm",    SVC(kernel="rbf", C=10, gamma="scale", probability=True)),
    ])
    clf.fit(X, y)

    # ── Grid for decision boundary ────────────────────────────────────────────
    s0, s1 = X[:,0].min(), X[:,0].max()
    d0, d1 = X[:,1].min(), X[:,1].max()
    ms, md = (s1-s0)*0.12, (d1-d0)*0.12

    gs = np.linspace(s0-ms, s1+ms, 600)
    gd = np.linspace(d0-md, d1+md, 600)
    GS, GD = np.meshgrid(gs, gd)
    Z = clf.predict_proba(np.c_[GS.ravel(), GD.ravel()])[:,1].reshape(GS.shape)

    # ── Plot ──────────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 6), dpi=130)
    cf = ax.contourf(GS, GD, Z, levels=np.linspace(0,1,101), cmap="RdYlGn_r", alpha=0.55)
    fig.colorbar(cf, ax=ax, label="Instability probability", pad=0.02)
    ax.contour(GS, GD, Z, levels=[0.5], colors="black", linewidths=2.2)
    ax.scatter(X_s[:,0], X_s[:,1], c="#2ecc71", edgecolors="k", s=60, lw=0.6, zorder=5)
    ax.scatter(X_u[:,0], X_u[:,1], c="#e74c3c", edgecolors="k", s=60, lw=0.6, zorder=5)
    ax.set_xlabel(f"Spindle Speed  ({speed_col})", fontsize=13)
    ax.set_ylabel(f"Axial Depth of Cut  ({doc_col})", fontsize=13)
    ax.set_title("Stability Lobe Diagram  (ML – SVM Decision Boundary)",
                 fontsize=14, fontweight="bold", pad=12)
    ax.legend(handles=[
        mpatches.Patch(color="#2ecc71", label="Stable"),
        mpatches.Patch(color="#e74c3c", label="Unstable"),
        plt.Line2D([0],[0], color="black", lw=2, label="SLD boundary (p=0.5)"),
    ], loc="upper right", fontsize=10)
    ax.grid(True, linestyle="--", alpha=0.4)
    fig.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)

    return StreamingResponse(buf, media_type="image/png",
        headers={"Content-Disposition":
                 'attachment; filename="stability_lobe_diagram.png"'})


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("stability_lobe_api:app", host="0.0.0.0", port=8002, reload=True)
