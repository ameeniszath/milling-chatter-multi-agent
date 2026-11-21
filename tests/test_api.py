import sys
from pathlib import Path

from fastapi.testclient import TestClient

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
from chatter_agents.api import app  # noqa: E402

client = TestClient(app)
SAMPLE = REPO / "data" / "sample_recordings"


def test_health():
    r = client.get("/health")
    assert r.status_code == 200 and r.json()["status"] == "ok"


def test_pipeline_flags_chatter_and_gives_safe_advice():
    with open(SAMPLE / "time3_49.mat", "rb") as f:
        r = client.post("/pipeline/analyse", files={"file": ("time3_49.mat", f, "application/octet-stream")})
    assert r.status_code == 200
    body = r.json()
    assert body["detection"]["detected"] == "chatter"
    assert body["advice"]["option_same_speed"]["doc_mm"] < 8.0


def test_agent1_then_agent2_then_agent3_over_http():
    with open(SAMPLE / "time3_43.mat", "rb") as f:
        sig = client.post("/agent1/extract?include_signal=true",
                          files={"file": ("time3_43.mat", f, "application/octet-stream")}).json()
    feats = client.post("/agent2/features", json={k: sig[k] for k in ("rpm", "doc_mm", "time", "acc")}).json()
    assert feats["chatter_ratio"] < 0.01
    assert client.post("/agent3/detect", json=feats).json()["detected"] == "stable"


def test_agent4_and_lobe_endpoints():
    a = client.post("/agent4/advise", json={"rpm": 9500, "doc_mm": 3.0}).json()
    assert a["status"] in ("stable", "chatter") and a["option_best_speed"]["mrr_cm3_min"] > 0
    lobe = client.get("/agent3/lobe").json()
    assert len(lobe["rpm"]) == len(lobe["doc_limit_mm"]) > 10
    assert client.get("/agent3/lobe.png").headers["content-type"] == "image/png"
