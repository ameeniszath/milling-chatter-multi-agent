import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
from chatter_agents.agent2_signal import extract_features  # noqa: E402
from chatter_agents.agent3_stability import LobeModel, monotonic_augment  # noqa: E402
from chatter_agents.agent4_decision import DecisionAgent, mrr_cm3_min  # noqa: E402
from chatter_agents.data import Recording, load_mat  # noqa: E402

FS = 25000.0


def synth(rpm=6000.0, chatter_amp=0.0, chatter_hz=1130.0, growth=0.0, seconds=0.6, seed=0):
    t = np.arange(int(FS * seconds)) / FS
    f_tp = rpm / 60 * 4
    x = sum(np.sin(2 * np.pi * k * f_tp * t) / k for k in range(1, 6))
    x = x + chatter_amp * np.exp(growth * t) * np.sin(2 * np.pi * chatter_hz * t)
    x = x + np.random.default_rng(seed).normal(0, 1e-4, len(t))
    return Recording(rpm=rpm, doc_mm=1.0, time=t, acc=x)


def test_forced_vibration_only_is_not_chatter():
    f = extract_features(synth())
    assert f["chatter_ratio"] < 0.01
    assert f["dominant_offset"] < 0.05


def test_off_harmonic_vibration_is_flagged():
    f = extract_features(synth(chatter_amp=3.0))
    assert f["chatter_ratio"] > 0.5
    assert f["dominant_freq_hz"] == pytest.approx(1130, abs=5)
    assert f["dominant_offset"] > 0.1


def test_growing_chatter_has_positive_growth():
    assert extract_features(synth(chatter_amp=0.05, growth=8.0))["chatter_growth"] > 0.5
    assert extract_features(synth(chatter_amp=0.05, growth=-8.0))["chatter_growth"] < -0.5


def test_monotonic_prior_adds_deeper_chatter_points():
    r, d, lab = monotonic_augment(np.array([5000.0, 5000.0]), np.array([1.0, 3.0]), np.array([1, 0]), 6.0)
    deeper = d[(r == 5000) & (d > 3)]
    assert len(deeper) and np.all(lab[(r == 5000) & (d > 3)] == 0)


def _toy_lobe():
    # true limit: 2 mm everywhere except a deep pocket (8 mm) around 15 000 rpm
    rng = np.random.default_rng(1)
    rpm = rng.uniform(2000, 20000, 900)
    doc = rng.uniform(0.2, 12, 900)
    lim = np.where(np.abs(rpm - 15000) < 1200, 8.0, 2.0)
    return LobeModel("Random Forest", doc_max_mm=12).fit(rpm, doc, (doc < lim).astype(int), monotonic=False)


def test_decision_agent_moves_to_the_lobe_pocket():
    agent = DecisionAgent(_toy_lobe(), safety_margin=0.15)
    a = agent.advise(6000, 5.0)
    assert a.status == "chatter"
    assert a.option_same_speed["doc_mm"] < 2.0
    assert abs(a.option_best_speed["rpm"] - 15000) < 1500
    assert a.option_best_speed["mrr_cm3_min"] > 3 * a.option_same_speed["mrr_cm3_min"]


def test_mrr_formula():
    # ae 5 mm x ap 2 mm x (0.05 mm x 4 teeth x 10 000 rpm) = 20 000 mm3/min = 20 cm3/min
    assert mrr_cm3_min(10000, 2.0) == pytest.approx(20.0)


def test_sample_recordings_are_classified_correctly():
    from chatter_agents.orchestrator import load_trained

    detector, _, _ = load_trained()
    expect = {"time3_43.mat": 1, "time3_49.mat": 0, "time3_133.mat": 1, "time3_157.mat": 0, "time3_7.mat": 0}
    for name, lab in expect.items():
        f = extract_features(load_mat(REPO / "data" / "sample_recordings" / name))
        assert int(detector.stable_probability([f])[0] >= 0.5) == lab, name
