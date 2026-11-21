"""Agent 2 - signal processing.

Turns a raw vibration recording into a handful of physically meaningful numbers.

Key idea: in a *stable* cut the tool is only forced by the teeth hitting the work, so the
vibration energy sits on the tooth-passing frequency and its harmonics
(f_tp = rpm / 60 x teeth). When *chatter* starts, the tool vibrates near its own natural
frequency, which is generally NOT a multiple of f_tp. The share of energy that falls
between the harmonics is therefore a direct chatter indicator.
"""
from __future__ import annotations

import numpy as np

from .data import N_TEETH, Recording

# Features the chatter classifier sees. Speed and depth are deliberately NOT included:
# chatter must be recognised from the vibration alone.
ML_FEATURES = ["log_peak_ratio", "dominant_offset", "log_chatter_ratio"]


def spectrum(signal: np.ndarray, fs: float):
    x = np.asarray(signal, float)
    x = x - x.mean()
    win = np.hanning(len(x))
    amp = np.abs(np.fft.rfft(x * win)) * 2.0 / win.sum()
    freq = np.fft.rfftfreq(len(x), 1.0 / fs)
    return freq, amp


def harmonic_mask(freq: np.ndarray, f_tp: float, half_width_hz: float) -> np.ndarray:
    """True for bins within +-half_width of k * f_tp (k >= 1)."""
    k = np.round(freq / f_tp)
    return (k >= 1) & (np.abs(freq - k * f_tp) <= half_width_hz)


def _split_power(sig: np.ndarray, fs: float, f_tp: float):
    """(off-harmonic power, total power, freq, amp) of one window, above 0.5 x f_tp."""
    freq, amp = spectrum(sig, fs)
    df = freq[1] - freq[0]
    half_width = max(3 * df, 0.02 * f_tp)          # harmonic band a few bins wide
    band = freq > 0.5 * f_tp                       # ignore DC / very low-frequency drift
    power = amp ** 2
    harm = harmonic_mask(freq, f_tp, half_width) & band
    return float(power[band & ~harm].sum()), float(power[band].sum()), freq, amp, band


def add_sensor_noise(rec: Recording, snr_db: float, rng: np.random.Generator) -> Recording:
    """Copy of `rec` with white accelerometer noise at the given signal-to-noise ratio."""
    sig_rms = float(np.sqrt(np.mean((rec.acc - rec.acc.mean()) ** 2)))
    noise = rng.normal(0.0, sig_rms / (10 ** (snr_db / 20.0)), size=rec.acc.shape)
    return Recording(rpm=rec.rpm, doc_mm=rec.doc_mm, time=rec.time, acc=rec.acc + noise,
                     disp=rec.disp, force=rec.force, source=rec.source)


def extract_features(rec: Recording, steady_fraction: float = 0.5, teeth: int = N_TEETH) -> dict:
    """Features from the steady-state part of the cut (last `steady_fraction` of the record).

    chatter_ratio  - share of vibration energy NOT on tooth-passing harmonics (0 = clean, 1 = chatter)
    log_peak_ratio - strongest off-harmonic peak / strongest harmonic peak (log10); robust to sensor noise
    dominant_offset- where the biggest peak sits between two harmonics (0 = on one, 0.5 = midway)
    chatter_growth - log10 change of that off-harmonic energy from the 2nd to the 4th quarter of the
                     record: > 0 means a self-excited vibration is still growing (unstable), < 0 means
                     the start-up transient is dying away (stable)
    """
    n = len(rec.acc)
    start = int(n * (1.0 - steady_fraction))
    acc = rec.acc[start:]
    fs = rec.fs
    f_tp = rec.rpm / 60.0 * teeth

    off, total, freq, amp, band = _split_power(acc, fs, f_tp)
    chatter_ratio = off / total if total > 0 else 0.0

    q = n // 4
    off_early, tot_early, *_ = _split_power(rec.acc[q:2 * q], fs, f_tp)
    off_late, tot_late, *_ = _split_power(rec.acc[3 * q:], fs, f_tp)
    eps = 1e-12 * max(tot_early, tot_late, 1e-30)
    growth = float(np.log10((off_late + eps) / (off_early + eps)))

    # strongest single off-harmonic peak vs strongest harmonic peak: broadband sensor noise spreads
    # over thousands of bins, so this stays small for stable cuts even on a noisy signal
    df = freq[1] - freq[0]
    harm = harmonic_mask(freq, f_tp, max(3 * df, 0.02 * f_tp)) & band
    peak_off = float(amp[band & ~harm].max()) if (band & ~harm).any() else 0.0
    peak_on = float(amp[harm].max()) if harm.any() else 1e-30
    log_peak_ratio = float(np.log10((peak_off + 1e-30) / peak_on))

    i_dom = int(np.argmax(np.where(band, amp, 0)))
    f_dom = float(freq[i_dom])
    offset = abs(f_dom / f_tp - round(f_dom / f_tp))   # 0 = on a harmonic, 0.5 = halfway between

    acc_rms = float(np.sqrt(np.mean((acc - acc.mean()) ** 2)))
    feats = {
        "rpm": rec.rpm,
        "doc_mm": rec.doc_mm,
        "tooth_freq_hz": f_tp,
        "chatter_ratio": float(chatter_ratio),
        "log_chatter_ratio": float(np.log10(chatter_ratio + 1e-12)),
        "chatter_growth": growth,
        "log_peak_ratio": log_peak_ratio,
        "dominant_offset": float(offset),
        "dominant_freq_hz": f_dom,
        "peak_amplitude": float(amp[i_dom]),
        "acc_rms": acc_rms,
        "log_acc_rms": float(np.log10(acc_rms + 1e-12)),
        "disp_p2p_um": float(np.ptp(rec.disp[start:]) * 1e6) if rec.disp is not None else float("nan"),
        "force_peak_n": float(np.max(np.abs(rec.force[start:]))) if rec.force is not None else float("nan"),
    }
    return feats


class SignalAgent:
    name = "agent2_signal"

    def run(self, rec: Recording) -> dict:
        return extract_features(rec)
