"""Rolling (past-only) ECG R-peak threshold for the streaming path."""
from pathlib import Path
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from driver_state.features import detect_ecg_peaks, rolling_peak_height


def ecg(seconds=200, fs=100., noisy_s=0., seed=0):
    rng = np.random.default_rng(seed)
    t = np.arange(0, seconds, 1 / fs)
    x = rng.normal(0, .02, t.size)
    beats = np.arange(1., seconds - 1, .8)
    for b in beats:
        x += np.exp(-.5 * ((t - b) / .012) ** 2)
    if noisy_s:
        m = t < noisy_s
        x[m] += rng.normal(0, 3., m.sum())          # contact noise far above the R waves
    return x, fs, beats


def recall(peaks, fs, beats, lo, hi):
    found = peaks / fs
    target = beats[(beats >= lo) & (beats < hi)]
    return np.mean([np.any(np.abs(found - b) < .05) for b in target])


def test_rolling_threshold_ignores_startup_noise():
    x, fs, beats = ecg(400, noisy_s=60)
    glob = detect_ecg_peaks(x, fs)
    roll = detect_ecg_peaks(x, fs, 'rolling')
    assert recall(glob, fs, beats, 160, 400) < .5     # buffer-wide percentile is dominated by the noise
    assert recall(roll, fs, beats, 160, 400) > .95


def test_rolling_threshold_is_past_only():
    x, fs, _ = ecg(200)
    full = rolling_peak_height(x ** 2, fs)
    cut = rolling_peak_height(x[:int(125 * fs)] ** 2, fs)
    # Blocks that started before the cut keep their threshold; nothing after a block start is used.
    assert np.array_equal(full[:int(120 * fs)], cut[:int(120 * fs)])


def test_clean_signal_matches_global_and_bad_mode_rejected():
    x, fs, beats = ecg(200)
    assert recall(detect_ecg_peaks(x, fs, 'rolling'), fs, beats, 0, 200) == recall(detect_ecg_peaks(x, fs), fs, beats, 0, 200) == 1
    with pytest.raises(ValueError):
        detect_ecg_peaks(x, fs, 'future')
