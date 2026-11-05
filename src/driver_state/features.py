"""윈도우 → 특징 벡터 (MLP·RF·XGBoost·LightGBM 공통 입력).

R-peak / 맥파 peak 검출은 기준 구현(reference)이다. 데이터/전처리 담당의 정식
파이프라인(Pan-Tompkins, PPG 보정 로직 등)이 나오면 같은 함수 이름·출력 형식으로 교체한다.
계산할 수 없는 특징(peak 부족 등)은 NaN 으로 두고, 정규화 단계에서 학습셋 중앙값으로 채운다.
"""
from __future__ import annotations

import numpy as np
from scipy.signal import butter, find_peaks, sosfiltfilt

from .data import WindowSet
from .spec import CAN_FEATURES, ECG_FEATURES, PPG_FEATURES, InputSpec

_MIN_IBI, _MAX_IBI = 0.3, 2.0   # 허용 박동 간격 [s] (= 30~200 bpm)


def _bandpass(x: np.ndarray, fs: float, lo: float, hi: float) -> np.ndarray:
    sos = butter(3, [lo, min(hi, 0.45 * fs)], btype="band", fs=fs, output="sos")
    return sosfiltfilt(sos, x)


def detect_ecg_peaks(ecg: np.ndarray, fs: float) -> np.ndarray:
    """간이 R-peak 검출: 5–15 Hz 대역통과 → 제곱 → 최소 간격 0.3초 조건의 peak."""
    f = _bandpass(ecg, fs, 5.0, 15.0)
    e = f ** 2
    peaks, _ = find_peaks(e, distance=int(_MIN_IBI * fs), height=0.3 * np.percentile(e, 99))
    return peaks


def detect_ppg_peaks(ppg: np.ndarray, fs: float) -> np.ndarray:
    """간이 맥파 peak 검출: 0.5–8 Hz 대역통과 후 최소 간격 0.3초 조건의 peak."""
    f = _bandpass(ppg, fs, 0.5, 8.0)
    peaks, _ = find_peaks(f, distance=int(_MIN_IBI * fs), prominence=0.3 * np.std(f))
    return peaks


def _ibi_stats(peaks: np.ndarray, fs: float) -> tuple[float, float, float, float, float]:
    """박동 간격(ms) 통계: (rate_bpm, mean_ms, sdnn_ms, rmssd_ms, pnn50)."""
    ibi = np.diff(peaks) / fs
    ibi = ibi[(ibi >= _MIN_IBI) & (ibi <= _MAX_IBI)] * 1000.0
    if len(ibi) < 3:
        return (np.nan,) * 5
    d = np.diff(ibi)
    return (60000.0 / ibi.mean(), ibi.mean(), ibi.std(ddof=1),
            float(np.sqrt(np.mean(d ** 2))), float(np.mean(np.abs(d) > 50.0)))


def window_features(win: np.ndarray, channels: tuple[str, ...], spec: InputSpec) -> np.ndarray:
    """윈도우 1개 (C, T) → 특징 벡터 (F,). 순서는 spec.feature_names 와 같다."""
    fs = spec.fs
    ch = {c: win[i].astype(np.float64) for i, c in enumerate(channels)}
    hr, rr, sdnn, rmssd, pnn50 = _ibi_stats(detect_ecg_peaks(ch["ecg"], fs), fs)
    pr, ppi, psdnn, prmssd, _ = _ibi_stats(detect_ppg_peaks(ch["ppg"], fs), fs)
    out = [hr, rr, sdnn, rmssd, pnn50, pr, psdnn, prmssd]
    if spec.use_can:
        sp, st = ch["speed"], ch["steering_angle"]
        out += [sp.mean(), sp.std(), st.std(), np.mean(np.abs(np.diff(st))) * fs,
                ch["accel_pedal"].mean(), float(np.mean(ch["brake"] > 0.5))]
    return np.asarray(out, dtype=np.float64)


def extract_features(ws: WindowSet, spec: InputSpec) -> np.ndarray:
    """WindowSet → (N, F) float32. NaN 은 그대로 둔다 (정규화 단계에서 처리)."""
    F = len(spec.feature_names)
    assert F == len(ECG_FEATURES) + len(PPG_FEATURES) + (len(CAN_FEATURES) if spec.use_can else 0)
    if len(ws) == 0:
        return np.zeros((0, F), np.float32)
    return np.stack([window_features(w, ws.channels, spec) for w in ws.x_seq]).astype(np.float32)
