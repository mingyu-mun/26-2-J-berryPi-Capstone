"""윈도우 분할 전 공통 전처리 (CNN 경로와 특징 경로가 함께 쓴다).

채널 하나의 처리 순서
1. 노치 필터 (원 주기에서, 전원 잡음 60 Hz). 원 주기 Nyquist 가 60 Hz 이하이면 생략.
2. 다운샘플이면 저역통과(anti-aliasing): 차단 주파수 = antialias_ratio * fs_out.
   이 단계 없이 줄이면 fs_out/2 이상의 잡음이 낮은 주파수로 접혀(aliasing) 다시 걸러낼 수 없다.
3. 공통 주기 fs 로 리샘플 (생체: cubic spline, CAN: 선형 보간).
4. 대역통과: ECG 0.5–40 Hz (기저선 흔들림·고주파 잡음 제거), PPG 0.5–8 Hz.
품질 검사는 필터 전(3단계 직후) 신호로 윈도우마다 평탄·포화 여부를 본다.

데이터 담당이 이미 필터를 적용해 넘기면 ``Recording.prefiltered=True`` 로 1·2·4단계를 건너뛴다.
"""
from __future__ import annotations

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.signal import butter, iirnotch, sosfiltfilt, filtfilt

from .spec import BIO_CHANNELS, InputSpec


def resample(x: np.ndarray, fs_in: float, fs_out: float, kind: str = "cubic") -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    if fs_in == fs_out:
        return x
    n_out = int(np.floor(len(x) * fs_out / fs_in))
    t_in = np.arange(len(x)) / fs_in
    t_out = np.arange(n_out) / fs_out
    if kind == "cubic":
        return CubicSpline(t_in, x)(t_out)
    return np.interp(t_out, t_in, x)


def _band(x: np.ndarray, fs: float, lo: float, hi: float) -> np.ndarray:
    hi = min(hi, 0.45 * fs)
    sos = butter(3, [lo, hi], btype="band", fs=fs, output="sos")
    return sosfiltfilt(sos, x)


def prepare_channel(x: np.ndarray, fs_in: float, name: str, spec: InputSpec,
                    prefiltered: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """채널 하나 → (필터된 신호, 필터 전 리샘플 신호). 둘 다 spec.fs 주기."""
    p = spec.preprocess
    x = np.asarray(x, dtype=np.float64)
    is_bio = name in BIO_CHANNELS
    if is_bio and not prefiltered:
        if p.notch_hz and fs_in / 2 > p.notch_hz + 1:
            b, a = iirnotch(p.notch_hz, Q=30.0, fs=fs_in)
            x = filtfilt(b, a, x)
        if fs_in > spec.fs:
            sos = butter(4, p.antialias_ratio * spec.fs, btype="low", fs=fs_in, output="sos")
            x = sosfiltfilt(sos, x)
    raw = resample(x, fs_in, spec.fs, "cubic" if is_bio else "linear")
    if not is_bio or prefiltered:
        return raw, raw
    band = p.ecg_band if name == "ecg" else p.ppg_band
    return _band(raw, spec.fs, *band), raw


def window_quality(raw_win: np.ndarray, spec: InputSpec, channels: tuple[str, ...]) -> bool:
    """필터 전 윈도우 (C, T) → 생체 채널이 모두 정상이면 True."""
    p = spec.preprocess
    for i, c in enumerate(channels):
        if c not in BIO_CHANNELS:
            continue
        x = raw_win[i]
        if x.std() < p.flat_std:
            return False
        lo, hi = x.min(), x.max()
        clip = np.mean((x <= lo + 1e-9 * abs(lo)) | (x >= hi - 1e-9 * abs(hi)))
        if clip > p.max_clip_ratio:
            return False
        if _longest_flat_run(x, 1e-6 * max(hi - lo, 1e-12)) > p.max_flat_run_sec * spec.fs:
            return False
    return True


def _longest_flat_run(x: np.ndarray, tol: float) -> int:
    """연속 샘플 차이가 tol 이하로 이어진 가장 긴 구간의 샘플 수."""
    still = np.abs(np.diff(x)) <= tol
    if not still.any():
        return 0
    d = np.diff(np.concatenate([[0], still.astype(np.int8), [0]]))
    return int((np.where(d == -1)[0] - np.where(d == 1)[0]).max()) + 1
