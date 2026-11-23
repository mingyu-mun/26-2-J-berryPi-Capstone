"""필터 적용 방식 스위치: 영위상(오프라인, filtfilt) / 인과(실시간, lfilter·sosfilt).

기기 실시간 추론은 미래 표본을 볼 수 없으므로 인과 필터를 쓴다. 세션 전체에 sosfilt 를 한 번 거는 것은
같은 필터를 표본이 들어올 때마다 상태(zi)를 이어 가며 거는 것과 같은 출력이다 (처음 과도 응답 포함).

    from driver_state.filtering import causal_filters, sos_filter
    with causal_filters(True):
        y = sos_filter(sos, x)       # → sosfilt

기본값은 영위상 (기존 교사·학생 결과 재현). 환경 변수 DS_CAUSAL=1 로도 켤 수 있다.
"""
from __future__ import annotations

import os
from contextlib import contextmanager

import numpy as np
from scipy.signal import filtfilt, lfilter, sosfilt, sosfiltfilt

_CAUSAL = os.environ.get("DS_CAUSAL", "0") == "1"


def is_causal() -> bool:
    return _CAUSAL


def set_causal(on: bool) -> None:
    global _CAUSAL
    _CAUSAL = bool(on)


@contextmanager
def causal_filters(on: bool | None):
    """on=None 이면 현재 설정 유지."""
    global _CAUSAL
    prev = _CAUSAL
    if on is not None:
        _CAUSAL = bool(on)
    try:
        yield
    finally:
        _CAUSAL = prev


def sos_filter(sos: np.ndarray, x: np.ndarray, causal: bool | None = None) -> np.ndarray:
    c = _CAUSAL if causal is None else causal
    return sosfilt(sos, x) if c else sosfiltfilt(sos, x)


def ba_filter(b: np.ndarray, a: np.ndarray, x: np.ndarray, causal: bool | None = None) -> np.ndarray:
    c = _CAUSAL if causal is None else causal
    return lfilter(b, a, x) if c else filtfilt(b, a, x)
