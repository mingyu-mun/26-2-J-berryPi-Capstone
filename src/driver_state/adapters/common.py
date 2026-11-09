"""어댑터 공용: 결측 처리 (규약 4장)."""
from __future__ import annotations

import numpy as np


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """True 가 이어지는 구간 [a, b) 목록."""
    m = np.concatenate([[False], mask, [False]]).astype(np.int8)
    d = np.diff(m)
    return list(zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)))


def finite_segments(signals: dict[str, np.ndarray], fs: float, max_gap_sec: float = 0.2,
                    min_seg_sec: float = 60.0) -> list[tuple[int, dict[str, np.ndarray]]]:
    """같은 길이·같은 fs 의 채널들을 함께 본다.

    모든 채널이 유한한 샘플만 유효. 짧은 결측(≤ max_gap_sec)은 선형 보간, 긴 결측에서는 구간을 나눈다.
    min_seg_sec 보다 짧은 구간은 버린다. 반환: [(시작 샘플, {채널: 배열}), ...]
    """
    keys = list(signals)
    n = len(signals[keys[0]])
    if any(len(signals[k]) != n for k in keys):
        raise ValueError("채널 길이가 다릅니다")
    ok = np.ones(n, bool)
    for k in keys:
        ok &= np.isfinite(signals[k])
    bad = ~ok
    max_gap = int(round(max_gap_sec * fs))
    # 짧은 내부 결측은 유효 구간으로 메운다 (앞뒤 끝 결측은 그대로 잘라냄)
    fill = np.zeros(n, bool)
    for a, b in _runs(bad):
        if a > 0 and b < n and b - a <= max_gap:
            fill[a:b] = True
    keep = ok | fill
    out = []
    for a, b in _runs(keep):
        if (b - a) < min_seg_sec * fs:
            continue
        seg = {}
        idx = np.arange(a, b)
        good = ok[a:b]
        for k in keys:
            x = np.asarray(signals[k][a:b], dtype=np.float64)
            if not good.all():
                x = np.interp(idx, idx[good], x[good])
            seg[k] = x
        out.append((int(a), seg))
    return out
