"""HRV 특징 함수 (numpy/scipy 만 사용). 입력은 정상 RR(NN) 간격 [ms] 과 그 시각 [s].

계산할 수 없으면 NaN 을 돌려준다 (LightGBM 이 결측을 직접 처리).
"""
from __future__ import annotations

import numpy as np
from scipy.signal import lombscargle

LF_BAND = (0.04, 0.15)
HF_BAND = (0.15, 0.40)
_FREQ = np.linspace(0.01, 0.5, 200)          # Hz
_DF = _FREQ[1] - _FREQ[0]
_W = 2 * np.pi * _FREQ


def time_domain(nn: np.ndarray, d: np.ndarray | None = None) -> list[float]:
    """[hr, rr_mean, sdnn, rmssd, pnn50, sd1, sd2, sd1_sd2]. d = 연속 NN 차이 (연속성 고려해 외부에서 줄 수 있음)."""
    if len(nn) < 3:
        return [np.nan] * 8
    if d is None:
        d = np.diff(nn)
    if len(d) < 2:
        return [np.nan] * 8
    m = float(nn.mean())
    sdnn = float(nn.std(ddof=1))
    rmssd = float(np.sqrt(np.mean(d ** 2)))
    pnn50 = float(np.mean(np.abs(d) > 50.0))
    sd1 = float(np.sqrt(0.5) * np.std(d, ddof=1))
    sd2_sq = 2 * sdnn ** 2 - sd1 ** 2
    sd2 = float(np.sqrt(sd2_sq)) if sd2_sq > 0 else np.nan
    return [60000.0 / m, m, sdnn, rmssd, pnn50, sd1, sd2, sd1 / sd2 if sd2 and sd2 == sd2 else np.nan]


def frequency(t: np.ndarray, nn: np.ndarray) -> list[float]:
    """Lomb–Scargle 로 [lf_log, hf_log, lf_hf_log, lf_nu]. 불균일 간격을 보간 없이 처리."""
    if len(nn) < 30 or (t[-1] - t[0]) < 60:
        return [np.nan] * 4
    y = nn - nn.mean()
    p = lombscargle(t, y, _W) * 2.0 / len(y)       # 진폭² 단위로 정규화 (창 길이에 덜 민감)
    lf = float(p[(_FREQ >= LF_BAND[0]) & (_FREQ < LF_BAND[1])].sum() * _DF)
    hf = float(p[(_FREQ >= HF_BAND[0]) & (_FREQ < HF_BAND[1])].sum() * _DF)
    if lf <= 0 or hf <= 0:
        return [np.nan] * 4
    return [np.log(lf), np.log(hf), np.log(lf / hf), lf / (lf + hf)]


def dfa_alpha1(nn: np.ndarray, scales: tuple[int, ...] = (4, 5, 6, 8, 10, 12, 16)) -> float:
    """단기 DFA 지수 α1 (4–16 박동)."""
    if len(nn) < 4 * max(scales):
        return np.nan
    y = np.cumsum(nn - nn.mean())
    fl = []
    for n in scales:
        k = len(y) // n
        seg = y[:k * n].reshape(k, n)
        x = np.arange(n)
        coef = np.polyfit(x, seg.T, 1)
        trend = np.outer(coef[0], x) + coef[1][:, None]
        fl.append(np.sqrt(np.mean((seg - trend) ** 2)))
    fl = np.asarray(fl)
    if np.any(fl <= 0):
        return np.nan
    return float(np.polyfit(np.log(scales), np.log(fl), 1)[0])


def sample_entropy(x: np.ndarray, m: int = 2, r_ratio: float = 0.2, max_n: int = 400) -> float:
    """표본 엔트로피 SampEn(m, r = r_ratio·SD). 길면 앞에서 max_n 개만 (계산량 O(n²))."""
    x = np.asarray(x, dtype=np.float64)[:max_n]
    n = len(x)
    if n < 50:
        return np.nan
    r = r_ratio * x.std()
    if r <= 0:
        return np.nan

    def count(mm: int) -> int:
        emb = np.lib.stride_tricks.sliding_window_view(x, mm)[: n - m]
        dist = np.max(np.abs(emb[:, None, :] - emb[None, :, :]), axis=2)
        return int((np.sum(dist <= r) - len(emb)) // 2)

    b, a = count(m), count(m + 1)
    if a == 0 or b == 0:
        return np.nan
    return float(-np.log(a / b))


def long_features(t: np.ndarray, nn: np.ndarray) -> list[float]:
    """[lf_log, hf_log, lf_hf_log, lf_nu, dfa_a1, sampen]"""
    return frequency(t, nn) + [dfa_alpha1(nn), sample_entropy(nn)]


def slope_per_min(t: np.ndarray, v: np.ndarray, min_points: int = 5) -> float:
    ok = np.isfinite(v)
    if ok.sum() < min_points or np.ptp(t[ok]) < 60:
        return np.nan
    return float(np.polyfit(t[ok] / 60.0, v[ok], 1)[0])
