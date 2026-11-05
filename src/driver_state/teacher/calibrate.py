"""확률 보정과 보정 지표: 온도 스케일링, ECE, Brier, 신뢰도 표.

온도 스케일링 p = sigmoid(logit / T) 는 순위를 바꾸지 않으므로 AUROC 는 그대로이고 확률 크기만 맞춘다.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar

from .models.base import sigmoid


def fit_temperature(logit: np.ndarray, y: np.ndarray, w: np.ndarray | None = None,
                    t_min: float = 0.25, t_max: float = 4.0) -> float:
    """NLL 최소 T. 검증셋이 작을 때 과하게 날카로워지지 않도록 [t_min, t_max] 로 제한."""
    logit = np.asarray(logit, np.float64)
    y = np.asarray(y, np.float64)
    w = np.ones_like(y) if w is None else np.asarray(w, np.float64)
    if len(np.unique(y)) < 2:
        return 1.0

    def nll(log_t):
        p = np.clip(sigmoid(logit / np.exp(log_t)), 1e-7, 1 - 1e-7)
        return -np.sum(w * (y * np.log(p) + (1 - y) * np.log(1 - p))) / w.sum()

    r = minimize_scalar(nll, bounds=(np.log(t_min), np.log(t_max)), method="bounded")
    return float(np.exp(r.x))


def ece(p: np.ndarray, y: np.ndarray, bins: int = 15) -> float:
    p, y = np.asarray(p), np.asarray(y)
    if len(p) == 0:
        return float("nan")
    idx = np.minimum((p * bins).astype(int), bins - 1)
    tot = 0.0
    for b in range(bins):
        m = idx == b
        if m.any():
            tot += m.mean() * abs(p[m].mean() - y[m].mean())
    return float(tot)


def brier(p: np.ndarray, y: np.ndarray) -> float:
    return float(np.mean((np.asarray(p) - np.asarray(y)) ** 2)) if len(p) else float("nan")


def reliability(p: np.ndarray, y: np.ndarray, bins: int = 10) -> pd.DataFrame:
    p, y = np.asarray(p), np.asarray(y)
    idx = np.minimum((p * bins).astype(int), bins - 1)
    rows = []
    for b in range(bins):
        m = idx == b
        rows.append({"bin": f"{b / bins:.1f}-{(b + 1) / bins:.1f}", "n": int(m.sum()),
                     "p_mean": float(p[m].mean()) if m.any() else np.nan,
                     "drowsy_rate": float(y[m].mean()) if m.any() else np.nan})
    return pd.DataFrame(rows)
