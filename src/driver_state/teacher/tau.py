"""τ 조정: LODO 예측(oof, 실제 라벨)에서 τ_hi / τ_lo 를 고른다.

- τ_hi: P ≥ τ_hi 인 창의 졸음 비율(정밀도)이 모든 데이터셋에서 목표 이상이 되는 가장 낮은 값
- τ_lo: P ≤ τ_lo 인 창의 정상 비율이 모든 데이터셋에서 목표 이상이 되는 가장 높은 값
데이터셋마다 졸음 비율이 달라 같은 τ 라도 정밀도가 다르므로 가장 나쁜 데이터셋 기준으로 고른다.
창이 min_count 개 미만인 데이터셋은 그 τ 의 판정에서 뺀다 (너무 적어 비율이 의미 없음).
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def tau_table(oof: pd.DataFrame, col: str, grid: np.ndarray | None = None, min_count: int = 30) -> pd.DataFrame:
    grid = np.round(np.arange(0.05, 0.96, 0.05), 2) if grid is None else grid
    rows = []
    for t in grid:
        r = {"tau": float(t)}
        prec_hi, prec_lo = [], []
        for d, g in oof.groupby("dataset_id"):
            y, p = g["y"].to_numpy(), g[col].to_numpy()
            hi, lo = p >= t, p <= t
            if hi.sum() >= min_count:
                prec_hi.append(y[hi].mean())
            if lo.sum() >= min_count:
                prec_lo.append(1 - y[lo].mean())
        y, p = oof["y"].to_numpy(), oof[col].to_numpy()
        r.update(drowsy_precision_min=min(prec_hi) if prec_hi else np.nan,
                 alert_precision_min=min(prec_lo) if prec_lo else np.nan,
                 frac_hi=float((p >= t).mean()), frac_lo=float((p <= t).mean()),
                 recall_hi=float(((p >= t) & (y == 1)).sum() / max((y == 1).sum(), 1)),
                 recall_lo=float(((p <= t) & (y == 0)).sum() / max((y == 0).sum(), 1)))
        rows.append(r)
    return pd.DataFrame(rows)


def choose_tau(oof: pd.DataFrame, col: str, precision: float = 0.85, npv: float = 0.90,
               min_count: int = 30, min_frac: float = 0.01) -> dict:
    """τ_hi 를 먼저 고르고, τ_lo 는 τ_hi 보다 작은 값 중에서 고른다.

    목표를 못 채우면 (모든 데이터셋 최저 기준) 정밀도가 가장 높은 값을 쓰되 notes 에 남긴다.
    이때 achieved=False 이며, 실제 라벨링 전에 사람이 표를 보고 정해야 한다.
    """
    tab = tau_table(oof, col, min_count=min_count)
    notes = []
    hi_ok = tab[(tab["drowsy_precision_min"] >= precision) & (tab["frac_hi"] >= min_frac)]
    achieved = True
    if len(hi_ok):
        tau_hi = float(hi_ok["tau"].min())
    else:
        achieved = False
        cand = tab[tab["frac_hi"] >= min_frac].dropna(subset=["drowsy_precision_min"])
        tau_hi = float(cand.loc[cand["drowsy_precision_min"].idxmax(), "tau"]) if len(cand) else 0.9
        notes.append(f"모든 데이터셋에서 졸음 정밀도 {precision} 를 만족하는 τ_hi 가 없어 정밀도가 가장 높은 값을 씀")
    below = tab[tab["tau"] < tau_hi - 1e-9]
    lo_ok = below[(below["alert_precision_min"] >= npv) & (below["frac_lo"] >= min_frac)]
    if len(lo_ok):
        tau_lo = float(lo_ok["tau"].max())
    else:
        achieved = False
        cand = below.dropna(subset=["alert_precision_min"])
        tau_lo = float(cand.loc[cand["alert_precision_min"].idxmax(), "tau"]) if len(cand) else max(tau_hi - 0.1, 0.0)
        notes.append(f"τ_hi 아래에서 정상 정밀도 {npv} 를 만족하는 τ_lo 가 없어 정밀도가 가장 높은 값을 씀")
    row_hi = tab.loc[(tab["tau"] - tau_hi).abs().idxmin()]
    row_lo = tab.loc[(tab["tau"] - tau_lo).abs().idxmin()]
    return {"model": col.removeprefix("p_"), "tau_hi": round(tau_hi, 2), "tau_lo": round(tau_lo, 2), "achieved": achieved,
            "target_drowsy_precision": precision, "target_alert_precision": npv,
            "drowsy_precision_min": float(row_hi["drowsy_precision_min"]),
            "alert_precision_min": float(row_lo["alert_precision_min"]),
            "frac_labeled_drowsy": float(row_hi["frac_hi"]), "frac_labeled_alert": float(row_lo["frac_lo"]),
            "notes": notes, "table": tab.to_dict(orient="records")}


def apply_tau(p: np.ndarray, tau_hi: float, tau_lo: float, valid: np.ndarray | None = None) -> np.ndarray:
    lab = np.full(len(p), -1, np.int64)
    lab[p >= tau_hi] = 1
    lab[p <= tau_lo] = 0
    if valid is not None:
        lab[~valid] = -1
    return lab
