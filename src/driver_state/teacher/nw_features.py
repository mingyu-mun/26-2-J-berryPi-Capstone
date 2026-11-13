"""NormWear 표현을 창·기준 벡터 열로 붙인다 (docs/teacher-normwear-stage1.md 1.1).

PCA 는 CLAS·MAUS 창으로만, 세션 평균을 뺀 표현으로 fit (32축). 모든 창·기준 창을 사영.
창: nw_00 … nw_31, 기준 벡터(profiles): m_nw_00 … (기준 창 사영값 중앙값).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

N_PC = 32
NW_COLS = [f"nw_{i:02d}" for i in range(N_PC)]
PCA_DATASETS = ("clas", "maus")


def _load(emb_dir: Path, ds: str):
    p = emb_dir / f"emb_{ds}.npz"
    if not p.exists():
        return None
    z = np.load(p)
    return pd.Series(list(z["emb"].astype(np.float32)), index=z["row_id"])


def fit_pca(ws: pd.DataFrame, emb_dir: Path):
    rows = []
    for ds in PCA_DATASETS:
        e = _load(emb_dir, ds)
        if e is None:
            continue
        d = ws[ws["dataset_id"] == ds][["row_id", "subject_id", "session_id"]]
        d = d[d["row_id"].isin(e.index)]
        X = np.stack(e.loc[d["row_id"]].to_numpy())
        key = (d["subject_id"].astype(str) + "/" + d["session_id"].astype(str)).to_numpy()
        Xc = X - pd.DataFrame(X).groupby(key).transform("mean").to_numpy()
        rows.append(Xc)
    Xc = np.concatenate(rows)
    Xc = Xc - Xc.mean(0)
    _, s, vt = np.linalg.svd(Xc, full_matrices=False)
    return vt[:N_PC], s[:N_PC] ** 2 / (s ** 2).sum()


def attach(win: pd.DataFrame, prof: pd.DataFrame, emb_dir: str | Path, comps: np.ndarray):
    emb_dir = Path(emb_dir)
    win = win.copy()
    prof = prof.copy()
    P = np.full((len(win), N_PC), np.nan)
    for ds in win["dataset_id"].unique():
        e = _load(emb_dir, ds)
        if e is None:
            continue
        m = (win["dataset_id"] == ds).to_numpy() & win["row_id"].isin(e.index).to_numpy()
        X = np.stack(e.loc[win.loc[m, "row_id"]].to_numpy())
        P[m] = X @ comps.T
    for i, c in enumerate(NW_COLS):
        win[c] = P[:, i]
    ref = {}
    for ds in prof["profile_key"].str.split("/").str[0].unique():
        p = emb_dir / f"ref_{ds}.npz"
        if not p.exists():
            continue
        z = np.load(p)
        R = z["emb"].astype(np.float32) @ comps.T
        df = pd.DataFrame(R, columns=NW_COLS)
        df["k"] = z["profile_key"]
        ref.update({k: g[NW_COLS].median().to_numpy() for k, g in df.groupby("k")})
    M = np.array([ref.get(k, np.full(N_PC, np.nan)) for k in prof["profile_key"]])
    for i, c in enumerate(NW_COLS):
        prof[f"m_{c}"] = M[:, i]
    return win, prof


def z_of(win_train: pd.DataFrame, win: pd.DataFrame, prof: pd.DataFrame, cols=NW_COLS) -> np.ndarray:
    """(x − m_ref) / σ_pop, σ_pop = 학습 창의 참가자별 표준편차 중앙값 (PersonalNormalizer 와 같은 방식)."""
    sig = np.array([np.nanmedian(win_train.groupby("group")[c].std(ddof=1).to_numpy()) for c in cols])
    sig = np.where(np.isfinite(sig) & (sig > 1e-9), sig, 1.0)
    m = win[["profile_key"]].merge(prof[["profile_key"] + [f"m_{c}" for c in cols]], on="profile_key", how="left")
    Z = (win[cols].to_numpy(float) - m[[f"m_{c}" for c in cols]].to_numpy(float)) / sig
    return np.clip(np.nan_to_num(Z, nan=0.0), -5, 5).astype(np.float32)
