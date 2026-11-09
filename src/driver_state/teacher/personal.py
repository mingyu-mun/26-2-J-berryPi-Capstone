"""개인 Baseline 정규화 (교사용, 기준 구간 5분).

    z_{j,k} = (x_{j,k} − m_{s,j,k}) / max(σ_{s,j,k}, σ_pop,{j,k})
    r_{j,k} = ln(x_{j,k} / m_{s,j,k})            (양수 특징만)

σ_pop 은 학습 참가자마다 그 사람 창들의 표준편차를 구해 중앙값을 취한 값이며,
fold 마다 학습 참가자로만 fit 한다 (baseline.fit_sigma_pop 과 같은 방식).
기준이 무효이거나 force_invalid 인 행은 z·r = NaN, baseline_valid = 0.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from .spec import TeacherSpec

CORE = ("hr", "rmssd", "sdnn")      # dev_core: 2분·5분 스케일의 심박·HRV 핵심 특징

# 열 묶음 (LightGBM 의 pers_cols 로 고름)
SIGNED_PREFIX = ("z_", "r_")
UNSIGNED_PREFIX = ("az_", "ar_", "dev_")


def pers_columns(cols, mode: str) -> list[str]:
    """mode: signed (기존) | signed_abs (부호 있는 + 없는) | abs (|z|·|r|·dev) | dev (스케일별 거리만)."""
    keep = {"signed": SIGNED_PREFIX, "signed_abs": SIGNED_PREFIX + UNSIGNED_PREFIX,
            "abs": UNSIGNED_PREFIX, "dev": ("dev_",)}[mode]
    return [c for c in cols if c.startswith(keep) or c == "baseline_valid"]


class PersonalNormalizer:
    def __init__(self, spec: TeacherSpec):
        self.spec = spec
        self.sigma_pop: dict[str, float] = {}

    def fit(self, win: pd.DataFrame) -> "PersonalNormalizer":
        for n in self.spec.baseline_feature_names:
            stds = win.groupby("group")[n].std(ddof=1)
            cnt = win.groupby("group")[n].count()
            v = stds[(cnt >= 3) & np.isfinite(stds)]
            s = float(np.median(v)) if len(v) else 1.0
            self.sigma_pop[n] = s if s > 1e-9 else 1.0
        return self

    def transform(self, win: pd.DataFrame, prof: pd.DataFrame,
                  force_invalid: np.ndarray | None = None) -> pd.DataFrame:
        names = self.spec.baseline_feature_names
        p = win[["profile_key"]].merge(prof, on="profile_key", how="left")
        valid = p["baseline_valid"].fillna(0.0).to_numpy() > 0.5
        if force_invalid is not None:
            valid &= ~force_invalid
        out = {}
        for n in names:
            x = win[n].to_numpy(np.float64)
            m = p[f"m_{n}"].to_numpy(np.float64)
            s = p[f"s_{n}"].to_numpy(np.float64)
            scale = np.fmax(np.nan_to_num(s, nan=0.0), self.sigma_pop[n])
            z = (x - m) / scale
            z[~valid] = np.nan
            out[f"z_{n}"] = z
        for n in self.spec.ratio_feature_names():
            x = win[n].to_numpy(np.float64)
            m = p[f"m_{n}"].to_numpy(np.float64)
            with np.errstate(divide="ignore", invalid="ignore"):
                r = np.where((x > 0) & (m > 0), np.log(x / m), np.nan)
            r[~valid] = np.nan
            out[f"r_{n}"] = r
        # 부호 없는 특징: 기준에서 "얼마나 벗어났나"만 본다. 데이터셋마다 졸음 방향이 반대인 문제
        # (MPD-DF 심박↓, DROZY·AdVitam 심박↑) 를 다루기 위한 것 (2026-11-09).
        for k in list(out):
            out[f"a{k}"] = np.abs(out[k])                     # az_*, ar_*
        groups = {f"dev_{tag}": [f"z_{n}" for n in names if n.rsplit("_", 1)[1] == tag]
                  for tag in sorted({n.rsplit("_", 1)[1] for n in names}, key=int)}
        groups["dev_core"] = [f"z_{n}" for n in names if n.rsplit("_", 1)[0] in CORE and n.rsplit("_", 1)[1] != "30"]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            for g, cols in groups.items():          # sqrt(Σz²/k): 대각 마할라노비스 거리
                if cols:
                    out[g] = np.sqrt(np.nanmean(np.column_stack([out[c] for c in cols]) ** 2, axis=1))
        out["baseline_valid"] = valid.astype(np.float64)
        return pd.DataFrame(out, index=win.index)

    def to_dict(self) -> dict:
        return {"sigma_pop": self.sigma_pop}

    @classmethod
    def from_dict(cls, spec: TeacherSpec, d: dict) -> "PersonalNormalizer":
        p = cls(spec)
        p.sigma_pop = {k: float(v) for k, v in d["sigma_pop"].items()}
        return p
