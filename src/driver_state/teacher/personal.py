"""개인 Baseline 정규화 (교사용, 기준 구간 5분).

    z_{j,k} = (x_{j,k} − m_{s,j,k}) / max(σ_{s,j,k}, σ_pop,{j,k})
    r_{j,k} = ln(x_{j,k} / m_{s,j,k})            (양수 특징만)

σ_pop 은 학습 참가자마다 그 사람 창들의 표준편차를 구해 중앙값을 취한 값이며,
fold 마다 학습 참가자로만 fit 한다 (baseline.fit_sigma_pop 과 같은 방식).
기준이 무효이거나 force_invalid 인 행은 z·r = NaN, baseline_valid = 0.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .spec import TeacherSpec


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
        out["baseline_valid"] = valid.astype(np.float64)
        return pd.DataFrame(out, index=win.index)

    def to_dict(self) -> dict:
        return {"sigma_pop": self.sigma_pop}

    @classmethod
    def from_dict(cls, spec: TeacherSpec, d: dict) -> "PersonalNormalizer":
        p = cls(spec)
        p.sigma_pop = {k: float(v) for k, v in d["sigma_pop"].items()}
        return p
