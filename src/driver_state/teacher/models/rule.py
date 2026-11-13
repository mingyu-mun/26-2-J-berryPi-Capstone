"""규칙 기준선: 학습된 교사가 넘어야 할 최소 기준.

점수 = −z(HR, 2분) + z(RMSSD, 2분). 졸음일수록 심박이 내려가고 부교감(RMSSD)이 오른다는 문헌상 경향을
부호만 고정해 쓴다. 개인 기준이 없으면 학습셋 평균·표준편차로 표준화한 값을 쓴다.
학습하는 것은 점수 → 확률 변환의 기울기·절편 2개뿐. 기울기는 양수로 고정해 AUROC 가 규칙 방향 그대로 나오게 한다.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .base import TeacherInput, TeacherModel


class RuleTeacher(TeacherModel):
    name = "rule"

    def __init__(self, seed: int = 0, hr_col: str = "hr_120", rmssd_col: str = "rmssd_120", **params):
        super().__init__(seed, hr_col=hr_col, rmssd_col=rmssd_col, **params)
        self.hr_col, self.rmssd_col = hr_col, rmssd_col
        self.pop = {}
        self.a, self.b = 1.0, 0.0

    def _score(self, X: TeacherInput) -> np.ndarray:
        parts = []
        for col, sign in ((self.hr_col, -1.0), (self.rmssd_col, 1.0)):
            z = X.pers[f"z_{col}"].to_numpy(np.float64)
            mu, sd = self.pop[col]
            zp = (X.raw[col].to_numpy(np.float64) - mu) / sd
            v = np.where(np.isfinite(z), z, zp)
            parts.append(sign * v)
        s = np.nanmean(np.column_stack(parts), axis=1)
        return np.nan_to_num(s, nan=0.0)

    def fit(self, X, y, w=None, X_val=None, y_val=None, w_val=None):
        for col in (self.hr_col, self.rmssd_col):
            v = X.raw[col].to_numpy(np.float64)
            self.pop[col] = (float(np.nanmean(v)), float(np.nanstd(v)) or 1.0)
        from sklearn.linear_model import LogisticRegression
        s = self._score(X)[:, None]
        lr = LogisticRegression(C=1.0).fit(s, y, sample_weight=w)
        self.a, self.b = float(lr.coef_[0, 0]), float(lr.intercept_[0])
        if self.a <= 0:
            # 학습셋에서 방향이 반대로 나와도 규칙의 부호(심박↓·RMSSD↑ = 졸음)는 지킨다.
            # 기울기가 음수면 순위가 뒤집혀 기준선이 0.5 아래로 떨어진다 (2026-11-09 실데이터 LODO).
            self.a = 1e-3
        return self

    def predict_logit(self, X):
        return self.a * self._score(X) + self.b

    def _save_state(self, path: Path) -> None:
        (path / "rule.json").write_text(json.dumps({"pop": self.pop, "a": self.a, "b": self.b}), encoding="utf-8")

    def _load_state(self, path: Path) -> None:
        d = json.loads((path / "rule.json").read_text(encoding="utf-8"))
        self.pop = {k: tuple(v) for k, v in d["pop"].items()}
        self.a, self.b = d["a"], d["b"]
