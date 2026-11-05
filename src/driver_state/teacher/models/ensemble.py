"""C. A + B 앙상블: 각자 온도 보정한 로짓의 가중 평균. A·B 를 다시 학습하지 않는다."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .base import TeacherInput, TeacherModel


class EnsembleTeacher(TeacherModel):
    name = "ensemble"

    def __init__(self, seed: int = 0, members=None, weights=None, **params):
        super().__init__(seed, **params)
        self.members: list[TeacherModel] = list(members or [])
        self.weights = list(weights) if weights is not None else [1.0 / max(len(self.members), 1)] * len(self.members)

    def fit(self, X, y, w=None, X_val=None, y_val=None, w_val=None):
        return self   # 구성원은 이미 학습·보정된 상태로 받는다

    def predict_logit(self, X: TeacherInput) -> np.ndarray:
        return sum(wt * m.predict_logit(X) / m.temperature for wt, m in zip(self.weights, self.members))

    def _save_state(self, path: Path) -> None:
        names = []
        for i, m in enumerate(self.members):
            m.save(path / f"member{i}_{m.name}")
            names.append(f"member{i}_{m.name}")
        (path / "ensemble.json").write_text(json.dumps({"members": names, "weights": self.weights}), encoding="utf-8")

    def _load_state(self, path: Path) -> None:
        from . import REGISTRY
        d = json.loads((path / "ensemble.json").read_text(encoding="utf-8"))
        self.members = []
        for n in d["members"]:
            meta = json.loads((path / n / "meta.json").read_text(encoding="utf-8"))
            self.members.append(REGISTRY[meta["name"]].load(path / n))
        self.weights = d["weights"]
