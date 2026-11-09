"""교사 모델 공통 인터페이스 (후보 모델의 CandidateModel 과 같은 메서드 이름, 이진 전용).

- fit(X, y, w, X_val, y_val, w_val)
- predict_logit(X) -> (N,)  보정 전 로짓
- predict_proba(X) -> (N, 2) 보정 후 [P(정상), P(졸음)] = sigmoid(logit / T)
- temperature: 온도 스케일링 값 (calibrate.fit_temperature 로 설정, 기본 1)
- save(dir) / load(dir)
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass
class TeacherInput:
    raw: pd.DataFrame                 # 스케일별 원 특징 + 추세 (입력 열만)
    pers: pd.DataFrame                # z_*, r_*, baseline_valid
    tacho: np.ndarray | None = None   # (N, 2, T) [RR ms, 결측 마스크]
    base_rr: np.ndarray | None = None # (N,) 기준 구간 평균 RR [ms], 무효면 NaN

    def __len__(self) -> int:
        return len(self.raw)

    def subset(self, idx: np.ndarray) -> "TeacherInput":
        return TeacherInput(self.raw.iloc[idx].reset_index(drop=True), self.pers.iloc[idx].reset_index(drop=True),
                            None if self.tacho is None else self.tacho[idx],
                            None if self.base_rr is None else self.base_rr[idx])


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -50, 50)))


def sample_weights(dataset: np.ndarray, group: np.ndarray, y: np.ndarray) -> np.ndarray:
    """데이터셋마다 합이 같고, 데이터셋 안에서는 참가자마다 같고, 정상·졸음 전체 합이 같은 가중치. 평균 1."""
    df = pd.DataFrame({"d": dataset, "g": group, "y": y})
    n_ds = df["d"].nunique()
    subj_per_ds = df.groupby("d")["g"].nunique()
    win_per_subj = df.groupby("g")["g"].transform("size")
    w = 1.0 / (n_ds * df["d"].map(subj_per_ds).to_numpy() * win_per_subj.to_numpy())
    for c in (0, 1):
        m = df["y"].to_numpy() == c
        if m.any():
            w[m] *= 0.5 / w[m].sum()
    return w * len(w) / w.sum()


class TeacherModel:
    name = "base"
    uses_tacho = False

    def __init__(self, seed: int = 0, **params):
        self.seed = seed
        self.params = params
        self.temperature = 1.0

    def fit(self, X: TeacherInput, y, w=None, X_val=None, y_val=None, w_val=None) -> "TeacherModel":
        raise NotImplementedError

    def predict_logit(self, X: TeacherInput) -> np.ndarray:
        raise NotImplementedError

    def predict_proba(self, X: TeacherInput) -> np.ndarray:
        p = sigmoid(self.predict_logit(X) / self.temperature)
        return np.column_stack([1 - p, p])

    # 저장: meta.json + 모델별 파일
    def _meta(self) -> dict:
        return {"name": self.name, "seed": self.seed, "params": self.params, "temperature": self.temperature}

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        (path / "meta.json").write_text(json.dumps(self._meta(), ensure_ascii=False, indent=2), encoding="utf-8")
        self._save_state(path)

    def _save_state(self, path: Path) -> None:
        raise NotImplementedError

    @classmethod
    def load(cls, path: str | Path) -> "TeacherModel":
        path = Path(path)
        meta = json.loads((path / "meta.json").read_text(encoding="utf-8"))
        m = cls(meta["seed"], **meta["params"])
        m.temperature = float(meta["temperature"])
        m._load_state(path)
        return m

    def _load_state(self, path: Path) -> None:
        raise NotImplementedError
