"""후보 모델 공통 인터페이스.

모든 후보 모델은 아래 메서드를 같은 의미로 제공한다. 평가 스크립트와 보드 추론 코드는
모델 종류를 몰라도 이 인터페이스만으로 동작한다.

- input_kind: InputPipeline.transform() 결과의 key. "features" (N,F+P) 또는
  "sequence+personal" ((N,C,T), (N,P)) 튜플. model_input() 으로 꺼낸다.
- fit(X, y, X_val, y_val): 학습. 검증셋이 있으면 early stopping 에 사용.
- predict_proba(X) -> (N, n_classes) float, 각 행의 합 = 1.
- predict(X) -> (N,) int
- save(dir) / load(dir), size_bytes()
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np


def model_input(inp: dict, kind: str):
    """InputPipeline.transform() 결과에서 모델 종류에 맞는 입력을 꺼낸다."""
    if kind == "sequence+personal":
        return (inp["sequence"], inp["personal"])
    return inp[kind]


def n_samples(X) -> int:
    return len(X[0]) if isinstance(X, (tuple, list)) else len(X)


class CandidateModel(ABC):
    name: str = "base"
    input_kind: str = "features"

    def __init__(self, n_classes: int, seed: int = 0, **params):
        self.n_classes = n_classes
        self.seed = seed
        self.params = params

    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray, X_val: np.ndarray | None = None,
            y_val: np.ndarray | None = None) -> "CandidateModel": ...

    @abstractmethod
    def predict_proba(self, X: np.ndarray) -> np.ndarray: ...

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.predict_proba(X).argmax(axis=1)

    @abstractmethod
    def save(self, path: str | Path) -> None: ...

    @classmethod
    @abstractmethod
    def load(cls, path: str | Path) -> "CandidateModel": ...

    def size_bytes(self) -> int:
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            self.save(d)
            return sum(p.stat().st_size for p in Path(d).rglob("*") if p.is_file())

    def _full_proba(self, proba: np.ndarray, present: np.ndarray) -> np.ndarray:
        """학습셋에 없던 클래스가 있어도 (N, n_classes) 형태를 보장."""
        out = np.zeros((proba.shape[0], self.n_classes), dtype=np.float64)
        out[:, present] = proba
        return out


def class_weights(y: np.ndarray, n_classes: int) -> np.ndarray:
    """클래스 불균형 보정 가중치 w_k = N / (K * N_k)."""
    cnt = np.bincount(y, minlength=n_classes).astype(np.float64)
    w = np.where(cnt > 0, len(y) / (n_classes * np.maximum(cnt, 1)), 0.0)
    return w
