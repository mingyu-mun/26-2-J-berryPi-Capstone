"""입력 정규화 규약. 모든 후보 모델이 이 파이프라인 하나를 거친 입력을 받는다.

규칙
1. 통계(평균·표준편차·중앙값)는 **학습 fold 의 학습 subject 로만** fit 한다.
   검증·테스트 subject 의 정보가 섞이면 LOSO 성능이 부풀려진다.
2. sequence 입력 (CNN+BiLSTM)
   - 생체 채널(ECG/PPG): 윈도우·채널별 z-score  x' = (x - mean_w) / (std_w + eps)
     (mean_w, std_w 는 그 윈도우 안에서 계산 → 센서별 진폭 단위 차이를 제거, fit 불필요)
   - CAN 채널: 학습셋 전체의 채널별 mean/std 로 z-score (물리 단위 유지 의미)
3. features 입력 (MLP·RF·XGBoost·LightGBM)
   - NaN → 학습셋 특징별 중앙값으로 대체
   - 학습셋 특징별 mean/std 로 z-score. 트리 모델은 스케일에 무관하지만 같은 입력을 써서
     모델 간 비교 조건을 맞추고, 보드에서 전처리 코드를 하나로 유지한다.
4. fit 결과는 JSON 으로 저장해 보드(추론) 쪽에서 같은 값을 그대로 쓴다.

개인화(Baseline 대비 정규화, subject 별 calibration)는 이 단계 범위 밖이다.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .data import WindowSet
from .features import extract_features
from .spec import BIO_CHANNELS, InputSpec

EPS = 1e-6


class InputPipeline:
    def __init__(self, spec: InputSpec):
        self.spec = spec
        self.seq_mean: np.ndarray | None = None   # (C,)
        self.seq_std: np.ndarray | None = None    # (C,)
        self.feat_median: np.ndarray | None = None  # (F,)
        self.feat_mean: np.ndarray | None = None
        self.feat_std: np.ndarray | None = None

    # ---- sequence ----
    def _bio_mask(self) -> np.ndarray:
        return np.array([c in BIO_CHANNELS for c in self.spec.channels])

    def _fit_seq(self, x: np.ndarray) -> None:
        self.seq_mean = x.mean(axis=(0, 2)).astype(np.float32)
        self.seq_std = x.std(axis=(0, 2)).astype(np.float32)

    def transform_sequence(self, x: np.ndarray) -> np.ndarray:
        x = x.astype(np.float32, copy=True)
        bio = self._bio_mask()
        if self.spec.sequence_bio_norm == "per_window":
            xb = x[:, bio]
            x[:, bio] = (xb - xb.mean(axis=2, keepdims=True)) / (xb.std(axis=2, keepdims=True) + EPS)
            glob = ~bio
        else:
            glob = np.ones_like(bio)
        if glob.any():
            m = self.seq_mean[glob][None, :, None]
            s = self.seq_std[glob][None, :, None]
            x[:, glob] = (x[:, glob] - m) / (s + EPS)
        return x

    # ---- features ----
    def _fit_feat(self, f: np.ndarray) -> None:
        med = np.nanmedian(f, axis=0)
        self.feat_median = np.where(np.isnan(med), 0.0, med).astype(np.float32)
        filled = self._impute(f)
        self.feat_mean = filled.mean(axis=0).astype(np.float32)
        self.feat_std = filled.std(axis=0).astype(np.float32)

    def _impute(self, f: np.ndarray) -> np.ndarray:
        f = f.astype(np.float32, copy=True)
        r, c = np.where(np.isnan(f))
        f[r, c] = self.feat_median[c]
        return f

    def transform_features(self, f: np.ndarray) -> np.ndarray:
        f = self._impute(f)
        if self.spec.feature_norm == "global":
            f = (f - self.feat_mean) / (self.feat_std + EPS)
        return f.astype(np.float32)

    # ---- public ----
    def fit(self, ws: WindowSet, feats: np.ndarray | None = None) -> "InputPipeline":
        self._fit_seq(ws.x_seq)
        self._fit_feat(extract_features(ws, self.spec) if feats is None else feats)
        return self

    def transform(self, ws: WindowSet, feats: np.ndarray | None = None) -> dict[str, np.ndarray]:
        """모델 입력 dict: {'sequence': (N,C,T), 'features': (N,F)}."""
        f = extract_features(ws, self.spec) if feats is None else feats
        return {"sequence": self.transform_sequence(ws.x_seq), "features": self.transform_features(f)}

    def save(self, path: str | Path) -> None:
        d = {
            "spec": self.spec.to_dict(),
            "seq_mean": self.seq_mean.tolist(), "seq_std": self.seq_std.tolist(),
            "feat_median": self.feat_median.tolist(),
            "feat_mean": self.feat_mean.tolist(), "feat_std": self.feat_std.tolist(),
        }
        Path(path).write_text(json.dumps(d, ensure_ascii=False, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "InputPipeline":
        d = json.loads(Path(path).read_text(encoding="utf-8"))
        p = cls(InputSpec.from_dict(d["spec"]))
        for k in ("seq_mean", "seq_std", "feat_median", "feat_mean", "feat_std"):
            setattr(p, k, np.asarray(d[k], dtype=np.float32))
        return p
