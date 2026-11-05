"""입력 정규화 규약. 모든 후보 모델이 이 파이프라인 하나를 거친 입력을 받는다.

규칙
1. 통계(평균·표준편차·중앙값·sigma_pop)는 **학습 fold 의 학습 참가자로만** fit 한다.
   검증·테스트 참가자의 정보가 섞이면 LOSO 성능이 부풀려진다.
2. sequence 입력 (CNN+BiLSTM)
   - 생체 채널(ECG/PPG): 윈도우·채널별 z-score  x' = (x - mean_w) / (std_w + eps)
     (mean_w, std_w 는 그 윈도우 안에서 계산 → 센서별 진폭 단위 차이를 제거, fit 불필요)
   - CAN 채널(선택): 학습셋 전체의 채널별 mean/std 로 z-score
3. features 입력 (MLP·RF·XGBoost·LightGBM) = [생체 특징 F | 개인화 특징 P]
   - 개인화 특징은 참가자 프로필과 sigma_pop 으로 계산 (baseline.py)
   - NaN → 학습셋 특징별 중앙값으로 대체
   - 학습셋 특징별 mean/std 로 z-score. 단 baseline_valid 열(0/1)은 그대로 둔다.
4. personal 입력 (CNN+BiLSTM 보조 입력) = features 의 개인화 열 P개 (정규화 후).
5. fit 결과는 JSON 으로 저장해 보드(추론) 쪽에서 같은 값을 그대로 쓴다.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .baseline import Profile, fit_sigma_pop, personal_matrix
from .data import WindowSet
from .features import extract_features
from .spec import BIO_CHANNELS, InputSpec

EPS = 1e-6


class InputPipeline:
    def __init__(self, spec: InputSpec):
        self.spec = spec
        self.seq_mean: np.ndarray | None = None   # (C,)
        self.seq_std: np.ndarray | None = None    # (C,)
        self.feat_median: np.ndarray | None = None  # (F+P,)
        self.feat_mean: np.ndarray | None = None
        self.feat_std: np.ndarray | None = None
        self.sigma_pop: dict[str, float] = {}

    # ---- sequence ----
    def _bio_mask(self) -> np.ndarray:
        return np.array([c in BIO_CHANNELS for c in self.spec.channels])

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
    def _full_features(self, base: np.ndarray, groups: np.ndarray, profiles: dict[str, Profile] | None,
                       force_invalid: np.ndarray | None = None) -> np.ndarray:
        if not self.spec.use_personal:
            return base
        pers = personal_matrix(base, groups, profiles or {}, self.sigma_pop, self.spec, force_invalid)
        return np.concatenate([base, pers], axis=1)

    def _impute(self, f: np.ndarray) -> np.ndarray:
        f = f.astype(np.float32, copy=True)
        r, c = np.where(np.isnan(f))
        f[r, c] = self.feat_median[c]
        return f

    def transform_features(self, f: np.ndarray) -> np.ndarray:
        f = self._impute(f)
        if self.spec.feature_norm == "global":
            f = (f - self.feat_mean) / np.where(self.feat_std > EPS, self.feat_std, 1.0)
        return f.astype(np.float32)

    # ---- public ----
    def fit(self, ws: WindowSet, base_feats: np.ndarray | None = None,
            profiles: dict[str, Profile] | None = None) -> "InputPipeline":
        self.seq_mean = ws.x_seq.mean(axis=(0, 2)).astype(np.float32)
        self.seq_std = ws.x_seq.std(axis=(0, 2)).astype(np.float32)
        base = extract_features(ws, self.spec) if base_feats is None else base_feats
        self.sigma_pop = fit_sigma_pop(base, ws.groups, self.spec)
        f = self._full_features(base, ws.groups, profiles)
        med = np.nanmedian(f, axis=0) if len(f) else np.zeros(f.shape[1])
        self.feat_median = np.where(np.isnan(med), 0.0, med).astype(np.float32)
        filled = self._impute(f)
        self.feat_mean = filled.mean(axis=0).astype(np.float32)
        self.feat_std = filled.std(axis=0).astype(np.float32)
        if self.spec.use_personal:   # baseline_valid(마지막 열)은 0/1 그대로 둔다
            self.feat_mean[-1], self.feat_std[-1] = 0.0, 1.0
        return self

    def transform(self, ws: WindowSet, base_feats: np.ndarray | None = None,
                  profiles: dict[str, Profile] | None = None, invalid_rate: float = 0.0,
                  seed: int = 0) -> dict[str, np.ndarray]:
        """모델 입력 dict: {'sequence': (N,C,T), 'features': (N,F+P), 'personal': (N,P)}.

        invalid_rate > 0 이면 그 비율의 윈도우를 무작위로 '기준 없음'으로 만든다 (학습셋 전용).
        """
        base = extract_features(ws, self.spec) if base_feats is None else base_feats
        force = None
        if invalid_rate > 0:
            force = np.random.default_rng(seed).random(len(ws)) < invalid_rate
        feats = self.transform_features(self._full_features(base, ws.groups, profiles, force))
        P = len(self.spec.personal_feature_names)
        return {"sequence": self.transform_sequence(ws.x_seq), "features": feats,
                "personal": feats[:, feats.shape[1] - P:] if P else feats[:, :0]}

    def save(self, path: str | Path) -> None:
        d = {
            "spec": self.spec.to_dict(),
            "seq_mean": self.seq_mean.tolist(), "seq_std": self.seq_std.tolist(),
            "feat_median": self.feat_median.tolist(),
            "feat_mean": self.feat_mean.tolist(), "feat_std": self.feat_std.tolist(),
            "sigma_pop": self.sigma_pop,
        }
        Path(path).write_text(json.dumps(d, ensure_ascii=False, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "InputPipeline":
        d = json.loads(Path(path).read_text(encoding="utf-8"))
        p = cls(InputSpec.from_dict(d["spec"]))
        for k in ("seq_mean", "seq_std", "feat_median", "feat_mean", "feat_std"):
            setattr(p, k, np.asarray(d[k], dtype=np.float32))
        p.sigma_pop = {k: float(v) for k, v in d.get("sigma_pop", {}).items()}
        return p
