"""A. 다중 스케일 HRV 특징 + LightGBM (기본안).

참가자 수가 적고 이웃 창이 많이 겹쳐 실제 표본 수가 작으므로 얕은 트리와 강한 정규화를 쓴다.
use_personal=False 면 개인 정규화 열 없이 원 특징만 쓴다 (비교용).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .base import TeacherInput, TeacherModel

DEFAULTS = dict(num_leaves=15, max_depth=4, min_child_samples=200, feature_fraction=0.6,
                bagging_fraction=0.7, bagging_freq=1, lambda_l2=5.0, learning_rate=0.03,
                num_boost_round=2000, early_stopping=100)


class LGBMTeacher(TeacherModel):
    name = "lgbm"

    def __init__(self, seed: int = 0, use_personal: bool = True, **params):
        super().__init__(seed, use_personal=use_personal, **{**DEFAULTS, **params})
        self.use_personal = use_personal
        self.booster = None
        self.columns: list[str] = []

    def _matrix(self, X: TeacherInput) -> pd.DataFrame:
        df = pd.concat([X.raw, X.pers], axis=1) if self.use_personal else X.raw
        return df[self.columns] if self.columns else df

    def fit(self, X, y, w=None, X_val=None, y_val=None, w_val=None):
        import lightgbm as lgb
        p = dict(self.params)
        p.pop("use_personal", None)
        rounds = int(p.pop("num_boost_round"))
        es = int(p.pop("early_stopping"))
        M = self._matrix(X)
        self.columns = list(M.columns)
        # 데이터가 작을 때(합성·소규모) 분할이 아예 안 생기지 않도록 학습 창 수의 1% 로 상한
        p["min_child_samples"] = int(min(p["min_child_samples"], max(20, len(M) // 100)))
        params = dict(objective="binary", verbose=-1, seed=self.seed, num_threads=0, **p)
        train = lgb.Dataset(M, y, weight=w, free_raw_data=False)
        cb, valid = [], []
        if X_val is not None and len(X_val) and len(np.unique(y_val)) == 2:
            valid = [lgb.Dataset(self._matrix(X_val), y_val, weight=w_val, reference=train)]
            cb = [lgb.early_stopping(es, verbose=False)]
        self.booster = lgb.train(params, train, num_boost_round=rounds, valid_sets=valid, callbacks=cb)
        return self

    def predict_logit(self, X):
        return self.booster.predict(self._matrix(X), raw_score=True,
                                    num_iteration=self.booster.best_iteration or None)

    def importance(self, top: int = 20) -> list[tuple[str, float]]:
        g = self.booster.feature_importance("gain")
        order = np.argsort(g)[::-1][:top]
        return [(self.columns[i], float(g[i])) for i in order]

    def _save_state(self, path: Path) -> None:
        self.booster.save_model(str(path / "model.txt"))

    def _load_state(self, path: Path) -> None:
        import lightgbm as lgb
        self.booster = lgb.Booster(model_file=str(path / "model.txt"))
        self.columns = self.booster.feature_name()
