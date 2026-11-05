"""트리 계열 후보: Random Forest, XGBoost, LightGBM (모두 features 입력)."""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np

from .base import CandidateModel, class_weights


class _SklearnLike(CandidateModel):
    input_kind = "features"
    defaults: dict = {}

    def __init__(self, n_classes: int, seed: int = 0, **params):
        super().__init__(n_classes, seed, **{**self.defaults, **params})
        self.model = None
        self.classes_: np.ndarray | None = None

    def _make(self):
        raise NotImplementedError

    def _sample_weight(self, y):
        return class_weights(y, self.n_classes)[y]

    def predict_proba(self, X):
        return self._full_proba(self.model.predict_proba(X), self.classes_)

    def save(self, path):
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        joblib.dump({"model": self.model, "classes": self.classes_}, path / "model.joblib", compress=3)
        (path / "meta.json").write_text(json.dumps(
            {"name": self.name, "n_classes": self.n_classes, "seed": self.seed, "params": self.params}, indent=2))

    @classmethod
    def load(cls, path):
        path = Path(path)
        meta = json.loads((path / "meta.json").read_text())
        m = cls(meta["n_classes"], meta["seed"], **meta["params"])
        d = joblib.load(path / "model.joblib")
        m.model, m.classes_ = d["model"], d["classes"]
        return m


class RandomForestModel(_SklearnLike):
    name = "random_forest"
    defaults = dict(n_estimators=200, max_depth=None, min_samples_leaf=2, max_features="sqrt")

    def fit(self, X, y, X_val=None, y_val=None):
        from sklearn.ensemble import RandomForestClassifier
        self.model = RandomForestClassifier(class_weight="balanced", random_state=self.seed, n_jobs=-1,
                                            **self.params)
        self.model.fit(X, y)
        self.model.set_params(n_jobs=1)   # 윈도우 1개 추론에서는 병렬화 오버헤드가 더 크다
        self.classes_ = self.model.classes_
        return self


class XGBoostModel(_SklearnLike):
    name = "xgboost"
    defaults = dict(n_estimators=300, max_depth=4, learning_rate=0.05, subsample=0.8,
                    colsample_bytree=0.8, early_stopping_rounds=30)

    def fit(self, X, y, X_val=None, y_val=None):
        from xgboost import XGBClassifier
        # XGBoost 는 라벨이 0..K'-1 로 연속이어야 하므로 학습셋에 있는 클래스만 다시 번호를 매긴다.
        self.classes_ = np.unique(y)
        remap = {c: i for i, c in enumerate(self.classes_)}
        yi = np.vectorize(remap.get)(y)
        params = dict(self.params)
        use_es = X_val is not None and len(X_val) > 0 and np.isin(y_val, self.classes_).all()
        if not use_es:
            params.pop("early_stopping_rounds", None)
        self.model = XGBClassifier(objective="multi:softprob" if len(self.classes_) > 2 else "binary:logistic",
                                   random_state=self.seed, n_jobs=-1, tree_method="hist",
                                   eval_metric="mlogloss" if len(self.classes_) > 2 else "logloss", **params)
        kw = {}
        if use_es:
            kw = dict(eval_set=[(X_val, np.vectorize(remap.get)(y_val))], verbose=False)
        self.model.fit(X, yi, sample_weight=self._sample_weight(y), **kw)
        return self


class LightGBMModel(_SklearnLike):
    name = "lightgbm"
    defaults = dict(n_estimators=300, num_leaves=15, learning_rate=0.05, subsample=0.8,
                    subsample_freq=1, colsample_bytree=0.8, min_child_samples=10)

    def fit(self, X, y, X_val=None, y_val=None):
        import lightgbm as lgb
        self.model = lgb.LGBMClassifier(class_weight="balanced", random_state=self.seed, verbose=-1,
                                        n_jobs=-1, **self.params)
        kw = {}
        if X_val is not None and len(X_val) > 0:
            kw = dict(eval_set=[(X_val, y_val)], callbacks=[lgb.early_stopping(30, verbose=False)])
        self.model.fit(X, y, **kw)
        self.classes_ = self.model.classes_
        return self
