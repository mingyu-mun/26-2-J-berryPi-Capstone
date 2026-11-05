"""후보 모델 비교 평가: 운전자 단위 분리(LOSO / GroupKFold).

각 fold 에서
1. 테스트 subject 를 떼어 두고,
2. 남은 subject 중 1명을 검증용(early stopping)으로 떼고,
3. InputPipeline 을 학습 subject 로만 fit → 학습·검증·테스트에 같은 변환 적용,
4. 5개 모델을 같은 입력으로 학습·평가한다.
특징 추출은 fold 와 무관(윈도우 단위 계산)하므로 한 번만 계산해 재사용한다.
"""
from __future__ import annotations

import time

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

from .data import WindowSet
from .features import extract_features
from .models import build_model
from .normalization import InputPipeline
from .spec import InputSpec


def subject_folds(groups: np.ndarray, n_folds: int | None, seed: int = 0):
    """n_folds=None 이면 LOSO(한 명씩), 아니면 subject 를 n_folds 묶음으로 나눈다."""
    subs = np.unique(groups)
    rng = np.random.default_rng(seed)
    subs = rng.permutation(subs)
    k = len(subs) if not n_folds else min(n_folds, len(subs))
    for chunk in np.array_split(subs, k):
        yield np.isin(groups, chunk)


def latency_ms(model, X: np.ndarray, repeats: int = 50) -> float:
    """윈도우 1개(batch=1) 추론 시간 중앙값 [ms]. PC 기준값이며 보드 측정은 별도."""
    x1 = X[:1]
    model.predict_proba(x1)   # warm-up
    ts = []
    for _ in range(repeats):
        t = time.perf_counter()
        model.predict_proba(x1)
        ts.append((time.perf_counter() - t) * 1000)
    return float(np.median(ts))


def run_benchmark(ws: WindowSet, spec: InputSpec, model_cfgs: dict[str, dict],
                  n_folds: int | None = None, seed: int = 0, label_mode: str = "multi",
                  log=print) -> dict:
    y_all = ws.y.copy()
    n_classes = spec.label.n_classes
    if label_mode == "binary":
        y_all = np.vectorize(spec.label.binary_map.get)(y_all)
        n_classes = 2
    feats_all = extract_features(ws, spec)
    labels = list(range(n_classes))

    per_model = {name: {"y_true": [], "y_pred": [], "fold_f1": [], "latency_ms": [], "size_bytes": []}
                 for name in model_cfgs}
    for fi, test_mask in enumerate(subject_folds(ws.groups, n_folds, seed)):
        trval_idx = np.where(~test_mask)[0]
        trval_subs = np.unique(ws.groups[trval_idx])
        val_sub = np.random.default_rng(seed + fi).choice(trval_subs) if len(trval_subs) > 1 else None
        val_mask = (ws.groups == val_sub) & ~test_mask
        tr_idx = np.where(~test_mask & ~val_mask)[0]
        va_idx, te_idx = np.where(val_mask)[0], np.where(test_mask)[0]

        pipe = InputPipeline(spec).fit(ws.subset(tr_idx), feats_all[tr_idx])
        inp = {k: pipe.transform(ws.subset(idx), feats_all[idx])
               for k, idx in (("tr", tr_idx), ("va", va_idx), ("te", te_idx))}
        log(f"[fold {fi}] test={sorted(set(ws.groups[te_idx]))} val={val_sub} "
            f"n_train={len(tr_idx)} n_val={len(va_idx)} n_test={len(te_idx)}")

        for name, params in model_cfgs.items():
            m = build_model(name, n_classes, seed=seed, **(params or {}))
            kind = m.input_kind
            t0 = time.perf_counter()
            m.fit(inp["tr"][kind], y_all[tr_idx], inp["va"][kind], y_all[va_idx])
            fit_s = time.perf_counter() - t0
            pred = m.predict(inp["te"][kind])
            f1 = f1_score(y_all[te_idx], pred, average="macro", labels=labels, zero_division=0)
            r = per_model[name]
            r["y_true"].append(y_all[te_idx]); r["y_pred"].append(pred); r["fold_f1"].append(f1)
            r["latency_ms"].append(latency_ms(m, inp["te"][kind]))
            r["size_bytes"].append(m.size_bytes())
            log(f"    {name:<14} macroF1={f1:.3f}  fit={fit_s:.1f}s")

    summary = {}
    for name, r in per_model.items():
        yt, yp = np.concatenate(r["y_true"]), np.concatenate(r["y_pred"])
        summary[name] = {
            "accuracy": float(accuracy_score(yt, yp)),
            "macro_f1": float(f1_score(yt, yp, average="macro", labels=labels, zero_division=0)),
            "macro_f1_fold_mean": float(np.mean(r["fold_f1"])),
            "macro_f1_fold_std": float(np.std(r["fold_f1"])),
            "per_class_f1": f1_score(yt, yp, average=None, labels=labels, zero_division=0).tolist(),
            "confusion_matrix": confusion_matrix(yt, yp, labels=labels).tolist(),
            "latency_ms_batch1": float(np.median(r["latency_ms"])),
            "model_size_kb": float(np.median(r["size_bytes"]) / 1024),
        }
    return {"n_windows": int(len(ws)), "n_subjects": int(len(np.unique(ws.groups))),
            "n_classes": n_classes, "label_mode": label_mode, "models": summary}


def summary_markdown(res: dict) -> str:
    rows = ["| 모델 | Accuracy | Macro-F1 | Fold별 Macro-F1 (평균±표준편차) | 지연 ms (batch 1, PC) | 크기 KB |",
            "| --- | --- | --- | --- | --- | --- |"]
    for name, s in res["models"].items():
        rows.append(f"| {name} | {s['accuracy']:.3f} | {s['macro_f1']:.3f} | "
                    f"{s['macro_f1_fold_mean']:.3f} ± {s['macro_f1_fold_std']:.3f} | "
                    f"{s['latency_ms_batch1']:.2f} | {s['model_size_kb']:.0f} |")
    return "\n".join(rows)
