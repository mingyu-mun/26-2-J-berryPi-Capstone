"""교사 후보 비교: LODO(데이터셋 하나씩 빼고 학습) + 보조 참가자 단위 K-fold.

fold 하나
1. 테스트 데이터셋(또는 참가자 묶음)을 떼어 둔다.
2. 남은 데이터에서 데이터셋마다 참가자 20% 를 검증용으로 뗀다 (early stopping + 온도 T).
3. σ_pop·가중치는 학습 참가자로만. 학습 창의 20% 는 기준을 일부러 무효로 (기준 없는 대상 데이터 대비).
4. 후보마다 학습 → 검증셋으로 T → 테스트 예측. 앙상블은 보정된 A·B 로짓 평균 (검증셋으로 T 다시).
모든 fold 의 테스트 예측을 oof 표 하나로 모은다 (τ 조정과 최종 T 가 이 표를 쓴다).
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, roc_auc_score

from .calibrate import brier, ece, fit_temperature
from .models import REGISTRY, EnsembleTeacher, TeacherInput, sample_weights
from .models.base import sigmoid
from .personal import PersonalNormalizer
from .spec import TeacherSpec

# 후보 이름 → (클래스 이름, 생성 인자)
CANDIDATES = {
    "rule": ("rule", {}),
    "lgbm": ("lgbm", {"use_personal": True}),
    "lgbm_nopers": ("lgbm", {"use_personal": False}),
    "lgbm_signabs": ("lgbm", {"use_personal": True, "pers_cols": "signed_abs"}),   # 부호 있는 z + |z|, Σz²
    "lgbm_abs": ("lgbm", {"use_personal": True, "pers_cols": "abs"}),              # |z|, |r|, dev 만 (부호 없음)
    "lgbm_dev": ("lgbm", {"use_personal": True, "pers_cols": "dev"}),              # 스케일별 거리 sqrt(mean z²) 만
    "lgbm_mspc": ("lgbm", {"use_personal": True, "pers_cols": "mspc"}),            # MSPC T²·Q 만
    "lgbm_abs_mspc": ("lgbm", {"use_personal": True, "pers_cols": "abs_mspc"}),    # 부호 없는 특징 + T²·Q
    "rrnet_mspc": ("rrnet", {"extra_cols": ["mspc_t2", "mspc_q"]}),               # 타코그램 + T²·Q
    "rrnet_dev": ("rrnet", {"extra_cols": ["mspc_t2", "mspc_q", "dev_120", "dev_300", "dev_core"]}),
    "rrnet": ("rrnet", {}),
    "ensemble": ("ensemble", {}),
}
SIMPLICITY = ["lgbm", "lgbm_nopers", "lgbm_dev", "lgbm_abs", "lgbm_signabs", "lgbm_mspc", "lgbm_abs_mspc",
              "rrnet", "rrnet_mspc", "rrnet_dev", "ensemble"]   # 동점일 때 앞쪽을 고른다
ENSEMBLE_MEMBERS = ("lgbm", "rrnet")


def build_candidate(name: str, seed: int, params: dict | None = None):
    cls_name, kw = CANDIDATES[name]
    return REGISTRY[cls_name](seed, **{**kw, **(params or {})})


def make_input(win: pd.DataFrame, prof: pd.DataFrame, spec: TeacherSpec, norm: PersonalNormalizer,
               tacho: np.ndarray | None = None, force_invalid: np.ndarray | None = None) -> TeacherInput:
    win = win.reset_index(drop=True)
    pers = norm.transform(win, prof, force_invalid)
    base_rr = None
    if tacho is not None:
        p = win[["profile_key"]].merge(prof, on="profile_key", how="left")
        col = f"m_rr_mean_{int(spec.scales[-1])}"
        base_rr = p[col].to_numpy(np.float64) if col in p else np.full(len(win), np.nan)
        fb = p.get(f"m_rr_mean_{int(spec.scales[0])}")
        if fb is not None:
            base_rr = np.where(np.isfinite(base_rr), base_rr, fb.to_numpy(np.float64))
        base_rr = np.where(pers["baseline_valid"].to_numpy() > 0.5, base_rr, np.nan)
    return TeacherInput(win[list(spec.raw_feature_names)], pers.reset_index(drop=True), tacho, base_rr)


def usable(win: pd.DataFrame) -> np.ndarray:
    return (win["valid"].to_numpy(bool)) & (win["y"].to_numpy() >= 0)


def split_val(win: pd.DataFrame, idx: np.ndarray, frac: float, rng) -> tuple[np.ndarray, np.ndarray]:
    """idx 안에서 데이터셋마다 참가자 frac 을 검증용으로 (각 데이터셋 최소 1명, 참가자 2명 이상일 때)."""
    sub = win.iloc[idx]
    val_groups = []
    for _, g in sub.groupby("dataset_id")["group"]:
        gs = np.unique(g.to_numpy())
        if len(gs) >= 2:
            k = max(1, int(round(frac * len(gs))))
            val_groups += list(rng.choice(gs, k, replace=False))
    is_val = sub["group"].isin(val_groups).to_numpy()
    return idx[~is_val], idx[is_val]


def folds(win: pd.DataFrame, mode: str, n_folds: int, seed: int):
    if mode == "lodo":
        for d in sorted(win["dataset_id"].unique()):
            yield d, (win["dataset_id"] == d).to_numpy()
    else:
        groups = np.random.default_rng(seed).permutation(win["group"].unique())
        for k, chunk in enumerate(np.array_split(groups, n_folds)):
            yield f"k{k}", win["group"].isin(chunk).to_numpy()


@dataclass
class FoldResult:
    name: str
    temperatures: dict = field(default_factory=dict)
    importance: list = field(default_factory=list)
    fit_sec: dict = field(default_factory=dict)


def run_cv(win: pd.DataFrame, prof: pd.DataFrame, spec: TeacherSpec, models: list[str],
           tacho: np.ndarray | None = None, mode: str = "lodo", n_folds: int = 5, seed: int = 0,
           params: dict | None = None, val_frac: float = 0.2, log=print) -> tuple[pd.DataFrame, list[FoldResult]]:
    params = params or {}
    keep = np.where(usable(win))[0]
    W = win.iloc[keep].reset_index(drop=True)
    TA = tacho[keep] if tacho is not None else None
    need_tacho = any(CANDIDATES[m][0] == "rrnet" for m in models) or "ensemble" in models
    if need_tacho and TA is None:
        raise ValueError("rrnet/ensemble 에는 타코그램이 필요합니다 (features 단계 --tacho).")
    if "ensemble" in models:
        for m in ENSEMBLE_MEMBERS:
            if m not in models:
                models = [m] + models
    order = [m for m in models if m != "ensemble"] + (["ensemble"] if "ensemble" in models else [])
    y_all = W["y"].to_numpy(np.int64)
    rng = np.random.default_rng(seed)
    oof_parts, results = [], []
    for fi, (fname, test_mask) in enumerate(folds(W, mode, n_folds, seed)):
        te = np.where(test_mask)[0]
        tr, va = split_val(W, np.where(~test_mask)[0], val_frac, rng)
        if len(np.unique(y_all[tr])) < 2 or len(te) == 0:
            log(f"[{fname}] 학습 라벨이 한 클래스뿐이거나 테스트가 비어 건너뜀")
            continue
        norm = PersonalNormalizer(spec).fit(W.iloc[tr], prof)   # MSPC 는 학습 fold 의 정상 창으로만
        force = np.random.default_rng(seed + fi).random(len(tr)) < spec.train_invalid_rate
        sl = (lambda a, i: None if a is None else a[i])
        X = {"tr": make_input(W.iloc[tr], prof, spec, norm, sl(TA, tr), force),
             "va": make_input(W.iloc[va], prof, spec, norm, sl(TA, va)),
             "te": make_input(W.iloc[te], prof, spec, norm, sl(TA, te))}
        w_tr = sample_weights(W["dataset_id"].to_numpy()[tr], W["group"].to_numpy()[tr], y_all[tr])
        w_va = sample_weights(W["dataset_id"].to_numpy()[va], W["group"].to_numpy()[va], y_all[va]) if len(va) else None
        log(f"[{fname}] train={len(tr)} ({W.iloc[tr]['group'].nunique()}명) val={len(va)} test={len(te)} "
            f"({W.iloc[te]['group'].nunique()}명, 졸음 {y_all[te].mean():.2f})")
        fr = FoldResult(fname)
        part = W.iloc[te][["row_id", "dataset_id", "group", "session_id", "t_start", "inducer", "y"]].copy()
        part["fold"] = fname
        fitted = {}
        for name in order:
            t0 = time.perf_counter()
            if name == "ensemble":
                mem = [fitted[m] for m in ENSEMBLE_MEMBERS]
                m = EnsembleTeacher(seed, members=mem)
            else:
                m = build_candidate(name, seed, params.get(name))
                m.fit(X["tr"], y_all[tr], w_tr, X["va"], y_all[va], w_va)
            if len(va) and len(np.unique(y_all[va])) == 2:
                m.temperature = fit_temperature(m.predict_logit(X["va"]), y_all[va], w_va)
            fitted[name] = m
            lg = m.predict_logit(X["te"])
            part[f"logit_{name}"] = lg
            part[f"p_{name}"] = sigmoid(lg / m.temperature)
            fr.temperatures[name] = m.temperature
            fr.fit_sec[name] = time.perf_counter() - t0
            if name == "lgbm":
                fr.importance = m.importance()
            a = _auc(y_all[te], part[f"p_{name}"].to_numpy())
            log(f"    {name:<12} AUROC={a:.3f}  T={m.temperature:.2f}  fit={fr.fit_sec[name]:.1f}s")
        oof_parts.append(part)
        results.append(fr)
    oof = pd.concat(oof_parts, ignore_index=True) if oof_parts else pd.DataFrame()
    return oof, results


def _auc(y, p) -> float:
    return float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else float("nan")


def metrics(oof: pd.DataFrame, models: list[str]) -> dict:
    """후보별: 데이터셋별·유발 요인별·전체 지표 + 데이터셋 평균."""
    out = {}
    for m in models:
        col = f"p_{m}"
        if col not in oof:
            continue
        per_ds = {}
        for d, g in oof.groupby("dataset_id"):
            y, p = g["y"].to_numpy(), g[col].to_numpy()
            per_ds[d] = {"n": int(len(g)), "drowsy_rate": float(y.mean()), "auroc": _auc(y, p),
                         "bal_acc": float(balanced_accuracy_score(y, p >= 0.5)) if len(np.unique(y)) == 2 else float("nan"),
                         "ece": ece(p, y), "brier": brier(p, y)}
        per_ind = {k: _auc(g["y"].to_numpy(), g[col].to_numpy()) for k, g in oof.groupby("inducer")}
        y, p = oof["y"].to_numpy(), oof[col].to_numpy()
        aucs = [v["auroc"] for v in per_ds.values() if np.isfinite(v["auroc"])]
        out[m] = {"auroc_mean": float(np.mean(aucs)) if aucs else float("nan"),
                  "bal_acc_mean": float(np.nanmean([v["bal_acc"] for v in per_ds.values()])),
                  "ece_mean": float(np.nanmean([v["ece"] for v in per_ds.values()])),
                  "auroc_pooled": _auc(y, p), "ece_pooled": ece(p, y), "brier_pooled": brier(p, y),
                  "per_dataset": per_ds, "per_inducer": per_ind}
    return out


def choose_best(summary: dict, tol: float = 0.01) -> str:
    """AUROC(데이터셋 평균) 최고 후보. 차이 tol 이하면 ECE 낮은 쪽, 그래도 비슷하면 단순한 쪽."""
    cands = [m for m in SIMPLICITY if m in summary and np.isfinite(summary[m]["auroc_mean"])]
    if not cands:
        raise ValueError("선택할 후보가 없습니다")
    best = max(summary[m]["auroc_mean"] for m in cands)
    near = [m for m in cands if summary[m]["auroc_mean"] >= best - tol]
    e_min = min(summary[m]["ece_mean"] for m in near)
    near = [m for m in near if summary[m]["ece_mean"] <= e_min + 0.005]
    return sorted(near, key=SIMPLICITY.index)[0]


def report_markdown(summary: dict, best: str, mode: str, results: list[FoldResult]) -> str:
    lines = [f"# 교사 후보 비교 ({mode.upper()})", "",
             f"선택: **{best}** (AUROC 데이터셋 평균 기준, 차이 0.01 이하면 ECE·단순성)", "",
             "| 후보 | AUROC (데이터셋 평균) | 균형 정확도 | ECE (평균) | AUROC (전체) | Brier |",
             "| --- | --- | --- | --- | --- | --- |"]
    for m, s in summary.items():
        lines.append(f"| {m} | {s['auroc_mean']:.3f} | {s['bal_acc_mean']:.3f} | {s['ece_mean']:.3f} | "
                     f"{s['auroc_pooled']:.3f} | {s['brier_pooled']:.3f} |")
    if "rule" in summary and best in summary:
        gap = summary[best]["auroc_mean"] - summary["rule"]["auroc_mean"]
        lines += ["", f"규칙 기준선 대비: {best} 가 AUROC {gap:+.3f}."]
    ds = sorted({d for s in summary.values() for d in s["per_dataset"]})
    lines += ["", "## 데이터셋별 AUROC", "", "| 후보 | " + " | ".join(ds) + " |",
              "| --- |" + " --- |" * len(ds)]
    for m, s in summary.items():
        lines.append(f"| {m} | " + " | ".join(f"{s['per_dataset'].get(d, {}).get('auroc', float('nan')):.3f}" for d in ds) + " |")
    ind = sorted({k for s in summary.values() for k in s["per_inducer"]})
    lines += ["", "## 유발 요인별 AUROC", "", "| 후보 | " + " | ".join(ind) + " |", "| --- |" + " --- |" * len(ind)]
    for m, s in summary.items():
        lines.append(f"| {m} | " + " | ".join(f"{s['per_inducer'].get(k, float('nan')):.3f}" for k in ind) + " |")
    lines += ["", "## fold 별 LightGBM 상위 특징 (gain)", ""]
    for r in results:
        if r.importance:
            lines.append(f"- {r.name}: " + ", ".join(n for n, _ in r.importance[:8]))
    return "\n".join(lines) + "\n"
