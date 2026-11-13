"""최종 교사: 전체 데이터로 다시 학습해 묶음으로 저장하고, 대상 데이터에 P(졸음)·라벨을 붙인다.

teacher_bundle/
  bundle.json      버전, 모델 이름, 학습 데이터 요약
  spec.json        TeacherSpec (창·스케일·Baseline 규칙)
  normalizer.json  σ_pop (학습 참가자 전체)
  model/           모델 파일 (+ 온도 T, LODO 예측으로 맞춤)
  tau.json         τ_hi, τ_lo (tau 단계 결과, 선택)
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from ..data_io import iter_session_dirs, load_session
from .calibrate import fit_temperature
from .evaluate import ENSEMBLE_MEMBERS, build_candidate, make_input, split_val, usable
from .features import build_tables, build_tachograms
from .models import REGISTRY, EnsembleTeacher, TeacherModel, sample_weights
from .personal import PersonalNormalizer
from .rr import extract_rr
from .spec import TeacherSpec
from .tau import apply_tau


class TeacherBundle:
    def __init__(self, spec: TeacherSpec, norm: PersonalNormalizer, model: TeacherModel, info: dict,
                 tau: dict | None = None):
        self.spec, self.norm, self.model, self.info, self.tau = spec, norm, model, info, tau

    @property
    def version(self) -> str:
        return f"{self.spec.version}+{self.info['model']}"

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        (path / "bundle.json").write_text(json.dumps(self.info, ensure_ascii=False, indent=2), encoding="utf-8")
        (path / "spec.json").write_text(json.dumps(self.spec.to_dict(), indent=2), encoding="utf-8")
        (path / "normalizer.json").write_text(json.dumps(self.norm.to_dict(), indent=2), encoding="utf-8")
        self.model.save(path / "model")
        if self.tau:
            (path / "tau.json").write_text(json.dumps(self.tau, ensure_ascii=False, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "TeacherBundle":
        path = Path(path)
        info = json.loads((path / "bundle.json").read_text(encoding="utf-8"))
        spec = TeacherSpec.from_dict(json.loads((path / "spec.json").read_text(encoding="utf-8")))
        norm = PersonalNormalizer.from_dict(spec, json.loads((path / "normalizer.json").read_text(encoding="utf-8")))
        meta = json.loads((path / "model" / "meta.json").read_text(encoding="utf-8"))
        model = REGISTRY[meta["name"]].load(path / "model")
        tau = json.loads((path / "tau.json").read_text(encoding="utf-8")) if (path / "tau.json").exists() else None
        return cls(spec, norm, model, info, tau)


def fit_final(win: pd.DataFrame, prof: pd.DataFrame, spec: TeacherSpec, model_name: str,
              tacho: np.ndarray | None = None, oof: pd.DataFrame | None = None, seed: int = 0,
              params: dict | None = None, log=print) -> TeacherBundle:
    params = params or {}
    keep = np.where(usable(win))[0]
    W = win.iloc[keep].reset_index(drop=True)
    TA = tacho[keep] if tacho is not None else None
    y = W["y"].to_numpy(np.int64)
    rng = np.random.default_rng(seed)
    tr, va = split_val(W, np.arange(len(W)), 0.1, rng)      # early stopping 용 10%
    norm = PersonalNormalizer(spec).fit(W.iloc[tr], prof)
    mspc_fit = norm.mspc
    force = rng.random(len(tr)) < spec.train_invalid_rate
    sl = (lambda a, i: None if a is None else a[i])
    Xtr = make_input(W.iloc[tr], prof, spec, norm, sl(TA, tr), force)
    Xva = make_input(W.iloc[va], prof, spec, norm, sl(TA, va))
    wtr = sample_weights(W["dataset_id"].to_numpy()[tr], W["group"].to_numpy()[tr], y[tr])
    wva = sample_weights(W["dataset_id"].to_numpy()[va], W["group"].to_numpy()[va], y[va]) if len(va) else None

    def one(name: str) -> TeacherModel:
        m = build_candidate(name, seed, params.get(name))
        m.fit(Xtr, y[tr], wtr, Xva, y[va], wva)
        # 최종 T 는 LODO 예측(처음 보는 데이터셋)으로 맞춘다. 없으면 검증셋으로.
        if oof is not None and f"logit_{name}" in oof:
            m.temperature = fit_temperature(oof[f"logit_{name}"].to_numpy(), oof["y"].to_numpy())
        elif len(va):
            m.temperature = fit_temperature(m.predict_logit(Xva), y[va], wva)
        log(f"  {name}: T={m.temperature:.2f}")
        return m

    if model_name == "ensemble":
        model = EnsembleTeacher(seed, members=[one(n) for n in ENSEMBLE_MEMBERS])
        if oof is not None and "logit_ensemble" in oof:
            model.temperature = fit_temperature(oof["logit_ensemble"].to_numpy(), oof["y"].to_numpy())
    else:
        model = one(model_name)
    norm = PersonalNormalizer(spec).fit(W)   # 라벨링 때 쓰는 σ_pop 은 전체 참가자로
    norm.mspc = mspc_fit                     # MSPC 는 모델이 학습한 것과 같은 PCA 를 써야 함
    info = {"model": model_name, "version": spec.version, "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "n_windows": int(len(W)), "n_subjects": int(W["group"].nunique()),
            "datasets": {d: int(n) for d, n in W["dataset_id"].value_counts().items()},
            "drowsy_rate": float(y.mean())}
    return TeacherBundle(spec, norm, model, info)


# ---------------------------------------------------------------- 라벨링
def predict_windows(bundle: TeacherBundle, rrs, n_jobs: int = 1) -> tuple[pd.DataFrame, np.ndarray]:
    spec = bundle.spec
    win, prof = build_tables(rrs, spec, n_jobs)
    tacho = build_tachograms(rrs, win, spec) if _uses_tacho(bundle.model) else None
    X = make_input(win, prof, spec, bundle.norm, tacho)
    return win, bundle.model.predict_proba(X)[:, 1]


def _uses_tacho(m: TeacherModel) -> bool:
    return m.uses_tacho or any(getattr(x, "uses_tacho", False) for x in getattr(m, "members", []))


def label_tree(bundle: TeacherBundle, roots: list[str | Path], out: str | Path | None = None,
               tau_hi: float | None = None, tau_lo: float | None = None, n_jobs: int = 1, log=print) -> pd.DataFrame:
    """규약 폴더들의 세션마다 teacher_p.csv, labels_window.csv, teacher_label.json 을 쓴다.

    out 을 주면 세션 폴더 대신 out/<dataset>/<subject>/<session>/ 에 쓴다 (원본 폴더를 건드리지 않음).
    """
    tau = bundle.tau or {}
    tau_hi = tau_hi if tau_hi is not None else tau.get("tau_hi", 0.8)
    tau_lo = tau_lo if tau_lo is not None else tau.get("tau_lo", 0.2)
    rrs, dirs = [], {}
    for root in roots:
        for d in iter_session_dirs(root):
            try:
                rec = load_session(d, channels=("ecg",))
            except KeyError:
                log(f"  ECG 없음, 건너뜀: {d}")
                continue
            r = extract_rr(rec, bundle.spec)
            rrs.append(r)
            dirs[(r.dataset_id, r.subject_id or "", r.session_id)] = Path(d)
    if not rrs:
        raise ValueError("ECG 가 있는 세션이 없습니다")
    win, p = predict_windows(bundle, rrs, n_jobs)
    win = win.assign(p_drowsy=p, label=apply_tau(p, tau_hi, tau_lo, win["valid"].to_numpy(bool)))
    for key, g in win.groupby(["dataset_id", "subject_id", "session_id"]):
        src = dirs[key]
        dst = Path(out) / key[0] / (key[1] or "anon") / key[2] if out else src
        dst.mkdir(parents=True, exist_ok=True)
        g2 = g.sort_values("t_rel")
        pd.DataFrame({"t_start": g2["t_start"], "t_end": g2["t_start"] + bundle.spec.window_sec,
                      "p_drowsy": g2["p_drowsy"].round(4), "valid": g2["valid"].astype(int),
                      "teacher_version": bundle.version}).to_csv(dst / "teacher_p.csv", index=False)
        g2[["t_start", "label"]].to_csv(dst / "labels_window.csv", index=False)
        (dst / "teacher_label.json").write_text(json.dumps(
            {"label_source": "teacher", "label_type": "drowsy_teacher_p", "teacher_version": bundle.version,
             "tau_hi": tau_hi, "tau_lo": tau_lo, "window_sec": bundle.spec.window_sec,
             "stride_sec": bundle.spec.stride_sec}, indent=2), encoding="utf-8")
    return win


def label_report(win: pd.DataFrame) -> str:
    lines = ["# 교사 라벨 분포", "", "| 데이터셋 | 창 | 졸음(1) | 정상(0) | 미라벨(-1) | P 중앙값 | P 90% |",
             "| --- | --- | --- | --- | --- | --- | --- |"]
    for d, g in win.groupby("dataset_id"):
        lab = g["label"]
        lines.append(f"| {d} | {len(g)} | {(lab == 1).mean():.1%} | {(lab == 0).mean():.1%} | {(lab == -1).mean():.1%} | "
                     f"{g['p_drowsy'].median():.2f} | {g['p_drowsy'].quantile(0.9):.2f} |")
    return "\n".join(lines) + "\n"
