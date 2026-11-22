"""실측 녹음 평가 (CPU, 재학습 없음). 어느 PC 에서든 같은 숫자가 나오게 설정을 고정한다. docs/selfmeasure.md 참고.

    python scripts/convert_jberry_sensor.py --raw "측정 20261119_20_14.zip" --out data/converted
    python scripts/selfmeasure_eval.py --session data/converted/jberry_sensor/self/2026-11-19_192922 --out out/selfmeasure_eval

scripts/baseline_exp_rec.py (부록 B 실험, 11-18 rec4 의 0.73 을 낸 스크립트) 와 같은 계산이다. 차이는 경로를 고정하지 않고
라벨을 세션의 labels (1 Hz) 에서 읽는다는 점뿐이다.

- 조건: F5 / F10 (처음 5·10분 고정 기준), A5 / A10 (+ 조건부 느린 갱신, src/driver_state/adaptive_baseline.py)
- 학생: S1-EP (ECG+PPG 파형 + ECG RR 특징) 시드 평균 logit, 상태 머신 v2 는 ECG 경로만
- 규칙: dev_core = 같은 기준의 z (심박·RMSSD·SDNN, 2분·5분 스케일) 의 RMS (자르지 않음, PersonalNormalizer 와 같음). 학습 없음
- 기준: 세션 t_rel 0 ~ L (자동 선택·품질 검사 없음), 파형 기준 = 그 안의 30초 창 전부 (10초 간격)
- 평가: --eval-from (기본 600초, 모든 조건 공통) 이후 창, 창 중심 시각의 라벨이 0/1 인 창
  * 창 AUROC: ECG 유효 창. 블록 AUROC: 원 녹음 시각 5분 블록 평균 (라벨이 섞인 블록은 다수결)

결과: <out>/results.csv (세션 × 조건), <out>/<session>__<cond>.csv (창별 t, label, valid, logit, rule, level)
infer_bundle.py 와 숫자가 다르다: 그쪽은 기준을 자동 선택하고 (60초 이후 깨끗한 창), 파형 기준을 30개로 제한하고, PPG 대체를 쓴다.
"""
from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from safetensors.torch import load_file
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]
warnings.filterwarnings("ignore")

from driver_state.adaptive_baseline import AdaptCfg, simulate  # noqa: E402
from driver_state.data_io import load_session  # noqa: E402
from driver_state.student.distill import Student  # noqa: E402
from driver_state.teacher.features import build_tables  # noqa: E402
from driver_state.teacher.personal import CORE  # noqa: E402
from driver_state.teacher.rr import extract_rr  # noqa: E402
from driver_state.teacher.spec import TeacherSpec  # noqa: E402
from student_cache import cut, prep  # noqa: E402
from window_quality import abn_ratio  # noqa: E402

CONDS = {"F5": (300.0, False), "F10": (600.0, False), "A5": (300.0, True), "A10": (600.0, True)}


def auc(y, s):
    y, s = np.asarray(y), np.asarray(s, float)
    ok = np.isfinite(s)
    return float(roc_auc_score(y[ok], s[ok])) if len(set(y[ok])) == 2 else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--session", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--models", default=str(ROOT / "models"))
    ap.add_argument("--conds", nargs="+", default=list(CONDS), choices=list(CONDS))
    ap.add_argument("--eval-from", type=float, default=600.0, help="이 세션 시각[초] 이후 창만 평가 (조건 공통)")
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    mdir = Path(a.models); B = json.loads((mdir / "bundle.json").read_text(encoding="utf-8"))
    spec = TeacherSpec.from_dict(B["spec"]); names = B["rr_feature_names"]
    sig_pop = np.array([B["normalizer"]["sigma_pop"][n] for n in names])
    core = [i for i, n in enumerate(names) if n.rsplit("_", 1)[0] in CORE and n.rsplit("_", 1)[1] != "30"]
    ms, params = [], []
    for p in sorted((mdir / "student").glob("S1-EP_s*.safetensors")):
        sd = load_file(str(p)); m = Student(2, "dist", sd["W"].numpy(), 0.0, 0.0, 36); m.load_state_dict(sd); m.eval(); ms.append(m)
        params.append({"W": sd["W"].numpy().astype(float), "alpha": float(sd["alpha"]), "b": float(sd["b"])})
    hr_idx = names.index("hr_120")
    rows = []
    for sdir in a.session:
        rec = load_session(sdir, channels=("ecg", "ppg")); rec.prefiltered = False
        meta = json.loads((Path(sdir) / "meta.json").read_text(encoding="utf-8"))
        off = float(meta.get("raw_offset_sec", 0.0))
        if rec.labels is None:
            raise ValueError(f"{sdir}: labels 가 없습니다")
        lfs = rec.label_fs or 1.0
        wav = {c: prep(rec.signals[c], rec.fs[c], c) for c in ("ecg", "ppg")}
        for cond in a.conds:
            L, adapt = CONDS[cond]
            r = extract_rr(rec, spec, "ecg"); r.meta["baseline_range"] = [0.0, L]
            win, prof = build_tables([r], spec)
            pr = prof.iloc[0]
            m0 = np.array([pr[f"m_{n}"] for n in names], float); s0 = np.array([pr[f"s_{n}"] for n in names], float)
            win = win.sort_values("t_rel").reset_index(drop=True)
            X = win[names].to_numpy(float); t = win["t_rel"].to_numpy(float)
            q = abn_ratio(r, t)
            W = np.zeros((len(win), 2, 1920), np.float32)
            rs = np.arange(0.0, L - 30 + 1e-6, 10.0)
            R = np.zeros((len(rs), 2, 1920), np.float32)
            for c, ch in enumerate(("ecg", "ppg")):
                W[:, c] = cut(wav[ch], t)[0]; R[:, c] = cut(wav[ch], rs)[0]
            with torch.no_grad():
                G = [m.g(torch.as_tensor(W)).numpy() for m in ms]
                g0 = [m.g(torch.as_tensor(R)).median(0).values.numpy() for m in ms]
            lg, lv = simulate(t + 30, X, win["valid"].to_numpy(bool), q, m0, s0, sig_pop, G, g0, params, hr_idx, L,
                              AdaptCfg(adapt=adapt))
            zr = (X - m0) / np.fmax(np.nan_to_num(s0, nan=0.0), sig_pop)   # dev_core 와 같게: 자르지 않고 nanmean
            with np.errstate(invalid="ignore"):
                rule = np.sqrt(np.nanmean(zr[:, core] ** 2, axis=1))
            li = np.clip(((t + 15) * lfs).astype(int), 0, len(rec.labels) - 1)
            d = pd.DataFrame({"t": t, "t_raw": t + off, "label": rec.labels[li], "valid": win["valid"].to_numpy(bool),
                              "q_ecg": q, "logit": lg, "rule": rule, "level": lv})
            d.to_csv(out / f"{rec.session_id}__{cond}.csv", index=False)
            e = d[(d.t >= a.eval_from) & d.label.isin([0, 1])]
            ev = e[e.valid]
            ev = ev.assign(blk=((ev.t_raw + 15) // 300).astype(int))
            bm = ev.groupby("blk").agg(y=("label", lambda s: int(s.mean() >= 0.5)), s=("logit", "mean"), r=("rule", "mean"))
            res = {"session": rec.session_id, "cond": cond, "eval_windows": len(ev),
                   "win_auc_student": auc(ev.label, ev.logit), "win_auc_rule": auc(ev.label, ev.rule),
                   "blk_auc_student": auc(bm.y, bm.s), "blk_auc_rule": auc(bm.y, bm.r), "n_blocks": len(bm),
                   "warn_normal": float((e[e.label == 0].level >= 1).mean()),
                   "warn_drowsy": float((e[e.label == 1].level >= 1).mean())}
            rows.append(res); print(res, flush=True)
    pd.DataFrame(rows).to_csv(out / "results.csv", index=False)


if __name__ == "__main__":
    main()
