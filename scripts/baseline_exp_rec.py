"""부록 B 기준 구간 실험: 실측 녹음 (CPU). docs/state-machine-v2.md 부록 B.

    python scripts/baseline_exp_rec.py --out out/baseline_exp
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

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]
warnings.filterwarnings("ignore")

from driver_state.adaptive_baseline import AdaptCfg, simulate  # noqa: E402
from driver_state.data_io import load_session  # noqa: E402
from driver_state.student.distill import Student  # noqa: E402
from driver_state.teacher.features import build_tables  # noqa: E402
from driver_state.teacher.rr import extract_rr  # noqa: E402
from driver_state.teacher.spec import TeacherSpec  # noqa: E402
from student_cache import cut, prep  # noqa: E402
from window_quality import abn_ratio  # noqa: E402

SELF = Path(r"C:/Users/NLL434-m/Downloads/datasets/selfmeasure/converted/jberry_sensor/self01")  # 연구실 PC 경로. 다른 PC 는 --self-dir
RECS = {  # 이름: (세션, 원본 기준 오프셋 초, 5분 블록 라벨 (원본 시각 기준, 1 = 졸음) 또는 None)
    "rec1 (정상, ECG 잡음)": ("2026-11-12_175237_s1", 0.88, None),
    "rec2 (졸린 상태 측정)": ("2026-11-12_222323_s1", 2.37, None),
    "rec4 (11-18, 5분 라벨)": ("2026-11-18_190640_s1", 60.0, {0: 0, 1: 0, 2: 1, 3: 1, 4: 1, 5: 0, 6: 0}),
}
CONDS = {"F5": (300.0, False), "F10": (600.0, False), "A5": (300.0, True), "A10": (600.0, True)}
EVAL_FROM = 600.0   # 두 기준 모두 끝난 뒤의 창만 비교


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True); ap.add_argument("--models", default=str(ROOT / "models"))
    ap.add_argument("--self-dir", default=str(SELF), help="실측 변환본 폴더 (jberry_sensor/self01). rec4 는 scripts/convert_rec4_orig.py 로 만든다")
    a = ap.parse_args(); out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    mdir = Path(a.models); B = json.loads((mdir / "bundle.json").read_text(encoding="utf-8"))
    spec = TeacherSpec.from_dict(B["spec"]); names = B["rr_feature_names"]
    sig_pop = np.array([B["normalizer"]["sigma_pop"][n] for n in names])
    ms, params = [], []
    for p in sorted((mdir / "student").glob("S1-EP_s*.safetensors")):
        sd = load_file(str(p)); m = Student(2, "dist", sd["W"].numpy(), 0.0, 0.0, 36); m.load_state_dict(sd); m.eval(); ms.append(m)
        params.append({"W": sd["W"].numpy().astype(float), "alpha": float(sd["alpha"]), "b": float(sd["b"])})
    hr_idx = names.index("hr_120")
    rows, tl = [], {}
    for name, (ses, off, lab) in RECS.items():
        if not (Path(a.self_dir) / ses).exists():
            print("건너뜀 (변환본 없음):", name); continue
        rec = load_session(Path(a.self_dir) / ses, channels=("ecg", "ppg")); rec.prefiltered = False
        wav = {c: prep(rec.signals[c], rec.fs[c], c) for c in ("ecg", "ppg")}
        for cond, (L, adapt) in CONDS.items():
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
            d = pd.DataFrame({"t": t, "valid": win["valid"].to_numpy(bool), "logit": lg, "level": lv})
            d["blk"] = ((d.t + off + 15) // 300).astype(int)
            tl[(name, cond)] = d
            e = d[(d.t >= EVAL_FROM)]
            res = {"rec": name, "cond": cond, "windows": len(e), "warn_all": float((e.level >= 1).mean())}
            if lab:
                e = e.assign(y=e.blk.map(lab)); ev = e[e.valid & e.y.notna()]
                from sklearn.metrics import roc_auc_score
                bm = ev.groupby("blk").agg(y=("y", "first"), s=("logit", "mean"))
                res |= {"win_auc": roc_auc_score(ev.y, ev.logit), "blk_auc": roc_auc_score(bm.y, bm.s),
                        "warn_normal": float((e[e.y == 0].level >= 1).mean()), "warn_drowsy": float((e[e.y == 1].level >= 1).mean())}
            rows.append(res); print(res, flush=True)
    R_ = pd.DataFrame(rows); R_.to_csv(out / "rec_results.csv", index=False)
    pd.concat({f"{k[0]}|{k[1]}": v for k, v in tl.items()}).to_parquet(out / "rec_timelines.parquet")


if __name__ == "__main__":
    main()
