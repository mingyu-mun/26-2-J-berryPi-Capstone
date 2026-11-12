"""실측 세션 (규약 폴더) 하나에 교사 F2C 와 최종 학생을 돌린다. GPU 서버에서 실행.

    python scripts/selfmeasure_infer.py --session <규약 세션 폴더> --out out/selfmeasure \
        --cache data/teacher_cache --student data/cache_student --nw-dir out/normwear \
        --normwear-dir ~/jberry --ckpt ~/jberry/normwear_ckpt/model.safetensors \
        --f2c-params out/f2c_label/f2c_params.npz --f2c-oof out/anomaly_c/oof_lodo.parquet \
        --student-dir out/student_final

결과: <out>/windows_scores.csv (창별 HR, 교사·학생 logit·확률·경보), <out>/summary.json
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

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
warnings.filterwarnings("ignore")

from driver_state.data_io import load_session  # noqa: E402
from driver_state.student.distill import CONDS, CH, Student  # noqa: E402
from driver_state.teacher.anomaly import z_matrix  # noqa: E402
from driver_state.teacher.evaluate import usable  # noqa: E402
from driver_state.teacher.features import baseline_plan, build_tables  # noqa: E402
from driver_state.teacher.metrics_time import weighted_median  # noqa: E402
from driver_state.teacher.nw_features import NW_COLS, attach, fit_pca, z_of  # noqa: E402
from driver_state.teacher.pairs import TEACHER_DATASETS  # noqa: E402
from driver_state.teacher.personal import PersonalNormalizer  # noqa: E402
from driver_state.teacher.rr import extract_rr  # noqa: E402
from driver_state.teacher.spec import TeacherSpec  # noqa: E402
from normwear_embed import Embedder, prep_ecg, segments  # noqa: E402
from student_cache import cut, prep  # noqa: E402


def teacher_logits(params, Z):
    out, i = [], 0
    while f"final/s{i}/W" in params:
        W, a, b = params[f"final/s{i}/W"], float(params[f"final/s{i}/alpha"]), float(params[f"final/s{i}/b"])
        s = np.sqrt(((Z @ W.T) ** 2).sum(1) + 1e-8) - 1e-4
        out.append(np.log1p(np.exp(a)) * s + b)
        i += 1
    return np.mean(out, 0)


def main():
    ap = argparse.ArgumentParser()
    for k in ("session", "out", "cache", "student", "nw-dir", "normwear-dir", "ckpt", "f2c-params", "f2c-oof", "student-dir"):
        ap.add_argument(f"--{k}", required=True)
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    spec = TeacherSpec()
    names = list(spec.baseline_feature_names)
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    # 1. RR 특징·기준 구간 (교사 파이프라인 그대로)
    rec = load_session(a.session, channels=("ecg", "ppg"))
    rr = extract_rr(rec, spec, "ecg")
    win, prof = build_tables([rr], spec)
    plan = baseline_plan([rr], spec)
    info = next(v for v in plan.values() if v is not None)
    key = f"{rr.dataset_id}/{rr.group}#{info.session_id}"
    print(f"창 {len(win)} (유효 {int(win['valid'].sum())}), 기준 {info.start:.0f}~{info.end:.0f}s, 기준 유효 {prof['baseline_valid'].tolist()}", flush=True)

    # 2. 교사 정규화 (교사 4종 유효 창) + NormWear PCA (CLAS·MAUS)
    wt, pt = pd.read_parquet(Path(a.cache) / "windows.parquet"), pd.read_parquet(Path(a.cache) / "profiles.parquet")
    ws = pd.read_parquet(Path(a.student) / "windows.parquet")
    okp = set(pt.loc[pt["baseline_valid"] > 0.5, "profile_key"])
    fit_rows = usable(wt) & wt["profile_key"].isin(okp).to_numpy()
    norm = PersonalNormalizer(spec).fit(wt[fit_rows])
    zr, dev_core = z_matrix(norm, win, prof, names)
    comps, _ = fit_pca(ws, Path(a.nw_dir))
    wt_nw, _ = attach(wt[fit_rows].reset_index(drop=True), pt, a.nw_dir, comps)

    # 3. NormWear 임베딩 (창 + 기준 창)
    emb = Embedder(a.normwear_dir, a.ckpt, dev)
    sig = prep_ecg(rec.signals["ecg"], rec.fs["ecg"])
    E = emb(segments(sig, win["t_rel"].to_numpy(float))).astype(np.float32) @ comps.T
    starts = np.arange(info.start, info.end - 30 + 1e-6, spec.stride_sec)
    Er = emb(segments(sig, starts)).astype(np.float32) @ comps.T
    for i, c in enumerate(NW_COLS):
        win[c] = E[:, i]
        prof[f"m_{c}"] = np.median(Er[:, i])
    zn = z_of(wt_nw, win, prof)
    Z = np.concatenate([zr, zn], 1)
    params = dict(np.load(a.f2c_params))
    np.savez(out / "inputs.npz", Z=Z, t_rel=win["t_rel"].to_numpy(float), valid=win["valid"].to_numpy(bool),
             names=np.array(names + NW_COLS), raw=win[names].to_numpy(float), ref=prof[[f"m_{n}" for n in names]].to_numpy(float))
    t_logit = teacher_logits(params, Z)

    # 교사 동작점 (LODO oof, 데이터셋별 졸음 창 가중 중앙값 평균 = 민감도 50%)
    oof = pd.read_parquet(a.f2c_oof)
    th_t = []
    for _, g in oof[(oof["y"] == 1) & np.isfinite(oof["s_F2"])].groupby("dataset_id"):
        wts = 1.0 / g.groupby("group")["group"].transform("size").to_numpy(float)
        th_t.append(weighted_median(g["s_F2"].to_numpy(float), wts))
    th_t = float(np.mean(th_t))

    # 4. 학생 (64 Hz 파형)
    sd = Path(a.student_dir)
    th_s = json.loads((sd / "thresholds.json").read_text()) if (sd / "thresholds.json").exists() else {}
    wav = {ch: prep(rec.signals[ch], rec.fs[ch], ch) for ch in ("ecg", "ppg")}
    rs = starts[:30]
    X = np.zeros((len(win), 2, 1920), np.float32); R = np.zeros((len(rs), 2, 1920), np.float32)
    for c, ch in enumerate(("ecg", "ppg")):
        X[:, c] = cut(wav[ch], win["t_rel"].to_numpy(float))[0]
        R[:, c] = cut(wav[ch], rs)[0]
    res = win[["t_rel", "valid"]].copy()
    res["hr_30"] = win["hr_30"].to_numpy() if "hr_30" in win else np.nan
    res["dev_core"] = dev_core
    res["teacher_logit"] = t_logit
    res["teacher_p"] = 1 / (1 + np.exp(-t_logit))
    res["teacher_alarm"] = t_logit >= th_t
    for cond, modes in (("S1-EP", {"": None, "_ecg": 0, "_ppg": 1}), ("S1-E", {"": None})):
        cfg = CONDS[cond]
        chs = [CH[c] for c in cfg["ch"]]
        seeds = sorted(sd.glob(f"{cond}_s*.pt"))
        if not seeds:
            continue
        for suf, only in modes.items():
            L = []
            for p in seeds:
                sdict = torch.load(p, map_location="cpu", weights_only=True)
                m = Student(len(chs), cfg["head"], sdict["W"].numpy(), 0.0, 0.0, 36).to(dev)
                m.load_state_dict(sdict); m.eval()
                with torch.no_grad():
                    x = torch.as_tensor(X[:, chs]); r = torch.as_tensor(R[:, chs])
                    if len(chs) == 2:
                        M = torch.tensor([0.0 if only == 1 else 1.0, 0.0 if only == 0 else 1.0])
                        x = x * M[None, :, None]; r = r * M[None, :, None]
                    gr = m.g(r.to(dev)).median(0).values
                    lg = m.logit(torch.as_tensor(zr).to(dev), m.g(x.to(dev)) - gr)
                L.append(lg.cpu().numpy())
            col = f"student_{cond}{suf}"
            res[f"{col}_logit"] = np.mean(L, 0)
            res[f"{col}_p"] = 1 / (1 + np.exp(-res[f"{col}_logit"]))
            if cond in th_s:
                res[f"{col}_alarm"] = res[f"{col}_logit"] >= th_s[cond]
    res.to_csv(out / "windows_scores.csv", index=False)
    v = res[res["valid"]]
    summ = {"n_windows": int(len(res)), "n_valid": int(len(v)), "baseline": [float(info.start), float(info.end)],
            "teacher_threshold": th_t, "student_thresholds": th_s,
            "alarm_rate": {c: float(v[c].mean()) for c in v.columns if c.endswith("_alarm")},
            "p_median": {c: float(v[c].median()) for c in v.columns if c.endswith("_p")}}
    (out / "summary.json").write_text(json.dumps(summ, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(summ, ensure_ascii=False, indent=1), flush=True)


if __name__ == "__main__":
    main()
