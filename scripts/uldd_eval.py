"""UL-DD 로 학생 PPG 경로 평가 (docs/dataset-ul-dd.md 평가 규칙). CPU 로 돈다.

    python scripts/uldd_eval.py --data <규약 폴더 루트> --out out/uldd_eval
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
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
warnings.filterwarnings("ignore")

from driver_state.data_io import load_session  # noqa: E402
from driver_state.student.distill import Student  # noqa: E402
from driver_state.teacher.anomaly import z_matrix  # noqa: E402
from driver_state.teacher.features import build_tables  # noqa: E402
from driver_state.teacher.metrics_time import bootstrap_m1, m1_dataset, m2_dataset, m4_dataset  # noqa: E402
from driver_state.teacher.personal import PersonalNormalizer  # noqa: E402
from driver_state.teacher.rr import extract_rr  # noqa: E402
from driver_state.teacher.spec import TeacherSpec  # noqa: E402
from student_cache import cut, prep  # noqa: E402
from window_quality import abn_ratio  # noqa: E402

SHORT = ("rmssd", "sd1", "pnn50", "sd1_sd2", "hf_log", "lf_hf_log", "lf_nu", "sampen", "dfa_a1")
HRK = ("hr", "rr_mean")
BASE = (60.0, 360.0)
KSS_SEC = 240.0


def stem(n):
    return n.rsplit("_", 1)[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--models", default=str(ROOT / "models")); ap.add_argument("--n-boot", type=int, default=2000)
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    mdir = Path(a.models)
    B = json.loads((mdir / "bundle.json").read_text(encoding="utf-8"))
    spec = TeacherSpec.from_dict(B["spec"]); names = B["rr_feature_names"]
    norm = PersonalNormalizer.from_dict(spec, B["normalizer"])
    ms = []
    for p in sorted((mdir / "student").glob("S1-EP_s*.safetensors")):
        sd = load_file(str(p)); m = Student(2, "dist", sd["W"].numpy(), 0.0, 0.0, 36); m.load_state_dict(sd); m.eval(); ms.append(m)
    keep = {"P-full": np.ones(36), "P-noshort": np.array([0.0 if stem(n) in SHORT else 1.0 for n in names]),
            "P-hr": np.array([1.0 if stem(n) in HRK else 0.0 for n in names]), "P-enc": np.zeros(36)}
    rows = []
    root = Path(a.data) / "uldd"
    for pdir in sorted(p for p in root.iterdir() if p.is_dir()):
        recs = [load_session(pdir / s, channels=("ppg",)) for s in ("alert", "drowsy") if (pdir / s).exists()]
        if not recs or recs[0].session_id != "alert":
            continue
        rrs = [extract_rr(r, spec, "ppg") for r in recs]
        rrs[0].meta["baseline_range"] = list(BASE)
        win, prof = build_tables(rrs, spec)
        if not len(win) or prof["baseline_valid"].max() < 0.5:
            print(f"{pdir.name}: 기준 무효, 건너뜀", flush=True); continue
        zr, dev = z_matrix(norm, win, prof, names)
        by = {r.session_id: (r, rc) for r, rc in zip(rrs, recs)}
        q = np.ones(len(win))
        X = np.zeros((len(win), 2, 1920), np.float32)
        for sid, g in win.groupby("session_id"):
            r, rc = by[sid]
            q[g.index.to_numpy()] = abn_ratio(r, g["t_rel"].to_numpy(float))
            X[g.index.to_numpy(), 1] = cut(prep(rc.signals["ppg"], rc.fs["ppg"], "ppg"), g["t_rel"].to_numpy(float))[0]
        rs = np.arange(BASE[0], BASE[1] - 30 + 1e-6, 10.0)
        R = np.zeros((len(rs), 2, 1920), np.float32)
        R[:, 1] = cut(prep(recs[0].signals["ppg"], recs[0].fs["ppg"], "ppg"), rs)[0]
        res = win[["session_id", "t_rel", "valid", "y"]].copy()
        res["group"] = pdir.name; res["dataset_id"] = "uldd"; res["q"] = q; res["R-dev"] = dev
        res["tbin"] = (win["t_rel"].to_numpy(float) // KSS_SEC).astype(int)
        with torch.no_grad():
            x, r = torch.as_tensor(X), torch.as_tensor(R)
            G = [(m.g(x), m.g(r).median(0).values) for m in ms]
            for cond, k in keep.items():
                zk = torch.as_tensor((zr * k[None, :]).astype(np.float32))
                res[cond] = np.mean([m.logit(zk, gx - gr).numpy() for m, (gx, gr) in zip(ms, G)], 0)
        rows.append(res)
        print(f"{pdir.name}: 창 {len(res)}, q≤0.3 {int((res.q <= 0.3).sum())}, 유효 {int(res.valid.sum())}", flush=True)
    D = pd.concat(rows, ignore_index=True)
    D.to_parquet(out / "uldd_scores.parquet")
    conds = list(keep) + ["R-dev"]
    rep = {}
    for tag, m in (("q30", (D["q"] <= 0.3)), ("valid", D["valid"].astype(bool))):
        t = D[m & (D["y"] >= 0) & np.isfinite(D[conds]).all(1)].reset_index(drop=True)
        r_ = {"n_windows": int(len(t)), "n_participants": int(t["group"].nunique()),
              "drowsy_frac": float(t["y"].mean())}
        Bt = bootstrap_m1(t, conds, n_boot=a.n_boot)
        for c in conds:
            m1 = m1_dataset(t, c)[0]
            r_[c] = {"M1": m1, "M1_lo": float(np.nanpercentile(Bt[:, conds.index(c)], 2.5)),
                     "M1_hi": float(np.nanpercentile(Bt[:, conds.index(c)], 97.5)),
                     "M2": float(m2_dataset(t, c)), "M4": float(m4_dataset(t, c))}
        d = Bt[:, conds.index("P-noshort")] - Bt[:, conds.index("P-full")]
        r_["P-noshort−P-full"] = {"diff": r_["P-noshort"]["M1"] - r_["P-full"]["M1"],
                                  "lo": float(np.nanpercentile(d, 2.5)), "hi": float(np.nanpercentile(d, 97.5))}
        rep[tag] = r_
    # 참가자별 세션 평균 (D − A)
    t = D[D["q"] <= 0.3]
    sess = t.groupby(["group", "session_id"])[conds].mean().unstack("session_id")
    pair = {}
    for c in conds:
        if ("drowsy" in sess[c]) and ("alert" in sess[c]):
            dd = (sess[c]["drowsy"] - sess[c]["alert"]).dropna()
            pair[c] = {"n": int(len(dd)), "drowsy_higher": int((dd > 0).sum())}
    rep["session_pairs"] = pair
    (out / "report_uldd.json").write_text(json.dumps(rep, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
    print(json.dumps(rep, ensure_ascii=False, indent=1, default=float))


if __name__ == "__main__":
    main()
