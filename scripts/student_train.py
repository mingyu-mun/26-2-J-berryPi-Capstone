"""학생 증류 학습·평가 (docs/student-f2c-stage1.md). GPU 서버에서 실행.

    python scripts/student_train.py --label-dir out/f2c_label --wave data/student_wave --cache data/teacher_cache \
        --prv data/cache_prv --s0-oof out/anomaly/oof_lodo.parquet --f2c-oof out/anomaly_c/oof_lodo.parquet --out out/student
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from driver_state.student.distill import CONDS, Data, n_folds, n_params, train_eval  # noqa: E402
from driver_state.teacher.anomaly import z_matrix  # noqa: E402
from driver_state.teacher.evaluate import usable  # noqa: E402
from driver_state.teacher.metrics_time import bootstrap_m1, m1_dataset, weighted_median  # noqa: E402
from driver_state.teacher.pairs import TEACHER_DATASETS  # noqa: E402
from driver_state.teacher.personal import PersonalNormalizer  # noqa: E402
from driver_state.teacher.spec import TeacherSpec  # noqa: E402


def prv_z(lab, cache, prv, spec, names):
    wt = pd.read_parquet(Path(cache) / "windows.parquet")
    norm = PersonalNormalizer(spec).fit(wt[usable(wt)])
    pw, pp = pd.read_parquet(Path(prv) / "windows.parquet"), pd.read_parquet(Path(prv) / "profiles.parquet")
    pw = pw[pw["valid"]].reset_index(drop=True)
    Z, _ = z_matrix(norm, pw, pp, names)
    k = ["dataset_id", "group", "session_id", "t_rel"]
    df = pd.concat([pw[k], pd.DataFrame(Z, columns=names)], axis=1)
    m = lab[k].merge(df, on=k, how="left")
    has = m[names[0]].notna().to_numpy()
    return np.nan_to_num(m[names].to_numpy(np.float32), nan=0.0), has


def stress_rate(t_oof, wes, col, cond):
    rates = []
    for f in TEACHER_DATASETS:
        td = t_oof[(t_oof["fold"] == f) & (t_oof["y"] == 1) & np.isfinite(t_oof[col])]
        if not len(td):
            continue
        wts = 1.0 / td.groupby("group")["group"].transform("size").to_numpy(float)
        th = weighted_median(td[col].to_numpy(float), wts)
        w = wes[(wes["fold"] == f) & (wes["cond"] == cond) & np.isfinite(wes[col])]
        if len(w):
            rates.append(float(np.mean(w[col].to_numpy(float) >= th)))
    return float(np.mean(rates)) if rates else float("nan")


def main():
    ap = argparse.ArgumentParser()
    for k in ("label-dir", "wave", "cache", "prv", "s0-oof", "f2c-oof", "out"):
        ap.add_argument(f"--{k}", required=True)
    ap.add_argument("--conds", nargs="+", default=list(CONDS))
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--n-boot", type=int, default=2000)
    a = ap.parse_args()
    import torch
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    spec = TeacherSpec()
    names = list(spec.baseline_feature_names)
    ld = Path(a.label_dir)
    lab = pd.read_parquet(ld / "f2c_labels.parquet")
    params = dict(np.load(ld / "f2c_params.npz"))
    W = Path(a.wave)
    index = pd.read_parquet(W / "index.parquet")
    assert (index["row_id"].to_numpy() == lab["row_id"].to_numpy()).all()
    t0 = time.time()
    wave = np.load(W / "wave.npy")                       # 메모리에 전부 (약 650 MB)
    ref_wave = np.load(W / "ref_wave.npy")
    ref_index = pd.read_parquet(W / "ref_index.parquet")
    pz, has_prv = prv_z(lab, a.cache, a.prv, spec, names)
    print(f"적재 {time.time() - t0:.0f}s, PRV 유효 (학생 데이터) {has_prv[lab['dataset_id'].isin(['clas', 'maus', 'wesad'])].mean():.3f}", flush=True)
    D = Data(lab, index, wave, ref_index, ref_wave, pz, names, params)
    hp = {"steps": a.steps} if a.steps else None

    res_p = out / "student_scores.parquet"
    if res_p.exists():
        res = pd.read_parquet(res_p)
    else:
        res = lab[["row_id", "dataset_id", "group", "session_id", "t_rel", "y", "cond"]].copy()
        res["gfold"] = D.gfold
    for cond in a.conds:
        if f"s_{cond}" in res.columns:
            print(f"{cond}: 있음, 건너뜀", flush=True)
            continue
        acc: dict[str, list] = {}
        tgt = np.full(len(lab), np.nan, np.float32)
        for fold in range(n_folds()):
            per_seed: dict[str, list] = {}
            for seed in a.seeds:
                o, _ = train_eval(D, cond, fold, seed, dev, hp=hp)
                for k, v in o.items():
                    if k != "target":
                        per_seed.setdefault(k, []).append(v)
                okt = np.isfinite(o["target"]); tgt[okt] = o["target"][okt]
            for k, vs in per_seed.items():
                m = np.nanmean(np.stack(vs), 0)
                acc.setdefault(k, []).append(m)
        for k, ms in acc.items():
            S = np.stack(ms)
            res[k] = np.where(np.isfinite(S).any(0), np.nanmax(np.where(np.isfinite(S), S, -np.inf), 0), np.nan)
        res["target"] = np.where(np.isfinite(tgt), tgt, res["target"] if "target" in res else np.nan)
        res.to_parquet(res_p)

    # ------------------------------------------------------------- 평가
    s0 = pd.read_parquet(a.s0_oof)[["row_id", "dataset_id", "tbin", "s_F2"]].rename(columns={"s_F2": "S0"})
    f2c = pd.read_parquet(a.f2c_oof)[["row_id", "dataset_id", "s_F2"]].rename(columns={"s_F2": "F2C"})
    t = res[res["dataset_id"].isin(TEACHER_DATASETS) & (res["y"] >= 0)]
    t = t.merge(s0, on=["row_id", "dataset_id"]).merge(f2c, on=["row_id", "dataset_id"])
    t["fold"] = t["dataset_id"]
    ecg_cols = [c for c in ("S0", "F2C", "s_S1-E", "s_S1-EP", "s_S2-E") if c in t.columns]
    t = t[np.isfinite(t[ecg_cols].to_numpy(float)).all(1)].reset_index(drop=True)
    rep = {"n_teacher_rows": int(len(t)), "M1": {}, "M1_ds": {}, "diff": {}, "S1": {}, "S2": {}, "agree": {}, "params": {}}
    for c in ecg_cols:
        per = {ds: m1_dataset(g, c)[0] for ds, g in t.groupby("dataset_id")}
        rep["M1_ds"][c] = per
        rep["M1"][c] = float(np.mean(list(per.values())))
    B = bootstrap_m1(t, ecg_cols, n_boot=a.n_boot)
    for c in ecg_cols:
        for ref in ("S0", "F2C"):
            if c != ref:
                d = B[:, ecg_cols.index(c)] - B[:, ecg_cols.index(ref)]
                rep["diff"][f"{c}-{ref}"] = {"diff": rep["M1"][c] - rep["M1"][ref],
                                             "lo": float(np.nanpercentile(d, 2.5)), "hi": float(np.nanpercentile(d, 97.5))}
    wes = res[(res["dataset_id"] == "wesad") & res["cond"].isin(["stress", "other"])].copy()
    wes["fold"] = [TEACHER_DATASETS[g] for g in wes["gfold"]]
    for c in [x for x in res.columns if x.startswith("s_S")]:
        base = c if c in ecg_cols else ("s_S1-EP" if c.startswith("s_S1-EP") else None)
        if base is None:
            continue
        tt = t.rename(columns={base: "_th"})
        ww = wes.rename(columns={c: "_th"})
        rep["S1"][c] = stress_rate(tt, ww, "_th", "stress")
        rep["S2"][c] = stress_rate(tt, ww, "_th", "other")
    st = res[res["dataset_id"].isin(["clas", "maus", "wesad"]) & np.isfinite(res["target"])]
    for c in [x for x in res.columns if x.startswith("s_S")]:
        r = {}
        for ds, g in st.groupby("dataset_id"):
            g = g[np.isfinite(g[c])]
            if len(g) < 50:
                continue
            hi = (g["target"] >= np.quantile(g["target"], 0.8)).astype(int)
            r[ds] = {"rho": float(spearmanr(g[c], g["target"]).statistic), "auroc_top20": float(roc_auc_score(hi, g[c])), "n": int(len(g))}
        if r:
            r["rho_mean"] = float(np.mean([v["rho"] for v in r.values()]))
        rep["agree"][c] = r
    for cond in CONDS:
        rep["params"][cond] = n_params(cond)
    s1e = rep["M1"].get("s_S1-EP", np.nan)
    rep["verdict"] = {
        "S1-EP M1 ≥ F2C−0.03": bool(s1e >= rep["M1"]["F2C"] - 0.03),
        "S1-EP − S0 하한 > 0": bool(rep["diff"].get("s_S1-EP-S0", {}).get("lo", -1) > 0),
        "S1-EP S1 ≤ 0.50": bool(rep["S1"].get("s_S1-EP", 1) <= 0.50),
        "PPG ρ ≥ 0.6 (S1-EP_ppg 또는 S1-P)": bool(max(rep["agree"].get("s_S1-EP_ppg", {}).get("rho_mean", -1),
                                                    rep["agree"].get("s_S1-P", {}).get("rho_mean", -1)) >= 0.6),
        "WESAD PPG S1 ≤ 0.55": bool(min(rep["S1"].get("s_S1-EP_ppg", 1), rep["S1"].get("s_S1-P", 1)) <= 0.55),
    }
    (out / "report_student.json").write_text(json.dumps(rep, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
    L = ["# 학생 증류 1단계 결과", "", f"교사 4종 평가 창 {len(t)}", "",
         "| 조건 | M1 | " + " | ".join(TEACHER_DATASETS) + " |", "|" + " --- |" * (2 + len(TEACHER_DATASETS))]
    for c in ecg_cols:
        L.append(f"| {c} | {rep['M1'][c]:.3f} | " + " | ".join(f"{rep['M1_ds'][c].get(d, float('nan')):.3f}" for d in TEACHER_DATASETS) + " |")
    L += ["", "| 차이 | 값 | 95% 구간 |", "| --- | --- | --- |"]
    L += [f"| {k} | {v['diff']:+.3f} | [{v['lo']:+.3f}, {v['hi']:+.3f}] |" for k, v in rep["diff"].items()]
    L += ["", "| 학생 | S1 | S2 | 교사 일치 ρ (CLAS / MAUS / WESAD) | 상위20% AUROC 평균 |", "| --- | --- | --- | --- | --- |"]
    for c in [x for x in res.columns if x.startswith("s_S")]:
        ag = rep["agree"].get(c, {})
        rho = " / ".join(f"{ag[d]['rho']:.2f}" if d in ag else "-" for d in ("clas", "maus", "wesad"))
        au = np.mean([ag[d]["auroc_top20"] for d in ("clas", "maus", "wesad") if d in ag]) if ag else float("nan")
        L.append(f"| {c} | {rep['S1'].get(c, float('nan')):.3f} | {rep['S2'].get(c, float('nan')):.3f} | {rho} | {au:.3f} |")
    L += ["", "파라미터: " + ", ".join(f"{k} {v:,}" for k, v in rep["params"].items()), "", "## 판정", ""]
    L += [f"- {k}: {v}" for k, v in rep["verdict"].items()]
    (out / "report_student.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L), flush=True)


if __name__ == "__main__":
    main()
