"""부록 B 기준 구간 실험: 공개 교사 4종 (GPU 서버). F5 (고정) vs A5 (조건부 느린 갱신).

    python scripts/baseline_exp_public.py --label-dir out/f2c_label --wave data/student_wave --cache data/teacher_cache \
        --oof out/anomaly/oof_lodo.parquet --out out/baseline_exp
최종 학생은 공개 데이터로 학습되었으므로 절대값은 낙관적이다. 조건 간 차이만 본다.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from safetensors.torch import load_file

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]

from driver_state.adaptive_baseline import AdaptCfg, simulate  # noqa: E402
from driver_state.student.distill import Student  # noqa: E402
from driver_state.teacher.metrics_time import bootstrap_m1, m1_dataset  # noqa: E402
from sm_v2_eval import entries, episodes  # noqa: E402

TEACHER = ("advitam", "dddb", "drozy", "mpddf")


def main():
    ap = argparse.ArgumentParser()
    for k in ("label-dir", "wave", "cache", "oof", "out"):
        ap.add_argument(f"--{k}", required=True)
    ap.add_argument("--models", default=str(ROOT / "models"))
    a = ap.parse_args(); out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    mdir = Path(a.models); B = json.loads((mdir / "bundle.json").read_text(encoding="utf-8"))
    names = B["rr_feature_names"]; sig_pop = np.array([B["normalizer"]["sigma_pop"][n] for n in names])
    ms, params = [], []
    for p in sorted((mdir / "student").glob("S1-EP_s*.safetensors")):
        sd = load_file(str(p)); m = Student(2, "dist", sd["W"].numpy(), 0.0, 0.0, 36); m.load_state_dict(sd); m.eval(); ms.append(m.to(dev))
        params.append({"W": sd["W"].numpy().astype(float), "alpha": float(sd["alpha"]), "b": float(sd["b"])})
    lab = pd.read_parquet(Path(a.label_dir) / "f2c_labels.parquet")
    q = pd.read_parquet(Path(a.label_dir) / "quality.parquet")[["row_id", "dataset_id", "q"]]
    W = Path(a.wave); wave = np.load(W / "wave.npy", mmap_mode="r"); ref_wave = np.load(W / "ref_wave.npy")
    ref_index = pd.read_parquet(W / "ref_index.parquet"); index = pd.read_parquet(W / "index.parquet")
    keep = lab["dataset_id"].isin(TEACHER).to_numpy()
    rows = np.flatnonzero(keep)
    wt = pd.read_parquet(Path(a.cache) / "windows.parquet"); pt = pd.read_parquet(Path(a.cache) / "profiles.parquet")
    L = lab.iloc[rows][["row_id", "dataset_id", "group", "session_id", "t_rel", "y"]].copy()
    L["wrow"] = rows
    L["profile_key"] = index["profile_key"].to_numpy()[rows]
    L = L.merge(wt[["row_id", "dataset_id", "valid"] + names], on=["row_id", "dataset_id"], how="left")
    L = L.merge(q, on=["row_id", "dataset_id"], how="left")
    # 인코더 출력
    G = [np.zeros((len(L), 32), np.float32) for _ in ms]
    with torch.no_grad():
        for i in range(0, len(L), 2048):
            x = torch.as_tensor(np.asarray(wave[L["wrow"].to_numpy()[i:i + 2048]], np.float32)).to(dev)
            for k, m in enumerate(ms):
                G[k][i:i + 2048] = m.g(x).cpu().numpy()
        gref = {}
        for pk, g in ref_index.groupby("profile_key"):
            r = torch.as_tensor(ref_wave[g.index.to_numpy()].astype(np.float32)).to(dev)
            gref[pk] = [m.g(r).median(0).values.cpu().numpy() for m in ms]
    P = pt.set_index("profile_key")
    hr_idx = names.index("hr_120")
    res = {c: np.full(len(L), np.nan) for c in ("F5", "A5")}
    lev = {c: np.zeros(len(L), int) for c in ("F5", "A5")}
    tls = []
    for (ds, grp, ses), g in L.groupby(["dataset_id", "group", "session_id"]):
        g = g.sort_values("t_rel")
        pk = g["profile_key"].iloc[0]
        if pk not in gref or pk not in P.index:
            continue
        m0 = P.loc[pk, [f"m_{n}" for n in names]].to_numpy(float); s0 = P.loc[pk, [f"s_{n}" for n in names]].to_numpy(float)
        grid = np.arange(g["t_rel"].min(), g["t_rel"].max() + 1e-6, 10.0)
        pos = np.searchsorted(grid, g["t_rel"].to_numpy(float) - 1e-6)
        n = len(grid)
        X = np.full((n, len(names)), np.nan); valid = np.zeros(n, bool); qq = np.ones(n)
        X[pos] = g[names].to_numpy(float); valid[pos] = g["valid"].fillna(False).to_numpy(bool); qq[pos] = g["q"].fillna(1.0).to_numpy(float)
        Gs = []
        for k in range(len(ms)):
            a_ = np.zeros((n, 32), np.float32); a_[pos] = G[k][g.index.to_numpy()]; Gs.append(a_)
        base_end = 300.0 if pk.split("#")[-1] == ses else None
        yg = np.full(n, -1); yg[pos] = g["y"].to_numpy(int)
        for c, adapt in (("F5", False), ("A5", True)):
            lg, lv = simulate(grid + 30, X, valid, qq, m0, s0, sig_pop, Gs, gref[pk], params, hr_idx, base_end, AdaptCfg(adapt=adapt))
            res[c][g.index.to_numpy()] = lg[pos]; lev[c][g.index.to_numpy()] = lv[pos]
            ent = entries(lv); eps = episodes(yg)
            hit = [np.flatnonzero(lv[a0:b0] >= 1) for a0, b0 in eps]
            tls.append({"ds": ds, "cond": c, "fa": int(np.sum(yg[ent] == 0)) if len(ent) else 0, "alert_h": float((yg == 0).sum() * 10 / 3600),
                        "n_ep": len(eps), "miss": int(sum(len(h) == 0 for h in hit)),
                        "warn_alert": float((lv[yg == 0] >= 1).mean()) if (yg == 0).any() else np.nan})
    for c in res:
        L[c] = res[c]
    tb = pd.read_parquet(a.oof)[["row_id", "dataset_id", "tbin"]]
    t = L[(L["y"] >= 0) & L["valid"].fillna(False) & np.isfinite(L["F5"]) & np.isfinite(L["A5"])].merge(tb, on=["row_id", "dataset_id"])
    M = {c: {ds: m1_dataset(gg, c)[0] for ds, gg in t.groupby("dataset_id")} for c in ("F5", "A5")}
    Bt = bootstrap_m1(t, ["F5", "A5"]); d = Bt[:, 1] - Bt[:, 0]
    T = pd.DataFrame(tls)
    rep = {"n": int(len(t)), "M1": {c: float(np.mean(list(v.values()))) for c, v in M.items()}, "M1_ds": M,
           "A5-F5": {"diff": float(np.mean(list(M["A5"].values())) - np.mean(list(M["F5"].values()))),
                     "lo": float(np.percentile(d, 2.5)), "hi": float(np.percentile(d, 97.5))}}
    for c, g in T.groupby("cond"):
        rep[c] = {"fa_per_h": float(g.fa.sum() / g.alert_h.sum()), "miss_rate": float(g.miss.sum() / max(g.n_ep.sum(), 1)),
                  "warn_alert": float(np.nanmean(g.warn_alert)),
                  "by_ds": {ds: {"fa_per_h": float(x.fa.sum() / max(x.alert_h.sum(), 1e-9)), "miss": float(x.miss.sum() / max(x.n_ep.sum(), 1))}
                            for ds, x in g.groupby("ds")}}
    (out / "public_results.json").write_text(json.dumps(rep, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
    print(json.dumps(rep, ensure_ascii=False, indent=1, default=float))


if __name__ == "__main__":
    main()
