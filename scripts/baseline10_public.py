"""공개 교사 4종에서 개인 기준 구간 5분 vs 10분 비교 (GPU 서버). 최종 학생 S1-EP (공개 데이터로 학습 → 절대값 낙관적).

같은 평가 창(두 기준 모두에서 기준 구간과 겹치지 않는 창)만 비교한다.

    python scripts/baseline10_public.py --label-dir out/f2c_label --wave data/student_wave --cache data/teacher_cache \
        --data data/converted --oof out/anomaly/oof_lodo.parquet --out out/baseline10
"""
from __future__ import annotations

import argparse
import dataclasses
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
from driver_state.data_io import load_session  # noqa: E402
from driver_state.student.distill import Student  # noqa: E402
from driver_state.teacher.features import baseline_plan, build_tables  # noqa: E402
from driver_state.teacher.metrics_time import bootstrap_m1, m1_dataset  # noqa: E402
from driver_state.teacher.rr import load_rr_cache  # noqa: E402
from driver_state.teacher.spec import TeacherSpec  # noqa: E402
from sm_v2_eval import entries, episodes  # noqa: E402
from student_cache import cut, prep  # noqa: E402

TEACHER = ("advitam", "dddb", "drozy", "mpddf")
KEY = ["dataset_id", "group", "session_id", "t_rel"]


def main():
    ap = argparse.ArgumentParser()
    for k in ("label-dir", "wave", "cache", "data", "oof", "out"):
        ap.add_argument(f"--{k}", required=True)
    ap.add_argument("--models", default=str(ROOT / "models")); ap.add_argument("--jobs", type=int, default=16)
    a = ap.parse_args(); out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    mdir = Path(a.models); B = json.loads((mdir / "bundle.json").read_text(encoding="utf-8"))
    names = B["rr_feature_names"]; sig_pop = np.array([B["normalizer"]["sigma_pop"][n] for n in names])
    ms, params = [], []
    for p in sorted((mdir / "student").glob("S1-EP_s*.safetensors")):
        sd = load_file(str(p)); m = Student(2, "dist", sd["W"].numpy(), 0.0, 0.0, 36); m.load_state_dict(sd); m.eval(); ms.append(m.to(dev))
        params.append({"W": sd["W"].numpy().astype(float), "alpha": float(sd["alpha"]), "b": float(sd["b"])})
    spec5 = TeacherSpec(); spec10 = dataclasses.replace(spec5, baseline_sec=600.0)
    rrs = [r for r in load_rr_cache(Path(a.cache) / "rr") if r.dataset_id in TEACHER]
    tabs = {}
    w5, p5 = pd.read_parquet(Path(a.cache) / "windows.parquet"), pd.read_parquet(Path(a.cache) / "profiles.parquet")
    tabs["F5"] = (w5[w5["dataset_id"].isin(TEACHER)], p5)
    print("10분 기준 특징 계산 중", flush=True)
    tabs["F10"] = build_tables(rrs, spec10, a.jobs)
    # 평가 창: 라벨 표 (파형이 있는 창) ∩ 두 표 모두에 있는 창
    lab = pd.read_parquet(Path(a.label_dir) / "f2c_labels.parquet")
    q = pd.read_parquet(Path(a.label_dir) / "quality.parquet")[["row_id", "dataset_id", "q"]]
    lab["wrow"] = np.arange(len(lab))
    lab = lab[lab["dataset_id"].isin(TEACHER)].merge(q, on=["row_id", "dataset_id"])
    w10 = tabs["F10"][0]
    lab = lab.merge(w10[KEY].assign(in10=True), on=KEY, how="left")
    lab = lab[lab["in10"].fillna(False)].reset_index(drop=True)
    print(f"공통 평가 창 {len(lab)}", flush=True)
    wave = np.load(Path(a.wave) / "wave.npy", mmap_mode="r")
    G = [np.zeros((len(lab), 32), np.float32) for _ in ms]
    with torch.no_grad():
        for i in range(0, len(lab), 2048):
            x = torch.as_tensor(np.asarray(wave[lab["wrow"].to_numpy()[i:i + 2048]], np.float32)).to(dev)
            for k, m in enumerate(ms):
                G[k][i:i + 2048] = m.g(x).cpu().numpy()
    # 기준 창 인코더 출력 (5분 / 10분, 기준 세션 원신호에서 다시 자름)
    groups: dict[str, list] = {}
    for r in rrs:
        groups.setdefault(f"{r.dataset_id}/{r.group}", []).append(r)
    gref = {"F5": {}, "F10": {}}
    for gname, lst in groups.items():
        for tag, spec in (("F5", spec5), ("F10", spec10)):
            plan = baseline_plan(lst, spec)
            info = next((v for v in plan.values() if v is not None), None)
            if info is None:
                continue
            key = f"{gname}#{info.session_id}"
            r0 = next(x for x in lst if x.session_id == info.session_id)
            rec = load_session(Path(a.data) / r0.dataset_id / (r0.subject_id or "anon") / r0.session_id, channels=("ecg",))
            sig = prep(rec.signals["ecg"], rec.fs["ecg"], "ecg")
            rs = np.arange(info.start, info.end - 30 + 1e-6, 10.0)[:60]
            R = np.zeros((max(len(rs), 1), 2, 1920), np.float32)
            if len(rs):
                R[:, 0] = cut(sig, rs)[0]
            with torch.no_grad():
                gref[tag][key] = [m.g(torch.as_tensor(R).to(dev)).median(0).values.cpu().numpy() for m in ms]
    print("기준 창 인코딩 끝", flush=True)
    hr_idx = names.index("hr_120")
    tb = pd.read_parquet(a.oof)[["row_id", "dataset_id", "tbin"]]
    res, tls = {}, []
    for tag in ("F5", "F10"):
        win, prof = tabs[tag]
        P = prof.set_index("profile_key")
        L = lab[KEY + ["row_id", "y", "q"]].merge(win[KEY + ["valid", "profile_key"] + names], on=KEY, how="left")
        L["G"] = np.arange(len(L))
        logit = np.full(len(L), np.nan)
        for (ds, grp, ses), g in L.groupby(["dataset_id", "group", "session_id"]):
            g = g.sort_values("t_rel"); pk = g["profile_key"].iloc[0]
            if pk not in P.index or pk not in gref[tag] or P.loc[pk, "baseline_valid"] < 0.5:
                continue
            m0 = P.loc[pk, [f"m_{n}" for n in names]].to_numpy(float); s0 = P.loc[pk, [f"s_{n}" for n in names]].to_numpy(float)
            grid = np.arange(g["t_rel"].min(), g["t_rel"].max() + 1e-6, 10.0)
            pos = np.searchsorted(grid, g["t_rel"].to_numpy(float) - 1e-6)
            n = len(grid); X = np.full((n, len(names)), np.nan); valid = np.zeros(n, bool); qq = np.ones(n)
            X[pos] = g[names].to_numpy(float); valid[pos] = g["valid"].fillna(False).to_numpy(bool); qq[pos] = g["q"].fillna(1.0).to_numpy(float)
            Gs = []
            for k in range(len(ms)):
                a_ = np.zeros((n, 32), np.float32); a_[pos] = G[k][g["G"].to_numpy()]; Gs.append(a_)
            lg, lv = simulate(grid + 30, X, valid, qq, m0, s0, sig_pop, Gs, gref[tag][pk], params, hr_idx, None, AdaptCfg(adapt=False))
            logit[g.index.to_numpy()] = lg[pos]
            yg = np.full(n, -1); yg[pos] = g["y"].to_numpy(int)
            ent = entries(lv); eps = episodes(yg)
            hit = [np.flatnonzero(lv[a0:b0] >= 1) for a0, b0 in eps]
            tls.append({"ds": ds, "cond": tag, "fa": int(np.sum(yg[ent] == 0)) if len(ent) else 0,
                        "alert_h": float((yg == 0).sum() * 10 / 3600), "n_ep": len(eps), "miss": int(sum(len(h) == 0 for h in hit))})
        res[tag] = logit
    E = lab[KEY + ["row_id", "y"]].copy(); E["F5"], E["F10"] = res["F5"], res["F10"]
    E = E[(E["y"] >= 0) & np.isfinite(E["F5"]) & np.isfinite(E["F10"])].merge(tb, on=["row_id", "dataset_id"])
    M = {c: {ds: m1_dataset(gg, c)[0] for ds, gg in E.groupby("dataset_id")} for c in ("F5", "F10")}
    Bt = bootstrap_m1(E, ["F5", "F10"]); d = Bt[:, 1] - Bt[:, 0]
    T = pd.DataFrame(tls)
    rep = {"n": int(len(E)), "M1": {c: float(np.mean(list(v.values()))) for c, v in M.items()}, "M1_ds": M,
           "F10-F5": {"diff": float(np.mean(list(M["F10"].values())) - np.mean(list(M["F5"].values()))),
                      "lo": float(np.percentile(d, 2.5)), "hi": float(np.percentile(d, 97.5))}}
    for c, g in T.groupby("cond"):
        rep[c] = {"fa_per_h": float(g.fa.sum() / g.alert_h.sum()), "miss_rate": float(g.miss.sum() / max(g.n_ep.sum(), 1)),
                  "by_ds": {ds: {"fa_per_h": float(x.fa.sum() / max(x.alert_h.sum(), 1e-9)), "miss": float(x.miss.sum() / max(x.n_ep.sum(), 1))}
                            for ds, x in g.groupby("ds")}}
    (out / "baseline10_results.json").write_text(json.dumps(rep, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
    print(json.dumps(rep, ensure_ascii=False, indent=1, default=float))


if __name__ == "__main__":
    main()
