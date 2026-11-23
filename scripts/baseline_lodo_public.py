"""공개 교사 4종에서 기준 조건 F5 / F10 / A5 / A10 의 LODO 점수 (fold 학생 모델, GPU 서버).

    python scripts/student_train.py ... --causal-input --conds S1-EP --seeds 0 --folds <f> --save-models --train-only --out <dir>/f<f>
    python scripts/baseline_lodo_public.py --label-dir out/f2c_label_c --quality out/f2c_label/quality.parquet \
        --wave data/student_wave_c --rr data/teacher_cache_c/rr --cache data/teacher_cache_c --oof out/anomaly/oof_lodo.parquet \
        --folds-dir out/student_causalA_folds --models models_causal --out out/baseline_lodo_causalA

- 시험 데이터셋 ds 의 창은 ds 를 빼고 학습한 fold 학생으로 점수를 낸다 (LODO).
- 기준 L (5분 / 10분): 데이터셋이 정한 기준 구간의 시작 a 부터 [a, a+L] 로 RR 프로필 (중앙값·IQR) 을 다시 계산한다.
  기준 세션과 같은 세션의 창은 모든 조건 공통으로 a+600초 이후만 평가한다 (실측 selfmeasure_eval 과 같은 방식).
- 파형 기준 (인코더 기준 벡터) 은 캐시의 기준 창 (최대 30개, 처음 약 5분) 을 모든 조건에 같이 쓴다 (한계).
- A5 / A10: src/driver_state/adaptive_baseline.py 의 조건부 느린 갱신.
M1 = 시간 층 안 참가자 가중 AUROC 의 데이터셋 평균, 95% 구간은 참가자 단위 부트스트랩.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]

from driver_state.adaptive_baseline import AdaptCfg, simulate  # noqa: E402
from driver_state.student.distill import Student  # noqa: E402
from driver_state.teacher.features import BaselineInfo, baseline_plan, baseline_profile  # noqa: E402
from driver_state.teacher.metrics_time import bootstrap_m1, m1_dataset  # noqa: E402
from driver_state.teacher.rr import load_rr_cache  # noqa: E402
from driver_state.teacher.spec import TeacherSpec  # noqa: E402

TEACHER = ("advitam", "dddb", "drozy", "mpddf")
CONDS = {"F5": (300.0, False), "F10": (600.0, False), "A5": (300.0, True), "A10": (600.0, True)}
KEY = ["dataset_id", "group", "session_id", "t_rel"]
EVAL_FROM = 600.0


def profiles(rr_root: str, spec: TeacherSpec):
    """profile_key → {L: (m0, s0)}, 기준 정보 (세션, 시작)."""
    names = list(spec.baseline_feature_names)
    groups: dict[str, list] = {}
    for r in load_rr_cache(rr_root):
        groups.setdefault(f"{r.dataset_id}/{r.group}", []).append(r)
    out, info = {}, {}
    for g, lst in groups.items():
        by = {x.session_id: x for x in lst}
        for bi in {v.session_id: v for v in baseline_plan(lst, spec).values() if v is not None}.values():
            key = f"{g}#{bi.session_id}"
            r = by[bi.session_id]
            info[key] = (bi.session_id, bi.start)
            out[key] = {}
            for L in (300.0, 600.0):
                p = baseline_profile(r, BaselineInfo(bi.session_id, bi.start, min(bi.start + L, r.duration)), spec)
                out[key][L] = (np.array([p[f"m_{n}"] for n in names], float), np.array([p[f"s_{n}"] for n in names], float))
    return out, info


def main():
    ap = argparse.ArgumentParser()
    for k in ("label-dir", "quality", "wave", "rr", "cache", "oof", "folds-dir", "models", "out"):
        ap.add_argument(f"--{k}", required=True)
    ap.add_argument("--n-boot", type=int, default=2000)
    a = ap.parse_args(); out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    B = json.loads((Path(a.models) / "bundle.json").read_text(encoding="utf-8"))
    spec = TeacherSpec.from_dict(B["spec"]); names = B["rr_feature_names"]
    sig_pop = np.array([B["normalizer"]["sigma_pop"][n] for n in names])
    hr_idx = names.index("hr_120")
    lab = pd.read_parquet(Path(a.label_dir) / "f2c_labels.parquet")
    q = pd.read_parquet(a.quality)[["row_id", "dataset_id", "q"]]
    W = Path(a.wave); wave = np.load(W / "wave.npy", mmap_mode="r"); ref_wave = np.load(W / "ref_wave.npy")
    ref_index = pd.read_parquet(W / "ref_index.parquet"); index = pd.read_parquet(W / "index.parquet")
    rows = np.flatnonzero(lab["dataset_id"].isin(TEACHER).to_numpy())
    L = lab.iloc[rows][["row_id", "dataset_id", "group", "session_id", "t_rel", "y"]].copy()
    L["wrow"] = rows
    L["profile_key"] = index["profile_key"].to_numpy()[rows]
    wt = pd.read_parquet(Path(a.cache) / "windows.parquet")
    w2 = wt[KEY + ["valid"] + names].assign(t_rel=wt["t_rel"].round(3))
    L = L.assign(t_rel=L["t_rel"].round(3)).merge(w2, on=KEY, how="left").merge(q, on=["row_id", "dataset_id"], how="left")
    prof, binfo = profiles(a.rr, spec)
    print(f"창 {len(L)}, 프로필 {len(prof)}", flush=True)

    res = {c: np.full(len(L), np.nan) for c in CONDS}
    evalable = np.zeros(len(L), bool)
    for ds in TEACHER:
        ms, params = [], []
        for p in sorted(Path(a.folds_dir).glob(f"*/fold_{ds}_S1-EP_s*.pt")):
            sd = torch.load(p, map_location="cpu", weights_only=True)
            m = Student(2, "dist", sd["W"].numpy(), 0.0, 0.0, 36); m.load_state_dict(sd); m.eval(); ms.append(m.to(dev))
            params.append({"W": sd["W"].numpy().astype(float), "alpha": float(sd["alpha"]), "b": float(sd["b"])})
        assert ms, f"fold 모델 없음: {ds}"
        Ld = L[L["dataset_id"] == ds]
        G = [np.zeros((len(Ld), 32), np.float32) for _ in ms]
        with torch.no_grad():
            wr = Ld["wrow"].to_numpy()
            for i in range(0, len(Ld), 2048):
                x = torch.as_tensor(np.asarray(wave[wr[i:i + 2048]], np.float32)).to(dev)
                for k, m in enumerate(ms):
                    G[k][i:i + 2048] = m.g(x).cpu().numpy()
            gref = {}
            for pk, g in ref_index[ref_index["profile_key"].str.startswith(ds + "/")].groupby("profile_key"):
                r = torch.as_tensor(ref_wave[g.index.to_numpy()].astype(np.float32)).to(dev)
                gref[pk] = [m.g(r).median(0).values.cpu().numpy() for m in ms]
        pos_in_L = {ix: j for j, ix in enumerate(Ld.index)}
        for (grp, ses), g in Ld.groupby(["group", "session_id"]):
            g = g.sort_values("t_rel")
            pk = g["profile_key"].iloc[0]
            if pk not in gref or pk not in prof:
                continue
            bses, bstart = binfo[pk]
            own = bses == ses
            grid = np.arange(g["t_rel"].min(), g["t_rel"].max() + 1e-6, 10.0)
            pos = np.searchsorted(grid, g["t_rel"].to_numpy(float) - 1e-6)
            n = len(grid)
            X = np.full((n, len(names)), np.nan); valid = np.zeros(n, bool); qq = np.ones(n)
            X[pos] = g[names].to_numpy(float); valid[pos] = g["valid"].fillna(False).to_numpy(bool)
            qq[pos] = g["q"].fillna(1.0).to_numpy(float)
            jj = np.array([pos_in_L[ix] for ix in g.index])
            Gs = []
            for k in range(len(ms)):
                a_ = np.zeros((n, 32), np.float32); a_[pos] = G[k][jj]; Gs.append(a_)
            ok_t = g["t_rel"].to_numpy(float) >= (bstart + EVAL_FROM) if own else np.ones(len(g), bool)
            evalable[L.index.get_indexer(g.index)] = ok_t
            for c, (Lb, adapt) in CONDS.items():
                m0, s0 = prof[pk][Lb]
                if not np.isfinite(m0[0]):
                    continue
                lg, _ = simulate(grid + 30, X, valid, qq, m0, s0, sig_pop, Gs, gref[pk], params, hr_idx,
                                 (bstart + Lb) if own else None, AdaptCfg(adapt=adapt))
                res[c][L.index.get_indexer(g.index)] = lg[pos]
        print(f"  {ds} 끝 (fold 모델 {len(ms)}개)", flush=True)
    for c in CONDS:
        L[c] = res[c]
    tb = pd.read_parquet(a.oof)[["row_id", "dataset_id", "tbin"]]
    ok = (L["y"] >= 0).to_numpy() & L["valid"].fillna(False).to_numpy(bool) & evalable
    for c in CONDS:
        ok &= np.isfinite(L[c].to_numpy(float))
    t = L[ok].merge(tb, on=["row_id", "dataset_id"])
    cs = list(CONDS)
    M = {c: {ds: m1_dataset(gg, c)[0] for ds, gg in t.groupby("dataset_id")} for c in cs}
    Bt = bootstrap_m1(t, cs, n_boot=a.n_boot)
    rep = {"n_windows": int(len(t)), "n_by_ds": t["dataset_id"].value_counts().to_dict(),
           "M1": {c: float(np.mean(list(M[c].values()))) for c in cs}, "M1_ds": M,
           "M1_ci": {c: [float(np.nanpercentile(Bt[:, i], 2.5)), float(np.nanpercentile(Bt[:, i], 97.5))] for i, c in enumerate(cs)},
           "diff": {}}
    for x, y in (("F10", "F5"), ("A5", "F5"), ("A10", "F10"), ("A10", "A5"), ("A10", "F5")):
        d = Bt[:, cs.index(x)] - Bt[:, cs.index(y)]
        rep["diff"][f"{x}-{y}"] = {"diff": rep["M1"][x] - rep["M1"][y], "lo": float(np.nanpercentile(d, 2.5)), "hi": float(np.nanpercentile(d, 97.5))}
    (out / "results.json").write_text(json.dumps(rep, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
    Lm = ["| 조건 | M1 | 95% 구간 | " + " | ".join(TEACHER) + " |", "|" + " --- |" * (3 + len(TEACHER))]
    for c in cs:
        Lm.append(f"| {c} | {rep['M1'][c]:.3f} | [{rep['M1_ci'][c][0]:.3f}, {rep['M1_ci'][c][1]:.3f}] | "
                  + " | ".join(f"{M[c].get(d, float('nan')):.3f}" for d in TEACHER) + " |")
    Lm += ["", "| 차이 | 값 | 95% 구간 |", "| --- | --- | --- |"]
    Lm += [f"| {k} | {v['diff']:+.3f} | [{v['lo']:+.3f}, {v['hi']:+.3f}] |" for k, v in rep["diff"].items()]
    Lm += ["", f"평가 창 {len(t)}: " + ", ".join(f"{k} {v}" for k, v in rep["n_by_ds"].items())]
    (out / "results.md").write_text("\n".join(Lm) + "\n", encoding="utf-8")
    print("\n".join(Lm), flush=True)


if __name__ == "__main__":
    main()
