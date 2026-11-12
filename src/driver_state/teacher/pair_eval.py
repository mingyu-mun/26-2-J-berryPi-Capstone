"""쌍 비교 교사 1단계 평가: LODO, BT·P0·P1·P2, 기준 차감, 보정, 지표 (docs/teacher-pair-stage1.md 6~8장)."""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from .calibrate import ece
from .evaluate import usable
from .metrics_time import (bootstrap_diff, bootstrap_m1, m1_dataset, m2_dataset, m3_dataset, m4_dataset, weighted_median)
from .models.base import sample_weights
from .models.pairnet import PairNet
from .pairs import (EVAL_ONLY, STUDENT_TRAIN, TEACHER_DATASETS, FeatureScaler, PairIndex, PairSampler,
                    annotate, feature_names, read_session_info, scale_groups)
from .personal import PersonalNormalizer
from .spec import TeacherSpec

README_B = {"rrnet": {"advitam": 0.529, "dddb": 0.558, "drozy": 0.744, "mpddf": 0.656},
            "dev_core": {"advitam": 0.571, "dddb": 0.536, "drozy": 0.898, "mpddf": 0.655}}
README_TIME = {"advitam": 0.707, "dddb": 0.346, "mpddf": 0.785}
README_M3_RRNET = {"advitam": 0.494, "dddb": 0.721, "mpddf": 0.592}


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


# ---------------------------------------------------------------- 데이터 준비
def load_all(cache: str, student: str, data_root: str | None, tbin_sec: float = 300.0, nw_dir: str | None = None):
    c, s = Path(cache), Path(student)
    wt, pt = pd.read_parquet(c / "windows.parquet"), pd.read_parquet(c / "profiles.parquet")
    ws, ps = pd.read_parquet(s / "windows.parquet"), pd.read_parquet(s / "profiles.parquet")
    info = read_session_info([c / "rr", s / "rr"])
    if nw_dir:                                   # NormWear 표현 열 붙이기 (teacher-normwear-stage1.md)
        from .nw_features import attach, fit_pca
        comps, var = fit_pca(ws, Path(nw_dir))
        print(f"NormWear PCA 32축 설명 분산 {var.sum():.3f}", flush=True)
        wt, pt = attach(wt, pt, nw_dir, comps)
        ws, ps = attach(ws, ps, nw_dir, comps)
    prof = pd.concat([pt, ps], ignore_index=True)
    at = annotate(wt, pt, info, data_root, tbin_sec)
    ast = annotate(ws, ps, info, data_root, tbin_sec)
    at["src"], ast["src"] = "teacher", "student"
    ann = pd.concat([at, ast], ignore_index=True)
    return ann, prof, wt, pt, ws, ps


def ref_index(ann: pd.DataFrame, prof: pd.DataFrame) -> np.ndarray:
    pos = pd.Series(np.arange(len(prof)), index=prof["profile_key"])
    return ann["profile_key"].map(pos).fillna(-1).astype(int).to_numpy()


def ref_matrix(prof: pd.DataFrame, names: list[str]) -> np.ndarray:
    cols = [f"m_{n}" for n in names]
    return prof.reindex(columns=cols).to_numpy(np.float64)


# ---------------------------------------------------------------- 한 fold 학습
class FoldRunner:
    def __init__(self, ann, prof, names, spec, log=print):
        self.ann, self.prof, self.names, self.spec, self.log = ann, prof, names, spec, log
        self.ref_of = ref_index(ann, prof)
        self.Mref_raw = ref_matrix(prof, names)
        self.groups = scale_groups(names)

    def run(self, train_mask: np.ndarray, teacher_train_mask: np.ndarray, score_rows: np.ndarray,
            conds: list[str], seeds: list[int], fold_index: int) -> dict:
        """train_mask: 쌍·표준화 학습 창 (교사 학습 데이터 + CLAS·MAUS, valid). teacher_train_mask: P0·BT·보정용.
        score_rows: 점수를 낼 행 번호. 반환: {cond: {"d_s": (seeds, n), "logit_s": (seeds, n), "raw_s": …}}"""
        ann = self.ann
        scaler = FeatureScaler(self.names).fit(ann[train_mask & ann["valid"].to_numpy()])
        Z = scaler.transform(ann[self.names].to_numpy())
        Zref = scaler.transform(self.Mref_raw)
        ref_of = self.ref_of
        out = {}
        tr_idx = np.flatnonzero(train_mask & ann["valid"].to_numpy())
        cal = np.flatnonzero(teacher_train_mask & usable(ann) & (ann["baseline_valid"].to_numpy() > 0.5))
        cal_w = sample_weights(ann["dataset_id"].to_numpy()[cal], ann["group"].to_numpy()[cal], ann["y"].to_numpy()[cal])
        Xr = Zref[np.clip(ref_of, 0, None)]
        Xr[ref_of < 0] = np.nan
        bv = ann["baseline_valid"].to_numpy() > 0.5

        def d_of(model, rows):
            d = model.score_diff(Z[rows], Xr[rows])
            d[~bv[rows]] = np.nan
            d[~np.isfinite(Xr[rows]).any(1) | ~np.isfinite(Z[rows]).any(1)] = np.nan   # 표현·기준 벡터 없음
            return d

        def calibrate(d_cal):
            ok = np.isfinite(d_cal)
            lr = LogisticRegression(C=1.0).fit(d_cal[ok, None], ann["y"].to_numpy()[cal][ok], sample_weight=cal_w[ok])
            a, b = float(lr.coef_[0, 0]), float(lr.intercept_[0])
            return (a if a > 0 else 1e-3), b

        sub = ann.iloc[tr_idx].reset_index(drop=True)
        index = None
        for cond in conds:
            res = {"d_s": [], "logit_s": [], "raw_s": [], "ab": []}
            for seed in seeds:
                t0 = time.perf_counter()
                net = PairNet(cond, seed, self.groups, d_in=len(self.names))
                if cond == "P0":
                    w_idx = np.flatnonzero(teacher_train_mask & usable(ann))
                    w = sample_weights(ann["dataset_id"].to_numpy()[w_idx], ann["group"].to_numpy()[w_idx],
                                       ann["y"].to_numpy()[w_idx])
                    net.fit_windows(Z[w_idx], ann["y"].to_numpy()[w_idx], w)
                else:
                    if index is None:
                        index = PairIndex(sub)
                    sampler = PairSampler(index, seed, fold_index)
                    net.fit_pairs(sampler, Z[tr_idx], ref_of[tr_idx], Zref, sub["dataset_id"].to_numpy())
                a, b = calibrate(d_of(net, cal))
                d = d_of(net, score_rows)
                res["d_s"].append(d)
                res["logit_s"].append(a * d + b)
                res["ab"].append((a, b))
                if cond == "P0":
                    res["raw_s"].append(net.score(Z[score_rows])[0])
                self.log(f"    {cond} seed {seed}: a={a:.3f} b={b:.3f} ({time.perf_counter() - t0:.0f}s)")
            out[cond] = {k: np.array(v) for k, v in res.items() if len(v)}
        return out


def bt_scores(ann, teacher_train_mask, score_rows) -> dict:
    t = ann["t_rel"].to_numpy(float)
    X = np.column_stack([t / 600, (t / 600) ** 2, ann["ses_rank"].to_numpy(float), ann["same_as_base"].to_numpy(float)])
    tr = np.flatnonzero(teacher_train_mask & usable(ann))
    w = sample_weights(ann["dataset_id"].to_numpy()[tr], ann["group"].to_numpy()[tr], ann["y"].to_numpy()[tr])
    lr = LogisticRegression(C=1.0, max_iter=1000).fit(X[tr], ann["y"].to_numpy()[tr], sample_weight=w)
    s = lr.decision_function(X[score_rows])
    return {"BTt": {"d": t[score_rows], "p": np.full(len(score_rows), np.nan)},
            "BTfit": {"d": s, "p": sigmoid(s)}}


# ---------------------------------------------------------------- LODO
def train_masks(ds: np.ndarray, test: str) -> tuple[np.ndarray, np.ndarray]:
    """(쌍·표준화 학습 마스크, 교사 학습 마스크). WESAD 는 어느 쪽에도 들어가지 않는다 (4.4)."""
    teacher_train = np.isin(ds, TEACHER_DATASETS) & (ds != test)
    return teacher_train | np.isin(ds, STUDENT_TRAIN), teacher_train


def run_lodo(ann, prof, wt, pt, ws, ps, b0_oof: str | None, conds, seeds, spec, log=print, names=None):
    names = names or feature_names(spec)
    runner = FoldRunner(ann, prof, names, spec, log)
    ds = ann["dataset_id"].to_numpy()
    is_teacher = np.isin(ds, TEACHER_DATASETS)
    is_student_train = np.isin(ds, STUDENT_TRAIN)
    is_wesad = np.isin(ds, EVAL_ONLY)
    nn_conds = [c for c in conds if c in ("P0", "P1", "P2")]
    parts, wes_parts = [], []
    for fi, test in enumerate(TEACHER_DATASETS):
        log(f"[fold {fi}: 시험 {test}]")
        train, teacher_train = train_masks(ds, test)
        test_rows = np.flatnonzero((ds == test) & usable(ann))
        wes_rows = np.flatnonzero(is_wesad & ann["valid"].to_numpy())
        rows = np.concatenate([test_rows, wes_rows])
        res = runner.run(train, teacher_train, rows, nn_conds, seeds, fi)
        part = ann.iloc[test_rows][["row_id", "dataset_id", "group", "session_id", "t_rel", "tbin", "y",
                                    "baseline_valid", "inducer"]].copy()
        part["fold"] = test
        wpart = ann.iloc[wes_rows][["row_id", "group", "session_id", "t_rel", "cond", "baseline_valid"]].copy()
        wpart["fold"] = test
        nt = len(test_rows)
        if "BT" in conds:
            bt = bt_scores(ann, teacher_train, test_rows)
            for k, v in bt.items():
                part[f"d_{k}"], part[f"p_{k}"] = v["d"], v["p"]
        for c, r in res.items():
            for k in range(len(seeds)):
                part[f"d_{c}_s{seeds[k]}"] = r["d_s"][k][:nt]
            part[f"d_{c}"] = np.nanmean(r["d_s"][:, :nt], 0)
            part[f"p_{c}"] = sigmoid(np.nanmean(r["logit_s"][:, :nt], 0))
            wpart[f"d_{c}"] = np.nanmean(r["d_s"][:, nt:], 0)
            wpart[f"p_{c}"] = sigmoid(np.nanmean(r["logit_s"][:, nt:], 0))
            if "raw_s" in r:
                part[f"d_{c}raw"] = np.nanmean(r["raw_s"][:, :nt], 0)
        # B1 dev_core (fold 마다 시험 창·WESAD 창)
        trw = wt[usable(wt) & wt["dataset_id"].isin([d for d in TEACHER_DATASETS if d != test]).to_numpy()]
        norm = PersonalNormalizer(spec).fit(trw, pt)
        tw = wt.set_index("row_id").loc[part["row_id"]].reset_index()
        part["d_B1"] = norm.transform(tw, pt)["dev_core"].to_numpy()
        wsel = ws.set_index("row_id").loc[wpart["row_id"]].reset_index()
        wpart["d_B1"] = norm.transform(wsel, ps)["dev_core"].to_numpy()
        parts.append(part)
        wes_parts.append(wpart)
    oof = pd.concat(parts, ignore_index=True)
    wes = pd.concat(wes_parts, ignore_index=True)
    if b0_oof:
        b0 = pd.read_parquet(b0_oof)[["row_id", "p_rrnet"]].rename(columns={"p_rrnet": "d_B0"})
        oof = oof.merge(b0, on="row_id", how="left")
        oof["p_B0"] = oof["d_B0"]
    return oof, wes


# ---------------------------------------------------------------- M7: DROZY 참가자 5-fold
def run_m7(ann, prof, conds, seeds, spec, log=print, n_folds=5, names=None):
    names = names or feature_names(spec)
    runner = FoldRunner(ann, prof, names, spec, log)
    ds = ann["dataset_id"].to_numpy()
    groups = np.random.default_rng(0).permutation(ann.loc[ds == "drozy", "group"].unique())
    out = []
    for k, chunk in enumerate(np.array_split(groups, n_folds)):
        log(f"[M7 fold {k}]")
        is_test = (ds == "drozy") & ann["group"].isin(chunk).to_numpy()
        teacher_train = (ds == "drozy") & ~is_test
        train = teacher_train | np.isin(ds, STUDENT_TRAIN)
        rows = np.flatnonzero(is_test & usable(ann))
        res = runner.run(train, teacher_train, rows, [c for c in conds if c in ("P0", "P1", "P2")], seeds, 100 + k)
        part = ann.iloc[rows][["row_id", "group", "y", "baseline_valid"]].copy()
        for c, r in res.items():
            part[f"d_{c}"] = np.nanmean(r["logit_s"], 0)
        out.append(part)
    return pd.concat(out, ignore_index=True)


# ---------------------------------------------------------------- 지표
def evaluate(oof, wes, m7, cond_cols: list[str], seeds) -> dict:
    rows = oof[(oof["y"] >= 0) & (oof["baseline_valid"] > 0.5)].copy()
    for c in cond_cols:
        rows = rows[np.isfinite(rows[f"d_{c}"])]
    rep = {"n_rows": int(len(rows)), "n_rows_all": int(len(oof)), "conds": {}}
    for c in cond_cols:
        col = f"d_{c}"
        r = {"M1": {}, "M1_used": {}, "M2": {}, "M3": {}, "M4": {}, "M6": {}}
        for d, g in rows.groupby("dataset_id"):
            r["M1"][d], r["M1_used"][d] = m1_dataset(g, col)
            r["M2"][d] = m2_dataset(g, col)
            r["M4"][d] = m4_dataset(g, col)
            if d in ("dddb", "mpddf"):
                r["M3"][d] = m3_dataset(g, col)
            if f"p_{c}" in g and np.isfinite(g[f"p_{c}"]).all():
                r["M6"][d] = ece(g[f"p_{c}"].to_numpy(), g["y"].to_numpy())
        for k in ("M1", "M2", "M3", "M4", "M6"):
            v = [x for x in r[k].values() if np.isfinite(x)]
            r[f"{k}_mean"] = float(np.mean(v)) if v else float("nan")
        seed_cols = [f"d_{c}_s{s}" for s in seeds if f"d_{c}_s{s}" in rows]
        if seed_cols:
            ms = [np.nanmean([m1_dataset(g, sc)[0] for _, g in rows.groupby("dataset_id")]) for sc in seed_cols]
            r["M1_seed_min"], r["M1_seed_max"] = float(np.min(ms)), float(np.max(ms))
        # M5
        if f"d_{c}" in wes:
            m5 = []
            for f, wf in wes.groupby("fold"):
                st = wf[(wf["cond"] == "stress") & np.isfinite(wf[f"d_{c}"])]
                td = oof[(oof["fold"] == f) & (oof["y"] == 1) & np.isfinite(oof[f"d_{c}"])]
                if len(st) == 0 or len(td) == 0:
                    continue
                wts = 1.0 / td.groupby("group")["group"].transform("size").to_numpy(float)
                theta = weighted_median(td[f"d_{c}"].to_numpy(float), wts)
                p = st[f"p_{c}"].to_numpy(float) if f"p_{c}" in st else np.full(len(st), np.nan)
                m5.append({"fold": f, "frac_p_ge_0.5": float(np.nanmean(p >= 0.5)) if np.isfinite(p).any() else None,
                           "p_median": float(np.nanmedian(p)) if np.isfinite(p).any() else None,
                           "fa_at_sens50": float(np.mean(st[f"d_{c}"].to_numpy(float) >= theta)), "n_stress": int(len(st))})
            if m5:
                r["M5_folds"] = m5
                r["M5_fa_at_sens50"] = float(np.mean([x["fa_at_sens50"] for x in m5]))
                ps_ = [x["frac_p_ge_0.5"] for x in m5 if x["frac_p_ge_0.5"] is not None]
                r["M5_frac_p_ge_0.5"] = float(np.mean(ps_)) if ps_ else None
                pm = [x["p_median"] for x in m5 if x["p_median"] is not None]
                r["M5_p_median"] = float(np.mean(pm)) if pm else None
        if m7 is not None and f"d_{c}" in m7:
            mm = m7[(m7["y"] >= 0) & np.isfinite(m7[f"d_{c}"])]
            r["M7"] = m2_dataset(mm, f"d_{c}")
        rep["conds"][c] = r
    # 부트스트랩 (M1)
    pairs = [("P1", "B0"), ("P2", "B0"), ("P2", "P1"), ("P1", "P0"), ("P2", "P0"), ("P1", "B1"), ("P2", "B1")]
    rep["bootstrap_M1"] = {}
    need = sorted({c for p in pairs for c in p if c in cond_cols})
    boot = bootstrap_m1(rows, [f"d_{c}" for c in need], n_boot=2000, seed=0)
    pos = {c: i for i, c in enumerate(need)}
    for a, b in pairs:
        if a in pos and b in pos:
            diff = boot[:, pos[a]] - boot[:, pos[b]]
            rep["bootstrap_M1"][f"{a}-{b}"] = {
                "diff": float(rep["conds"][a]["M1_mean"] - rep["conds"][b]["M1_mean"]),
                "lo": float(np.nanpercentile(diff, 2.5)), "hi": float(np.nanpercentile(diff, 97.5)), "n_boot": 2000}
    # 판정 (8.3)
    bt = rep["conds"].get("BTt", {}).get("M1", {})
    valid_metric = bool(bt) and all(0.45 <= v <= 0.55 for v in bt.values() if np.isfinite(v))
    m1 = {c: rep["conds"][c]["M1_mean"] for c in rep["conds"]}
    best_p = max(m1.get("P1", -1), m1.get("P2", -1))
    bs = rep["bootstrap_M1"]
    rep["verdict"] = {
        "metric_valid": valid_metric,
        "gate_pass": bool(valid_metric and "B0" in m1 and best_p > m1["B0"]),
        "heads": 2 if bs.get("P2-P1", {}).get("lo", -1) > 0 else 1,
        "pair_effect": bool(bs.get("P1-P0", {}).get("lo", -1) > 0),
    }
    return rep


def report_markdown(rep: dict, repro: dict, stats: dict | None = None) -> str:
    L = ["# teacher-pair 1단계 결과", "", f"평가 행 {rep['n_rows']:,} / 전체 {rep['n_rows_all']:,} 창", ""]
    L += ["## 재현 확인", "", "```", str(repro), "```", ""]
    conds = list(rep["conds"])
    L += ["## M1 시간 층화 AUROC (주 지표)", "", "| 조건 | 평균 | advitam | dddb | drozy | mpddf | 시드 범위 |",
          "| --- | --- | --- | --- | --- | --- | --- |"]
    for c in conds:
        r = rep["conds"][c]
        rng = f"{r['M1_seed_min']:.3f}~{r['M1_seed_max']:.3f}" if "M1_seed_min" in r else ""
        L.append(f"| {c} | {r['M1_mean']:.3f} | " + " | ".join(f"{r['M1'].get(d, float('nan')):.3f}" for d in TEACHER_DATASETS)
                 + f" | {rng} |")
    L += ["", "M1 비교에 쓰인 창 비율: " + ", ".join(f"{d} {v:.2f}" for d, v in rep["conds"][conds[0]]["M1_used"].items()), ""]
    for k, title in (("M2", "M2 원 AUROC"), ("M4", "M4 참가자 내 AUROC")):
        L += [f"## {title}", "", "| 조건 | 평균 | advitam | dddb | drozy | mpddf |", "| --- | --- | --- | --- | --- | --- |"]
        for c in conds:
            r = rep["conds"][c]
            L.append(f"| {c} | {r[f'{k}_mean']:.3f} | " + " | ".join(f"{r[k].get(d, float('nan')):.3f}" for d in TEACHER_DATASETS) + " |")
        L.append("")
    L += ["## M3 선형 추세 제거 AUROC", "", "| 조건 | dddb | mpddf |", "| --- | --- | --- |"]
    for c in conds:
        r = rep["conds"][c]
        L.append(f"| {c} | {r['M3'].get('dddb', float('nan')):.3f} | {r['M3'].get('mpddf', float('nan')):.3f} |")
    L += ["", "## M5 WESAD TSST 오경보, M6 ECE, M7 DROZY 5-fold", "",
          "| 조건 | 오경보 (민감도 50%) | P≥0.5 비율 | P 중앙값 | ECE 평균 | M7 |", "| --- | --- | --- | --- | --- | --- |"]
    f = lambda v: "-" if v is None or (isinstance(v, float) and not np.isfinite(v)) else f"{v:.3f}"
    for c in conds:
        r = rep["conds"][c]
        L.append(f"| {c} | {f(r.get('M5_fa_at_sens50'))} | {f(r.get('M5_frac_p_ge_0.5'))} | {f(r.get('M5_p_median'))} | "
                 f"{f(r.get('M6_mean'))} | {f(r.get('M7'))} |")
    L += ["", "B0 의 M5 는 fold 모델을 저장하지 않아 v1 값(P≥0.5 77%, 중앙값 0.87)만 있다.", "",
          "## 부트스트랩 (M1 차이, 참가자 단위, 2,000회)", "", "| 비교 | 차이 | 95% 구간 |", "| --- | --- | --- |"]
    for k, v in rep["bootstrap_M1"].items():
        L.append(f"| {k} | {v['diff']:+.3f} | [{v['lo']:+.3f}, {v['hi']:+.3f}] |")
    L += ["", "## 판정 (8.3)", "", "```", str(rep["verdict"]), "```"]
    return "\n".join(L) + "\n"
