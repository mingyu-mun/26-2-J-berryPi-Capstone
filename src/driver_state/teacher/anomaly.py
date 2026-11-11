"""생체 이상 점수 1단계: 방향 없는 학습 거리 (docs/teacher-anomaly-stage1.md).

점수 s(z) ≥ 0, z = 개인 기준 대비 정규화 특징 36차원. z = 0 (기준과 같음) 이면 s = 0.
A1·A2: ‖W z‖₂ (W: 8×36), A3: ‖f(z) − f(0)‖₂ (f: 36→32→16 MLP). P = σ(softplus(α)·s + b).
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd
import torch
from torch import nn
from sklearn.linear_model import LogisticRegression

from .evaluate import usable
from .metrics_time import bootstrap_m1, m1_dataset, m2_dataset, m4_dataset, weighted_median
from .calibrate import ece
from .models.base import sample_weights
from .pairs import TEACHER_DATASETS
from .personal import PersonalNormalizer
from .spec import TeacherSpec

HP = dict(k=8, lr_lin=1e-2, lr_mlp=1e-3, wd=1e-3, batch=1024, steps=1500, noise=0.1)


def wesad_folds(groups, n=4) -> dict[str, int]:
    g = np.random.default_rng(0).permutation(sorted(set(groups)))
    return {x: k for k, chunk in enumerate(np.array_split(g, n)) for x in chunk}


def z_matrix(norm: PersonalNormalizer, win: pd.DataFrame, prof: pd.DataFrame, names: list[str]) -> np.ndarray:
    P = norm.transform(win.reset_index(drop=True), prof)
    Z = P[[f"z_{n}" for n in names]].to_numpy(np.float64)
    Z = np.nan_to_num(Z, nan=0.0)
    return np.clip(Z, -5, 5).astype(np.float32), P["dev_core"].to_numpy(np.float64)


class _Score(nn.Module):
    def __init__(self, kind: str, k: int = 8):
        super().__init__()
        self.kind = kind
        if kind == "lin":
            self.W = nn.Parameter(torch.randn(k, 36) * 0.1)
        else:
            self.f = nn.Sequential(nn.Linear(36, 32), nn.GELU(), nn.Linear(32, 16))
        self.alpha = nn.Parameter(torch.zeros(()))
        self.b = nn.Parameter(torch.zeros(()))

    def dist(self, z):
        if self.kind == "lin":
            v = z @ self.W.T
        else:
            v = self.f(z) - self.f(torch.zeros_like(z[:1]))
        return torch.sqrt((v ** 2).sum(-1) + 1e-8) - 1e-4     # z = 0 → 0

    def forward(self, z):
        s = self.dist(z)
        return nn.functional.softplus(self.alpha) * s + self.b, s


def train_model(kind: str, Z: np.ndarray, y: np.ndarray, w: np.ndarray, seed: int, hp=None) -> _Score:
    hp = {**HP, **(hp or {})}
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed + 31)
    m = _Score(kind, hp["k"])
    opt = torch.optim.AdamW(m.parameters(), lr=hp["lr_lin"] if kind == "lin" else hp["lr_mlp"], weight_decay=hp["wd"])
    p = w / w.sum()
    Zt, yt = torch.as_tensor(Z), torch.as_tensor(y, dtype=torch.float32)
    for _ in range(hp["steps"]):
        b = torch.as_tensor(rng.choice(len(y), hp["batch"], p=p))
        z = Zt[b] + hp["noise"] * torch.randn(len(b), 36) * (Zt[b] != 0)
        logit, _ = m(z)
        loss = nn.functional.binary_cross_entropy_with_logits(logit, yt[b])
        opt.zero_grad(); loss.backward(); opt.step()
    m.eval()
    return m


@torch.no_grad()
def predict(m: _Score, Z: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    logit, s = m(torch.as_tensor(Z))
    return s.numpy().astype(np.float64), logit.numpy().astype(np.float64)


def run(win_t, prof_t, ann_t, win_s, prof_s, ann_s, b0_oof, conds, seeds, spec: TeacherSpec, log=print,
        rr_ctx: dict | None = None):
    """rr_ctx: {"tacho_t": (N_t,2,T), "tacho_s": (N_s,2,T), "refs": {profile_key: (k,2,T)}, "base": {profile_key: RR ms},
    "device": str}. 있으면 평가 행을 기준 타코그램이 있는 참가자로 제한하고 R1~R3 를 돌린다."""
    names = list(spec.baseline_feature_names)
    ds = win_t["dataset_id"].to_numpy()
    use_t = usable(win_t) & (ann_t["baseline_valid"].to_numpy() > 0.5)
    if rr_ctx is not None:
        has_ref = lambda df: df["profile_key"].isin(set(rr_ctx["refs"]) & {k for k, v in rr_ctx["base"].items() if np.isfinite(v)}).to_numpy()
        use_t &= has_ref(win_t)
    wes = ann_s["dataset_id"].to_numpy() == "wesad"
    wf = wesad_folds(ann_s.loc[wes, "group"])
    wes_fold = ann_s["group"].map(wf).fillna(-1).astype(int).to_numpy()
    wes_ok = wes & win_s["valid"].to_numpy() & (ann_s["baseline_valid"].to_numpy() > 0.5)
    if rr_ctx is not None:
        wes_ok &= has_ref(win_s)
    parts, wparts = [], []
    for fi, test in enumerate(TEACHER_DATASETS):
        log(f"[fold {fi}: 시험 {test}, WESAD 묶음 {fi}]")
        tr = use_t & (ds != test)
        te = use_t & (ds == test)
        norm = PersonalNormalizer(spec).fit(win_t[tr])
        Zt_tr, _ = z_matrix(norm, win_t[tr], prof_t, names)
        Zt_te, dev_te = z_matrix(norm, win_t[te], prof_t, names)
        ws_tr = wes_ok & (wes_fold != fi) & (ann_s["cond"].to_numpy() == "stress")
        ws_te = wes_ok & (wes_fold == fi) & np.isin(ann_s["cond"].to_numpy(), ["stress", "other"])
        Zs_tr, _ = z_matrix(norm, win_s[ws_tr], prof_s, names)
        Zs_te, dev_ws = z_matrix(norm, win_s[ws_te], prof_s, names)
        y_tr = win_t.loc[tr, "y"].to_numpy(int)
        w_tr = sample_weights(ds[tr], win_t.loc[tr, "group"].to_numpy(), y_tr)
        gs = win_s.loc[ws_tr, "group"].to_numpy()
        w_s = 1.0 / pd.Series(gs).map(pd.Series(gs).value_counts()).to_numpy(float)
        w_s = w_s / w_s.sum() * (w_tr.sum() / len(np.unique(ds[tr])))
        part = win_t.loc[te, ["row_id", "dataset_id", "group", "session_id", "t_rel", "y"]].copy()
        part["tbin"] = ann_t.loc[te, "tbin"].to_numpy()
        part["fold"] = test
        part["s_B1"] = dev_te
        wpart = win_s.loc[ws_te, ["row_id", "group"]].copy()
        wpart["cond"] = ann_s.loc[ws_te, "cond"].to_numpy()
        wpart["fold"] = test
        wpart["s_B1"] = dev_ws
        # B1 보정 (P 용)
        dev_tr = z_matrix(norm, win_t[tr], prof_t, names)[1]
        okd = np.isfinite(dev_tr)
        lr = LogisticRegression().fit(dev_tr[okd, None], y_tr[okd], sample_weight=w_tr[okd])
        a = max(float(lr.coef_[0, 0]), 1e-3)
        part["p_B1"] = 1 / (1 + np.exp(-(a * dev_te + float(lr.intercept_[0]))))
        for c in conds:
            if c.startswith("R"):
                lt, lw = _run_rr(c, seeds, fi, win_t, ann_t, tr, te, y_tr, w_tr, win_s, ws_tr, ws_te, w_s, rr_ctx, log)
                part[f"s_{c}"] = np.mean(lt, 0)
                part[f"p_{c}"] = 1 / (1 + np.exp(-part[f"s_{c}"].to_numpy()))
                for k, sc in zip(seeds, lt):
                    part[f"s_{c}_s{k}"] = sc
                wpart[f"s_{c}"] = np.mean(lw, 0)
                continue
            kind = "lin" if c in ("A1", "A2", "F2") else "mlp"
            if c == "A1":
                Z, y, w = Zt_tr, y_tr, w_tr
            else:
                Z = np.concatenate([Zt_tr, Zs_tr]); y = np.r_[y_tr, np.zeros(len(Zs_tr), int)]; w = np.r_[w_tr, w_s]
            lt, lw = [], []
            for seed in seeds:
                t0 = time.perf_counter()
                m = train_model(kind, Z, y, w, seed)
                lt.append(predict(m, Zt_te)[1])
                lw.append(predict(m, Zs_te)[1])
                log(f"    {c} seed {seed} ({time.perf_counter() - t0:.0f}s)")
            # 점수 = 시드 평균 logit (a > 0 이라 거리와 순위가 같다). θ 와 WESAD 도 같은 척도.
            part[f"s_{c}"] = np.mean(lt, 0)
            part[f"p_{c}"] = 1 / (1 + np.exp(-part[f"s_{c}"].to_numpy()))
            for k, s in zip(seeds, lt):
                part[f"s_{c}_s{k}"] = s
            wpart[f"s_{c}"] = np.mean(lw, 0)
        parts.append(part)
        wparts.append(wpart)
    oof = pd.concat(parts, ignore_index=True)
    if b0_oof:
        b0 = pd.read_parquet(b0_oof)[["row_id", "p_rrnet"]].rename(columns={"p_rrnet": "s_B0"})
        oof = oof.merge(b0, on="row_id", how="left")
        oof["p_B0"] = oof["s_B0"]
    return oof, pd.concat(wparts, ignore_index=True)


def _run_rr(c, seeds, fi, win_t, ann_t, tr, te, y_tr, w_tr, win_s, ws_tr, ws_te, w_s, ctx, log):
    """R1 (스트레스 음성 없음), R2 (+스트레스 음성), R3 (R2 + 같은 참가자 순위 보조 손실)."""
    from .anomaly_rr import Trainer
    from .pairs import PairIndex, PairSampler
    Tt, Ts, refs, base = ctx["tacho_t"], ctx["tacho_s"], ctx["refs"], ctx["base"]
    keys_t = win_t.loc[tr, "profile_key"].tolist()
    T_tr, keys, y, w = Tt[tr], list(keys_t), y_tr.copy(), w_tr.copy()
    ds_tr = win_t.loc[tr, "dataset_id"].tolist()
    grp = win_t.loc[tr, "group"].tolist()
    if c != "R1":
        T_tr = np.concatenate([T_tr, Ts[ws_tr]])
        keys += win_s.loc[ws_tr, "profile_key"].tolist()
        y = np.r_[y, np.zeros(int(ws_tr.sum()), int)]
        w = np.r_[w, w_s]
        ds_tr += ["wesad"] * int(ws_tr.sum())
        grp += win_s.loc[ws_tr, "group"].tolist()
    # 검증: 데이터셋마다 참가자 20% (fold·조건 고정 시드)
    rng = np.random.default_rng(500 + fi)
    gdf = pd.DataFrame({"d": ds_tr, "g": grp})
    val_g = []
    for _, gg in gdf.groupby("d")["g"]:
        u = np.unique(gg)
        if len(u) >= 2:
            val_g += list(rng.choice(u, max(1, int(round(0.2 * len(u)))), replace=False))
    val_idx = np.flatnonzero(gdf["g"].isin(val_g).to_numpy())
    rpairs = None
    if c == "R3":
        sub = ann_t[tr].reset_index(drop=True)
        sub = sub[~sub["group"].isin(val_g)]
        pos = sub.index.to_numpy()
        ix = PairIndex(sub.reset_index(drop=True))

        def rpairs(rng_, n, _ix=ix, _pos=pos):
            sm = PairSampler(_ix, 0, 0, n_r=n, n_t=0, n_s=0)
            sm.rng = rng_
            a, b, _ = sm.sample_r()
            return _pos[a], _pos[b]
    te_keys = win_t.loc[te, "profile_key"].tolist()
    ws_keys = win_s.loc[ws_te, "profile_key"].tolist()
    lt, lw = [], []
    for seed in seeds:
        t0 = time.perf_counter()
        trn = Trainer(c, seed, ctx.get("device", "cpu")).fit(T_tr, keys, y, w, refs, base, val_idx, rpairs)
        lt.append(trn.predict(Tt[te], te_keys, refs, base, np.arange(int(te.sum()))))
        lw.append(trn.predict(Ts[ws_te], ws_keys, refs, base, np.arange(int(ws_te.sum()))))
        log(f"    {c} seed {seed} ({time.perf_counter() - t0:.0f}s)")
    return lt, lw


# ---------------------------------------------------------------- 지표 (4장)
def _fold_alarm(oof, wes, col, cond, fold, groups=None):
    td = oof[(oof["fold"] == fold) & (oof["y"] == 1)]
    wts = 1.0 / td.groupby("group")["group"].transform("size").to_numpy(float)
    theta = weighted_median(td[col].to_numpy(float), wts)
    w = wes[(wes["fold"] == fold) & (wes["cond"] == cond)]
    if groups is not None:
        w = pd.concat([w[w["group"] == g] for g in groups], ignore_index=True)
    return float(np.mean(w[col].to_numpy(float) >= theta)) if len(w) else float("nan")


def stress_rates(oof, wes, col, cond="stress") -> float:
    return float(np.nanmean([_fold_alarm(oof, wes, col, cond, f) for f in TEACHER_DATASETS]))


def bootstrap_stress(oof, wes, col_a, col_b, n_boot=2000, seed=0) -> dict:
    """WESAD 시험 참가자를 fold 안에서 복원 추출해 S1(a) − S1(b) 의 95% 구간."""
    rng = np.random.default_rng(seed)
    th = {}
    for f in TEACHER_DATASETS:
        td = oof[(oof["fold"] == f) & (oof["y"] == 1)]
        wts = 1.0 / td.groupby("group")["group"].transform("size").to_numpy(float)
        th[f] = {c: weighted_median(td[c].to_numpy(float), wts) for c in (col_a, col_b)}
    per = {}
    for f in TEACHER_DATASETS:
        w = wes[(wes["fold"] == f) & (wes["cond"] == "stress")]
        per[f] = [(g, (d[col_a].to_numpy() >= th[f][col_a]).astype(float), (d[col_b].to_numpy() >= th[f][col_b]).astype(float))
                  for g, d in w.groupby("group")]
    diffs = []
    for _ in range(n_boot):
        ra, rb = [], []
        for f, gl in per.items():
            if not gl:
                continue
            pick = rng.integers(len(gl), size=len(gl))
            ra.append(np.concatenate([gl[i][1] for i in pick]).mean())
            rb.append(np.concatenate([gl[i][2] for i in pick]).mean())
        diffs.append(np.mean(ra) - np.mean(rb))
    diffs = np.asarray(diffs)
    return {"diff": float(stress_rates(oof, wes, col_a) - stress_rates(oof, wes, col_b)),
            "lo": float(np.percentile(diffs, 2.5)), "hi": float(np.percentile(diffs, 97.5))}


def evaluate(oof, wes, conds_all, seeds) -> dict:
    rep = {"n_rows": int(len(oof)), "conds": {}}
    for c in conds_all:
        col = f"s_{c}"
        rows = oof[np.isfinite(oof[col])]
        r = {"M1": {}, "M2": {}, "M4": {}, "ECE": {}}
        for d, g in rows.groupby("dataset_id"):
            r["M1"][d] = m1_dataset(g, col)[0]
            r["M2"][d] = m2_dataset(g, col)
            r["M4"][d] = m4_dataset(g, col)
            if f"p_{c}" in g:
                r["ECE"][d] = ece(g[f"p_{c}"].to_numpy(), g["y"].to_numpy())
        for k in ("M1", "M2", "M4", "ECE"):
            v = [x for x in r[k].values() if np.isfinite(x)]
            r[f"{k}_mean"] = float(np.mean(v)) if v else float("nan")
        if col in wes:
            r["S1"] = stress_rates(rows, wes, col, "stress")
            r["S2"] = stress_rates(rows, wes, col, "other")
        sc = [f"{col}_s{s}" for s in seeds if f"{col}_s{s}" in rows]
        if sc:
            ms = [np.nanmean([m1_dataset(g, x)[0] for _, g in rows.groupby("dataset_id")]) for x in sc]
            r["M1_seed_min"], r["M1_seed_max"] = float(min(ms)), float(max(ms))
        rep["conds"][c] = r
    rows = oof.copy()
    for c in conds_all:
        rows = rows[np.isfinite(rows[f"s_{c}"])]
    boot = bootstrap_m1(rows, [f"s_{c}" for c in conds_all], 2000, 0)
    pos = {c: i for i, c in enumerate(conds_all)}
    rep["boot_M1"] = {}
    for a, b in [("R2", "B1"), ("R3", "B1"), ("R2", "B0"), ("R3", "B0"), ("R2", "F2"), ("R2", "R1"), ("R3", "R2"), ("F2", "B1")]:
        if a in pos and b in pos:
            d = boot[:, pos[a]] - boot[:, pos[b]]
            rep["boot_M1"][f"{a}-{b}"] = {"diff": rep["conds"][a]["M1_mean"] - rep["conds"][b]["M1_mean"],
                                          "lo": float(np.nanpercentile(d, 2.5)), "hi": float(np.nanpercentile(d, 97.5))}
    rep["boot_S1"] = {}
    for a, b in [("R2", "B1"), ("R3", "B1"), ("R2", "R1"), ("F2", "B1"), ("R2", "F2")]:
        if a in pos and b in pos and f"s_{a}" in wes and f"s_{b}" in wes:
            rep["boot_S1"][f"{a}-{b}"] = bootstrap_stress(rows, wes, f"s_{a}", f"s_{b}")
    # 판정 (5장)
    m1 = {c: rep["conds"][c]["M1_mean"] for c in rep["conds"]}
    ref = max(m1.get("B0", -1), m1.get("B1", -1))
    ok = {}
    for c in ("R2", "R3"):
        if c not in rep["conds"]:
            continue
        r = rep["conds"][c]
        keep = r["M1_mean"] >= ref - 0.01
        better = r.get("S1", 1) < rep["conds"]["B1"].get("S1", 0) and rep["boot_S1"].get(f"{c}-B1", {}).get("hi", 1) < 0
        ok[c] = {"keep_drowsy": bool(keep), "fewer_stress_alarms": bool(better)}
    passing = [c for c, v in ok.items() if v["keep_drowsy"] and v["fewer_stress_alarms"]]
    if len(passing) == 2 and abs(m1["R3"] - m1["R2"]) <= 0.01:
        chosen = "R2"
    else:
        chosen = max(passing, key=lambda c: m1[c]) if passing else None
    rep["verdict"] = {"reference_M1": ref, "per_model": ok, "adopted": chosen,
                      "encoder_effect": bool(rep["boot_M1"].get("R2-F2", {}).get("lo", -1) > 0),
                      "stress_negative_effect": bool(rep["boot_S1"].get("R2-R1", {}).get("hi", 1) < 0)}
    return rep


def report_markdown(rep, repro) -> str:
    C = list(rep["conds"])
    f = lambda v: "-" if v is None or not np.isfinite(v) else f"{v:.3f}"
    L = ["# 생체 이상 점수 1단계 결과", "", "## 재현 확인", "", "```", str(repro), "```", "",
         "## 졸음 판별과 스트레스 오경보", "",
         "| 모델 | M1 평균 | advitam | dddb | drozy | mpddf | M2 평균 | M4 평균 | S1 스트레스 오경보 | S2 평상시 오경보 | ECE |",
         "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for c in C:
        r = rep["conds"][c]
        L.append(f"| {c} | {f(r['M1_mean'])} | " + " | ".join(f(r["M1"].get(d)) for d in TEACHER_DATASETS)
                 + f" | {f(r['M2_mean'])} | {f(r['M4_mean'])} | {f(r.get('S1'))} | {f(r.get('S2'))} | {f(r.get('ECE_mean'))} |")
    L += ["", "S1·S2: 시험 데이터셋 졸음 창 점수의 참가자 균등 중앙값(민감도 50%)을 넘는 WESAD 시험 참가자 창 비율, 4 fold 평균. B0 는 fold 모델이 없어 S1 없음.", "",
          "## 부트스트랩 95% 구간", "", "| 비교 | 지표 | 차이 | 95% 구간 |", "| --- | --- | --- | --- |"]
    for k, v in rep["boot_M1"].items():
        L.append(f"| {k} | M1 | {v['diff']:+.3f} | [{v['lo']:+.3f}, {v['hi']:+.3f}] |")
    for k, v in rep["boot_S1"].items():
        L.append(f"| {k} | S1 | {v['diff']:+.3f} | [{v['lo']:+.3f}, {v['hi']:+.3f}] |")
    L += ["", "## 판정 (5장)", "", "```", str(rep["verdict"]), "```"]
    return "\n".join(L) + "\n"
