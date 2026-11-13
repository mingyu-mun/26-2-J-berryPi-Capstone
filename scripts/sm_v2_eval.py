"""상태 머신 v2 평가 (docs/state-machine-v2.md 9장).

    python scripts/sm_v2_eval.py --scores out/student/student_scores.parquet --quality out/f2c_label/quality.parquet \
        --rec1 out/sm_rec1/windows_scores.csv --rec2 out/sm_rec2/windows_scores.csv --out out/sm_v2_eval
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from driver_state.state_machine_v2 import StateMachineV2, V2Config, WindowIn  # noqa: E402

TEACHER = ("advitam", "dddb", "drozy", "mpddf")


def sig(x):
    return 1 / (1 + np.exp(-np.asarray(x, float)))


class V03:
    """v0.3 (docs/state-machine.md) 시연 설정: EMA 0.5, on 0.6 / off 0.4, 10초."""

    def __init__(self, alpha=0.5, on=0.6, off=0.4, up=10, down=10, hold=60, dt=10):
        self.a, self.on, self.off, self.upn, self.dnn, self.hold, self.dt = alpha, on, off, up, down, hold, dt
        self.s, self.lvl, self.u, self.d, self.inv = None, 0, 0, 0, 0

    def step(self, p):
        if p is None:
            self.inv += self.dt
            if self.inv >= self.hold:
                self.s, self.lvl, self.u, self.d = None, 0, 0, 0
                return -1
            return self.lvl
        self.inv = 0
        self.s = p if self.s is None else self.a * p + (1 - self.a) * self.s
        if self.s > self.on:
            self.d = 0
            self.u += self.dt
            if self.u >= self.upn and self.lvl < 2:
                self.lvl += 1
                self.u = 0
        elif self.s < self.off:
            self.u = 0
            self.d += self.dt
            if self.d >= self.dnn and self.lvl > 0:
                self.lvl -= 1
                self.d = 0
        return self.lvl


def _f(r, k):
    v = r.get(k, np.nan)
    return float(v) if v is not None and np.isfinite(v) else None


def timeline(df: pd.DataFrame) -> pd.DataFrame:
    """10초 격자로 펴고, 빈 칸은 판정 불가 (p NaN)."""
    df = df.sort_values("t_rel")
    t = np.arange(df["t_rel"].min(), df["t_rel"].max() + 1e-6, 10.0)
    return pd.DataFrame({"t_rel": np.round(t, 3)}).merge(df.assign(t_rel=df["t_rel"].round(3)), on="t_rel", how="left")


def run_both(g: pd.DataFrame, cfg: V2Config | None = None):
    v2, v3 = StateMachineV2(cfg or V2Config()), V03()
    L2, C2, L3 = [], [], []
    for r in g.to_dict("records"):
        p = _f(r, "p")
        o = v2.step(WindowIn(t=r["t_rel"] + 30, p_ep=p, q_ecg=_f(r, "q") or 0.0,
                             p_ppg=_f(r, "p_ppg"), q_ppg=_f(r, "q_ppg"), hr_ecg=_f(r, "hr_ecg"), hr_ppg=_f(r, "hr_ppg")))
        L2.append(o.level)
        C2.append(o.candidate or o.level == 2)
        L3.append(v3.step(p))
    return np.array(L2), np.array(C2), np.array(L3)


def entries(levels):
    lv = np.asarray(levels)
    ent = np.flatnonzero((lv[1:] >= 1) & (lv[:-1] <= 0)) + 1
    return np.r_[0, ent] if len(lv) and lv[0] >= 1 else ent


def episodes(y, min_len=6):
    out, i = [], 0
    while i < len(y):
        if y[i] == 1:
            j = i
            while j < len(y) and y[j] == 1:
                j += 1
            if j - i >= min_len:
                out.append((i, j))
            i = j
        else:
            i += 1
    return out


def score_timelines(groups, inject=None, rng=None):
    rows = []
    for key, g in groups:
        g = timeline(g)
        if inject is not None:
            g = inject(g, rng)
        L2, C2, L3 = run_both(g)
        y = g["y"].fillna(-1).to_numpy()
        for name, L, C in (("v2", L2, C2), ("v0.3", L3, L3 >= 2)):
            ent = entries(L)
            eps = episodes(y)
            lat1, lat2, miss = [], [], 0
            for a, b in eps:
                hit = np.flatnonzero(L[a:b] >= 1)
                hit2 = np.flatnonzero(C[a:b])
                if len(hit):
                    lat1.append(hit[0] * 10)
                else:
                    miss += 1
                if len(hit2):
                    lat2.append(hit2[0] * 10)
            rows.append(dict(key=str(key), sm=name, fa=int(np.sum(y[ent] == 0)) if len(ent) else 0,
                             alert_h=float((y == 0).sum() * 10 / 3600), n_ep=len(eps), miss=miss, lat1=lat1, lat2=lat2,
                             warn_frac_alert=float(np.mean(L[y == 0] >= 1)) if (y == 0).any() else np.nan))
    return pd.DataFrame(rows)


def summarize(df):
    out = {}
    for sm, g in df.groupby("sm"):
        lat1 = [x for l_ in g["lat1"] for x in l_]
        lat2 = [x for l_ in g["lat2"] for x in l_]
        h = g["alert_h"].sum()
        out[sm] = {"fa_per_h": float(g["fa"].sum() / h) if h else float("nan"), "alert_h": round(float(h), 2),
                   "episodes": int(g["n_ep"].sum()), "miss_rate": float(g["miss"].sum() / max(g["n_ep"].sum(), 1)),
                   "lat_warn_median_s": float(np.median(lat1)) if lat1 else float("nan"),
                   "lat_drowsy_median_s": float(np.median(lat2)) if lat2 else float("nan"),
                   "warn_time_frac_alert": float(np.nanmean(g["warn_frac_alert"]))}
    return out


def main():
    ap = argparse.ArgumentParser()
    for k in ("scores", "quality", "rec1", "rec2", "out"):
        ap.add_argument(f"--{k}", required=True)
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    sc = pd.read_parquet(a.scores).merge(pd.read_parquet(a.quality)[["row_id", "dataset_id", "q"]],
                                         on=["row_id", "dataset_id"], how="left")
    sc["p"] = sig(sc["s_S1-EP"])
    rep = {}
    pub = sc[sc["dataset_id"].isin(TEACHER) & np.isfinite(sc["p"])]
    rep["public"] = {ds: summarize(score_timelines(g.groupby(["group", "session_id"]))) for ds, g in pub.groupby("dataset_id")}
    rep["public_all"] = summarize(score_timelines(pub.groupby(["dataset_id", "group", "session_id"])))
    w = sc[(sc["dataset_id"] == "wesad") & np.isfinite(sc["p"])].copy()
    w["y"] = 0
    rep["wesad"] = summarize(score_timelines(w.groupby(["group", "session_id"])))
    recs = {}
    for name, f in (("rec1", a.rec1), ("rec2", a.rec2)):
        d = pd.read_csv(f)
        d = d[d["valid"]].copy()
        d["p"] = sig(d["student_S1-EP_logit"]); d["q"] = d["q_abn"]; d["y"] = 0
        if "student_S1-EP_prv_logit" in d:
            d["p_ppg"] = sig(d["student_S1-EP_prv_logit"])
        recs[name] = d
        rep[name] = summarize(score_timelines(d.groupby("session_id")))
        rep[name]["baseline_check_v2"] = json.loads((Path(f).parent / "summary.json").read_text(encoding="utf-8")).get("baseline_check_v2")
    # 실측 1차 잡음 덩어리 이식: 공개 정상 구간에 시간당 1개, 1차의 q>5% 창부터 9창 (p, q 그대로)
    r1 = recs["rec1"].sort_values("t_rel").reset_index(drop=True)
    noisy = np.flatnonzero(r1["q"].to_numpy() > 0.05)
    blocks = [r1.iloc[max(0, i - 1): i + 8][["p", "q"]].to_numpy() for i in noisy if i + 8 <= len(r1)]

    def inject(g, rng):
        y = g["y"].fillna(-1).to_numpy()
        ok = np.flatnonzero((y == 0) & np.isfinite(g["p"].to_numpy(float)))
        if len(ok) < 20 or not blocks:
            return g
        g = g.copy()
        for _ in range(max(1, int(len(ok) * 10 / 3600))):
            s = int(rng.choice(ok[:-9]))
            b = blocks[rng.integers(len(blocks))]
            idx = g.index[s: min(s + len(b), len(g))]
            g.loc[idx, "p"] = b[: len(idx), 0]
            g.loc[idx, "q"] = b[: len(idx), 1]
        return g
    rep["inject_public_alert"] = summarize(score_timelines(pub.groupby(["dataset_id", "group", "session_id"]), inject,
                                                           np.random.default_rng(0)))
    rep["n_noise_blocks"] = len(blocks)
    (out / "report_sm_v2.json").write_text(json.dumps(rep, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
    print(json.dumps(rep, ensure_ascii=False, indent=1, default=float))


if __name__ == "__main__":
    main()
