"""NormWear 교체 실험 비교 (docs/teacher-normwear-stage1.md 3~5장).

    python scripts/normwear_compare.py --pair-rr out/pair --pair-nw out/pair_nw --pair-c out/pair_c \
        --an-rr out/anomaly --an-nw out/anomaly_nw --an-c out/anomaly_c --out out/normwear_compare

각 폴더의 oof_lodo.parquet·wesad_scores.parquet 를 row_id 로 합치고, 모든 조건 점수가 유한한 창만 남겨
M1 (부트스트랩 차이)·S1 을 계산해 판정한다.
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

from driver_state.teacher.anomaly import bootstrap_stress, stress_rates  # noqa: E402
from driver_state.teacher.metrics_time import bootstrap_m1, per_dataset  # noqa: E402

KEY = ["row_id", "fold"]


def pick(d: Path, cols: dict[str, str]):
    """cols: {원래 열: 새 이름}."""
    o = pd.read_parquet(d / "oof_lodo.parquet")
    w = pd.read_parquet(d / "wesad_scores.parquet")
    oc = [c for c in cols if c in o.columns]
    wc = [c for c in cols if c in w.columns]
    return (o[KEY + oc].rename(columns=cols), w[KEY + wc].rename(columns=cols))


def main():
    ap = argparse.ArgumentParser()
    for k in ("pair-rr", "pair-nw", "pair-c", "an-rr", "an-nw", "an-c", "out"):
        ap.add_argument(f"--{k}", required=True)
    ap.add_argument("--n-boot", type=int, default=2000)
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)

    base = pd.read_parquet(Path(a.pair_rr) / "oof_lodo.parquet")[KEY + ["dataset_id", "group", "session_id", "t_rel", "tbin", "y", "baseline_valid"]]
    wbase = pd.read_parquet(Path(a.pair_rr) / "wesad_scores.parquet")[KEY + ["group", "cond"]]
    parts = [pick(Path(a.pair_rr), {"d_B0": "B0", "d_B1": "B1", "d_P1": "P1"}),
             pick(Path(a.pair_nw), {"d_P0": "P0N", "d_P1": "P1N", "d_P2": "P2N"}),
             pick(Path(a.pair_c), {"d_P1": "P1C"}),
             pick(Path(a.an_rr), {"s_F2": "F2"}),
             pick(Path(a.an_nw), {"s_F2": "F2N"}),
             pick(Path(a.an_c), {"s_F2": "F2C"})]
    oof, wes = base, wbase
    for o, w in parts:
        oof = oof.merge(o, on=KEY, how="inner")
        wes = wes.merge(w, on=KEY, how="inner")
    conds = ["B0", "B1", "P1", "F2", "P0N", "P1N", "P2N", "P1C", "F2N", "F2C"]
    wconds = [c for c in conds if c in wes.columns]
    n0 = len(oof)
    oof = oof[(oof["y"] >= 0) & (oof["baseline_valid"] == 1)]
    oof = oof[np.isfinite(oof[conds].to_numpy(float)).all(1)].reset_index(drop=True)
    wes = wes[np.isfinite(wes[wconds].to_numpy(float)).all(1)].reset_index(drop=True)
    print(f"평가 창 {len(oof)} / 합친 창 {n0}, WESAD 창 {len(wes)}", flush=True)

    rep = {"n_rows": int(len(oof)), "n_wesad": int(len(wes)), "M1": {}, "M1_ds": {}, "S1": {}, "S2": {}, "diff": {}}
    for c in conds:
        pd_ = per_dataset(oof, c, "M1")
        rep["M1_ds"][c] = pd_
        rep["M1"][c] = float(np.nanmean(list(pd_.values())))
    for c in wconds:
        rep["S1"][c] = stress_rates(oof, wes, c)
        rep["S2"][c] = stress_rates(oof, wes, c, cond="other")
    pairs = [("P1N", "P1"), ("F2N", "F2"), ("P1C", "P1"), ("F2C", "F2")]
    bcols = sorted({c for p in pairs for c in p})
    B = bootstrap_m1(oof, bcols, n_boot=a.n_boot)          # 같은 재표본에서 모든 조건 (쌍체)
    for x, y in pairs:
        d = B[:, bcols.index(x)] - B[:, bcols.index(y)]
        rep["diff"][f"{x}-{y}"] = {"diff": rep["M1"][x] - rep["M1"][y],
                                   "lo": float(np.nanpercentile(d, 2.5)), "hi": float(np.nanpercentile(d, 97.5))}
    best = max(rep["M1"]["B0"], rep["M1"]["B1"])
    cands = []
    for c in ["P0N", "P1N", "P2N", "P1C", "F2N", "F2C"]:
        if c not in rep["S1"]:
            continue
        sd = bootstrap_stress(oof, wes, c, "B1", n_boot=a.n_boot)
        rep["diff"][f"S1 {c}-B1"] = sd
        if rep["M1"][c] >= best - 0.01 and rep["S1"][c] < rep["S1"]["B1"] and sd["hi"] < 0:
            cands.append(c)
    rep["verdict"] = {
        "NW 효과 (쌍)": rep["diff"]["P1N-P1"]["lo"] > 0,
        "NW 효과 (거리)": rep["diff"]["F2N-F2"]["lo"] > 0,
        "결합 효과 (쌍)": rep["diff"]["P1C-P1"]["lo"] > 0,
        "결합 효과 (거리)": rep["diff"]["F2C-F2"]["lo"] > 0,
        "채택": max(cands, key=lambda c: rep["M1"][c]) if cands else None,
    }
    (out / "report_normwear.json").write_text(json.dumps(rep, ensure_ascii=False, indent=1, default=float), encoding="utf-8")

    L = ["# NormWear 교체 실험 결과", "", f"평가 창 {len(oof)}, WESAD 창 {len(wes)}", "",
         "| 조건 | M1 | " + " | ".join(sorted(rep["M1_ds"]["B0"])) + " | S1 | S2 |", "|" + " --- |" * (5 + len(rep["M1_ds"]["B0"]))]
    for c in conds:
        ds = " | ".join(f"{rep['M1_ds'][c][k]:.3f}" for k in sorted(rep["M1_ds"][c]))
        s1 = f"{rep['S1'][c]:.3f}" if c in rep["S1"] else "-"
        s2 = f"{rep['S2'][c]:.3f}" if c in rep["S2"] else "-"
        L.append(f"| {c} | {rep['M1'][c]:.3f} | {ds} | {s1} | {s2} |")
    L += ["", "| 차이 | 값 | 95% 구간 |", "| --- | --- | --- |"]
    for k, v in rep["diff"].items():
        L.append(f"| {k} | {v['diff']:+.3f} | [{v['lo']:+.3f}, {v['hi']:+.3f}] |")
    L += ["", "## 판정", ""] + [f"- {k}: {v}" for k, v in rep["verdict"].items()]
    (out / "report_normwear.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L), flush=True)


if __name__ == "__main__":
    main()
