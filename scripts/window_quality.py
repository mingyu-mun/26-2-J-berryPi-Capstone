"""라벨 표 각 창의 30초 안 이상 간격 비율 q (docs/student-f2c-stage1.md 부록 A).

    python scripts/window_quality.py --labels out/f2c_label/f2c_labels.parquet --rr data/teacher_cache/rr data/cache_student/rr \
        --out out/f2c_label/quality.parquet
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from driver_state.teacher.rr import load_rr_cache  # noqa: E402

W = 30.0


def abn_ratio(rr, starts: np.ndarray) -> np.ndarray:
    t = rr.rr_t
    i = np.searchsorted(t, starts); j = np.searchsorted(t, starts + W)
    c = np.r_[0, np.cumsum(rr.abnormal.astype(int))]
    n = j - i
    return np.where(n > 0, (c[j] - c[i]) / np.maximum(n, 1), 1.0)


def rr_weight(q: np.ndarray, lo: float = 0.02, hi: float = 0.10) -> np.ndarray:
    return np.clip((hi - q) / (hi - lo), 0.0, 1.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", required=True); ap.add_argument("--rr", nargs="+", required=True); ap.add_argument("--out", required=True)
    a = ap.parse_args()
    lab = pd.read_parquet(a.labels)[["row_id", "dataset_id", "group", "session_id", "t_rel"]]
    rrs = {(r.dataset_id, f"{r.dataset_id}/{r.group}", r.session_id): r for root in a.rr for r in load_rr_cache(root)}
    q = np.full(len(lab), np.nan)
    for (ds, g, s), idx in lab.groupby(["dataset_id", "group", "session_id"]).groups.items():
        r = rrs.get((ds, g, s))
        if r is None:
            continue
        q[np.asarray(idx)] = abn_ratio(r, lab.loc[idx, "t_rel"].to_numpy(float))
    lab["q"] = q
    lab["w_rr"] = rr_weight(np.nan_to_num(q, nan=1.0))
    lab[["row_id", "dataset_id", "q", "w_rr"]].to_parquet(a.out)
    print(lab.groupby("dataset_id")["q"].describe(percentiles=[.5, .9]).round(3).to_string())
    print("결측", int(np.isnan(q).sum()))


if __name__ == "__main__":
    main()
