"""공개 4종 LODO 비교: 인과 학생 vs 기존 학생 vs 규칙 dev_core (docs/causal-student.md 4장).

    python scripts/causal_compare.py --old out/student/student_scores.parquet \
        --new out/student_causal/student_scores.parquet [--new-ct out/student_causal_ct/student_scores.parquet] \
        --labels out/f2c_label_c/f2c_labels.parquet --s0-oof out/anomaly/oof_lodo.parquet --out out/causal_compare

M1 = 시간 층(tbin, s0-oof 의 값) 안 참가자 가중 AUROC 의 데이터셋 평균 (student_train 과 같은 정의).
평가 창 = 모든 열이 유한한 교사 4종 라벨 창 (같은 창 집합에서 비교). 95% 구간은 참가자 단위 부트스트랩.
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

from driver_state.teacher.metrics_time import bootstrap_m1, m1_dataset  # noqa: E402
from driver_state.teacher.pairs import TEACHER_DATASETS  # noqa: E402

K = ["row_id", "dataset_id"]


def main():
    ap = argparse.ArgumentParser()
    for k in ("old", "new", "labels", "s0-oof", "out"):
        ap.add_argument(f"--{k}", required=True)
    ap.add_argument("--new-ct", dest="new_ct")
    ap.add_argument("--n-boot", type=int, default=2000)
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    lab = pd.read_parquet(a.labels)[K + ["group", "session_id", "y", "dev_core", "dev_core_c"]]
    lab = lab.rename(columns={"dev_core": "rule_old", "dev_core_c": "rule_causal"})
    t = lab[lab["dataset_id"].isin(TEACHER_DATASETS) & (lab["y"] >= 0)]
    s0 = pd.read_parquet(a.s0_oof)[K + ["tbin"]]
    t = t.merge(s0, on=K)
    cols = {"rule_old": None, "rule_causal": None}
    srcs = [("old", a.old), ("causal", a.new)] + ([("causal_ct", a.new_ct)] if a.new_ct else [])
    for tag, p in srcs:
        sc = pd.read_parquet(p)
        for c in ("s_S1-EP", "s_S1-E", "s_S1-EP_ecg", "s_S1-EP_ppg"):
            if c in sc:
                t = t.merge(sc[K + [c]].rename(columns={c: f"{tag}:{c[2:]}"}), on=K, how="left")
                cols[f"{tag}:{c[2:]}"] = None
    cols = [c for c in cols if c in t]
    main_cols = [c for c in cols if not c.endswith(("_ecg", "_ppg"))]
    t = t[np.isfinite(t[main_cols].to_numpy(float)).all(1)].reset_index(drop=True)
    rep = {"n_windows": int(len(t)), "M1": {}, "M1_ds": {}, "diff": {}}
    for c in cols:
        g = t[np.isfinite(t[c].to_numpy(float))]
        per = {ds: m1_dataset(d, c)[0] for ds, d in g.groupby("dataset_id")}
        rep["M1_ds"][c] = per
        rep["M1"][c] = float(np.mean(list(per.values())))
    B = bootstrap_m1(t, main_cols, n_boot=a.n_boot)
    pairs = [("causal:S1-EP", "old:S1-EP"), ("causal:S1-EP", "rule_causal"), ("old:S1-EP", "rule_old"),
             ("causal:S1-E", "old:S1-E"), ("causal_ct:S1-EP", "causal:S1-EP")]
    for x, y in pairs:
        if x in main_cols and y in main_cols:
            d = B[:, main_cols.index(x)] - B[:, main_cols.index(y)]
            rep["diff"][f"{x} − {y}"] = {"diff": rep["M1"][x] - rep["M1"][y],
                                         "lo": float(np.nanpercentile(d, 2.5)), "hi": float(np.nanpercentile(d, 97.5))}
    (out / "compare.json").write_text(json.dumps(rep, ensure_ascii=False, indent=1), encoding="utf-8")
    L = [f"# 공개 4종 LODO: 인과 학생 vs 기존 학생 vs 규칙", "", f"평가 창 {len(t)} (모든 비교 열이 유한한 창)", "",
         "| 점수 | M1 | " + " | ".join(TEACHER_DATASETS) + " |", "|" + " --- |" * (2 + len(TEACHER_DATASETS))]
    for c in cols:
        L.append(f"| {c} | {rep['M1'][c]:.3f} | " + " | ".join(f"{rep['M1_ds'][c].get(d, float('nan')):.3f}" for d in TEACHER_DATASETS) + " |")
    L += ["", "| 차이 | 값 | 95% 구간 |", "| --- | --- | --- |"]
    L += [f"| {k} | {v['diff']:+.3f} | [{v['lo']:+.3f}, {v['hi']:+.3f}] |" for k, v in rep["diff"].items()]
    (out / "compare.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
