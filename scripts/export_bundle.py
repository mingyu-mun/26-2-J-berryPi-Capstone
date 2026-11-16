"""다른 기기에서 추론하기 위한 모델 번들 만들기 (models/). GPU 서버에서 한 번 실행.

    python scripts/export_bundle.py --cache data/teacher_cache --student data/cache_student --nw-dir out/normwear \
        --f2c-params out/f2c_label/f2c_params.npz --f2c-oof out/anomaly_c/oof_lodo.parquet \
        --student-dir out/student_final --out models

결과 (전부 합쳐 약 2 MB, pickle 없음):
  models/student/S1-EP_s{0,1,2}.safetensors, S1-E_s*.safetensors   학생 가중치
  models/teacher/f2c_final.npz                                     교사 F2C 최종 W·α·b (시드 5개)
  models/teacher/nw_pca.npz                                        NormWear 768 → 32 PCA 축
  models/bundle.json                                               정규화 통계, 동작점, 규약, NormWear 출처
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from safetensors.torch import save_file

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from driver_state.teacher.evaluate import usable  # noqa: E402
from driver_state.teacher.metrics_time import weighted_median  # noqa: E402
from driver_state.teacher.nw_features import NW_COLS, attach, fit_pca  # noqa: E402
from driver_state.teacher.personal import PersonalNormalizer  # noqa: E402
from driver_state.teacher.spec import TeacherSpec  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    for k in ("cache", "student", "nw-dir", "f2c-params", "f2c-oof", "student-dir", "out"):
        ap.add_argument(f"--{k}", required=True)
    ap.add_argument("--normwear-ckpt", help="sha256 기록용 (선택)")
    a = ap.parse_args()
    out = Path(a.out)
    (out / "student").mkdir(parents=True, exist_ok=True)
    (out / "teacher").mkdir(parents=True, exist_ok=True)
    spec = TeacherSpec()
    names = list(spec.baseline_feature_names)

    # 1. 학생 가중치 → safetensors
    for p in sorted(Path(a.student_dir).glob("S1-*_s*.pt")):
        sd = torch.load(p, map_location="cpu", weights_only=True)
        save_file({k: v.contiguous() for k, v in sd.items()}, str(out / "student" / f"{p.stem}.safetensors"))
    th_s = json.loads((Path(a.student_dir) / "thresholds.json").read_text())

    # 2. 교사 최종 파라미터
    P = dict(np.load(a.f2c_params))
    final = {k: v for k, v in P.items() if k.startswith("final/")}
    np.savez(out / "teacher" / "f2c_final.npz", **{k.replace("/", "__"): v for k, v in final.items()})

    # 3. 정규화 통계 (교사 4종 유효 창, selfmeasure_infer 와 같은 방식)
    wt, pt = pd.read_parquet(Path(a.cache) / "windows.parquet"), pd.read_parquet(Path(a.cache) / "profiles.parquet")
    ws = pd.read_parquet(Path(a.student) / "windows.parquet")
    okp = set(pt.loc[pt["baseline_valid"] > 0.5, "profile_key"])
    fit_rows = usable(wt) & wt["profile_key"].isin(okp).to_numpy()
    norm = PersonalNormalizer(spec).fit(wt[fit_rows])
    comps, var = fit_pca(ws, Path(a.nw_dir))
    np.savez(out / "teacher" / "nw_pca.npz", comps=comps.astype(np.float32), explained=var.astype(np.float32))
    wt_nw, _ = attach(wt[fit_rows].reset_index(drop=True), pt, a.nw_dir, comps)
    nw_sig = [float(np.nanmedian(wt_nw.groupby("group")[c].std(ddof=1).to_numpy())) for c in NW_COLS]

    # 4. 교사 동작점 (LODO oof, 데이터셋별 졸음 창 가중 중앙값 평균 = 민감도 50%)
    oof = pd.read_parquet(a.f2c_oof)
    th_t = []
    for _, g in oof[(oof["y"] == 1) & np.isfinite(oof["s_F2"])].groupby("dataset_id"):
        w = 1.0 / g.groupby("group")["group"].transform("size").to_numpy(float)
        th_t.append(weighted_median(g["s_F2"].to_numpy(float), w))

    ckpt_sha = None
    if a.normwear_ckpt:
        h = hashlib.sha256()
        with open(a.normwear_ckpt, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        ckpt_sha = h.hexdigest()
    bundle = {
        "version": "2026-11-13-milestone",
        "spec": spec.to_dict(),
        "rr_feature_names": names,
        "normalizer": norm.to_dict(),
        "nw_cols": NW_COLS,
        "nw_sigma_pop": nw_sig,
        "nw_pca_explained": float(var.sum()),
        "thresholds": {"student_logit": th_s, "teacher_logit": float(np.mean(th_t))},
        "student": {"conds": {"S1-EP": {"channels": ["ecg", "ppg"], "rr": "ecg", "head": "dist"},
                              "S1-E": {"channels": ["ecg"], "rr": "ecg", "head": "dist"}},
                    "wave": {"fs": 64, "win_sec": 30, "bands": {"ecg": [0.5, 25.0], "ppg": [0.5, 8.0]},
                             "zscore_per_window": True, "clip": 10, "max_ref_windows": 30}},
        "teacher": {"normwear": {"repo": "https://github.com/Mobile-Sensing-and-UbiComp-Laboratory/NormWear",
                                 "repo_commit": "07517fc",
                                 "checkpoint": "https://huggingface.co/mosaic-laboratory/normwear (model.safetensors, 775 MB)",
                                 "sha256": ckpt_sha, "input": "ECG 0.5–40 Hz, 65 Hz, 6초 구간 5개, 창마다 z-score"}},
        "license_note": "학생·교사 가중치는 WESAD(비상업 연구용)로 학습했다. 비상업 용도로만 쓴다.",
    }
    (out / "bundle.json").write_text(json.dumps(bundle, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"번들 저장: {out}, 학생 {len(list((out / 'student').glob('*.safetensors')))}개, 교사 동작점 {np.mean(th_t):.4f}", flush=True)


if __name__ == "__main__":
    main()
