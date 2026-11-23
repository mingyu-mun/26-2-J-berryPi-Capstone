"""인과(실시간) 학생 입력을 기존 F2C 라벨 표에 붙인다 (docs/causal-student.md 2장).

    python scripts/teacher.py --causal rr       --data <교사 4종> --out data/teacher_cache_c/rr --jobs 24
    python scripts/teacher.py --causal features --rr data/teacher_cache_c/rr --out data/teacher_cache_c --no-tacho
    (학생 3종 cache_student_c, PRV cache_prv_c 도 같은 방식)
    python scripts/causal_relabel.py --label-dir out/f2c_label --cache data/teacher_cache_c \
        --student data/cache_student_c --out out/f2c_label_c

결과 (out):
  f2c_labels.parquet  기존 열 그대로 + zc_<특징> (인과 RR z, 학생 입력), dev_core_c, logit_c (인과 z 로 다시 매긴 교사 logit)
  f2c_params.npz      기존 교사 파라미터 복사 (교사는 다시 학습하지 않음)
  normalizer_c.json   인과 창으로 맞춘 PersonalNormalizer (번들용)

기존 zr_ (창 중심 정렬 RR z) 와 logit (오프라인 교사 목표) 는 바꾸지 않는다. 교사는 오프라인 라벨러라 미래 문맥을 봐도 된다.
logit_c 는 "목표도 인과로" 비교 조건 (student_train --causal-target) 용이다.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from driver_state.student.distill import fold_logits  # noqa: E402
from driver_state.teacher.anomaly import z_matrix  # noqa: E402
from driver_state.teacher.evaluate import usable  # noqa: E402
from driver_state.teacher.personal import PersonalNormalizer  # noqa: E402
from driver_state.teacher.spec import TeacherSpec  # noqa: E402

KEY = ["dataset_id", "group", "session_id", "t_rel"]


def main():
    ap = argparse.ArgumentParser()
    for k in ("label-dir", "cache", "student", "out"):
        ap.add_argument(f"--{k}", required=True)
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    spec = json.loads((Path(a.cache) / "spec.json").read_text(encoding="utf-8"))
    spec = TeacherSpec.from_dict(spec)
    assert spec.causal, "인과 캐시가 아닙니다 (teacher.py --causal features 로 만드세요)"
    names = list(spec.baseline_feature_names)
    ld = Path(a.label_dir)
    lab = pd.read_parquet(ld / "f2c_labels.parquet")
    params = dict(np.load(ld / "f2c_params.npz"))

    wt, pt = pd.read_parquet(Path(a.cache) / "windows.parquet"), pd.read_parquet(Path(a.cache) / "profiles.parquet")
    ws, ps = pd.read_parquet(Path(a.student) / "windows.parquet"), pd.read_parquet(Path(a.student) / "profiles.parquet")
    okp = set(pt.loc[pt["baseline_valid"] > 0.5, "profile_key"])
    norm = PersonalNormalizer(spec).fit(wt[usable(wt) & wt["profile_key"].isin(okp).to_numpy()])
    (out / "normalizer_c.json").write_text(json.dumps(norm.to_dict(), indent=1), encoding="utf-8")

    parts = []
    for w, p in ((wt, pt), (ws, ps)):
        w = w.reset_index(drop=True)
        Z, dev = z_matrix(norm, w, p, names)
        d = pd.concat([w[KEY], pd.DataFrame(Z, columns=[f"zc_{n}" for n in names])], axis=1)
        d["dev_core_c"] = dev
        d["valid_c"] = w["valid"].to_numpy(bool)
        parts.append(d)
    zc = pd.concat(parts, ignore_index=True)
    zc["t_rel"] = zc["t_rel"].round(3)
    m = lab.assign(t_rel=lab["t_rel"].round(3)).merge(zc, on=KEY, how="left", validate="one_to_one")
    hit = m["zc_hr_30"].notna().to_numpy()
    print(f"라벨 창 {len(lab)}, 인과 특징 매칭 {hit.mean():.4f}", flush=True)
    for ds, g in m.groupby("dataset_id"):
        print(f"  {ds}: 매칭 {g['zc_hr_30'].notna().mean():.4f}, 인과 창 유효 {g['valid_c'].mean():.3f}", flush=True)
    zcols = [f"zc_{n}" for n in names]
    m[zcols] = m[zcols].fillna(0.0).astype(np.float32)
    m["t_rel"] = lab["t_rel"].to_numpy()

    # 인과 목표 logit_c: 원 logit 과 같은 교사로, z 의 RR 부분만 인과로
    zn = m[[c for c in m.columns if c.startswith("zn_")]].to_numpy(np.float32)
    Z = np.concatenate([m[zcols].to_numpy(np.float32), zn], 1)
    lc = np.full(len(m), np.nan, np.float32)
    ds = m["dataset_id"].to_numpy()
    src = m["label_src"].astype(str).to_numpy()
    # label_src: lodo_oof (교사 데이터 oof → 그 데이터셋을 뺀 fold 교사), final, final_wes<k>
    src = np.where(src == "lodo_oof", np.char.add("lodo_", ds.astype(str)), src)
    for tag in np.unique(src):
        r = src == tag
        lc[r] = fold_logits(params, tag, Z[r])
    m["logit_c"] = lc
    zr_old = m[[f"zr_{n}" for n in names]].to_numpy(np.float32)
    corr = [np.corrcoef(zr_old[hit, i], m.loc[hit, zcols[i]])[0, 1] for i in range(len(names))]
    print("원 zr 과 인과 zc 상관 (스케일별 중앙값): "
          + ", ".join(f"{s}: {np.nanmedian([c for c, n in zip(corr, names) if n.endswith('_' + s)]):.3f}" for s in ("30", "120", "300")),
          flush=True)
    ok = np.isfinite(lc) & np.isfinite(m["logit"].to_numpy(float))
    print(f"logit 과 logit_c 상관 {np.corrcoef(m.loc[ok, 'logit'], lc[ok])[0, 1]:.3f}", flush=True)
    m.drop(columns=["valid_c"]).to_parquet(out / "f2c_labels.parquet")
    shutil.copy(ld / "f2c_params.npz", out / "f2c_params.npz")


if __name__ == "__main__":
    main()
