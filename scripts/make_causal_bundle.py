"""인과(실시간) 학생 번들 만들기. 기존 models/ 는 그대로 두고 models_causal/ 에 새로 만든다.

    python scripts/make_causal_bundle.py --student-dir out/student_causal_final \
        --normalizer out/f2c_label_c/normalizer_c.json --out models_causal

교사 파일(f2c_final.npz, nw_pca.npz)과 NormWear 정보는 기존 번들에서 복사한다 (교사는 오프라인 라벨러라 바꾸지 않음).
바뀌는 것: spec.causal = true (infer_bundle·selfmeasure_eval 이 인과 필터·과거 전용 RR 창을 씀), 정규화 통계, 학생 가중치, 동작점.
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--student-dir", required=True, help="student_train --final 결과 (*.pt, thresholds.json)")
    ap.add_argument("--normalizer", required=True)
    ap.add_argument("--base", default=str(ROOT / "models"))
    ap.add_argument("--out", default=str(ROOT / "models_causal"))
    ap.add_argument("--version", default="causal-1")
    a = ap.parse_args()
    import torch
    from safetensors.torch import save_file
    base, out = Path(a.base), Path(a.out)
    (out / "student").mkdir(parents=True, exist_ok=True)
    shutil.copytree(base / "teacher", out / "teacher", dirs_exist_ok=True)
    for p in sorted(Path(a.student_dir).glob("S1-*_s*.pt")):
        sd = torch.load(p, map_location="cpu", weights_only=True)
        save_file({k: v.contiguous() for k, v in sd.items()}, str(out / "student" / f"{p.stem}.safetensors"))
    B = json.loads((base / "bundle.json").read_text(encoding="utf-8"))
    B["version"] = a.version
    B["spec"]["causal"] = True
    B["normalizer"] = json.loads(Path(a.normalizer).read_text(encoding="utf-8"))
    B["thresholds"]["student_logit"] = json.loads((Path(a.student_dir) / "thresholds.json").read_text())
    B["causal_note"] = ("실시간 학생: 창 [t, t+30] 의 RR 특징은 [t+30−L, t+30] 만, 필터·리샘플은 인과. "
                        "teacher_logit 동작점과 교사 파일은 오프라인 교사 그대로.")
    (out / "bundle.json").write_text(json.dumps(B, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"번들 저장: {out}, 학생 {len(list((out / 'student').glob('*.safetensors')))}개")


if __name__ == "__main__":
    main()
