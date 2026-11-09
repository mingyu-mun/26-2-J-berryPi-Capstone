"""공개 데이터 원본 → 전달 규약 폴더 (docs/data-handoff.md). 원본은 읽기만 한다.

교사 학습용 (라벨 있음)
    python scripts/convert.py advitam --src <Exp4 압축 해제 폴더> --out D:\\data
    python scripts/convert.py mpddf   --src <MPD-DF 폴더>        --out D:\\data
    python scripts/convert.py dddb    --src <DD-Database 폴더>   --out D:\\data
    python scripts/convert.py drozy   --src <DROZY 폴더>         --out D:\\data
교사 라벨링 대상
    python scripts/convert.py maus    --src <MAUS 폴더>          --out D:\\data
    python scripts/convert.py mand    --src <manD extracted>     --out D:\\data [--participants P13 P14] [--dehold]

결과: <out>/<dataset_id>/<subject>/<session>/{meta.json, data.npz, labels_orig.csv}
그다음: python scripts/teacher.py rr --data <out>/advitam <out>/mpddf <out>/dddb <out>/drozy --out cache/rr
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import yaml  # noqa: E402

from driver_state.adapters import ADAPTERS, TEACHER  # noqa: E402
from driver_state.teacher.spec import LabelRule  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dataset", choices=sorted(ADAPTERS))
    ap.add_argument("--src", required=True, help="압축 해제한 원본 폴더")
    ap.add_argument("--out", required=True, help="규약 폴더 루트 (레포 밖 권장)")
    ap.add_argument("--participants", nargs="*", help="manD: 변환할 참가자만 (예: P13 P14)")
    ap.add_argument("--dehold", action="store_true", help="manD: 반복 값(표본 유지)을 바뀐 시점 사이 선형 보간")
    ap.add_argument("--kss-label-sec", type=float, default=300.0, help="AdVitam: 주행 끝 몇 초에 KSS 라벨을 줄지")
    ap.add_argument("--config", default=str(ROOT / "configs/teacher.yaml"))
    a = ap.parse_args()
    if Path(a.out).resolve().is_relative_to(ROOT):
        print("경고: --out 이 레포 안입니다. 데이터가 커밋되지 않게 레포 밖을 권장합니다.")
    mod = ADAPTERS[a.dataset]
    if a.dataset in TEACHER:
        cfg = yaml.safe_load(open(a.config, encoding="utf-8"))
        rule = LabelRule(**{k: v for k, v in (cfg.get("labels") or {}).items()
                            if k in LabelRule.__dataclass_fields__})
        if a.dataset == "advitam":
            mod.convert(a.src, a.out, rule, a.kss_label_sec)
        else:
            mod.convert(a.src, a.out, rule)
    elif a.dataset == "mand":
        mod.convert(a.src, a.out, a.participants, a.dehold)
    else:
        mod.convert(a.src, a.out)


if __name__ == "__main__":
    main()
