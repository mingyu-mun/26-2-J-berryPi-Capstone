"""UL-DD (주행 시뮬레이터 졸음, Empatica E4 등) → 규약 폴더. 학생 PPG 경로 검증용 (docs/dataset-ul-dd.md).

원본 (CSV_Files/CSV_Files/<참가자 A~S>/<A|D>/<p>_<채널>_<s>.csv), 머리글 없음:
  BVP 64 Hz (손목 PPG), HR 1 Hz, ACC 32 Hz, EDA·TEMP 4 Hz, O2M 0.5 Hz, LGP·RGP 3 Hz,
  Labels: KSS 10개 (40분 세션, 4분마다), Telemetry: 시뮬레이터 60 Hz (D 와 대부분의 A)

변환:
- 참가자 U<id>, 세션 alert (A) / drowsy (D). ppg 채널만 (ECG 없음). IBI 파일은 잘려 있어 쓰지 않는다.
- KSS i 번째 값을 [240·i, 240·(i+1)) 초 구간에 적용 → 1 Hz 이진 라벨 (LabelRule.kss: ≥7 졸음, ≤5 정상, 6 제외)
- 시작 시각이 없어 alert 를 drowsy 보다 하루 앞으로 둔다 (순서만 의미). 기준 구간은 alert 세션 앞부분.
- 원 KSS 는 labels_orig.csv, 텔레메트리는 meta 의 경로 메모만 (재배포 금지 데이터, 레포 밖에 둘 것)
"""
from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from ..data_io import write_session
from ..teacher.spec import LabelRule

DATASET_ID = "uldd"
FS_BVP = 64.0
KSS_SEC = 240.0


def _root(src: Path) -> Path:
    for c in (src / "CSV_Files" / "CSV_Files", src / "CSV_Files", src):
        if c.is_dir() and any(p.is_dir() and len(p.name) == 1 for p in c.iterdir()):
            return c
    raise FileNotFoundError(f"참가자 폴더(A~S)를 찾지 못함: {src}")


def convert(src: str | Path, out: str | Path, rule: LabelRule | None = None, log=print) -> dict:
    src, out = Path(src), Path(out)
    rule = rule or LabelRule()
    root = _root(src)
    n_ok, skipped = 0, []
    base = datetime(2000, 1, 1, 9, 0, 0)
    for pdir in sorted(p for p in root.iterdir() if p.is_dir()):
        pid = pdir.name
        for s, ses, day in (("A", "alert", 0), ("D", "drowsy", 1)):
            d = pdir / s
            f = d / f"{pid}_BVP_{s}.csv"
            if not f.exists() or f.stat().st_size == 0:
                skipped.append(f"{pid}/{s}: BVP 없음")
                continue
            bvp = pd.read_csv(f, header=None)[0].to_numpy(float)
            dur = len(bvp) / FS_BVP
            kf = d / f"{pid}_Labels_{s}.csv"
            kss = np.array([float(x) for x in kf.read_text().strip().split(",")]) if kf.exists() else np.array([])
            lab = np.full(int(np.ceil(dur)), -1, np.int64)
            for i, v in enumerate(kss):
                a, b = int(i * KSS_SEC), int(min((i + 1) * KSS_SEC, len(lab)))
                lab[a:b] = rule.kss(v)
            meta = {"dataset_id": DATASET_ID, "subject_id": f"U{pid}", "session_id": ses,
                    "t0": (base + timedelta(days=day)).isoformat(), "fs": {"ppg": FS_BVP}, "channels": ["ppg"],
                    "label_fs": 1, "is_rest": False, "prefiltered": False,
                    "drowsiness_inducer": "none" if s == "A" else "sleep_deprivation",
                    "sensor": {"ppg": "Empatica E4 BVP (wrist)"},
                    "baseline_per_session": False,
                    "notes": f"UL-DD {pid}/{s}, {dur / 60:.1f}분. KSS 4분마다 {len(kss)}개. 텔레메트리: {d.name}/{pid}_Telemetry_{s}.csv"}
            orig = pd.DataFrame({"t": np.arange(len(kss)) * KSS_SEC, "t_end": (np.arange(len(kss)) + 1) * KSS_SEC, "kss": kss})
            write_session(out / DATASET_ID / f"U{pid}" / ses, meta, {"ppg": bvp}, lab, orig)
            n_ok += 1
    log(f"UL-DD: 세션 {n_ok}개 → {out / DATASET_ID}")
    for m in skipped:
        log(f"  건너뜀 {m}")
    return {"dataset": DATASET_ID, "sessions": n_ok, "skipped": skipped}
