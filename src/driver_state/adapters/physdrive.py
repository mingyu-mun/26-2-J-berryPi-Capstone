"""PhysDrive (실차 주행, 생체 신호 부분) → 규약 폴더. 상태 머신·학생의 실차 잡음 오경보 점검용.

원본 (labels_stmaps 아카이브를 푼 폴더): source/<subject>/<session>/Label/{ECG,BVP,RESP,SPO2,HR}.mat
  - 모든 신호가 영상 프레임에 맞춘 명목 30 Hz, 기록당 약 5분, 세션 A1·A2·B1·B2·C1·C2
  - 졸음 라벨 없음 → 전부 정상 상태로 간주 (라벨 -1, 오경보 비율만 본다)

변환: 참가자 P<subject>, 세션 이름 그대로. ecg, ppg(BVP) 30 Hz. 세션 순서를 t0 로 (A1 이 가장 앞, 기준 세션).
학술용, 재배포 금지 (레포 밖에 둘 것). MAT 은 scipy.io.loadmat 으로 읽는다 (pickle 아님).
"""
from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from ..data_io import write_session

DATASET_ID = "physdrive"
FS = 30.0
ORDER = ["A1", "A2", "B1", "B2", "C1", "C2"]


def _mat(p: Path, key: str) -> np.ndarray | None:
    from scipy.io import loadmat
    if not p.exists():
        return None
    d = loadmat(p)
    return np.asarray(d[key], float).squeeze() if key in d else None


def convert(src: str | Path, out: str | Path, log=print) -> dict:
    src, out = Path(src), Path(out)
    root = src / "source" if (src / "source").is_dir() else src
    n_ok, skipped = 0, []
    base = datetime(2000, 1, 1, 9, 0, 0)
    for sdir in sorted(p for p in root.iterdir() if p.is_dir()):
        for ses in sorted((p for p in sdir.iterdir() if p.is_dir()), key=lambda p: ORDER.index(p.name) if p.name in ORDER else 99):
            lab = ses / "Label"
            ecg, bvp = _mat(lab / "ECG.mat", "ECG"), _mat(lab / "BVP.mat", "BVP")
            if ecg is None or bvp is None or len(ecg) < FS * 60:
                skipped.append(f"{sdir.name}/{ses.name}: ECG/BVP 없음 또는 60초 미만")
                continue
            n = min(len(ecg), len(bvp))
            k = ORDER.index(ses.name) if ses.name in ORDER else 9
            meta = {"dataset_id": DATASET_ID, "subject_id": f"P{sdir.name}", "session_id": ses.name,
                    "t0": (base + timedelta(hours=k)).isoformat(), "fs": {"ecg": FS, "ppg": FS},
                    "channels": ["ecg", "ppg"], "is_rest": False, "prefiltered": False,
                    "drowsiness_inducer": "none", "sensor": {"ecg": "PhysioLAB Pro (30 Hz 재표본)", "ppg": "PhysioLAB Pro BVP"},
                    "notes": f"PhysDrive {sdir.name}/{ses.name}, {n / FS:.0f}초, 명목 30 Hz, 졸음 라벨 없음"}
            write_session(out / DATASET_ID / meta["subject_id"] / ses.name, meta, {"ecg": ecg[:n], "ppg": bvp[:n]},
                          None, pd.DataFrame([{"t": 0.0, "session": ses.name}]))
            n_ok += 1
    log(f"PhysDrive: 세션 {n_ok}개 → {out / DATASET_ID}")
    for m in skipped:
        log(f"  건너뜀 {m}")
    return {"dataset": DATASET_ID, "sessions": n_ok, "skipped": skipped}
