"""MPD-DF (장시간 주행 피로 PSG) → 규약 폴더. 교사 학습용.

원본:
  MPDDF_raw_<NN>_PSG.edf         E1-M2, E2-M2 (256 Hz), ECG (1024 Hz, mV), Thor (32 Hz). 약 2 시간
  MPDDF_raw_<NN>_Annotation.txt  "시각,30초 에폭 번호,피로 단계" 의 변화점 목록. 단계 0–4, 그 밖에
                                 'Severe Artifacts', 'Signal Abnormality' 같은 표시가 있음
  Questionnaire information.xlsx 성별·나이·MEQ·PSQI (meta 로 옮기지 않음)

변환:
- 참가자 F01–F50, 세션 drive 하나. t0 = EDF 시작 시각
- 변화점 시각(EDF 시작 기준 초)부터 다음 변화점 전까지 같은 단계. 첫 변화점 전은 -1
- 단계 → 이진: LabelRule.fatigue (기본 0 정상, 3–4 졸음, 1–2 제외). 잡음 표시 구간은 -1
- 기준 구간은 기본 규칙 (세션 앞 5 분). 전원 50 Hz (중국)
"""
from __future__ import annotations

import re
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from ..data_io import write_session
from ..teacher.spec import LabelRule
from .common import finite_segments

DATASET_ID = "mpddf"


def read_annotation(path: Path, start: datetime) -> pd.DataFrame:
    """변화점 → DataFrame(t 초, clock, epoch, value). 자정을 넘으면 하루를 더한다."""
    rows = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 3 or not re.fullmatch(r"\d{1,2}:\d{2}:\d{2}", parts[0]):
            continue
        h, m, s = (int(x) for x in parts[0].split(":"))
        t = start.replace(hour=h, minute=m, second=s, microsecond=0)
        if t < start - timedelta(hours=12):
            t += timedelta(days=1)
        rows.append({"t": (t - start).total_seconds(), "clock": parts[0], "epoch": parts[1], "value": parts[2]})
    return pd.DataFrame(rows, columns=["t", "clock", "epoch", "value"])


def per_second_labels(ann: pd.DataFrame, dur: float, rule: LabelRule) -> np.ndarray:
    n = int(np.ceil(dur))
    lab = np.full(n, -1, np.int64)
    ann = ann.sort_values("t").reset_index(drop=True)
    for i, r in ann.iterrows():
        a = max(int(np.floor(r["t"])), 0)
        b = int(np.floor(ann["t"].iloc[i + 1])) if i + 1 < len(ann) else n
        v = str(r["value"])
        lab[a:max(a, min(b, n))] = rule.fatigue(int(v)) if v.isdigit() else -1
    return lab


def convert(src: str | Path, out: str | Path, rule: LabelRule | None = None, log=print) -> dict:
    import pyedflib

    src, out = Path(src), Path(out)
    rule = rule or LabelRule()
    n_ok, skipped = 0, []
    for edf in sorted(src.glob("MPDDF_raw_*_PSG.edf")):
        num = int(re.search(r"raw_(\d+)_PSG", edf.name).group(1))
        ann_path = src / f"MPDDF_raw_{num:02d}_Annotation.txt"
        f = pyedflib.EdfReader(str(edf))
        try:
            labels = f.getSignalLabels()
            if "ECG" not in labels:
                skipped.append(f"{edf.name}: ECG 채널 없음")
                continue
            i = labels.index("ECG")
            fs = float(f.getSampleFrequency(i))
            ecg = f.readSignal(i)
            start = f.getStartdatetime()
        finally:
            f.close()
        if not ann_path.exists():
            skipped.append(f"{edf.name}: 주석 파일 없음 (라벨 없이 변환)")
            ann = pd.DataFrame(columns=["t", "clock", "epoch", "value"])
        else:
            ann = read_annotation(ann_path, start)
        lab_all = per_second_labels(ann, len(ecg) / fs, rule)
        segs = finite_segments({"ecg": ecg}, fs)
        for j, (a, seg) in enumerate(segs):
            ses = "drive" if len(segs) == 1 else f"drive_seg{j + 1}"
            off = a / fs
            dur = len(seg["ecg"]) / fs
            lab = lab_all[int(off):int(off) + int(np.ceil(dur))]
            orig = ann.assign(t=ann["t"] - off)
            meta = {
                "dataset_id": DATASET_ID, "subject_id": f"F{num:02d}", "session_id": ses,
                "t0": (start + timedelta(seconds=off)).isoformat(), "fs": {"ecg": fs}, "channels": ["ecg"],
                "label_fs": 1, "is_rest": False, "prefiltered": False, "mains_hz": 50.0,
                "drowsiness_inducer": "prolonged_driving", "sensor": {"ecg": "PSG (MPD-DF)"},
                "notes": "피로 단계 변화점을 다음 변화점까지 유지해 초 단위 라벨로 펼침",
            }
            write_session(out / DATASET_ID / meta["subject_id"] / ses, meta, {"ecg": seg["ecg"]}, lab, orig)
            n_ok += 1
    log(f"MPD-DF: 세션 {n_ok}개 → {out / DATASET_ID}")
    for m in skipped:
        log(f"  {m}")
    return {"dataset": DATASET_ID, "sessions": n_ok, "skipped": skipped}
