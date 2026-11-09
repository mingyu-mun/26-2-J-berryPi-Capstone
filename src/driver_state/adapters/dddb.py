"""DD-Database (운전 시뮬레이터 졸음, 10명 × 2회 × 2시간) → 규약 폴더. 교사 학습용.

원본 (README.md):
  <NN><M|F>_<trial>.edf              O1, O2, C3, C4, LOC, ROC, EKG (모두 128 Hz, uV)
  <NN><M|F>_<trial>_annotations.edf  졸음 버튼 시각 [s] ('Event')
  날짜는 무작위로 바꿔 두었고, 두 회차는 서로 다른 날

변환:
- 참가자 DD01–DD10, 세션 trial1·trial2. 날짜가 가짜라 t0 는 회차 순서만 (회차마다 하루)
- 녹화 시작 30 초 이내, 끝 30 초 이내의 Event 는 시작·끝 표시로 보고 뺀다 (모든 파일에 있음)
- 라벨 (LabelRule): 버튼 ±event_drowsy_sec 졸음, 가장 가까운 버튼에서 event_alert_gap_sec 이상 정상, 사이 -1
- 회차가 다른 날이라 baseline_per_session (회차마다 앞 5 분). 전원 50 Hz (아르헨티나)
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

from ..data_io import write_session
from ..teacher.spec import LabelRule
from .common import finite_segments

DATASET_ID = "dddb"
MARKER_SEC = 30.0


def event_labels(events: np.ndarray, dur: float, rule: LabelRule) -> np.ndarray:
    n = int(np.ceil(dur))
    if len(events) == 0:
        return np.zeros(n, np.int64)
    sec = np.arange(n) + 0.5
    dist = np.min(np.abs(sec[:, None] - np.asarray(events)[None, :]), axis=1)
    lab = np.full(n, -1, np.int64)
    lab[dist <= rule.event_drowsy_sec] = 1
    lab[dist >= rule.event_alert_gap_sec] = 0
    return lab


def convert(src: str | Path, out: str | Path, rule: LabelRule | None = None, log=print) -> dict:
    import pyedflib

    src, out = Path(src), Path(out)
    rule = rule or LabelRule()
    n_ok, skipped = 0, []
    for edf in sorted(src.glob("*.edf")):
        m = re.fullmatch(r"(\d+)([MF])_(\d+)", edf.stem)
        if not m:
            continue
        num, sex, trial = int(m.group(1)), m.group(2), int(m.group(3))
        f = pyedflib.EdfReader(str(edf))
        try:
            labels = f.getSignalLabels()
            name = next((x for x in labels if x.upper() in ("EKG", "ECG")), None)
            if name is None:
                skipped.append(f"{edf.name}: ECG 채널 없음")
                continue
            i = labels.index(name)
            fs = float(f.getSampleFrequency(i))
            ecg = f.readSignal(i)
        finally:
            f.close()
        dur_all = len(ecg) / fs
        ann_path = edf.with_name(f"{edf.stem}_annotations.edf")
        events = np.array([])
        if ann_path.exists():
            a = pyedflib.EdfReader(str(ann_path))
            try:
                events = np.asarray(a.readAnnotations()[0], dtype=float)
            finally:
                a.close()
        else:
            skipped.append(f"{edf.name}: 주석 없음 (전 구간 정상으로 처리됨)")
        used = events[(events > MARKER_SEC) & (events < dur_all - MARKER_SEC)]
        lab_all = event_labels(used, dur_all, rule)
        segs = finite_segments({"ecg": ecg}, fs)
        for j, (a0, seg) in enumerate(segs):
            ses = f"trial{trial}" if len(segs) == 1 else f"trial{trial}_seg{j + 1}"
            off = a0 / fs
            dur = len(seg["ecg"]) / fs
            lab = lab_all[int(off):int(off) + int(np.ceil(dur))]
            orig = pd.DataFrame({"t": events - off, "event": 1,
                                 "used": (events > MARKER_SEC) & (events < dur_all - MARKER_SEC)})
            meta = {
                "dataset_id": DATASET_ID, "subject_id": f"DD{num:02d}", "session_id": ses,
                "t0": float((trial - 1) * 86400 + off), "fs": {"ecg": fs}, "channels": ["ecg"],
                "label_fs": 1, "is_rest": False, "prefiltered": False, "mains_hz": 50.0,
                "baseline_per_session": True, "drowsiness_inducer": "prolonged_driving", "sex": sex,
                "sensor": {"ecg": "AKONIC Neurotrace (DIII)"},
                "notes": f"졸음 버튼 {len(used)}개 (시작·끝 표시 {len(events) - len(used)}개 제외)",
            }
            write_session(out / DATASET_ID / meta["subject_id"] / ses, meta, {"ecg": seg["ecg"]}, lab, orig)
            n_ok += 1
    log(f"DD-Database: 세션 {n_ok}개 → {out / DATASET_ID}")
    for m in skipped:
        log(f"  {m}")
    return {"dataset": DATASET_ID, "sessions": n_ok, "skipped": skipped}
