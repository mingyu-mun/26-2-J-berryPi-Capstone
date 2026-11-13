"""MAUS (Mental workload, IEEE DataPort) → 규약 폴더. 교사 라벨링 대상 (학생 학습용).

원본 (Data/Raw_data/<id>/):
  inf_ecg.csv, inf_ppg.csv  열 6개 = Trial 1–6 (n-back 0/2/3), 256 Hz × 300 초 (76 800 행)
  inf_resting.csv           Resting_ECG, Resting_PPG, Resting_GSR (256 Hz, 약 5 분)
  record.xlsx               블록별 n-back 응답 시각 (time, time.1 … time.5) → 시험 시작 시각
  (pixart*.csv 손목 PPG 는 쓰지 않음)

변환:
- 참가자 M<id>, 세션 rest (is_rest = 기준 구간) + trial1–trial6
- t0: trial 은 record.xlsx 블록 첫 응답 시각, rest 는 기록이 없어 trial1 보다 1 시간 앞으로 둠 (순서만 의미)
- n-back 수준과 NASA-TLX 는 labels_orig.csv 와 meta 에만 (졸음 라벨 아님)
- 전원 주파수는 문서에서 확인하지 못해 mains_hz 를 넣지 않음 (기본 노치)
"""
from __future__ import annotations

import re
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

from ..data_io import write_session
from .common import finite_segments

DATASET_ID = "maus"
FS = 256.0


def _ts(v) -> datetime | None:
    m = re.fullmatch(r"(\d+)_(\d+)_(\d+)_(\d+)_(\d+)_(\d+)(?:\.(\d+))?", str(v).strip())
    if not m:
        return None
    y, mo, d, h, mi, s = (int(x) for x in m.groups()[:6])
    ms = int((m.group(7) or "0").ljust(3, "0")[:3])
    return datetime(y, mo, d, h, mi, s, ms * 1000)


def _trial_starts(raw: Path) -> list[datetime | None]:
    p = raw / "record.xlsx"
    if not p.exists():
        return [None] * 6
    df = pd.read_excel(p, nrows=1)
    cols = ["time"] + [f"time.{i}" for i in range(1, 6)]
    return [_ts(df[c].iloc[0]) if c in df else None for c in cols]


def _tlx(src: Path, sid: str) -> dict[int, float]:
    p = src / "Subjective_rating" / sid / "NASA_TLX.csv"
    if not p.exists():
        return {}
    df = pd.read_csv(p)
    df = df[df["Scale Title"].isin(["Mental Demand", "Physical Demand", "Temporal Demand",
                                    "Performance", "Effort", "Frustration"])]
    out = {}
    for c in df.columns:
        m = re.match(r"Trial (\d+)", c)
        if m:
            out[int(m.group(1))] = float(pd.to_numeric(df[c], errors="coerce").mean())
    return out


def _write(out: Path, sid: str, ses: str, t0: datetime, sig: dict, extra: dict, orig: pd.DataFrame) -> int:
    n = 0
    segs = finite_segments(sig, FS)
    for j, (a, seg) in enumerate(segs):
        name = ses if len(segs) == 1 else f"{ses}_seg{j + 1}"
        meta = {"dataset_id": DATASET_ID, "subject_id": f"M{sid}", "session_id": name,
                "t0": (t0 + timedelta(seconds=a / FS)).isoformat(), "fs": {k: FS for k in seg},
                "channels": list(seg), "prefiltered": False, "drowsiness_inducer": "none",
                "sensor": {"ecg": "Infiniti (MAUS)", "ppg": "Infiniti fingertip PPG"}} | extra
        write_session(out / DATASET_ID / f"M{sid}" / name, meta, seg, None, orig)
        n += 1
    return n


def convert(src: str | Path, out: str | Path, log=print) -> dict:
    src, out = Path(src), Path(out)
    raw_root = src / "Data" / "Raw_data"
    n_ok, skipped = 0, []
    for raw in sorted(p for p in raw_root.iterdir() if p.is_dir()):
        sid = raw.name
        starts = _trial_starts(raw)
        tlx = _tlx(src, sid)
        first = next((t for t in starts if t), datetime(2000, 1, 1))
        rest = raw / "inf_resting.csv"
        if rest.exists():
            r = pd.read_csv(rest)
            k = _write(out, sid, "rest", first - timedelta(hours=1),
                       {"ecg": r["Resting_ECG"].to_numpy(float), "ppg": r["Resting_PPG"].to_numpy(float)},
                       {"is_rest": True, "notes": "MAUS 휴식 녹화"}, pd.DataFrame([{"t": 0.0, "nback": "rest"}]))
            if not k:
                skipped.append(f"{sid}: 휴식 녹화가 비었거나 60초 미만 (기준 구간은 가장 이른 trial 앞부분)")
            n_ok += k
        else:
            skipped.append(f"{sid}: inf_resting.csv 없음")
        if not (raw / "inf_ecg.csv").exists() or not (raw / "inf_ppg.csv").exists():
            skipped.append(f"{sid}: inf_ecg/inf_ppg 없음")
            continue
        ecg, ppg = pd.read_csv(raw / "inf_ecg.csv"), pd.read_csv(raw / "inf_ppg.csv")
        for i, col in enumerate(ecg.columns):
            k = i + 1
            nb = col.split(":")[-1].strip()            # '0back' 등
            t0 = starts[i] if i < len(starts) and starts[i] else first + timedelta(minutes=8 * i)
            orig = pd.DataFrame([{"t": 0.0, "nback": nb, "nasa_tlx_raw": tlx.get(k)}])
            n_ok += _write(out, sid, f"trial{k}", t0,
                           {"ecg": ecg[col].to_numpy(float), "ppg": ppg[ppg.columns[i]].to_numpy(float)},
                           {"is_rest": False, "nback": nb, "notes": f"MAUS {col}"}, orig)
    log(f"MAUS: 세션 {n_ok}개 → {out / DATASET_ID}")
    for m in skipped:
        log(f"  건너뜀 {m}")
    return {"dataset": DATASET_ID, "sessions": n_ok, "skipped": skipped}
