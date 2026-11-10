"""CLAS (Cognitive Load, Affect and Stress, Mendeley Data) → 규약 폴더. 교사 라벨링 대상 (학생 학습용).

원본 (CLAS/Participants/Part<N>/):
  full_ecg.csv      Timestamp, PythonTimestamp, ecg1, ecg2  (Shimmer ECG, 256 Hz)
  full_gsr_ppg.csv  Timestamp, PythonTimestamp, accelx..z, ppg, gsr  (Shimmer GSR+, 256 Hz)
  두 파일 모두 중간중간 'Start Next Stimulus,{...}' 같은 표시 줄이 끼어 있음 (자극 시작)
  Block_details/Part<N>_Block_Details.csv: 블록 순서와 종류 (Baseline, Math Test, Video clip, Neutral …)

변환:
- 참가자 C<N>, 세션 full 하나. ECG·PPG 를 PythonTimestamp(같은 PC 시계) 로 맞춰 겹치는 구간만,
  256 Hz 격자에 선형 보간. ECG 리드는 ecg1·ecg2 중 신호가 살아 있는 쪽 (ecg1 이 상수로 끊긴 참가자가 많음)
- 'begin' 휴식 화면(60초, 첫 블록 Baseline)을 baseline_range 로. 자극 표시는 labels_orig.csv 에 (졸음 라벨 아님)
- 전원 주파수 50 Hz (인도 수집)
"""
from __future__ import annotations

import ast
import re
from pathlib import Path

import numpy as np
import pandas as pd

from ..data_io import write_session
from .common import finite_segments

DATASET_ID = "clas"
FS = 256.0


def read_shimmer(path: Path, cols: list[str]) -> tuple[pd.DataFrame, list[tuple[float, str]]]:
    """숫자 행만 표로, 표시 줄은 (직전 시각, 내용) 목록으로."""
    rows, events, last_t = [], [], np.nan
    with open(path, encoding="utf-8", errors="replace") as f:
        header = f.readline().strip().split(",")
        idx = [header.index(c) for c in ["PythonTimestamp"] + cols]
        for line in f:
            parts = line.rstrip("\n").split(",")
            try:
                vals = [float(parts[i]) for i in idx]
            except (ValueError, IndexError):
                if line.startswith("Start Next Stimulus"):
                    m = re.search(r"'fname':\s*'([^']*)'", line)
                    events.append((last_t, m.group(1) if m else line[:80]))
                continue
            last_t = vals[0]
            rows.append(vals)
    return pd.DataFrame(rows, columns=["t"] + cols), events


def convert(src: str | Path, out: str | Path, log=print) -> dict:
    src, out = Path(src), Path(out)
    n_ok, skipped = 0, []
    parts = sorted((p for p in (src / "Participants").iterdir() if p.is_dir()),
                   key=lambda p: int(re.sub(r"\D", "", p.name) or 0))
    for pdir in parts:
        num = int(re.sub(r"\D", "", pdir.name))
        fe, fp = pdir / "full_ecg.csv", pdir / "full_gsr_ppg.csv"
        if not fe.exists() or not fp.exists():
            skipped.append(f"{pdir.name}: full_ecg/full_gsr_ppg 없음")
            continue
        e, ev = read_shimmer(fe, ["ecg1", "ecg2"])
        # 리드 선택: ecg1 이 상수(-605)로 끊긴 참가자가 있어, 값이 바뀌는 비율이 높은 리드를 쓴다
        lead = max(("ecg1", "ecg2"), key=lambda c: float((np.diff(e[c].to_numpy()) != 0).mean()))
        e = e.rename(columns={lead: "ecg_sel"})
        p, _ = read_shimmer(fp, ["ppg"])
        if len(e) < FS * 60 or len(p) < FS * 60:
            skipped.append(f"{pdir.name}: 기록이 1분 미만")
            continue
        t0, t1 = max(e.t.iloc[0], p.t.iloc[0]), min(e.t.iloc[-1], p.t.iloc[-1])
        grid = np.arange(t0, t1, 1 / FS)
        sig = {"ecg": np.interp(grid, e.t.to_numpy(), e.ecg_sel.to_numpy()),
               "ppg": np.interp(grid, p.t.to_numpy(), p.ppg.to_numpy())}
        orig = pd.DataFrame([{"t": round(t - t0, 3), "stimulus": name} for t, name in ev if np.isfinite(t)])
        # 기준 구간: 'begin' 화면(휴식 60초). 없으면 녹화 첫 60초
        b = orig[orig.stimulus.str.contains("begin", case=False)] if len(orig) else orig
        b0 = float(b.t.iloc[0]) if len(b) else 0.0
        for j, (a, seg) in enumerate(finite_segments(sig, FS)):
            ses = "full" if j == 0 else f"full_seg{j + 1}"
            br = [max(0.0, b0 - a / FS), max(0.0, b0 - a / FS) + 60.0]
            meta = {"dataset_id": DATASET_ID, "subject_id": f"C{num:02d}", "session_id": ses,
                    "t0": float(t0 + a / FS), "fs": {"ecg": FS, "ppg": FS}, "channels": ["ecg", "ppg"],
                    "is_rest": False, "prefiltered": False, "mains_hz": 50.0, "drowsiness_inducer": "none",
                    "baseline_range": br,
                    "sensor": {"ecg": f"Shimmer ECG ({lead})", "ppg": "Shimmer GSR+ PPG"},
                    "notes": "CLAS 전체 녹화. 'begin' 휴식 화면 60초 = 기준 구간. 자극 표시는 labels_orig"}
            write_session(out / DATASET_ID / meta["subject_id"] / ses, meta, seg, None,
                          orig.assign(t=orig.t - a / FS) if len(orig) else None)
            n_ok += 1
        log(f"  {pdir.name}: 완료 ({len(grid) / FS / 60:.1f}분)")
    log(f"CLAS: 세션 {n_ok}개 → {out / DATASET_ID}")
    for m in skipped:
        log(f"  건너뜀 {m}")
    return {"dataset": DATASET_ID, "sessions": n_ok, "skipped": skipped}
