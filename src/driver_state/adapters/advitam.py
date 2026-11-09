"""AdVitam Exp4 (조건부 자율주행, 경미한 수면 박탈, 63명) → 규약 폴더. 교사 학습용.

원본 (Exp4_README.md, Exp4.zip 에서 Raw/Physio/Txt 와 Preprocessed/Questionnaire 만 풀어 둠):
  Raw/Physio/Txt/<code>.txt          BioPac 1000 Hz, 탭 구분. 머리말 뒤 'min CH1 CH2 CH3' (EDA, ECG, RESP).
                                     ECG 는 장비에서 0.5–35 Hz 로 걸러진 값. 첫 열은 경과 분
  Raw/Physio/Txt/<code>-markers.txt  구간 표시 (초): Baseline / Training / Block1 / Block2 Start·End
  Preprocessed/Questionnaire/Exp4_Database.csv
      KSS_0   실험 전, KSS_B_k  k 번째 주행 끝(인계 요청 직전) 상태를 주행 뒤에 평가, KSS_k 질문 뒤
  code 예: 01_AC16 → A(정상 수면)/D(수면 박탈), C/U(첫 시나리오), 10/16(실험 시각)

변환:
- 참가자 = code, 세션 baseline (is_rest, 자율주행 관찰 5 분) · block1 · block2. Training(수동 주행)은 뺌
- 라벨: block k 의 마지막 kss_label_sec(기본 5 분)에 KSS_B_k 를 LabelRule.kss 로 (≥7 졸음, ≤5 정상), 나머지 -1
- t0 는 같은 녹화 안의 구간 시작 [s] (순서만 의미). 전원 50 Hz (스위스)
- 수면 박탈 여부는 drowsiness_inducer (D → sleep_deprivation, A → none) 와 meta.label_sleep 에 둔다
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

from ..data_io import write_session
from ..teacher.spec import LabelRule
from .common import finite_segments

DATASET_ID = "advitam"


def read_markers(path: Path) -> dict[str, float]:
    out = {}
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        m = re.match(r"Event \d+:\s+([\d.]+)\s+(.+?),", line)
        if m:
            out[m.group(2).strip()] = float(m.group(1))
    return out


def read_physio(path: Path) -> tuple[np.ndarray, float]:
    """ECG 열과 fs 를 돌려준다. 머리말에서 채널 순서와 표본 간격을 읽는다."""
    names, dt_ms, skip = [], None, 0
    with open(path, encoding="utf-8", errors="replace") as f:
        for i, line in enumerate(f):
            s = line.strip()
            m = re.match(r"([\d.]+)\s*msec/sample", s)
            if m:
                dt_ms = float(m.group(1))
            if re.match(r"(EDA|ECG|Respiration|RESP)", s, re.I):
                names.append(s.split()[0].upper())
            if re.match(r"(min|hrs|sec)\s+CH1", s):   # 시간 단위가 파일마다 다름 (53_AC16 은 hrs)
                skip = i + 2          # 다음 줄은 채널별 표본 수
                break
    if "ECG" not in names or not skip:
        raise KeyError(f"{path.name}: ECG 채널 또는 데이터 시작 줄을 찾지 못함 ({names})")
    col = 1 + names.index("ECG")
    x = pd.read_csv(path, sep="\t", skiprows=skip, header=None, usecols=[col], dtype=np.float64,
                    engine="c").iloc[:, 0].to_numpy()
    return x, 1000.0 / (dt_ms or 1.0)


def convert(src: str | Path, out: str | Path, rule: LabelRule | None = None,
            kss_label_sec: float = 300.0, log=print) -> dict:
    src, out = Path(src), Path(out)
    rule = rule or LabelRule()
    db = pd.read_csv(src / "Preprocessed" / "Questionnaire" / "Exp4_Database.csv", sep=None, engine="python")
    db.columns = [c.lstrip("﻿") for c in db.columns]
    db = db.set_index("participant_code")
    n_ok, skipped = 0, []
    txt = src / "Raw" / "Physio" / "Txt"
    for mk in sorted(txt.glob("*-markers.txt")):
        code = mk.name[: -len("-markers.txt")]
        phys = txt / f"{code}.txt"
        if not phys.exists():
            skipped.append(f"{code}: 신호 파일 없음")
            continue
        marks = read_markers(mk)
        try:
            ecg, fs = read_physio(phys)
        except KeyError as e:
            skipped.append(str(e))
            continue
        row = db.loc[code] if code in db.index else None
        sd = code.split("_")[1][0] if "_" in code else "?"
        periods = [("baseline", "Baseline", None)] + [(f"block{k}", f"Block{k}", f"KSS_B_{k}") for k in (1, 2, 3)]
        for ses, key, kss_col in periods:
            a, b = marks.get(f"{key} Start"), marks.get(f"{key} End")
            if a is None or b is None or b <= a:
                if key != "Block3":
                    skipped.append(f"{code}/{ses}: 구간 표시 없음")
                continue
            i0, i1 = int(round(a * fs)), min(int(round(b * fs)), len(ecg))
            kss = float(row[kss_col]) if (row is not None and kss_col and kss_col in row and pd.notna(row[kss_col])) else np.nan
            for j, (o, seg) in enumerate(finite_segments({"ecg": ecg[i0:i1]}, fs)):
                name = ses if j == 0 else f"{ses}_seg{j + 1}"
                dur = len(seg["ecg"]) / fs
                lab = None
                orig = None
                if kss_col:
                    lab = np.full(int(np.ceil(dur)), -1, np.int64)
                    end_rel = (i1 - i0) / fs - o / fs          # 구간 끝 (이 조각 기준 초)
                    lo = max(0, int(np.floor(end_rel - kss_label_sec)))
                    hi = min(len(lab), int(np.ceil(end_rel)))
                    if hi > lo:
                        lab[lo:hi] = rule.kss(kss)
                    orig = pd.DataFrame([{"t": end_rel, "kss": kss, "item": kss_col}])
                meta = {
                    "dataset_id": DATASET_ID, "subject_id": code, "session_id": name,
                    "t0": float(a + o / fs), "fs": {"ecg": fs}, "channels": ["ecg"],
                    "label_fs": 1 if lab is not None else None, "is_rest": ses == "baseline",
                    "prefiltered": False, "mains_hz": 50.0,
                    "drowsiness_inducer": "sleep_deprivation" if sd == "D" else "none",
                    "label_sleep": 1 if sd == "D" else 0,
                    "sensor": {"ecg": "BioPac MP36 (0.5–35 Hz 장비 필터)"},
                    "notes": f"AdVitam Exp4 {code} {key}. 라벨은 구간 끝 {kss_label_sec:.0f}s 에 {kss_col}",
                }
                write_session(out / DATASET_ID / code / name, meta, {"ecg": seg["ecg"]}, lab, orig)
                n_ok += 1
        log(f"  {code}: 완료")
    log(f"AdVitam: 세션 {n_ok}개 → {out / DATASET_ID}")
    for m in skipped:
        log(f"  {m}")
    return {"dataset": DATASET_ID, "sessions": n_ok, "skipped": skipped}
