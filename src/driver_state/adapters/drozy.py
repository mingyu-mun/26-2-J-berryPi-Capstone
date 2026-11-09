"""DROZY (ULg Multimodality Drowsiness Database) → 규약 폴더. 교사 학습용.

원본 (README.md):
- psg/<S>-<T>.edf : 512 Hz PSG, 'ECG' 채널 사용 (1-1, 2-1 은 Oz 채널이 하나 더 있어 이름으로 찾음)
- KSS.txt         : 14 줄(참가자) × 3 열(테스트), 0 = 기록 없음
- 테스트 = 10 분 PVT 3 회. 수면 박탈이 진행되며 1 → 3 으로 졸려짐.
- 빠진 테스트: 9-1, 10-2, 12-2, 12-3, 13-3 (유실), 7-1 (미실시)

변환:
- 참가자 D01–D14, 세션 test1–test3, t0 = EDF 시작 시각 → 가장 이른 test1 앞 5 분이 기준 구간
- KSS 는 테스트 하나에 한 번이라 세션 전체에 같은 이진 라벨 (KSS ≥ 7 졸음, ≤ 5 정상, 6 제외)
- labels_orig.csv 에 원 KSS 보존, 전원 50 Hz (벨기에)
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ..data_io import write_session
from ..teacher.spec import LabelRule
from .common import finite_segments

DATASET_ID = "drozy"


def read_kss(src: Path) -> np.ndarray:
    k = np.loadtxt(src / "KSS.txt", dtype=float, ndmin=2)
    k[k <= 0] = np.nan
    return k


def convert(src: str | Path, out: str | Path, rule: LabelRule | None = None, log=print) -> dict:
    import pyedflib

    src, out = Path(src), Path(out)
    rule = rule or LabelRule()
    kss = read_kss(src)
    n_ok, skipped = 0, []
    for edf in sorted((src / "psg").glob("*.edf"), key=lambda p: tuple(int(x) for x in p.stem.split("-"))):
        s, t = (int(x) for x in edf.stem.split("-"))
        f = pyedflib.EdfReader(str(edf))
        try:
            labels = f.getSignalLabels()
            if "ECG" not in labels:
                skipped.append(f"{edf.name}: ECG 채널 없음")
                continue
            i = labels.index("ECG")
            fs = float(f.getSampleFrequency(i))
            ecg = f.readSignal(i)
            t0 = f.getStartdatetime()
        finally:
            f.close()
        v = kss[s - 1, t - 1] if s - 1 < len(kss) else np.nan
        segs = finite_segments({"ecg": ecg}, fs)
        for j, (a, seg) in enumerate(segs):
            dur = len(seg["ecg"]) / fs
            lab = np.full(int(np.ceil(dur)), rule.kss(v), np.int64)
            ses = f"test{t}" if len(segs) == 1 else f"test{t}_seg{j + 1}"
            meta = {
                "dataset_id": DATASET_ID, "subject_id": f"D{s:02d}", "session_id": ses,
                "t0": (pd.Timestamp(t0) + pd.Timedelta(seconds=a / fs)).isoformat(),
                "fs": {"ecg": fs}, "channels": ["ecg"], "label_fs": 1, "is_rest": False,
                "prefiltered": False, "mains_hz": 50.0,
                "drowsiness_inducer": "none" if t == 1 else "sleep_deprivation",
                "sensor": {"ecg": "PSG (DROZY)"},
                "notes": f"DROZY {edf.stem}, 10분 PVT. KSS={'' if v != v else int(v)} 를 세션 전체에 적용",
            }
            orig = pd.DataFrame([{"t": dur, "kss": v, "test": t}])
            write_session(out / DATASET_ID / meta["subject_id"] / ses, meta, {"ecg": seg["ecg"]}, lab, orig)
            n_ok += 1
    log(f"DROZY: 세션 {n_ok}개 → {out / DATASET_ID}")
    for m in skipped:
        log(f"  건너뜀 {m}")
    return {"dataset": DATASET_ID, "sessions": n_ok, "skipped": skipped}
