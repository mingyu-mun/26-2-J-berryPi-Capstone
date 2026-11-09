"""manD 1.0 (Harvard Dataverse, doi:10.7910/DVN/SG9TMD) → 규약 폴더. 교사 라벨링 대상 (학생 학습용).

원본 (압축 해제본 extracted/):
  P##/<Session>[/<part>]/DriverState/PhysiologicalData/
    EEG_ECG.txt  CSV, time[s] + 'BIOPAC/ECG' (256 Hz). 앞부분 행은 기록 시작 전이라 비어 있을 수 있음
    PPG_EDA.txt  CSV, time[s] + 'Empatica/BVP' (E4 64 Hz 를 256 Hz 로 올린 값)
  Session = Familiarization | Anger | Fear | NoEmotion | Sadness | Surprise
  Familiarization 은 일부 참가자에서 01/02/03 파트로 나뉨

변환:
- ECG·BVP 를 time 열(1/256 s 격자)로 맞춘 뒤, 둘 다 있는 샘플만 사용. 0.2 초 이하 결측은 보간, 그보다
  긴 결측에서는 세션을 나눔 (_seg2 …). 60 초 미만 조각은 버림.
- EEG_ECG.txt 가 없는 세션은 교사 라벨을 만들 수 없으므로 건너뜀.
- 실제 주행 순서 정보가 없어 t0 는 상대값: Familiarization(파트 순) 이 먼저, 감정 시나리오는 그 뒤.
  → 개인 기준 구간은 적응 주행(Familiarization) 앞부분.
- 감정 시나리오 이름은 labels_orig.csv 와 meta.scenario 에만 둔다 (졸음 라벨 아님). 전원 50 Hz (독일).

주의 (2026-11-09 확인): 256 Hz 라고 되어 있지만 같은 값이 반복된 표본 유지 신호다. 값이 바뀌는 실효 주기가
ECG 약 28–41 Hz, BVP 약 6 Hz 라 R-peak 시각이 ±15–30 ms 흔들린다. 교사 rr/features 에서 유효 창이
0.1% (dehold 보간 후에도 0.5%) 뿐이었다. meta.effective_hz 에 세션별 실효 주기를 남긴다.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ..data_io import write_session
from .common import finite_segments

DATASET_ID = "mand"
FS = 256.0
ORDER = ("Familiarization", "NoEmotion", "Anger", "Fear", "Sadness", "Surprise")


def _read(path: Path, col: str) -> pd.Series:
    df = pd.read_csv(path, usecols=["time", col])
    df = df[np.isfinite(df["time"])]
    k = np.round(df["time"].to_numpy() * FS).astype(np.int64)
    s = pd.Series(df[col].to_numpy(dtype=np.float64), index=k)
    return s[~s.index.duplicated(keep="first")]


def dehold(x: np.ndarray) -> np.ndarray:
    """표본 유지(같은 값 반복)로 256 Hz 에 맞춘 신호를, 값이 바뀐 시점만 남겨 선형 보간한다.

    manD 는 ECG 가 실효 약 30–40 Hz, BVP 가 약 6 Hz 로 기록된 뒤 256 Hz 격자에 반복 값으로 채워져 있다.
    계단 모양이 R-peak 검출을 흔들어서, 바뀐 시점 사이를 직선으로 잇는다 (정보가 늘지는 않음).
    """
    x = np.asarray(x, dtype=np.float64)
    ok = np.isfinite(x)
    ch = np.flatnonzero(ok & np.concatenate([[True], x[1:] != x[:-1]]))
    if len(ch) < 2:
        return x
    out = np.interp(np.arange(len(x)), ch, x[ch])
    out[~ok] = np.nan
    return out


def effective_hz(x: np.ndarray) -> float:
    x = x[np.isfinite(x)]
    return float(FS * (np.diff(x) != 0).mean()) if len(x) > 1 else 0.0


def _sessions(src: Path):
    """(참가자, 세션 id, 정렬 키, 시나리오, 생리 폴더)"""
    for phys in sorted(src.glob("P*/**/DriverState/PhysiologicalData")):
        rel = phys.relative_to(src).parts      # P##, Session, [part], DriverState, PhysiologicalData
        subj, scen = rel[0], rel[1]
        part = rel[2] if len(rel) == 5 else None
        ses = scen if part is None else f"{scen}_{part}"
        rank = ORDER.index(scen) if scen in ORDER else len(ORDER)
        yield subj, ses, (rank, part or ""), scen, phys


def convert(src: str | Path, out: str | Path, participants: list[str] | None = None,
            do_dehold: bool = False, log=print) -> dict:
    src, out = Path(src), Path(out)
    by_subj: dict[str, list] = {}
    for item in _sessions(src):
        if participants and item[0] not in participants:
            continue
        by_subj.setdefault(item[0], []).append(item)
    n_ok, skipped = 0, []
    for subj in sorted(by_subj, key=lambda p: int(p[1:])):
        items = sorted(by_subj[subj], key=lambda it: it[2])
        for order, (_, ses, _, scen, phys) in enumerate(items):
            fe, fp = phys / "EEG_ECG.txt", phys / "PPG_EDA.txt"
            if not fe.exists():
                skipped.append(f"{subj}/{ses}: EEG_ECG.txt 없음")
                continue
            ecg = _read(fe, "BIOPAC/ECG")
            if fp.exists():
                bvp = _read(fp, "Empatica/BVP")
                lo, hi = max(ecg.index.min(), bvp.index.min()), min(ecg.index.max(), bvp.index.max())
            else:
                bvp, (lo, hi) = None, (ecg.index.min(), ecg.index.max())
            if hi <= lo:
                skipped.append(f"{subj}/{ses}: ECG·BVP 시간 겹침 없음")
                continue
            grid = np.arange(lo, hi + 1)
            sig = {"ecg": ecg.reindex(grid).to_numpy()}
            if bvp is not None:
                sig["ppg"] = bvp.reindex(grid).to_numpy()
            eff = {k: round(effective_hz(v), 1) for k, v in sig.items()}
            if do_dehold:
                sig = {k: dehold(v) for k, v in sig.items()}
            segs = finite_segments(sig, FS)
            if not segs:
                skipped.append(f"{subj}/{ses}: 60초 이상 연속 구간 없음")
                continue
            for j, (a, seg) in enumerate(segs):
                sid = ses if len(segs) == 1 else f"{ses}_seg{j + 1}"
                t_file = (lo + a) / FS
                meta = {
                    "dataset_id": DATASET_ID, "subject_id": subj, "session_id": sid,
                    "t0": float(order * 86400 + t_file), "fs": {k: FS for k in seg},
                    "channels": list(seg), "is_rest": False, "prefiltered": False, "mains_hz": 50.0,
                    "drowsiness_inducer": "none", "scenario": scen, "t_file_start": t_file,
                    "effective_hz": eff, "dehold": bool(do_dehold),
                    "sensor": {"ecg": "BIOPAC", "ppg": "Empatica E4 BVP (64 Hz → 256 Hz)"},
                    "notes": "t0 는 상대값 (Familiarization 먼저). t_file_start = 원 파일 time 열 기준 시작 [s]",
                }
                orig = pd.DataFrame([{"t": 0.0, "scenario": scen}])
                write_session(out / DATASET_ID / subj / sid, meta, seg, None, orig)
                n_ok += 1
        log(f"  {subj}: 완료")
    log(f"manD: 세션 {n_ok}개 → {out / DATASET_ID}")
    for m in skipped:
        log(f"  건너뜀 {m}")
    return {"dataset": DATASET_ID, "sessions": n_ok, "skipped": skipped}
