"""WESAD (Wearable Stress and Affect Detection, UCI) → 규약 폴더. 교사 라벨링 대상 (학생 학습용).

원본 (WESAD/S<N>/):  ※ S<N>.pkl 은 열지 않는다 (pickle 은 임의 코드 실행 위험, 같은 신호가 txt 에 있음)
  S<N>_respiban.txt  가슴 RespiBAN, 700 Hz, 탭 구분. 머리말 JSON 의 sensor 순서상 CH1 = ECG
                     머리말의 date·time = 녹화 시작 (현지 시각, 독일 여름 UTC+2)
  S<N>_E4_Data.zip   손목 Empatica E4. BVP.csv: 첫 줄 시작 unix 시각, 둘째 줄 64 Hz, 이후 값
  S<N>_quest.csv     ORDER / START / END: 조건(Base, TSST, Medi 1, Fun, Medi 2, sRead, fRead)의
                     RespiBAN 시작 기준 시각. 값은 '분.초' 형식 (예: 39.55 = 39분 55초)

변환:
- 참가자 W<N>, 세션 full 하나. ECG 700 Hz, PPG(손목 BVP) 64 Hz 를 각자 원 주기로 저장
- 두 기기 시계를 머리말 시각으로 1차 정렬한 뒤, ECG·BVP 의 1초 심박 곡선 상호상관으로 ±120초 안에서
  어긋남을 보정 (meta.e4_offset_sec). 상관이 0.4 미만이면 경고 (손목 BVP 잡음 때문에 0.5~0.9 정도가 보통, 보정량은 대개 ±2초)
- Base 조건(약 20분 앉은 휴식)을 baseline_range 로. 조건 구간은 labels_orig.csv 에 (졸음 라벨 아님)
- 전원 주파수 50 Hz (독일)
"""
from __future__ import annotations

import io
import json
import re
import zipfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from ..data_io import write_session
from ..features import detect_ecg_peaks, detect_ppg_peaks

DATASET_ID = "wesad"
FS_ECG, FS_PPG = 700.0, 64.0
LOCAL_TZ = timezone(timedelta(hours=2))      # 수집 기간(2017년 5–7월) 독일 서머타임


def _mmss(v) -> float:
    m = re.fullmatch(r"\s*(\d+)(?:\.(\d+))?\s*", str(v))
    if not m:
        return np.nan
    sec = m.group(2) or "0"
    return int(m.group(1)) * 60 + int(sec.ljust(2, "0")[:2])


def read_quest(path: Path) -> pd.DataFrame:
    rows = {}
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        parts = [p.strip() for p in line.split(";")]
        if parts and parts[0].startswith("#") and len(parts) > 1:
            rows[parts[0].lstrip("# ").upper()] = [p for p in parts[1:] if p]
    order = rows.get("ORDER", [])
    st, en = rows.get("START", []), rows.get("END", [])
    return pd.DataFrame([{"condition": o, "t_start": _mmss(st[i]) if i < len(st) else np.nan,
                          "t_end": _mmss(en[i]) if i < len(en) else np.nan} for i, o in enumerate(order)])


def read_respiban(path: Path) -> tuple[np.ndarray, datetime]:
    with open(path, encoding="utf-8", errors="replace") as f:
        f.readline()
        hdr = json.loads(f.readline().lstrip("# ").strip())
    dev = next(iter(hdr.values()))
    sensors = dev["sensor"]
    col = 2 + sensors.index("ECG")             # nSeq, DI, CH1 …
    start = datetime.strptime(f"{dev['date']} {dev['time']}", "%Y-%m-%d %H:%M:%S.%f").replace(tzinfo=LOCAL_TZ)
    x = pd.read_csv(path, sep="\t", comment="#", header=None, usecols=[col], dtype=np.float64).iloc[:, 0].to_numpy()
    return x, start


def read_bvp(path: Path) -> tuple[np.ndarray, datetime]:
    with zipfile.ZipFile(path) as z:
        lines = z.read("BVP.csv").decode().splitlines()
    start = datetime.fromtimestamp(float(lines[0].split(",")[0]), tz=timezone.utc)
    return np.array([float(v.split(",")[0]) for v in lines[2:]], np.float64), start


def _hr_curve(peaks: np.ndarray, fs: float, n_sec: int) -> np.ndarray:
    t = peaks[1:] / fs
    rr = np.diff(peaks) / fs
    ok = (rr > 0.33) & (rr < 1.5)
    hr = np.full(n_sec, np.nan)
    if ok.sum() > 10:
        hr = np.interp(np.arange(n_sec), t[ok], 60 / rr[ok], left=np.nan, right=np.nan)
        # 손목 BVP 박동 검출은 잡음이 많아 15초 이동 중앙값으로 느린 심박 흐름만 비교
        hr = pd.Series(hr).rolling(15, center=True, min_periods=5).median().to_numpy()
    return hr


def estimate_offset(ecg, ppg_aligned_start_sec, ppg, max_lag=120) -> tuple[float, float]:
    """ECG 시작 기준 PPG 시작 시각(1차 추정)을 받아, 1초 심박 곡선 상관이 최대인 보정량과 그 상관."""
    n = int(len(ecg) / FS_ECG)
    he = _hr_curve(detect_ecg_peaks(ecg, FS_ECG), FS_ECG, n)
    hp_full = _hr_curve(detect_ppg_peaks(ppg, FS_PPG), FS_PPG, int(len(ppg) / FS_PPG))
    best, best_r = 0.0, -1.0
    for lag in range(-max_lag, max_lag + 1):
        s = int(round(ppg_aligned_start_sec)) + lag          # ECG 초 k ↔ PPG 초 k - s
        idx = np.arange(n) - s
        ok = (idx >= 0) & (idx < len(hp_full))
        a, b = he[ok], hp_full[idx[ok]]
        m = np.isfinite(a) & np.isfinite(b)
        if m.sum() < 300:
            continue
        r = np.corrcoef(a[m], b[m])[0, 1]
        if r > best_r:
            best, best_r = float(lag), float(r)
    return best, best_r


def convert(src: str | Path, out: str | Path, log=print) -> dict:
    src, out = Path(src), Path(out)
    n_ok, skipped = 0, []
    for sdir in sorted((p for p in src.iterdir() if p.is_dir() and p.name.startswith("S")),
                       key=lambda p: int(p.name[1:])):
        sid = sdir.name
        fr, fe, fq = sdir / f"{sid}_respiban.txt", sdir / f"{sid}_E4_Data.zip", sdir / f"{sid}_quest.csv"
        if not fr.exists():
            skipped.append(f"{sid}: respiban 없음")
            continue
        ecg, t_ecg = read_respiban(fr)
        quest = read_quest(fq) if fq.exists() else pd.DataFrame(columns=["condition", "t_start", "t_end"])
        base = quest[quest.condition.str.lower().str.startswith("base")]
        signals, fs, extra = {"ecg": ecg}, {"ecg": FS_ECG}, {}
        if fe.exists():
            bvp, t_bvp = read_bvp(fe)
            s0 = (t_ecg - t_bvp).total_seconds()             # ECG 시작이 BVP 시작보다 s0 초 늦음
            lag, r = estimate_offset(ecg, -s0, bvp)
            s = s0 - lag                                     # 보정 후 BVP 에서 ECG 시작에 해당하는 초
            i0 = int(round(s * FS_PPG))
            dur = min(len(ecg) / FS_ECG, (len(bvp) - i0) / FS_PPG)
            if i0 >= 0 and dur > 60:
                signals = {"ecg": ecg[:int(dur * FS_ECG)], "ppg": bvp[i0:i0 + int(dur * FS_PPG)]}
                fs = {"ecg": FS_ECG, "ppg": FS_PPG}
                extra = {"e4_offset_sec": round(s, 2), "e4_align_lag_sec": lag, "e4_align_r": round(r, 3)}
                if r < 0.4:
                    skipped.append(f"{sid}: ECG·BVP 정렬 상관 {r:.2f} (낮음, 확인 필요)")
            else:
                skipped.append(f"{sid}: BVP 가 ECG 구간을 덮지 못해 ECG 만 저장")
        meta = {"dataset_id": DATASET_ID, "subject_id": f"W{int(sid[1:]):02d}", "session_id": "full",
                "t0": t_ecg.isoformat(), "fs": fs, "channels": list(signals), "is_rest": False,
                "prefiltered": False, "mains_hz": 50.0, "drowsiness_inducer": "none",
                "sensor": {"ecg": "RespiBAN chest ECG", "ppg": "Empatica E4 wrist BVP"},
                "notes": "WESAD 전체 녹화. Base 조건 = 기준 구간. 조건 구간은 labels_orig"} | extra
        if len(base) and np.isfinite(base.t_start.iloc[0]):
            meta["baseline_range"] = [float(base.t_start.iloc[0]), float(base.t_end.iloc[0])]
        orig = quest.rename(columns={"t_start": "t"})
        write_session(out / DATASET_ID / meta["subject_id"] / "full", meta, signals, None, orig)
        n_ok += 1
        log(f"  {sid}: 완료 ({len(signals['ecg']) / FS_ECG / 60:.0f}분, 정렬 상관 {extra.get('e4_align_r', '-')})")
    log(f"WESAD: 세션 {n_ok}개 → {out / DATASET_ID}")
    for m in skipped:
        log(f"  {m}")
    return {"dataset": DATASET_ID, "sessions": n_ok, "skipped": skipped}
