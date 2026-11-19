"""J-Berry 센서 측정 zip/폴더 → 전달 규약 세션 폴더 (docs/data-handoff.md). 원본은 읽기만 한다.

    python scripts/convert_jberry_sensor.py --raw "측정 20261119_20_14.zip" --out data/converted [--subject self]

입력 (측정 앱이 내보내는 파일, 이름은 날짜마다 조금 다름):
- ecg_1_*.csv : 시간(초), ECG원본, ECG_mV, 전극분리, 라벨     (약 435 Hz, 녹음 60초부터)
- ppg_1_*.csv : 시간(초), 시간_FIR지연보정(초), PPG적색광원본, PPG적외선원본, ..., 라벨  (약 100.7 Hz)
- 원본 파일 이름의 YYYYMMDD_HHMMSS 로 녹음 시작 시각(t0)을 정한다 (없으면 --t0).

변환 규칙 (고정):
- ECG = ECG_mV. PPG = 'AC보정' 열, 시간은 'FIR지연보정' 시간 열 (11-18 rec4 를 낸 연구실 PC 변환과 같다.
  --ppg-col 적외선 으로 바꿀 수 있지만, 학생 결과가 크게 달라진다: docs/selfmeasure.md 5장).
  필터는 파이프라인이 한다 (prefiltered=false).
- 세션 시작 = 원 녹음 --start 초 (기본 60.0, 측정 앱이 처음 60초를 버림). 끝 = 두 채널 공통 끝.
  채널별 실측 평균 주기의 균일 격자에 타임스탬프 기준 선형 보간.
  0.2초 넘게 끊긴 곳이 있으면 멈춘다 (규약 4장).
- 라벨 열이 표본마다 있으면 (정상/졸음) 1 Hz labels (0 정상, 1 졸음, -1 미라벨) 로 저장.
  5분마다 표시만 있는 형식 ("0~5분 정상" 등, 표시 시각 = 그 블록의 끝) 이면 블록 라벨로 바꾼다.
- 세션 t_rel 0 = 두 채널 공통 시작 (보통 원 녹음 60초). meta.raw_offset_sec, meta.notes, labels_orig.csv 에
  원 시각 오프셋을 남긴다.

결과: <out>/jberry_sensor/<subject>/<YYYY-MM-DD_HHMMSS>/{meta.json, data.npz, labels_orig.csv}

기본 설정은 11-18 rec4 (연구실 PC 변환) 의 학생 창 AUROC 를 ±0.005 로 재현한다 (docs/selfmeasure.md 5장).
"""
from __future__ import annotations

import argparse
import json
import re
import tempfile
import zipfile
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

LAB_MAP = {"정상": 0, "졸음": 1}


def _read(path: Path) -> pd.DataFrame:
    for enc in ("utf-8-sig", "cp949"):
        try:
            return pd.read_csv(path, encoding=enc)
        except UnicodeDecodeError:
            continue
    raise ValueError(f"{path}: 인코딩을 읽지 못했습니다")


def _find(root: Path, prefix: str) -> Path:
    c = [p for p in root.rglob(f"{prefix}_1_*.csv")]
    if len(c) != 1:
        raise FileNotFoundError(f"{root}: {prefix}_1_*.csv 가 정확히 1개여야 합니다 ({[p.name for p in c]})")
    return c[0]


def _t0(root: Path) -> str | None:
    for p in root.rglob("*.csv"):
        m = re.search(r"(20\d{6})_(\d{6})", p.name)
        if m:
            return datetime.strptime(m.group(1) + m.group(2), "%Y%m%d%H%M%S").strftime("%Y-%m-%dT%H:%M:%S+09:00")
    return None


def _uniform(t: np.ndarray, x: np.ndarray, t_a: float, t_b: float, name: str):
    dt = np.diff(t)
    if (dt <= 0).any():
        raise ValueError(f"{name}: 시간이 단조 증가하지 않습니다")
    if dt.max() > 0.2:
        raise ValueError(f"{name}: {dt.max():.3f}초 끊긴 곳이 있습니다. 규약 4장대로 세션을 나누세요")
    fs = round((len(t) - 1) / (t[-1] - t[0]), 2)
    g = t_a + np.arange(int((t_b - t_a) * fs)) / fs
    return fs, np.interp(g, t, x)


def _labels(e: pd.DataFrame, te: np.ndarray, t_a: float, t_b: float):
    col = e.columns[-1]
    s = e[col]
    g1 = t_a + np.arange(int(t_b - t_a))
    if s.notna().mean() > 0.9:                              # 표본마다 라벨
        code = s.map(LAB_MAP).fillna(-1).to_numpy(int)
        lab = code[np.clip(np.searchsorted(te, g1), 0, len(te) - 1)]
        ch = np.r_[0, np.flatnonzero(np.diff(code)) + 1]
        orig = pd.DataFrame({"t_orig_sec": te[ch], "mark": s.iloc[ch].to_numpy()})
    elif s.notna().any():                                   # 블록 끝 표시
        marks = e.loc[s.notna(), [e.columns[0], col]].to_numpy()
        lab = np.full(len(g1), -1)
        prev = 0.0
        for t_end, txt in marks:
            y = next((v for k, v in LAB_MAP.items() if k in str(txt)), -1)
            lab[(g1 >= prev) & (g1 < float(t_end))] = y
            prev = float(t_end)
        orig = pd.DataFrame({"t_orig_sec": marks[:, 0].astype(float), "mark": marks[:, 1]})
    else:
        return None, None
    orig["t_rel_sec"] = orig["t_orig_sec"] - t_a
    return lab.astype(np.int8), orig


def convert(raw: Path, out: Path, subject: str, t0: str | None, ppg_col: str = "AC보정", ppg_time: str = "fir",
            start: float | None = 60.0) -> Path:
    tmp = None
    if raw.suffix.lower() == ".zip":
        tmp = tempfile.TemporaryDirectory()
        with zipfile.ZipFile(raw) as z:
            z.extractall(tmp.name)
        root = Path(tmp.name)
    else:
        root = raw
    e, p = _read(_find(root, "ecg")), _read(_find(root, "ppg"))
    te, xe = e.iloc[:, 0].to_numpy(float), e["ECG_mV"].to_numpy(float)
    ir = [c for c in p.columns if ppg_col in c]
    if not ir:
        raise KeyError(f"PPG '{ppg_col}' 열이 없습니다: {list(p.columns)}")
    tcol = [c for c in p.columns if "FIR" in c] if ppg_time == "fir" else [p.columns[0]]
    if not tcol:
        raise KeyError(f"PPG FIR 지연 보정 시간 열이 없습니다: {list(p.columns)}")
    tp, xp = p[tcol[0]].to_numpy(float), p[ir[0]].to_numpy(float)
    t_b = min(te[-1], tp[-1])
    t_a = float(start) if start is not None else max(te[0], tp[0])
    if max(te[0], tp[0]) - t_a > 0.2:
        raise ValueError(f"--start {t_a}s 가 채널 시작({te[0]:.3f}, {tp[0]:.3f}s)보다 0.2초 넘게 앞섭니다")
    fs_e, ecg = _uniform(te, xe, t_a, t_b, "ECG")
    fs_p, ppg = _uniform(tp, xp, t_a, t_b, "PPG")
    lab, orig = _labels(e, te, t_a, t_b)
    t0 = t0 or _t0(root)
    if t0 is None:
        raise ValueError("녹음 시작 시각을 파일 이름에서 못 찾았습니다. --t0 를 주세요")
    sid = datetime.fromisoformat(t0).strftime("%Y-%m-%d_%H%M%S")
    d = out / "jberry_sensor" / subject / sid
    d.mkdir(parents=True, exist_ok=True)
    arr = {"ecg": ecg.astype(np.float32), "ppg": ppg.astype(np.float32)}
    if lab is not None:
        arr["labels"] = lab
    np.savez_compressed(d / "data.npz", **arr)
    meta = {"dataset_id": "jberry_sensor", "subject_id": subject, "session_id": sid, "t0": t0,
            "fs": {"ecg": fs_e, "ppg": fs_p}, "label_fs": 1 if lab is not None else None,
            "is_rest": False, "prefiltered": False, "channels": ["ecg", "ppg"],
            "raw_offset_sec": round(float(t_a), 3),
            "notes": f"{raw.name}: ecg_1 ECG_mV / ppg_1 {ir[0]} (시간 {tcol[0]}), 원 시각 {t_a:.3f}~{t_b:.3f}s (t_rel 0 = 원 {t_a:.3f}s), 균일 격자 선형 보간"}
    (d / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    if orig is not None:
        orig.to_csv(d / "labels_orig.csv", index=False, encoding="utf-8")
    print(f"{d}: {t_b - t_a:.0f}s, fs ecg {fs_e} ppg {fs_p}, 원 시각 오프셋 {t_a:.3f}s, 라벨 {'있음' if lab is not None else '없음'}")
    if tmp:
        tmp.cleanup()
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True, nargs="+", help="측정 zip 또는 압축 푼 폴더 (여러 개 가능)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--subject", default="self")
    ap.add_argument("--t0", help="녹음 시작 시각 ISO 8601 (파일 이름에서 못 찾을 때)")
    ap.add_argument("--ppg-col", default="AC보정", help="PPG 열 이름에 들어 있는 글자 (기본 AC보정, 예: 적외선, 적색광)")
    ap.add_argument("--ppg-time", default="fir", choices=["fir", "raw"], help="PPG 시간 열: FIR 지연 보정(기본) 또는 첫 열")
    ap.add_argument("--start", type=float, default=60.0, help="세션 시작 원 시각[초] (기본 60.0). 음수면 두 채널 겹침 시작")
    a = ap.parse_args()
    for r in a.raw:
        convert(Path(r), Path(a.out), a.subject, a.t0, a.ppg_col, a.ppg_time, None if a.start < 0 else a.start)


if __name__ == "__main__":
    main()
