"""J-Berry 센서 측정 → 전달 규약 세션 폴더 (표준 변환기). 원본은 읽기만 한다.

2026-11-18 연구실 PC 에서 rec4 를 실제로 변환한 코드 (11-18 Claude 대화 기록에서 복구) 를 표준으로 정했다 (2026-11-23).
계산은 그 코드 그대로이고, 녹음마다 다른 값 (파일 위치, 시작 시각, 5분 라벨) 만 인자·파일에서 읽는다.
기본 설정으로 11-18 rec4 를 변환하면 연구실 PC 의 원 변환본과 data.npz·meta.json·labels_orig.csv 가 모두 같다.
(PR #12 의 역추정 변환기는 폐기했다. 그 출력은 끝 표본 수가 달라 rec4 평가 창이 142 → 132 개였다.)

    python scripts/convert_jberry_sensor.py --raw drive-download-20261118T124939Z-1-001.zip --out data/converted

입력 (측정 앱이 내보내는 파일):
- ecg_1_*.csv : 시간(초), ECG원본, ECG_mV, 전극분리, 라벨링   (녹음 약 60초부터)
- ppg_1_*.csv : 시간(초), 시간_FIR지연보정(초), PPG적색광원본, PPG적외선원본, 상태, 필터안정화중, DC추정, AC보정신호, ...
- 파일 이름의 YYYYMMDD_HHMMSS = 녹음 시작 시각 (없으면 --t0).

변환 규칙 (원 코드 그대로):
- ECG = ecg_1 의 0번 열(시간) · 2번 열(ECG_mV). PPG = ppg_1 의 1번 열(시간_FIR지연보정) · 7번 열(AC보정신호).
- 각 채널을 원 시각 60.0초부터 그 채널 마지막 시각까지, 고정 fs (ECG 435.33, PPG 100.66 Hz) 균일 격자에 선형 보간.
  필터는 파이프라인이 한다 (prefiltered=false). 세션 0초 = 원 녹음 60초, t0 = 녹음 시작 + 60초.
- 5분 블록 라벨은 labels_orig.csv (block_min, label) 로 남긴다. 라벨은 --labels 로 주거나,
  없으면 라벨링 열의 블록 끝 표시 ("0~5분 정상" 등) 에서 읽는다.
- --labels-array 를 주면 1 Hz labels (0 정상, 1 졸음, -1 미라벨) 도 data.npz 에 넣는다 (selfmeasure_eval.py 용).
  원 변환본에는 없던 배열이라 기본은 끔.

결과: <out>/jberry_sensor/<subject>/<YYYY-MM-DD_HHMMSS>_s1/{meta.json, data.npz, labels_orig.csv}
"""
from __future__ import annotations

import argparse
import re
import sys
import tempfile
import zipfile
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from driver_state.data_io import write_session  # noqa: E402

T0 = 60.0
FE, FP = 435.33, 100.66
LAB_MAP = {"정상": 0, "졸음": 1}


def uni(t, x, fs, t0=T0):
    g = np.arange(t0, t[-1], 1 / fs)
    return np.interp(g, t, x)


def _find(root: Path, prefix: str) -> Path:
    c = sorted(root.rglob(f"{prefix}_1_*.csv"))
    if len(c) != 1:
        raise FileNotFoundError(f"{root}: {prefix}_1_*.csv 가 정확히 1개여야 합니다 ({[p.name for p in c]})")
    return c[0]


def _rec_start(root: Path) -> datetime | None:
    for p in sorted(root.rglob("*.csv")):
        m = re.search(r"(20\d{6})_(\d{6})", p.name)
        if m:
            return datetime.strptime(m.group(1) + m.group(2), "%Y%m%d%H%M%S")
    return None


def _label_col(ecg_path: Path) -> tuple[np.ndarray, pd.Series] | None:
    d = pd.read_csv(ecg_path, encoding="utf-8-sig")
    col = [c for c in d.columns if "라벨" in c]
    if not col:
        return None
    return d.iloc[:, 0].to_numpy(float), d[col[0]]


def _block_labels(ecg_path: Path) -> list[str] | None:
    """라벨링 열의 블록 끝 표시 ("0~5분 정상") 를 5분 블록 순서대로 읽는다."""
    tc = _label_col(ecg_path)
    if tc is None:
        return None
    s = tc[1]
    if not s.notna().any() or s.notna().mean() > 0.9:
        return None
    out = [next((k for k in LAB_MAP if k in txt), None) for txt in s[s.notna()].astype(str)]
    out = [y for y in out if y is not None]
    return out or None


def _label_array(ecg_path: Path, t_end: float, t0: float = T0) -> np.ndarray | None:
    """1 Hz 라벨 (세션 시각 기준). 표본마다 라벨이 있거나 블록 끝 표시가 있는 형식 모두."""
    tc = _label_col(ecg_path)
    if tc is None:
        return None
    t, s = tc
    g = t0 + np.arange(int(t_end - t0))
    if s.notna().mean() > 0.9:
        code = s.map(LAB_MAP).fillna(-1).to_numpy(int)
        return code[np.clip(np.searchsorted(t, g), 0, len(t) - 1)].astype(np.int8)
    if not s.notna().any():
        return None
    lab, prev = np.full(len(g), -1), 0.0
    for te, txt in zip(t[s.notna().to_numpy()], s[s.notna()].astype(str)):
        lab[(g >= prev) & (g < te)] = next((v for k, v in LAB_MAP.items() if k in txt), -1)
        prev = te
    return lab.astype(np.int8)


def convert(raw: Path, out: Path, subject: str = "self01", t0_iso: str | None = None,
            labels: list[str] | None = None, labels_array: bool = False) -> Path:
    tmp = None
    if raw.suffix.lower() == ".zip":
        tmp = tempfile.TemporaryDirectory()
        with zipfile.ZipFile(raw) as z:
            z.extractall(tmp.name)
        R = Path(tmp.name)
    else:
        R = raw
    fe_path, fp_path = _find(R, "ecg"), _find(R, "ppg")
    e = pd.read_csv(fe_path, encoding="utf-8-sig", usecols=[0, 2])
    p = pd.read_csv(fp_path, encoding="utf-8-sig", usecols=[1, 7])
    ecg = uni(e.iloc[:, 0].to_numpy(float), e.iloc[:, 1].to_numpy(float), FE)
    ppg = uni(p.iloc[:, 0].to_numpy(float), p.iloc[:, 1].to_numpy(float), FP)

    start = datetime.fromisoformat(t0_iso).replace(tzinfo=None) if t0_iso else _rec_start(R)
    if start is None:
        raise ValueError("녹음 시작 시각을 파일 이름에서 못 찾았습니다. --t0 를 주세요")
    sid = start.strftime("%Y-%m-%d_%H%M%S") + "_s1"
    labels = labels or _block_labels(fe_path)
    lab_arr = _label_array(fe_path, float(e.iloc[-1, 0])) if labels_array else None

    D = out / "jberry_sensor" / subject / sid
    note = f"{start:%m-%d} 측정. ecg_1 ECG_mV, ppg_1 AC보정신호(FIR 지연 보정 시각). 세션 0초 = 원본 60초."
    if labels:
        note += " 5분 라벨: " + ",".join(labels)
    meta = {"dataset_id": "jberry_sensor", "subject_id": subject, "session_id": sid,
            "t0": (start + timedelta(seconds=T0)).strftime("%Y-%m-%dT%H:%M:%S+09:00"),
            "fs": {"ecg": FE, "ppg": FP}, "channels": ["ecg", "ppg"], "is_rest": False, "prefiltered": False,
            "sensor": {"ecg": "AD8232", "ppg": "MAX30102"}, "notes": note}
    if lab_arr is not None:
        meta["label_fs"] = 1
    orig = None
    if labels:
        orig = pd.DataFrame({"block_min": [f"{5 * k}-{5 * k + 5}" for k in range(len(labels))], "label": labels})
    write_session(D, meta, {"ecg": ecg, "ppg": ppg}, lab_arr, orig)
    print(D, "ecg", len(ecg), "ppg", len(ppg), "라벨", ",".join(labels) if labels else "없음")
    if tmp:
        tmp.cleanup()
    return D


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True, nargs="+", help="측정 zip 또는 압축 푼 폴더 (여러 개 가능)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--subject", default="self01")
    ap.add_argument("--t0", help="녹음 시작 시각 ISO 8601 (파일 이름에서 못 찾을 때)")
    ap.add_argument("--labels", help="5분 블록 라벨, 쉼표로 (예: 정상,정상,졸음,졸음,졸음,정상,정상)")
    ap.add_argument("--labels-array", action="store_true", help="1 Hz labels 배열도 저장 (selfmeasure_eval.py 용)")
    a = ap.parse_args()
    for r in a.raw:
        convert(Path(r), Path(a.out), a.subject, a.t0, a.labels.split(",") if a.labels else None, a.labels_array)


if __name__ == "__main__":
    main()
