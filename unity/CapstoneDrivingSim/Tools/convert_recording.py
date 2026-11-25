"""
convert_recording.py — 팀 측정 프로그램 기록(zip 또는 폴더) → 시뮬레이터 Replay 세션 폴더

입력: 측정 프로그램이 저장한 zip(예: "측정 26.11.18 19 42.zip") 또는 그 압축을 푼 폴더
  ecg_*_라벨링.csv  열: 시간(초), ECG원본, ECG_mV, 전극분리, 라벨링
  ppg_*_라벨링.csv  열: 시간(초), …, PPG적색광원본, PPG적외선원본, …, 라벨링
출력: <out>/meta.json, ecg.csv, ppg.csv (value 열 하나 — 저장소 docs/data-handoff.md 형식), labels.csv

변환 규칙 (simulator 2026-11-18 결정, README 4장 참고)
  · ECG = 'ECG원본'(ADC), PPG = 'PPG적외선원본' (--ppg red 로 적색광 선택 가능). 필터·전처리는 하지 않는다 (서버가 저장소 코드로 함)
  · 두 채널 중 늦게 시작한 쪽에 시작 시각을 맞춘다 → meta.t_offset (원본 기록 시각)
  · 타임스탬프가 묶음으로 들어와 간격이 고르지 않으므로 fs = (표본 수 − 1) / (마지막 − 처음 시각) 으로 두고 균일 간격으로 본다
  · labels.csv: '라벨링' 열에 블록 끝마다 찍힌 표시(예: "11~15분 졸음")를 [이전 표시, 이 표시) 구간으로 바꾼다.
    모델 입력이 아니라 화면 참고용 (참가자 자기보고)

사용:
  # 리플레이용 (유니티 안에 넣기)
  python Tools/convert_recording.py "C:/Users/<me>/Downloads/측정 26.11.18 19 42.zip" --out Assets/StreamingAssets/Replays/rec_20261118
  # 개인 기준(baseline) 측정 파일 — 안정 상태 구간만 잘라서 (예: 처음 8분)
  python Tools/convert_recording.py "<측정 zip>" --out Assets/StreamingAssets/Replays/rec_20261118_baseline --role baseline --end 480
주의: 결과물은 원신호(개인 생체 데이터)다. StreamingAssets/Replays/ 와 SimLogs/ 는 .gitignore 대상이니 저장소에 올리지 말 것.
"""
from __future__ import annotations

import argparse
import io
import json
import re
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd


def _read(src: Path, pattern: str) -> tuple[str, pd.DataFrame]:
    rx = re.compile(pattern)
    if src.is_file() and src.suffix.lower() == ".zip":
        with zipfile.ZipFile(src) as z:
            names = [n for n in z.namelist() if rx.search(Path(_decode(n)).name)]
            if not names:
                raise FileNotFoundError(f"zip 안에 {pattern} 파일 없음")
            raw = z.read(names[0])
            return _decode(names[0]), _csv(raw)
    files = [p for p in src.rglob("*.csv") if rx.search(p.name)]
    if not files:
        raise FileNotFoundError(f"{src} 안에 {pattern} 파일 없음")
    return files[0].name, _csv(files[0].read_bytes())


def _decode(name: str) -> str:
    try:
        return name.encode("cp437").decode("cp949")      # 한글 파일명이 들어간 Windows zip
    except (UnicodeEncodeError, UnicodeDecodeError):
        return name


def _csv(raw: bytes) -> pd.DataFrame:
    for enc in ("utf-8-sig", "cp949"):
        try:
            return pd.read_csv(io.BytesIO(raw), encoding=enc, low_memory=False)
        except UnicodeDecodeError:
            continue
    raise ValueError("CSV 인코딩을 알 수 없음")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src", help="측정 zip 또는 폴더")
    ap.add_argument("--out", required=True, help="세션 폴더 (예: SimLogs/bio/rec_20261118)")
    ap.add_argument("--ppg", choices=["ir", "red"], default="ir")
    ap.add_argument("--start", type=float, default=0.0, help="자를 구간 시작 [s, 두 채널 맞춘 뒤 기준]")
    ap.add_argument("--end", type=float, default=None, help="자를 구간 끝 [s]")
    ap.add_argument("--role", choices=["replay", "baseline"], default="replay",
                    help="baseline: 메인 메뉴의 '기준 측정 파일' 목록에 표시 (개인 기준 측정, 보통 5~10분 안정 상태)")
    a = ap.parse_args()
    src, out = Path(a.src), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    ename, e = _read(src, r"^ecg_\d+_라벨링\.csv$")
    pname, p = _read(src, r"^ppg_\d+_라벨링\.csv$")
    te, ve = e.iloc[:, 0].to_numpy(float), e["ECG원본"].to_numpy(float)
    pcol = "PPG적외선원본" if a.ppg == "ir" else "PPG적색광원본"
    tp, vp = p.iloc[:, 0].to_numpy(float), p[pcol].to_numpy(float)

    t0 = max(te[0], tp[0])
    ke, kp = te >= t0, tp >= t0
    te, ve, tp, vp = te[ke], ve[ke], tp[kp], vp[kp]
    fs_e = (len(ve) - 1) / (te[-1] - te[0])
    fs_p = (len(vp) - 1) / (tp[-1] - tp[0])
    # 구간 자르기 (균일 간격 가정이므로 표본 번호로 자른다)
    c0 = max(0.0, a.start)
    c1 = a.end
    ve = ve[int(c0 * fs_e): (int(c1 * fs_e) if c1 else len(ve))]
    vp = vp[int(c0 * fs_p): (int(c1 * fs_p) if c1 else len(vp))]
    dur = min(len(ve) / fs_e, len(vp) / fs_p)
    pd.DataFrame({"value": ve}).to_csv(out / "ecg.csv", index=False)
    pd.DataFrame({"value": vp}).to_csv(out / "ppg.csv", index=False)

    # 라벨: 블록 끝 표시 → 구간 (잘라낸 구간 기준으로 옮기고 잘라냄)
    rows = []
    if "라벨링" in e.columns:
        lab = e["라벨링"].iloc[np.flatnonzero(ke)].fillna("").astype(str).to_numpy()
        prev = 0.0
        for i in np.flatnonzero(lab != ""):
            t_end = float(te[i] - t0)
            m = re.search(r"(정상|졸음|주의|피로)", lab[i])
            s0, s1 = max(prev, c0) - c0, min(t_end, c0 + dur) - c0
            if s1 > s0:
                rows.append((round(float(s0), 2), round(float(s1), 2), m.group(1) if m else lab[i], lab[i]))
            prev = t_end
    pd.DataFrame(rows, columns=["start_s", "end_s", "label", "source_marker"]).to_csv(out / "labels.csv", index=False, encoding="utf-8")

    meta = {"dataset_id": "jberry_sensor", "subject_id": "User1", "session_id": out.name, "role": a.role,
            "fs": {"ecg": fs_e, "ppg": fs_p}, "duration_s": round(dur, 3), "is_rest": a.role == "baseline",
            "prefiltered": False, "t_offset": float(t0 + c0),
            "source": {"ecg": f"{ename} · ECG원본 (ADC)", "ppg": f"{pname} · {pcol}", "input": src.name,
                       "range_s": [c0, c0 + dur]},
            "note": "두 채널 시작 시각을 맞춤(t_offset=원본 기록 시각). fs=(표본수-1)/기간, 균일 간격 가정",
            "labels": "labels.csv: 원본 '라벨링' 열의 블록 끝 표시를 구간으로 바꾼 참고 라벨 (모델 입력 아님)"}
    (out / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"저장: {out}  [{a.role}] ECG {fs_e:.2f} Hz · PPG {fs_p:.2f} Hz · {dur / 60:.1f}분 · 라벨 구간 {len(rows)}개 (t_offset {t0 + c0:.2f}s)")
    for r in rows:
        print("   ", r)


if __name__ == "__main__":
    main()
