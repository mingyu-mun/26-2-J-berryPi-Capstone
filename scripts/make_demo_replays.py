#!/usr/bin/env python3
"""시연 영상용: 실측 원본 기록(zip 또는 폴더)을 웹보드 원본 기록 폴더 여러 개로 자른다.

입력: 측정 프로그램 zip (예: "측정 26.11.18 19 42.zip") 또는 압축을 푼 폴더.
      '원본_ecg_*.csv' (수신시간, ECG원본, 라벨링) 과 '원본_ppg_*.csv' (시간, 적색광, 적외선, 라벨링) 를 쓴다.
출력: <out>/<클립 이름>/ecg.csv (time,adc), ppg.csv (time,red,ir), meta.json, recording.json
      → `python scripts/run_dashboard.py --recordings-dir <out> --rpeak-threshold rolling`

시각 규칙
- 모든 구간은 **원본 시각**(원본 CSV 첫 열, 측정 시작 = 0 초)으로 적는다.
  시뮬레이터(Unity) Replay 는 라벨링 파일(약 60 초부터 시작)로 만들어서 Unity 기록 시각 = 원본 시각 - t_offset (약 60 초).
- 타임스탬프가 묶음으로 들어와 간격이 고르지 않으므로 fs = (표본 수 - 1) / (마지막 - 처음 시각) 으로 두고 균일 간격으로 본다
  (Unity convert_recording.py 와 같은 규칙). 클립은 그 표본 번호로 자른다. 값은 바꾸지 않는다.
- recording.json 의 fast_forward_s: 웹보드 판정 재생이 클립 앞쪽 이 길이만큼을 기다리지 않고 바로 처리한다
  (모델의 300 초 장기 특징 창과 상태 머신 지속 조건 채우기). 실제 측정값만 넣고 값을 만들지 않는다.
  화면의 재생 위치와 시뮬레이터 동기화 기준은 fast_forward_s 가 끝나는 순간 = display_start_s (원본 시각).

결과물은 개인 생체 원신호다. outputs/ 아래에 두고 저장소에 올리지 않는다.

예)
  python scripts/make_demo_replays.py "측정 26.11.18 19 42.zip" --out outputs/local-recordings/demo_1118 --preset rec20261118
  python scripts/make_demo_replays.py <zip> --out <dir> --clip full:0:2100 --clip mine:600:1500:420
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

# 11-18 실측(rec4) 시연 구간. 근거: 전체 기록 판정 재생(rolling)에서 정상 310~590초, 졸음 1010~1500초 연속.
# (이름, 제목, 클립 시작, 클립 끝, 빠르게 넘길 앞부분 길이) — 시각은 모두 원본 시각 [s]
PRESETS = {
    "rec20261118": [
        ("1118_baseline", "11-18 기준 (원본 0~540초, 기준 계산용)", 0, 540, 0),
        ("1118_normal", "11-18 시연 정상 (화면 원본 320~580초)", 0, 580, 320),
        ("1118_drowsy", "11-18 시연 졸음 1분 (화면 원본 1285~1345초, 졸음 확률 최고 구간)", 0, 1345, 1285),
        ("1118_to_drowsy", "11-18 졸음 전환 30초 (화면 원본 990~1020초, 1010초에 졸음)", 0, 1020, 990),
        ("1118_full", "11-18 전체 (원본 0~2100초)", 0, 2100, 0),
    ],
}


def _decode(info: zipfile.ZipInfo) -> str:
    if info.flag_bits & 0x800:          # 이름이 이미 UTF-8
        return info.filename
    raw = info.filename.encode("cp437")
    for encoding in ("utf-8", "cp949"):
        try:
            return raw.decode(encoding)
        except UnicodeDecodeError:
            continue
    return info.filename


def _read_csv(raw: bytes) -> pd.DataFrame:
    for encoding in ("utf-8-sig", "cp949"):
        try:
            return pd.read_csv(io.BytesIO(raw), encoding=encoding)
        except UnicodeDecodeError:
            continue
    raise ValueError("CSV 인코딩을 읽을 수 없습니다 (UTF-8 / CP949)")


def load_source(src: Path, pattern: str) -> tuple[str, pd.DataFrame]:
    rx = re.compile(pattern)
    if src.is_file() and zipfile.is_zipfile(src):
        with zipfile.ZipFile(src) as z:
            infos = [i for i in z.infolist() if rx.search(Path(_decode(i)).name)]
            if len(infos) != 1:
                raise SystemExit(f"{src}: '{pattern}' 파일이 하나여야 합니다 ({len(infos)}개)")
            return Path(_decode(infos[0])).name, _read_csv(z.read(infos[0]))
    files = [p for p in src.rglob("*.csv") if rx.search(p.name)]
    if len(files) != 1:
        raise SystemExit(f"{src}: '{pattern}' 파일이 하나여야 합니다 ({len(files)}개)")
    return files[0].name, _read_csv(files[0].read_bytes())


def load_raw(src: Path) -> dict:
    """원본 ECG(time, ADC) · PPG(time, red, IR). 라벨링 열은 참고용으로만 읽는다."""
    ecg_name, ecg = load_source(src, r"^원본_ecg_.*\.csv$")
    ppg_name, ppg = load_source(src, r"^원본_ppg_.*\.csv$")
    out = {}
    for ch, name, df, cols in (("ecg", ecg_name, ecg, 1), ("ppg", ppg_name, ppg, 2)):
        t = df.iloc[:, 0].to_numpy(float)
        values = df.iloc[:, 1:1 + cols].to_numpy(np.int64)
        fs = (len(t) - 1) / (t[-1] - t[0])
        labels = []
        if df.shape[1] > 1 + cols:
            lab = df.iloc[:, -1]
            labels = [(round(float(t[i]), 1), str(v)) for i, v in lab.dropna().items()]
        out[ch] = dict(name=name, t0=float(t[0]), t_end=float(t[-1]), fs=float(fs), values=values, labels=labels)
    return out


def write_clip(raw: dict, out: Path, name: str, title: str, start: float, end: float, fast_forward: float,
               session: str) -> dict:
    folder = out / name
    folder.mkdir(parents=True, exist_ok=True)
    info = {}
    for ch, header in (("ecg", "time,adc"), ("ppg", "time,red,ir")):
        src = raw[ch]
        a = max(0, int(round((start - src["t0"]) * src["fs"])))
        b = min(len(src["values"]), int(round((end - src["t0"]) * src["fs"])))
        part = src["values"][a:b]
        t = np.arange(len(part)) / src["fs"]
        with (folder / f"{ch}.csv").open("w", encoding="utf-8", newline="") as handle:
            handle.write(header + "\n")
            body = np.column_stack([t, part])
            fmt = ["%.4f"] + ["%d"] * part.shape[1]
            np.savetxt(handle, body, fmt=fmt, delimiter=",")
        info[ch] = dict(samples=int(len(part)), first_sample=a, source=src["name"])
    meta = dict(fs={c: raw[c]["fs"] for c in ("ecg", "ppg")}, session_id=f"{session}_{name}",
                source_start_s=float(start), source_end_s=float(end),
                note="실측 원본에서 표본 번호로 자른 구간. 값은 바꾸지 않았다. 시각은 원본 CSV 기준(측정 시작 0초).",
                channels=info)
    manifest = dict(title=title, ecg="ecg.csv", ppg="ppg.csv", meta="meta.json",
                    source_start_s=float(start), source_end_s=float(end))
    if fast_forward > 0:
        manifest.update(fast_forward_s=float(fast_forward), display_start_s=float(start + fast_forward))
    (folder / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
    (folder / "recording.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=1), encoding="utf-8")
    return manifest


def parse_clip(text: str):
    parts = text.split(":")
    if len(parts) not in (3, 4):
        raise argparse.ArgumentTypeError("--clip 이름:시작:끝[:빠르게넘길초]")
    name, a, b = parts[0], float(parts[1]), float(parts[2])
    ff = float(parts[3]) if len(parts) == 4 else 0.
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,40}", name) or not 0 <= a < b or not 0 <= ff < b - a:
        raise argparse.ArgumentTypeError(f"잘못된 클립: {text}")
    return (name, f"{name} (원본 {a:.0f}~{b:.0f}초)", a, b, ff)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("source", type=Path, help="측정 zip 또는 폴더")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--preset", choices=sorted(PRESETS))
    p.add_argument("--clip", type=parse_clip, action="append", default=[], help="이름:시작:끝[:빠르게넘길초] (원본 시각)")
    p.add_argument("--session", default="rec20261118")
    a = p.parse_args()
    clips = list(PRESETS.get(a.preset, [])) + a.clip
    if not clips:
        p.error("--preset 또는 --clip 이 필요합니다")
    raw = load_raw(a.source)
    for ch in ("ecg", "ppg"):
        r = raw[ch]
        print(f"{ch}: {r['name']}  fs={r['fs']:.3f} Hz  {r['t0']:.3f}~{r['t_end']:.3f} s  표본 {len(r['values'])}")
    if raw["ecg"]["labels"]:
        print("라벨 표시 (원본 시각):", ", ".join(f"{t:.0f}s {v}" for t, v in raw["ecg"]["labels"]))
    for clip in clips:
        m = write_clip(raw, a.out, *clip, session=a.session)
        extra = f", 화면 시작 {m['display_start_s']:.0f}s (앞 {m['fast_forward_s']:.0f}s 빠르게)" if "fast_forward_s" in m else ""
        print(f"→ {a.out / clip[0]}: 원본 {clip[2]:.0f}~{clip[3]:.0f}s{extra}")


if __name__ == "__main__":
    main()
