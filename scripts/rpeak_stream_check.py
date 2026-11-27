#!/usr/bin/env python3
"""실시간 경로의 ECG R-peak 검출을 오프라인 기준과 비교 (모델 가중치·학습 불필요).

웹보드 판정 재생·Pi 실시간(Runtime)과 같은 방식으로 10초마다 버퍼를 잘라 R-peak 를 다시 찾고,
녹음 전체로 찾은 박동(학습·오프라인 평가와 같은 방식)과 마지막 30초 창에서 얼마나 같은지 센다.
버퍼: 기준 준비 전(약 390초까지)은 처음부터 현재까지, 그 뒤는 최근 330초 (pi_runtime._snapshot 과 같음).

  임계 global : 학습과 같음. 버퍼 전체 99 백분위 → 초반 잡음이 크면 임계가 올라가 박동을 놓친다.
  임계 rolling: 과거 10초 블록 8개의 99 백분위 중앙값 (--rpeak-threshold rolling).

예)
  python scripts/rpeak_stream_check.py --recording outputs/local-recordings/rec_1118
  python scripts/rpeak_stream_check.py --session data/converted/jberry_sensor/self01/2026-11-18_190640_s1
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from driver_state.data import Recording  # noqa: E402
from driver_state.teacher.rr import extract_rr  # noqa: E402
from driver_state.teacher.spec import TeacherSpec  # noqa: E402

READY_S = 390.      # 기준 수집(360초) + 첫 입력 창(30초). 그 뒤 Runtime 은 최근 330초만 넘긴다.
KEEP_S = 330.
TOL_S = 0.05


def load_ecg(args):
    if args.session:
        from driver_state.data_io import load_session
        rec = load_session(args.session, ('ecg',))
        return np.asarray(rec.signals["ecg"], float), float(rec.fs["ecg"])
    folder = Path(args.recording)
    from driver_state.local_recordings import RecordingCatalog
    catalog = RecordingCatalog(folder.parent)
    listing = catalog.refresh()["recordings"]
    match = [r for r in listing if Path(catalog.get(r["id"])["paths"]["ecg"]).parent == folder.resolve()]
    if not match:
        raise SystemExit(f"{folder}: 웹보드 원본 기록 폴더(recording.json)가 아닙니다.")
    entry = catalog.get(match[0]["id"])
    import pandas as pd
    ecg = pd.read_csv(entry["paths"]["ecg"], encoding="utf-8-sig").iloc[:, 1].to_numpy(float)
    return ecg, float(entry["rates"]["ecg"])


def beats(ecg, fs, spec, mode):
    rec = Recording("check", "stream", {"ecg": ecg}, {"ecg": fs}, prefiltered=False)
    return extract_rr(rec, spec, "ecg", mode).beat_t


def recall(found, ref):
    if len(ref) == 0:
        return np.nan, len(found)
    hit = np.array([np.any(np.abs(found - t) <= TOL_S) for t in ref])
    extra = int(sum(not np.any(np.abs(ref - t) <= TOL_S) for t in found))
    return float(hit.mean()), extra


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--recording", help="웹보드 원본 기록 폴더 (recording.json, ecg.csv)")
    src.add_argument("--session", help="규약 세션 폴더 (convert_jberry_sensor.py 결과)")
    parser.add_argument("--step", type=float, default=10.)
    parser.add_argument("--print-every", type=float, default=60.)
    parser.add_argument("--json", help="창별 결과 저장 경로")
    args = parser.parse_args()

    spec = TeacherSpec.from_dict(json.loads((ROOT / "models/bundle.json").read_text(encoding="utf-8"))["spec"])
    ecg, fs = load_ecg(args)
    duration = len(ecg) / fs
    ref = beats(ecg, fs, spec, "global")
    rows = []
    print(f"ECG {duration:.0f}초, {fs:g} Hz, 전체 녹음 기준 박동 {len(ref)}개 (허용 ±{TOL_S * 1000:.0f} ms)")
    print(f"{'창 끝':>6} | {'global 재현율':>12} {'추가':>4} | {'rolling 재현율':>13} {'추가':>4}")
    for end in np.arange(60., duration + 1e-9, args.step):
        start = 0. if end < READY_S else end - KEEP_S
        part = ecg[int(round(start * fs)):int(round(end * fs))]
        target = ref[(ref >= end - 30) & (ref < end - 0.5)]       # 마지막 0.5초는 양쪽 모두 불확실
        row = dict(end=float(end))
        for mode in ("global", "rolling"):
            b = beats(part, fs, spec, mode) + start
            r, extra = recall(b[(b >= end - 30) & (b < end - 0.5)], target)
            row[mode] = r
            row[f"{mode}_extra"] = extra
        rows.append(row)
        if end % args.print_every < args.step / 2:
            print(f"{end:6.0f} | {row['global']:12.0%} {row['global_extra']:4d} | {row['rolling']:13.0%} {row['rolling_extra']:4d}")
    print("\n구간별 평균 재현율 (마지막 30초 창)")
    for lo, hi in ((60, 400), (400, 800), (800, 1e9)):
        sel = [r for r in rows if lo <= r["end"] < hi]
        if sel:
            g = np.nanmean([r["global"] for r in sel])
            ro = np.nanmean([r["rolling"] for r in sel])
            label = f"{lo:.0f}~{hi:.0f}초" if hi < 1e9 else f"{lo:.0f}초~"
            print(f"  {label:>10}: global {g:.0%} / rolling {ro:.0%}  (창 {len(sel)}개)")
    for mode in ("global", "rolling"):
        ok = next((r["end"] for r in rows if all(x[mode] >= .8 for x in rows if x["end"] >= r["end"])), None)
        print(f"  {mode}: 이후 계속 80% 이상인 첫 시점 = {'없음' if ok is None else f'{ok:.0f}초'}")
    if args.json:
        Path(args.json).write_text(json.dumps(rows, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
