#!/usr/bin/env python3
"""시뮬레이터 CAN 시나리오 만들기·점검 (통합 테스트용, Unity·CAN 장치·모델 가중치 불필요).

  make   합성 주행 JSONL 생성 (--can-sim-scenario 로 웹보드에 넣는 형식)
  check  시나리오의 10초 창별 can_drowsy 와 원인 출력 (Unity 기록 점검에도 사용)
  bio    합성 원본 ECG·PPG 기록 폴더 생성 (웹보드 '원본 기록' 재생 흐름 점검용. 모델 판정 결과는 의미 없음)
  e2e    고정 고점수 가짜 모델 + 실제 상태 머신 v2 + Runtime 으로 CAN 유무를 비교하고
         모든 이벤트를 웹보드 v1.2 규약으로 검증. 모델 성능과 무관한 배선 확인용.

예)
  python scripts/can_sim_scenario.py make outputs/can-sim/warnings.jsonl --minutes 15
  python scripts/can_sim_scenario.py make outputs/can-sim/quiet.jsonl --minutes 15 --no-warnings
  python scripts/can_sim_scenario.py check outputs/can-sim/warnings.jsonl --until 180
  python scripts/can_sim_scenario.py e2e outputs/can-sim/warnings.jsonl
  python scripts/can_sim_scenario.py bio outputs/local-recordings/synthetic_bio --minutes 20
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from driver_state.can_assist import CanAssist, CODES, read_samples, synthetic_highway  # noqa: E402


def make(args):
    out = Path(args.out)
    if out.exists() and not args.force:
        raise SystemExit(f"{out} 이(가) 이미 있습니다. --force 로 덮어쓰세요.")
    out.parent.mkdir(parents=True, exist_ok=True)
    every = 0 if args.no_warnings else args.every
    rows = 0
    with out.open("w", encoding="utf-8") as file:
        meta = dict(source="synthetic", generator="scripts/can_sim_scenario.py", minutes=args.minutes,
                    warning_every_s=every, first_warning_s=args.first, lane_change_every_s=args.lane_change_every,
                    note="합성 시연 데이터. 성능 근거가 아님.")
        file.write(json.dumps(meta, ensure_ascii=False) + "\n")
        for s in synthetic_highway(args.minutes * 60, deviation_every_s=every, first_deviation_s=args.first,
                                   lane_change_every_s=args.lane_change_every, steering_step_ms=args.steering_ms):
            row = {"t_us": s.t_us}
            if s.has_speed:
                row["speed_kmh"] = s.speed
            if s.has_steer:
                row["steering_deg"] = s.steer
            row.update(s.codes)
            file.write(json.dumps(row) + "\n")
            rows += 1
    print(f"{out}: {rows} 행, {args.minutes}분")


def bio(args):
    """Gaussian PQRST ECG (ADC 0~1023) + pulse PPG (RED·IR) with slowly falling heart rate."""
    import numpy as np
    out = Path(args.out)
    if out.exists() and any(out.iterdir()) and not args.force:
        raise SystemExit(f"{out} 이(가) 비어 있지 않습니다. --force 로 덮어쓰세요.")
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    duration = args.minutes * 60
    beats, t = [], 1.0
    while t < duration - 1:
        frac = t / duration
        hr = 72 - 12 * max(0., (frac - .5) * 2)                        # 후반부 심박 서서히 감소
        rr = 60 / hr * (1 + .04 * np.sin(2 * np.pi * t / 4.5)) + rng.normal(0, .015 + .02 * frac)
        beats.append(t)
        t += max(.45, rr)
    beats = np.array(beats)
    fe, fp = args.ecg_fs, 100.
    te = np.arange(0, duration, 1 / fe)
    ecg = np.full_like(te, 512.)
    for amp, off, width in ((25, -.18, .025), (-30, -.03, .01), (330, 0., .012), (-60, .03, .012), (60, .25, .045)):
        for b in beats:
            sl = slice(int((b + off - 5 * width) * fe), int((b + off + 5 * width) * fe) + 1)
            ecg[sl] += amp * np.exp(-.5 * ((te[sl] - b - off) / width) ** 2)
    ecg += rng.normal(0, 3, te.size)
    tp = np.arange(0, duration, 1 / fp)
    pulse = np.zeros_like(tp)
    for b in beats:
        sl = slice(int((b + .2) * fp), int((b + 1.2) * fp))
        x = tp[sl] - b - .2
        pulse[sl] += np.exp(-.5 * ((x - .15) / .07) ** 2) + .4 * np.exp(-.5 * ((x - .4) / .1) ** 2)
    ir = 60000 + 1500 * pulse + rng.normal(0, 20, tp.size)
    red = 50000 + 1000 * pulse + rng.normal(0, 20, tp.size)
    with (out / "ecg.csv").open("w") as f:
        f.write("time,adc\n")
        f.writelines(f"{a:.4f},{int(np.clip(round(v), 0, 1023))}\n" for a, v in zip(te, ecg))
    with (out / "ppg.csv").open("w") as f:
        f.write("time,red,ir\n")
        f.writelines(f"{a:.2f},{int(r)},{int(i)}\n" for a, r, i in zip(tp, red, ir))
    (out / "meta.json").write_text(json.dumps(dict(fs={"ecg": fe, "ppg": fp}, session_id=f"synthetic-bio-{args.seed}",
                                                 note="합성 원본. 측정값이 아님."), ensure_ascii=False), encoding="utf-8")
    (out / "recording.json").write_text(json.dumps(dict(title=f"합성 원본 {args.minutes:g}분 (시험용)", ecg="ecg.csv",
                                                      ppg="ppg.csv", meta="meta.json"), ensure_ascii=False),
                                        encoding="utf-8")
    print(f"{out}: ECG {te.size}표본({fe:g} Hz), PPG {tp.size}표본, 박동 {beats.size}개")


def timeline(a, until_s, window_s=10):
    for end in range(int(window_s), int(until_s) + 1, int(window_s)):
        for step in range((end - int(window_s)) * 10 + 1, end * 10 + 1):
            a.advance(step / 10)
        yield end, a.window_flag(end), a.window_reasons(end), {k: a.code(k) for k in CODES}


def check(args):
    a = CanAssist(lambda: read_samples(args.scenario))
    counts = {True: 0, False: 0, None: 0}
    for end, flag, reasons, codes in timeline(a, args.until):
        counts[flag] += 1
        if flag or reasons or args.verbose:
            print(f"{end - 10:6.0f}-{end:<6.0f}s  can_drowsy={str(flag):5}  원인={','.join(reasons) or '-'}")
    print(f"창 합계: True {counts[True]} / False {counts[False]} / None {counts[None]}")
    if counts[None] == sum(counts.values()):
        print("경고: 모든 창이 None 입니다. 경고 상태(LaneSafe_indi·Careless_Warn)나 CAN 수신을 확인하세요.")


def e2e(args):
    from driver_state.pi_runtime import Runtime, load_config
    from driver_state.pi_sensors import Sample
    from driver_state.state_machine_v2 import StateMachineV2, WindowIn
    from driver_state.web_dashboard import Dashboard, validate_event

    class FixedScore:
        def __init__(self):
            self.reset()

        def reset(self):
            self.sm = StateMachineV2()

        def consume(self, signals, fs, start, end, can_drowsy=None):
            p = args.score if end >= args.drowsy_from else 0.2
            wi = WindowIn(t=end, p_ep=p, q_ecg=0., p_ppg=p, q_ppg=0., hr_ecg=70., hr_ppg=70., can_drowsy=can_drowsy)
            return dict(status="ready", clean_windows=20, baseline_range=[0., 30.], input=asdict(wi),
                        output=asdict(self.sm.step(wi)))

    class Sink(list):
        def put(self, event): self.append(event)
        def clear(self): del self[:]
        def flush(self): return True

    def run(can):
        now, sink = [0.], Sink()
        runtime = Runtime(load_config(ROOT / "tests/sensors.example.json"), FixedScore(), sink,
                          clock=lambda: now[0], origin="recorded", synchronous=True, can_assist=can)
        for c in ("ecg", "ppg"):
            runtime.state(c, True, "e2e fixture")
        for second in range(1, int(args.seconds) + 1):
            for c, v in (("ecg", 512.), ("ppg", 60000.)):
                runtime.emit(c, Sample(second - .5, v))
            now[0] = float(second)
            runtime.tick()
        return sink

    validator = Dashboard().validator
    for label, can in (("CAN 없음", None), ("CAN 시나리오", CanAssist(lambda: read_samples(args.scenario)))):
        events = run(can)
        for event in events:
            validate_event(event, validator)
        board = Dashboard()
        accepted = sum(board.ingest(events[i:i + 1000])["accepted"] for i in range(0, len(events), 1000))
        out = [e["payload"] for e in events if e["type"] == "ai.decision" and e["payload"]["output"]]
        def when(pred):
            return next((p["output"]["t"] for p in out if pred(p)), None)
        caution = when(lambda p: p["output"]["level"] == 1)
        cand = when(lambda p: p["output"]["candidate"])
        drowsy = next((p for p in out if p["output"]["level"] == 2), None)
        sec = lambda v: "없음" if v is None else f"{v:g}s"
        print(f"[{label}] 이벤트 {len(events)}개 규약 통과, 웹보드 수신 {accepted}개 · "
              f"주의 {sec(caution)} · 졸음 후보 {sec(cand)} · 졸음 "
              f"{sec(drowsy and drowsy['output']['t'])} (근거 {drowsy['confirmation_source'] if drowsy else '-'})")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("make")
    p.add_argument("out")
    p.add_argument("--minutes", type=float, default=15)
    p.add_argument("--every", type=float, default=30, help="이상 상황(차선·부주의 경고 번갈아) 간격 초")
    p.add_argument("--first", type=float, default=20, help="첫 이상 상황 시각 초")
    p.add_argument("--lane-change-every", type=float, default=60, help="깜빡이 켠 차로 변경 간격 초 (0=없음)")
    p.add_argument("--no-warnings", action="store_true", help="경고 없이 차로 변경만 (False·제외 확인용)")
    p.add_argument("--steering-ms", type=int, default=20, help="조향 표본 간격 ms (Unity 50 Hz=20)")
    p.add_argument("--force", action="store_true")
    p.set_defaults(func=make)
    p = sub.add_parser("check")
    p.add_argument("scenario")
    p.add_argument("--until", type=float, default=300)
    p.add_argument("--verbose", action="store_true", help="경고 없는 창도 출력")
    p.set_defaults(func=check)
    p = sub.add_parser("bio")
    p.add_argument("out")
    p.add_argument("--minutes", type=float, default=20)
    p.add_argument("--ecg-fs", type=float, default=250.)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--force", action="store_true")
    p.set_defaults(func=bio)
    p = sub.add_parser("e2e")
    p.add_argument("scenario")
    p.add_argument("--seconds", type=float, default=600)
    p.add_argument("--score", type=float, default=0.9, help="가짜 모델 점수")
    p.add_argument("--drowsy-from", type=float, default=120, help="이 시각부터 고점수")
    p.set_defaults(func=e2e)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
