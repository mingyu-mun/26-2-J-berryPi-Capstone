"""기록된 세션을 웹보드로 재생한다 (origin = "recorded").

1) 먼저 번들로 추론:   python scripts/infer_bundle.py --session <세션> --out out/infer [--ppg-noshort]
2) 대시보드 실행:      python scripts/run_dashboard.py            (http://localhost:8000)
3) 재생:              python scripts/webboard_replay.py --session <세션> --infer out/infer --speed 10

보내는 이벤트 (web/dashboard/contract 규약 v1.2):
- bio.wave       : 공통 전처리 경로(100 Hz, ECG 0.5–40 Hz / PPG 0.5–8 Hz) 파형을 1초 묶음으로. 단위 a.u.
                   quality_ok 는 그 1초 원신호에 대한 기존 품질 검사(window_quality) 결과
- ai.decision    : windows.csv 의 학생 점수·품질로 WindowIn 을 만들고 StateMachineV2.step() 의 StepOut 그대로
- system.status  : 생체 기준 수집 상태 (첫 세션 기준 구간이 끝나기 전 collecting, 이후 ready / 실패면 retry_required)
CAN 은 보내지 않는다 (can_origin = "unavailable", can_drowsy = null).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from driver_state.data_io import load_session  # noqa: E402
from driver_state.preprocessing import prepare_channel, window_quality  # noqa: E402
from driver_state.spec import InputSpec  # noqa: E402
from driver_state.state_machine_v2 import StateMachineV2, V2Config, WindowIn  # noqa: E402
from driver_state.webboard_connector import WebboardPublisher, ai_decision, bio_wave, system_status, wave_channel  # noqa: E402


def sig(x):
    return float(1 / (1 + np.exp(-x)))


def _f(r, k):
    v = r.get(k)
    return float(v) if v is not None and np.isfinite(v) else None


def build_events(sessions: list[Path], infer_dir: Path, sid_prefix: str, ppg_col: str, chunk_sec: float = 1.0) -> list[dict]:
    W = pd.read_csv(infer_dir / "windows.csv")
    summ = json.loads((infer_dir / "summary.json").read_text(encoding="utf-8"))
    bcheck = summ.get("baseline_check", {})
    base_ses, b0, b1 = summ["baseline_used"]
    spec = InputSpec()
    events = []
    for d in sessions:
        rec = load_session(d, channels=("ecg", "ppg"))
        rec.prefiltered = False
        sid = f"{sid_prefix}-{rec.session_id}"
        # 파형 (공통 전처리 100 Hz)
        ch = {}
        for c in ("ecg", "ppg"):
            if c in rec.signals:
                filt, raw = prepare_channel(rec.signals[c], rec.fs[c], c, spec, False)
                ch[c] = (filt, raw)
        fs = spec.fs
        n = max(len(v[0]) for v in ch.values())
        step = int(round(chunk_sec * fs))
        for a in range(0, n - step + 1, step):
            t0 = a / fs
            waves = {}
            for c, (filt, raw) in ch.items():
                if a + step <= len(filt):
                    ok = bool(window_quality(raw[None, a:a + step], spec, (c,)))
                    waves[c] = wave_channel(t0, fs, filt[a:a + step], "a.u.", "common_preprocessed", ok)
            events.append(bio_wave(sid, t0 + chunk_sec, ecg=waves.get("ecg"), ppg=waves.get("ppg"), origin="recorded"))
        # 기준 상태
        is_base = rec.session_id == base_ses
        ready_t = b1 if is_base else 0.0
        if is_base:
            for t in np.arange(0.0, b1, 10.0):
                events.append(system_status(sid, t, bio_status="collecting", clean_windows=None, bio_elapsed_s=t,
                                            connections={"ecg": "connected", "ppg": "connected", "can": "disconnected",
                                                         "model": "connected"}, origin="recorded"))
        status = "ready"                                   # infer_bundle 이 기준을 잡은 경우에만 windows.csv 가 있다
        events.append(system_status(sid, ready_t, bio_status=status,
                                    clean_windows=bcheck.get("clean_windows") if is_base else None,
                                    bio_elapsed_s=ready_t if is_base else None,
                                    connections={"ecg": "connected", "ppg": "connected", "can": "disconnected", "model": "connected"},
                                    origin="recorded"))
        # 판정 (StepOut 은 실제 StateMachineV2 출력)
        sm = StateMachineV2(V2Config())
        g = W[W["session_id"] == rec.session_id].sort_values("t_rel")
        for r in g.to_dict("records"):
            ok_e, ok_p = bool(r["valid"]), bool(r.get("valid_ppg", False))
            win = WindowIn(t=float(r["t_rel"]) + 30.0,
                           p_ep=sig(r["student_EP_logit"]) if ok_e and np.isfinite(r["student_EP_logit"]) else None,
                           q_ecg=float(r["q_ecg"]) if np.isfinite(r["q_ecg"]) else 1.0,
                           p_ppg=sig(r[ppg_col]) if ok_p and np.isfinite(r[ppg_col]) else None,
                           q_ppg=_f(r, "q_ppg"), hr_ecg=_f(r, "hr_ecg"), hr_ppg=_f(r, "hr_ppg"),
                           ok_ecg=ok_e, ok_ppg=ok_p, can_drowsy=None)
            ui_in = {**win.__dict__, "q_ecg": _f(r, "q_ecg")}
            if win.t < ready_t:                       # 기준이 준비되기 전의 창: 판정하지 않는다
                events.append(ai_decision(sid, ui_in, None, baseline_status="collecting", bio_origin="recorded",
                                          can_origin="unavailable", origin="recorded"))
                continue
            out = sm.step(win)
            events.append(ai_decision(sid, ui_in, out, baseline_status="ready", bio_origin="recorded",
                                      can_origin="unavailable", origin="recorded"))
    return events


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--session", nargs="+", required=True, help="infer_bundle 에 넣은 세션 폴더들 (같은 순서)")
    ap.add_argument("--infer", required=True, help="infer_bundle --out 폴더 (windows.csv, summary.json)")
    ap.add_argument("--url", default="http://127.0.0.1:8000")
    ap.add_argument("--speed", type=float, default=1.0, help="재생 배속. 0 이면 기다리지 않고 한꺼번에")
    ap.add_argument("--session-id", default="replay")
    ap.add_argument("--ppg-noshort", dest="ppg_noshort", action="store_true")
    ap.add_argument("--dump", help="보내지 않고 JSONL 로 저장 (웹 화면의 '기록 파일' 재생용)")
    a = ap.parse_args()
    ev = build_events([Path(s) for s in a.session], Path(a.infer), a.session_id,
                      "student_PPG_ns_logit" if a.ppg_noshort else "student_PPG_logit")
    order = {s: i for i, s in enumerate(dict.fromkeys(e["session_id"] for e in ev))}
    ev.sort(key=lambda e: (order[e["session_id"]], e["t_rel_s"]))
    counts = ", ".join(f"{k} {sum(e['type'] == k for e in ev)}" for k in ("bio.wave", "ai.decision", "system.status"))
    print(f"이벤트 {len(ev)}개 ({counts})")
    if a.dump:
        with open(a.dump, "w", encoding="utf-8") as f:
            for e in ev:
                f.write(json.dumps(e, ensure_ascii=False, allow_nan=False) + "\n")
        print(f"저장: {a.dump}")
        return
    pub = WebboardPublisher(a.url)
    if a.speed <= 0:
        print(pub.send(ev)[-1]); return
    t_wall, last_sid, t_base = time.monotonic(), None, 0.0
    for e in ev:
        if e["session_id"] != last_sid:
            last_sid, t_wall, t_base = e["session_id"], time.monotonic(), e["t_rel_s"]
        wait = (e["t_rel_s"] - t_base) / a.speed - (time.monotonic() - t_wall)
        if wait > 0:
            time.sleep(wait)
        pub.send(e)
    print("재생 끝")


if __name__ == "__main__":
    main()
