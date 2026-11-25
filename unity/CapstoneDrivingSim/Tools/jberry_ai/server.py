"""
server.py — Unity 시뮬레이터 ↔ J_Berry PI 실제 모델(S1-EP + 상태 머신 v2) 실시간 추론 서버.

  Unity (센서 역할)  ──UDP 5005──▶  server.py  ──UDP 5006──▶  Unity (화면)
     hello / bio / can                 │
                                       ├─ SimLogs/ai/events_<세션>.jsonl  (웹보드 규약 v1.2 이벤트 4종 그대로)
                                       ├─ SimLogs/ai/windows_<세션>.csv   (창마다 WindowIn·StepOut·logit)
                                       └─ (선택) --webboard http://127.0.0.1:8000 → /api/events

규약(WEBBOARD_DATA_CONTRACT v1.2) 지키는 점
  · 생체 출처 bio_origin = "recorded" (기록 재생) 또는 "live" (실측 모드), CAN 출처 can_origin = "simulated" (시뮬레이터 합성)
  · can_drowsy 는 검증된 규칙이 없으므로 항상 null. 상태는 StateMachineV2.step 출력만 쓴다 (UI 가 따로 판정하지 않음)
  · 값이 없으면 null (Unity 로 보내는 UI 패킷은 JsonUtility 제약 때문에 -1 + has_* 플래그로 표시)

실행 (보통 Unity AiLink 가 .venv 의 python 으로 자동 실행):
  .venv/Scripts/python -u server.py --mode causal --log-dir ../../SimLogs/ai
"""
from __future__ import annotations

import argparse
import json
import math
import queue
import socket
import sys
import threading
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from engine import REPO, StreamEngine  # noqa: E402
from driver_state import webboard_connector as wb  # noqa: E402

LEVEL_NAME = {-1: "측정 불가", 0: "정상", 1: "주의", 2: "졸음"}


def r3(x, nd=3):
    if x is None:
        return -1.0
    try:
        v = float(x)
    except (TypeError, ValueError):
        return -1.0
    return round(v, nd) if math.isfinite(v) else -1.0


def has(x):
    try:
        return x is not None and math.isfinite(float(x))
    except (TypeError, ValueError):
        return False


def align_base(bn: np.ndarray, xn: np.ndarray, fs: float) -> np.ndarray:
    """기준 파형 bn 에서 현재 파형 xn 과 길이가 같은 구간을 고름. 시작점은 bn 의 처음 3초 안에서,
    xn 의 처음 1.5초와 상관이 가장 큰 곳 → 첫 박동의 위상이 맞고, 그 뒤로 박동이 어긋나는 정도가 심박 차이를 보여 줌."""
    n, m = len(xn), min(len(xn), int(1.5 * fs))
    lags = max(1, min(int(3.0 * fs), len(bn) - n + 1))
    c = np.correlate(bn[:lags + m - 1], xn[:m], "valid")
    L = int(np.argmax(c)) if len(c) else 0
    out = bn[L:L + n]
    return np.pad(out, (0, n - len(out))) if len(out) < n else out

class Session:
    """Unity Play 한 번 = 세션 하나. 새 hello(세션 id 다름) 가 오면 새로 만든다 (규약: 새 session_id + 상태 머신 reset)."""

    def __init__(self, hello: dict, args, log_dir: Path):
        self.id = hello.get("session") or time.strftime("sim_%Y%m%d_%H%M%S")
        self.fs_ecg = float(hello.get("fs_ecg") or 0)
        self.fs_ppg = float(hello.get("fs_ppg") or 0)
        self.mode = hello.get("mode") or args.mode
        if self.mode not in ("causal", "exact"):
            self.mode = "causal"
        self.replay_name = hello.get("replay") or ""
        self.replay_offset = float(hello.get("replay_offset") or 0.0)
        self.origin = hello.get("origin") if hello.get("origin") in ("recorded", "live") else "recorded"
        self.eng = StreamEngine(self.fs_ecg, self.fs_ppg, mode=self.mode, session_id=self.id,
                                force_baseline=True) if self.fs_ecg > 0 and self.fs_ppg > 0 else None
        self.t_start = time.time()
        self.last_bio_rx = None
        self.last_can_rx = None
        self.can = {}                  # 최근 CAN 값 + 관측 시각
        self.hist = []                 # (t_end, p_used, level)
        self.last = None               # 최근 WindowResult
        self.last_conf_src = None
        self.n_windows = 0
        self.preroll = 0.0
        self.err = ""
        log_dir.mkdir(parents=True, exist_ok=True)
        self.ev = open(log_dir / f"events_{self.id}.jsonl", "w", encoding="utf-8")
        self.wc = open(log_dir / f"windows_{self.id}.csv", "w", encoding="utf-8")
        self.wc.write("t_start,t_end,replay_t_end,valid,valid_ppg,logit_ep,logit_ppg,p_ep,p_ppg,q_ecg,q_ppg,hr_ecg,hr_ppg,"
                      "level,level_name,candidate,score,p_used,confidence,channel,frozen,proc_ms\n")
        print(f"[server] 새 세션 {self.id}  mode={self.mode}  fs ecg={self.fs_ecg:.2f} ppg={self.fs_ppg:.2f}  "
              f"replay={self.replay_name}@{self.replay_offset:.0f}s", flush=True)

    # 세션 경과 시간 = 엔진이 받은 생체 표본 길이 (측정 시각 기준)
    def t_rel(self):
        return self.eng.duration() if self.eng else 0.0

    def close(self):
        for f in (self.ev, self.wc):
            try:
                f.close()
            except Exception:
                pass


class Server:
    def __init__(self, args):
        self.a = args
        self.log_dir = Path(args.log_dir).resolve()
        self.rx = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.rx.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 4 * 1024 * 1024)
        self.rx.bind((args.host, args.in_port))
        self.tx = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.dst = (args.host, args.out_port)
        self.s: Session | None = None
        self.last_rx = time.time()
        self.pub_q: queue.Queue | None = None
        if args.webboard:
            self.pub_q = queue.Queue(maxsize=5000)
            threading.Thread(target=self._pub_loop, daemon=True).start()
        print(f"[server] udp://{args.host}:{args.in_port} → :{args.out_port}  repo={REPO.name}  "
              f"webboard={args.webboard or '-'}", flush=True)

    # ------------------------------------------------------------ 규약 이벤트
    def emit(self, ev: dict):
        ev = wb.clean(ev)
        if self.s:
            self.s.ev.write(json.dumps(ev, ensure_ascii=False) + "\n")
        if self.pub_q is not None:
            try:
                self.pub_q.put_nowait(ev)
            except queue.Full:
                pass

    def _pub_loop(self):
        pub = wb.WebboardPublisher(self.a.webboard)
        while True:
            batch = [self.pub_q.get()]
            while not self.pub_q.empty() and len(batch) < 200:
                batch.append(self.pub_q.get_nowait())
            try:
                pub.send(batch)
            except Exception as e:  # 웹보드가 꺼져 있어도 추론은 계속
                print(f"[server] webboard 전송 실패: {e}", flush=True)
                time.sleep(2)

    # ------------------------------------------------------------ 수신 처리
    def on_packet(self, m: dict):
        kind = m.get("type")
        if kind == "hello":
            if self.s is None or m.get("session") != self.s.id:
                if self.s:
                    self.s.close()
                self.s = Session(m, self.a, self.log_dir)
                if m.get("baseline_dir"):
                    self.baseline_file(Path(m["baseline_dir"]))
                pre_dir, pre_sec = m.get("preroll_dir"), float(m.get("preroll_sec") or 0)
                if pre_dir and pre_sec > 0:
                    # 자동 기준: 기록 0 s ~ 시작 위치 / 기준 파일: 시작 위치 앞 최대 300 s (장기 특징 창 채우기)
                    self.preroll(Path(pre_dir), max(0.0, self.s.replay_offset - pre_sec), pre_sec)
            return
        s = self.s
        if s is None or s.eng is None:
            return
        if kind == "bio":
            s.eng.push("ecg", m.get("ecg") or [])
            s.eng.push("ppg", m.get("ppg") or [])
            s.last_bio_rx = time.time()
        elif kind == "can":
            self.on_can(m)

    def baseline_file(self, d: Path):
        """별도 기준 측정 파일(세션 폴더)로 개인 기준을 먼저 정한다 → 입력 0 s 부터 바로 판정."""
        import pandas as pd
        s = self.s
        try:
            meta = json.loads((d / "meta.json").read_text(encoding="utf-8"))
            e = pd.read_csv(d / "ecg.csv")["value"].to_numpy(np.float64)
            p = pd.read_csv(d / "ppg.csv")["value"].to_numpy(np.float64)
            t0 = time.time()
            s.eng.use_baseline_recording(e, p, float(meta["fs"]["ecg"]), float(meta["fs"]["ppg"]), d.name)
            print(f"[server] 기준 파일 {d.name} 처리 {time.time() - t0:.1f}s", flush=True)
        except Exception as ex:
            s.err = f"기준 파일 실패: {ex}"
            print("[server] " + s.err, flush=True)

    def preroll(self, d: Path, start: float, sec: float):
        """기록의 [start, start+sec) 를 빠르게 먼저 넣는다 (시연용 옵션, 화면에 표시됨). 실제 측정값을 그대로 넣을 뿐 값을 만들지 않는다.
        세션 시각 0 = 기록 start 초가 되므로 재생 시각 = replay_offset − preroll + 세션 시각."""
        import pandas as pd
        s = self.s
        try:
            e = pd.read_csv(d / "ecg.csv")["value"].to_numpy(np.float64)
            p = pd.read_csv(d / "ppg.csv")["value"].to_numpy(np.float64)
            s.eng.push("ecg", e[int(start * s.fs_ecg): int((start + sec) * s.fs_ecg)])
            s.eng.push("ppg", p[int(start * s.fs_ppg): int((start + sec) * s.fs_ppg)])
            s.preroll = sec
            t0 = time.time()
            for r in s.eng.poll():
                self.on_window(r)
            print(f"[server] preroll {sec:.0f}s 처리 {time.time() - t0:.1f}s, 창 {s.n_windows}", flush=True)
        except Exception as ex:
            s.err = f"preroll 실패: {ex}"
            print("[server] " + s.err, flush=True)

    def on_can(self, m: dict):
        """Unity 차량 신호 → 규약 can.telemetry (origin simulated). Can list.xlsx 표현 범위로 양자화."""
        s = self.s
        ts = m.get("t") or []
        if not ts:
            return
        s.last_can_rx = time.time()
        i = len(ts) - 1
        off = s.preroll                      # Unity 시각(Play 기준) → 세션 시각 (preroll 만큼 앞당겨 시작)
        t = float(ts[i]) + off
        spd = m.get("speed") or []
        st = m.get("steer") or []
        ac = m.get("accel") or []
        br = m.get("brake") or []
        gear = m.get("gear") or []
        lt = m.get("turn") or []
        sw = float(st[i]) if st else None
        prev = s.can.get("steer"), s.can.get("t_steer")
        s.can.update(t=t, speed=int(max(0, min(255, round(spd[i])))) if spd else None,
                     steer=round(sw, 1) if sw is not None and abs(sw) < 3276.7 else None, t_steer=t,
                     accel=int(max(0, min(254, round(ac[i] * 254)))) if ac else None,
                     brake=int(max(0, min(127, round(br[i] * 127)))) if br else None,
                     gear=int(gear[i]) if gear else None, turn=int(lt[i]) if lt else 0)
        rate = None
        if prev[0] is not None and s.can["steer"] is not None and t > prev[1]:
            rate = (s.can["steer"] - prev[0]) / (t - prev[1])
        turn = s.can["turn"]
        payload = {
            "speed_kmh": s.can["speed"], "steering_angle_deg": s.can["steer"],
            "road_wheel_angle_deg": None, "steering_rate_deg_s": None if rate is None else round(rate, 1),
            "accel_pedal_raw": s.can["accel"], "brake_pedal_raw": s.can["brake"],
            "accelerator_pct": None, "brake_pct": None, "baseline_coverage_pct": None,
            "deviation_status": "unavailable", "can_drowsy": None, "rule_id": None, "signal_map_id": None,
            "status_codes": {"GearPosition": s.can["gear"], "LTurnSig_Indi": 1 if turn < 0 else 0,
                             "RTurnSig_Indi": 1 if turn > 0 else 0},
            "signal_times_s": {"speed": t, "steering": t, "accelerator": t, "brake": t},
        }
        self.emit(wb.envelope("can.telemetry", s.id, "simulated", t, payload))

    # ------------------------------------------------------------ 판정
    def on_window(self, r):
        s = self.s
        st = r.step
        s.last = r
        s.n_windows += 1
        p_used = st.p_used
        s.hist.append((r.t_end, -1.0 if p_used is None else float(p_used), int(st.level)))
        s.hist = s.hist[-120:]
        bstat = "ready" if s.eng.baseline_ready else s.eng.base_status
        ev = wb.ai_decision(s.id, r.win_in, st, baseline_status=bstat, bio_origin=s.origin,
                            can_origin="simulated" if s.last_can_rx else "unavailable")
        s.last_conf_src = ev["payload"].get("confirmation_source")
        self.emit(ev)
        w = r.win_in
        rep = s.replay_offset - s.preroll + r.t_end
        s.wc.write(",".join(str(x) for x in [
            r.t_start, r.t_end, round(rep, 2), int(r.valid), int(r.valid_ppg), r3(r.logit_ep, 4), r3(r.logit_ppg, 4),
            r3(w.p_ep, 4), r3(w.p_ppg, 4), r3(w.q_ecg, 4), r3(w.q_ppg, 4), r3(w.hr_ecg, 2), r3(w.hr_ppg, 2),
            st.level, st.level_name, int(st.candidate), r3(st.score, 4), r3(st.p_used, 4), r3(st.confidence, 3),
            st.channel, int(st.frozen), round(r.proc_ms, 1)]) + "\n")
        s.wc.flush()
        print(f"[win] t={r.t_end:7.1f}s replay={rep:7.1f}s lv={st.level}({st.level_name}) cand={int(st.candidate)} "
              f"p={r3(st.p_used)} ch={st.channel} q={r3(w.q_ecg, 2)} hr={r3(w.hr_ecg, 1)} {r.proc_ms:.0f}ms", flush=True)

    # ------------------------------------------------------------ Unity 로 보낼 화면 패킷
    def ui_packet(self, wave: bool) -> dict:
        s = self.s
        e = s.eng
        pk = {"type": "ui", "session": s.id, "mode": s.mode, "model_id": "S1-EP", "sm_id": "v2",
              "bio_origin": s.origin, "can_origin": "simulated" if s.last_can_rx else "unavailable",
              "t_rel": round(s.t_rel(), 2), "preroll": s.preroll, "error": s.err,
              "decision_delay": round(e.lookahead + e.edge_margin + 30, 1) if e else -1}
        if e is None:
            pk["base_status"] = "unavailable"
            return pk
        bstat = e.base_status if not e.baseline_ready or e.base_status == "retry_required" else "ready"
        pk.update(feat_ready=round(min(1.0, s.t_rel() / max(e.spec.scales)), 3),
                  base_status=bstat, base_forced=e.base_forced, base_source=e.base_source, clean_windows=e.base_clean,
                  required_clean=e.cfg.base_min_clean, base_wait=e.cfg.base_wait, base_max=e.cfg.base_max_wait,
                  base_a=r3(e.base_range[0] if e.base_range else None, 1),
                  base_b=r3(e.base_range[1] if e.base_range else None, 1),
                  n_windows=s.n_windows, ecg_rx=bool(s.last_bio_rx and time.time() - s.last_bio_rx < 3),
                  can_rx=bool(s.last_can_rx and time.time() - s.last_can_rx < 2))
        r = s.last
        if r is not None:
            st, w = r.step, r.win_in
            pk.update(has_out=True, level=int(st.level), level_name=st.level_name, candidate=bool(st.candidate),
                      score=r3(st.score), p_used=r3(st.p_used), confidence=r3(st.confidence), channel=st.channel,
                      frozen=bool(st.frozen), confirmation_source=s.last_conf_src or "",
                      p_ep=r3(w.p_ep), p_ppg=r3(w.p_ppg), q_ecg=r3(w.q_ecg), q_ppg=r3(w.q_ppg),
                      hr_ecg=r3(w.hr_ecg, 1), hr_ppg=r3(w.hr_ppg, 1), logit_ep=r3(r.logit_ep),
                      win_end=round(r.t_end, 1), proc_ms=round(r.proc_ms, 1),
                      out_age=round(s.t_rel() - r.t_end, 1), th_on=e.cfg.th_on, th_off=e.cfg.th_off)
            for ch in ("ecg", "ppg"):
                bm = ((getattr(e, "base_rr", None) or {}).get(ch) or (None, None))[0]
                pk[f"{ch}_hrb"] = round(60000.0 / bm, 1) if bm else -1
            if r.zr is not None:
                top = np.argsort(-np.abs(r.zr))[:3]
                pk["z_names"] = [e.names[i] for i in top]
                pk["z_vals"] = [round(float(r.zr[i]), 2) for i in top]
        if s.hist:
            pk["hist_t"] = [round(h[0], 1) for h in s.hist]
            pk["hist_p"] = [round(h[1], 3) for h in s.hist]
            pk["hist_lv"] = [h[2] for h in s.hist]
        if wave:
            for ch in ("ecg", "ppg"):
                pv = None
                try:
                    pv = e.preview(ch, self.a.wave_sec + 2.0)
                except Exception:
                    pv = None
                if pv is not None:
                    fs, x, t0 = pv
                    n = int(self.a.wave_sec * fs)
                    x = x[-n - int(1.0 * fs): len(x) - int(1.0 * fs)]       # 끝 1초는 필터 가장자리라 버림
                    bw = (getattr(e, "base_wave", None) or {}).get(ch)
                    if bw is not None:
                        # 기준 파형의 산포로 둘 다 나눔 → 진폭 차이가 그대로 보임
                        sd = float(np.std(bw)) or 1.0
                        xn = (x - np.median(x)) / sd
                        bn = (bw - np.median(bw)) / sd
                        pk[f"{ch}_b"] = [round(float(v), 3) for v in align_base(bn, xn, fs)]
                    else:
                        sd = float(np.std(x)) or 1.0
                        xn = (x - np.median(x)) / sd
                    pk[f"{ch}_w"] = [round(float(v), 3) for v in xn]
                    pk["wave_fs"] = fs
                    pk[f"{ch}_w_t1"] = round(t0 + (len(x) + int(1.0 * fs)) / fs - 1.0, 2)
        return pk

    def status_event(self):
        s = self.s
        e = s.eng
        bstat = ("ready" if e.baseline_ready and e.base_status != "retry_required" else e.base_status) if e else "waiting"
        conn = {"ecg": "connected" if s.last_bio_rx and time.time() - s.last_bio_rx < 3 else "disconnected",
                "ppg": "connected" if s.last_bio_rx and time.time() - s.last_bio_rx < 3 else "disconnected",
                "can": "connected" if s.last_can_rx and time.time() - s.last_can_rx < 2 else "disconnected",
                "model": "connected" if e else "disconnected"}
        self.emit(wb.system_status(s.id, s.t_rel(), bio_status=bstat, clean_windows=e.base_clean if e else None,
                                   bio_elapsed_s=s.t_rel(), can_status="unavailable", connections=conn,
                                   origin=s.origin))

    def bio_wave_event(self):
        s = self.s
        out = {}
        for ch, unit in (("ecg", "a.u."), ("ppg", "a.u.")):
            pv = s.eng.preview(ch, 3.0)
            if pv is None:
                continue
            fs, x, t0 = pv
            t_end = t0 + len(x) / fs
            x = x[-int(2.0 * fs): -int(1.0 * fs)]            # 직전 1초 (끝 1초는 필터 가장자리)
            out[ch] = wb.wave_channel(t_end - 2.0, fs, x.tolist(), unit, "common_preprocessed", None)
        if out:
            self.emit(wb.bio_wave(s.id, s.t_rel() - 1.0, ecg=out.get("ecg"), ppg=out.get("ppg"), origin=s.origin))

    def send(self, obj):
        b = json.dumps(obj, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        if len(b) > 60000:
            obj.pop("ecg_w", None); obj.pop("ppg_w", None)
            b = json.dumps(obj, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        try:
            self.tx.sendto(b, self.dst)
        except OSError:
            pass

    # ------------------------------------------------------------ 메인 루프
    def run(self):
        t_ui = t_wave = t_stat = 0.0
        while True:
            # 쌓인 패킷을 비우되 최대 30 ms 만 쓰고, 그다음 추론·화면 전송을 한다 (입력이 몰려도 출력이 굶지 않게)
            t_end = time.time() + 0.03
            self.rx.settimeout(0.02)
            while time.time() < t_end:
                try:
                    data, _ = self.rx.recvfrom(1 << 20)
                except socket.timeout:
                    break
                except OSError:
                    break
                self.last_rx = time.time()
                self.rx.settimeout(0.0005)
                try:
                    self.on_packet(json.loads(data.decode("utf-8")))
                except Exception as ex:
                    print(f"[server] 패킷 처리 오류: {ex}", flush=True)
            now = time.time()
            if self.a.idle_exit > 0 and now - self.last_rx > self.a.idle_exit:
                print("[server] Unity 입력이 없어 종료", flush=True)
                break
            s = self.s
            if s is None:
                if now - t_ui > 0.5:
                    self.send({"type": "ui", "session": "", "base_status": "no_session", "error": ""})
                    t_ui = now
                continue
            if s.eng is not None:
                try:
                    for r in s.eng.poll():
                        self.on_window(r)
                except Exception as ex:
                    s.err = f"추론 오류: {ex}"
                    print("[server] " + s.err, flush=True)
            if now - t_ui > 0.1:
                wave = now - t_wave > 0.1
                self.send(self.ui_packet(wave))
                t_ui = now
                if wave:
                    t_wave = now
            if now - t_stat > 1.0 and s.eng is not None:
                self.status_event()
                try:
                    self.bio_wave_event()
                except Exception:
                    pass
                s.ev.flush()
                t_stat = now


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--in-port", type=int, default=5005)
    ap.add_argument("--out-port", type=int, default=5006)
    ap.add_argument("--mode", default="causal", choices=["causal", "exact"])
    ap.add_argument("--log-dir", default=str(HERE.parent.parent / "SimLogs" / "ai"))
    ap.add_argument("--webboard", default="", help="예: http://127.0.0.1:8000 (비우면 웹보드 전송 안 함)")
    ap.add_argument("--wave-sec", type=float, default=6.0)
    ap.add_argument("--idle-exit", type=float, default=0, help="Unity 입력이 이 초 동안 없으면 종료 (0=계속)")
    a = ap.parse_args()
    Server(a).run()


if __name__ == "__main__":
    main()
