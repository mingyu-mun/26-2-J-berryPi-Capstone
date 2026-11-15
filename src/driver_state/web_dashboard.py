"""LAN dashboard transport for model outputs; no neural network or sensor driver.

The server runs the repository's Python StateMachineV2, sharing one timeline across
all connected displays. Model/sensor code supplies one validated window every 10 s.
"""
from __future__ import annotations

import csv
import io
import json
import math
import threading
import time
from collections import deque
from dataclasses import asdict, fields
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit

from .state_machine_v2 import StateMachineV2, V2Config, WindowIn

STATIC_DIR = Path(__file__).resolve().parents[2] / "web" / "dashboard"
MAX_BODY = 2 * 1024 * 1024
STALE_SEC = 20.0
SCENARIOS = {"normal", "drowsy", "recover", "noise", "invalid", "manual"}


class APIError(ValueError):
    def __init__(self, message: str, status: int = 400):
        super().__init__(message)
        self.status = status


def number(value, name, *, low=None, high=None, nullable=True):
    if value is None and nullable:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise APIError(f"{name}: 유한한 숫자가 필요합니다.")
    if low is not None and value < low or high is not None and value > high:
        raise APIError(f"{name}: 허용 범위를 벗어났습니다.")
    return float(value)


def boolean(value, name, *, nullable=False):
    if value is None and nullable:
        return None
    if type(value) is not bool:
        raise APIError(f"{name}: true 또는 false가 필요합니다.")
    return value


def parse_window(data: dict) -> WindowIn:
    if not isinstance(data, dict):
        raise APIError("창 입력은 JSON 객체여야 합니다.")
    unknown = set(data) - {f.name for f in fields(WindowIn)}
    if unknown:
        raise APIError("알 수 없는 창 필드: " + ", ".join(sorted(unknown)))
    t = number(data.get("t"), "t", low=0, nullable=False)
    p = number(data.get("p_ep"), "p_ep", low=0, high=1)
    pp = number(data.get("p_ppg"), "p_ppg", low=0, high=1)
    q = number(data.get("q_ecg"), "q_ecg", low=0, high=1)
    qp = number(data.get("q_ppg"), "q_ppg", low=0, high=1)
    # Omitted quality/status never makes a channel valid by accident.
    ok = boolean(data.get("ok_ecg", False), "ok_ecg") and q is not None
    okp = boolean(data.get("ok_ppg", False), "ok_ppg") and qp is not None
    return WindowIn(t=t, p_ep=p, p_ppg=pp, q_ecg=q if q is not None else 1.0,
                    q_ppg=qp, ok_ecg=ok, ok_ppg=okp,
                    hr_ecg=number(data.get("hr_ecg"), "hr_ecg", low=1, high=300),
                    hr_ppg=number(data.get("hr_ppg"), "hr_ppg", low=1, high=300),
                    can_drowsy=boolean(data.get("can_drowsy"), "can_drowsy", nullable=True))


def csv_number(value, name):
    if value is None or str(value).strip().lower() in {"", "nan", "none", "null"}:
        return None
    try:
        return number(float(value), name)
    except (ValueError, TypeError):
        raise APIError(f"CSV {name}: 숫자 형식이 잘못됐습니다.") from None


def csv_bool(value, name):
    if value is None or str(value).strip().lower() in {"", "nan", "none", "null"}:
        return None
    val = str(value).strip().lower()
    if val in {"true", "1", "yes"}:
        return True
    if val in {"false", "0", "no"}:
        return False
    raise APIError(f"CSV {name}: 불리언 형식이 잘못됐습니다.")


def parse_replay(text):
    if not isinstance(text, str) or not text.strip():
        raise APIError("CSV 내용이 비어 있습니다.")
    rows = csv.DictReader(io.StringIO(text.lstrip("\ufeff")))
    headers = rows.fieldnames or []
    if not ({"student_S1-EP_p", "student_S1-EP_logit"} & set(headers)):
        raise APIError("student_S1-EP_p 또는 student_S1-EP_logit 열이 필요합니다.")
    if not ({"t_rel", "t"} & set(headers)):
        raise APIError("창 시작 시각 t_rel 또는 창 끝 시각 t 열이 필요합니다.")
    result = []
    for row in rows:
        if len(result) >= 10000:
            raise APIError("CSV는 최대 10,000개 창까지 지원합니다.")
        if None in row:
            raise APIError("CSV 열 개수가 헤더와 다릅니다.")
        def n(key):
            return csv_number(row.get(key), key)
        p = n("student_S1-EP_p")
        if p is None:
            lg = n("student_S1-EP_logit")
            if lg is not None:
                p = 1.0 / (1.0 + math.exp(-max(-700.0, min(700.0, lg))))
        t = n("t") if "t" in headers else n("t_rel")
        if t is None:
            raise APIError("CSV의 모든 행에 시각이 필요합니다.")
        if "t" not in headers:
            t += 30.0  # selfmeasure_infer stores window START; WindowIn uses END.
        w = parse_window({"t": t, "p_ep": p, "p_ppg": n("student_S1-EP_prv_p"),
                          "q_ecg": n("q_abn"), "q_ppg": n("q_ppg"),
                          "hr_ecg": n("hr_ecg") if "hr_ecg" in headers else n("hr_30"),
                          "hr_ppg": n("hr_ppg"),
                          "ok_ecg": csv_bool(row.get("valid"), "valid") is True,
                          "ok_ppg": csv_bool(row.get("valid_ppg"), "valid_ppg") is True,
                          "can_drowsy": csv_bool(row.get("can_drowsy"), "can_drowsy")})
        ses = (row.get("dataset_id", ""), row.get("group", ""), row.get("session_id", ""))
        if result and ses == result[-1][0] and w.t <= result[-1][1].t:
            raise APIError("같은 세션의 CSV 시각은 증가해야 합니다.")
        result.append((ses, w))
    if not result:
        raise APIError("CSV 데이터 행이 없습니다.")
    return result


class Dashboard:
    def __init__(self, mode="demo", interval_sec=0.55, clock=time.monotonic):
        if mode not in {"demo", "live"}:
            raise ValueError("mode must be demo or live")
        self.lock = threading.RLock()
        self.stop_event = threading.Event()
        self.clock = clock
        self.mode, self.interval_sec = mode, interval_sec
        self.cfg = V2Config()
        self.sm = StateMachineV2(self.cfg)
        self.scenario, self.can_scenario, self.manual_p = "normal", "none", 0.7
        self.running = False
        self.loop_csv = False
        self.rows = []
        self.csv_pos = 0
        self.source_name = "시연용 모의 데이터" if mode == "demo" else "실시간 입력 대기"
        self.events, self.trend = deque(maxlen=30), deque(maxlen=36)
        self.last_input = self.last_output = self.last_received = self.last_session = None
        self.worker = None
        self.event("대시보드 준비: " + self.source_name)

    def event(self, message, kind="normal"):
        self.events.appendleft({"t": self.last_input.t if self.last_input else 0, "message": message, "kind": kind})

    def _reset_timeline(self, reason):
        self.sm.reset()
        self.last_input = self.last_output = self.last_received = self.last_session = None
        self.trend.clear()
        self.event(reason)

    def _demo(self):
        t = self.last_input.t + self.cfg.dt if self.last_input else self.cfg.dt
        k = t / 10
        p = .24 + .045 * math.sin(k * .43)
        pp, q, qp, he, hp, ok, okp = p, .02, .02, 73.0, 72.0, True, True
        if self.scenario == "drowsy":
            p, pp, he, hp = .82 + .035 * math.sin(k * .4), .8, 68, 69
        elif self.scenario == "recover":
            p, pp = .17, .19
        elif self.scenario == "noise":
            p, pp, q, he, hp = .7, .62, .16, 92, 74
        elif self.scenario == "invalid":
            p = pp = he = hp = None
            q, qp, ok, okp = .23, .2, False, False
        elif self.scenario == "manual":
            p = pp = self.manual_p
        can = None if self.can_scenario == "offline" else self.can_scenario == "confirm"
        return WindowIn(t, p, q, pp, qp, he, hp, ok, okp, can)

    def _apply(self, w, session=None):
        if self.last_input is not None:
            delta = w.t - self.last_input.t
            boundary = session != self.last_session
            gap = abs(delta - self.cfg.dt) > .001
            if boundary or gap:
                self._reset_timeline("세션 변경 또는 10초 간격 단절: 누적 판정 초기화")
        previous = self.last_output
        out = self.sm.step(w)
        self.last_input, self.last_output, self.last_session = w, out, session
        self.last_received = self.clock()
        self.trend.append({"t": w.t, "p": out.p_used, "score": out.score})
        if previous is None or previous.level != out.level:
            self.event("상태: " + out.level_name, "drowsy" if out.level == 2 else "caution" if out.level == 1 else "normal")
        if out.candidate and (previous is None or not previous.candidate):
            self.event("졸음 후보: 지속시간 또는 외부 확인 대기", "caution")
        if previous and previous.channel != out.channel:
            self.event("판정 채널: " + out.channel)

    def _advance(self):
        if self.mode == "live":
            raise APIError("실시간 모드는 /api/window 입력으로만 진행합니다.", 409)
        if self.mode == "demo":
            self._apply(self._demo())
            return
        if self.csv_pos >= len(self.rows):
            if not self.loop_csv:
                self.running = False
                return
            self._reset_timeline("CSV 반복 재생: 이전 누적 판정 초기화")
            self.csv_pos = 0
        session, w = self.rows[self.csv_pos]
        self.csv_pos += 1
        self._apply(w, session)
        if self.csv_pos == len(self.rows) and not self.loop_csv:
            self.running = False
            self.event("CSV 재생 완료")

    def status(self):
        with self.lock:
            age = None if self.last_received is None else max(0, self.clock() - self.last_received)
            stale = self.mode == "live" and (age is None or age >= STALE_SEC)
            return {"mode": self.mode, "source_name": self.source_name, "running": self.running,
                    "scenario": self.scenario, "can_scenario": self.can_scenario, "manual_p": self.manual_p,
                    "interval_sec": self.interval_sec, "config": asdict(self.cfg),
                    "input": asdict(self.last_input) if self.last_input and not stale else None,
                    "output": asdict(self.last_output) if self.last_output and not stale else None,
                    "trend": list(self.trend), "events": list(self.events),
                    "csv": {"position": self.csv_pos, "count": len(self.rows), "loop": self.loop_csv},
                    "connection": {"stale": stale, "age_sec": age},
                    "baseline": {"status": "demo" if self.mode == "demo" else "unknown"},
                    "model_connected": False}

    def control(self, data):
        if not isinstance(data, dict):
            raise APIError("JSON 객체가 필요합니다.")
        with self.lock:
            action = data.get("action")
            if not isinstance(action, str):
                raise APIError("action은 문자열이어야 합니다.")
            if action == "settings":
                # Validate all fields before mutating anything.
                updates = {}
                for name, choices in (("scenario", SCENARIOS), ("can_scenario", {"none", "confirm", "offline"})):
                    if name in data:
                        if not isinstance(data[name], str) or data[name] not in choices:
                            raise APIError(name + ": 알 수 없는 설정입니다.")
                        updates[name] = data[name]
                if "manual_p" in data:
                    updates["manual_p"] = number(data["manual_p"], "manual_p", low=0, high=1, nullable=False)
                if "interval_sec" in data:
                    updates["interval_sec"] = number(data["interval_sec"], "interval_sec", low=.1, high=10, nullable=False)
                if "loop_csv" in data:
                    updates["loop_csv"] = boolean(data["loop_csv"], "loop_csv")
                for name, value in updates.items():
                    setattr(self, name, value)
            elif action == "source":
                if not isinstance(data.get("mode"), str) or data["mode"] not in {"demo", "live"}:
                    raise APIError("mode는 demo 또는 live여야 합니다.")
                self.mode, self.running = data["mode"], False
                self.source_name = "시연용 모의 데이터" if self.mode == "demo" else "실시간 입력 대기"
                self.csv_pos = 0
                self.rows = []
                self._reset_timeline("입력 모드 변경: " + self.mode)
            elif action == "reset":
                self.running, self.csv_pos = False, 0
                self.events.clear()
                self._reset_timeline("판정 초기화: 새 입력 대기")
            elif action == "pause":
                self.running = False
            elif action in {"start", "step"}:
                if self.mode == "live":
                    raise APIError("실시간 모드는 재생 버튼을 사용하지 않습니다.", 409)
                if self.mode == "csv" and self.csv_pos >= len(self.rows) and not self.loop_csv:
                    raise APIError("CSV 재생이 끝났습니다. 초기화 후 다시 시작하세요.", 409)
                if action == "start":
                    self.running = True
                else:
                    self._advance()
            else:
                raise APIError("알 수 없는 action입니다.")
            return self.status()

    def replay(self, data):
        if not isinstance(data, dict):
            raise APIError("JSON 객체가 필요합니다.")
        try:
            rows = parse_replay(data.get("csv"))
        except csv.Error:
            raise APIError("CSV 형식을 읽을 수 없습니다.") from None
        name = data.get("name", "windows_scores.csv")
        if not isinstance(name, str) or len(name) > 200:
            raise APIError("파일 이름이 잘못됐습니다.")
        with self.lock:
            self.mode, self.running = "csv", False
            self.rows, self.csv_pos = rows, 0
            self.source_name = name
            self.events.clear()
            self._reset_timeline(f"CSV {len(rows)}개 창 불러옴: " + name)
            return self.status()

    def window(self, data):
        w = parse_window(data)
        with self.lock:
            if self.mode != "live":
                raise APIError("먼저 실시간 입력 모드로 전환하세요.", 409)
            if self.last_input:
                delta = w.t - self.last_input.t
                if delta <= 0:
                    raise APIError("t는 이전 창보다 커야 합니다.")
                if delta < self.cfg.dt - .001:
                    raise APIError("최소 10초 간격의 새 창만 받습니다.")
                if abs(delta / self.cfg.dt - round(delta / self.cfg.dt)) > .0001:
                    raise APIError("창 시각은 이전 창에서 10초의 배수만큼 증가해야 합니다.")
            if self.last_received is not None and self.clock() - self.last_received >= STALE_SEC:
                self._reset_timeline("실시간 수신 재개: 오래된 누적 판정 초기화")
            self._apply(w)
            return self.status()

    def start_worker(self):
        if self.worker:
            return
        def run():
            while not self.stop_event.wait(self.interval_sec):
                with self.lock:
                    if self.running:
                        self._advance()
        self.worker = threading.Thread(target=run, name="dashboard-replay", daemon=True)
        self.worker.start()

    def close(self):
        self.stop_event.set()
        if self.worker:
            self.worker.join(timeout=2)


class DashboardServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True

    def server_close(self):
        if getattr(self, "dashboard", None) is not None:
            self.dashboard.close()
        super().server_close()


def create_server(host="0.0.0.0", port=8000, *, dashboard=None, static_dir=STATIC_DIR):
    engine = dashboard or Dashboard()
    static = Path(static_dir)

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):
            pass

        def reply(self, status, data, content_type="application/json; charset=utf-8"):
            body = json.dumps(data, ensure_ascii=False, allow_nan=False).encode("utf-8") if not isinstance(data, bytes) else data
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            path = urlsplit(self.path).path
            if path == "/api/status":
                return self.reply(200, engine.status())
            files = {"/": ("index.html", "text/html"), "/index.html": ("index.html", "text/html"),
                     "/app.js": ("app.js", "application/javascript"), "/styles.css": ("styles.css", "text/css")}
            if path in files:
                name, mime = files[path]
                try:
                    return self.reply(200, (static / name).read_bytes(), mime + "; charset=utf-8")
                except FileNotFoundError:
                    return self.reply(404, {"error": "화면 파일을 찾을 수 없습니다."})
            self.reply(404, {"error": "경로가 없습니다."})

        def do_POST(self):
            try:
                # Browser writes must originate from this same site. CLI/loopback
                # producers have no Origin header. Do not expose this LAN server publicly.
                origin = self.headers.get("Origin")
                if origin:
                    parsed = urlsplit(origin)
                    if parsed.scheme not in {"http", "https"} or parsed.netloc.lower() != self.headers.get("Host", "").lower():
                        raise APIError("다른 사이트의 요청은 허용하지 않습니다.", 403)
                if self.headers.get("Sec-Fetch-Site") == "cross-site":
                    raise APIError("다른 사이트의 요청은 허용하지 않습니다.", 403)
                if self.headers.get_content_type() != "application/json":
                    raise APIError("Content-Type: application/json이 필요합니다.", 415)
                try:
                    size = int(self.headers.get("Content-Length", "0"))
                except ValueError:
                    raise APIError("Content-Length가 잘못됐습니다.") from None
                if not 0 < size <= MAX_BODY:
                    raise APIError("요청은 비어 있지 않은 2MB 이하 JSON이어야 합니다.", 413)
                try:
                    data = json.loads(self.rfile.read(size).decode("utf-8"))
                except (ValueError, UnicodeDecodeError):
                    raise APIError("JSON을 읽을 수 없습니다.") from None
                actions = {"/api/control": engine.control, "/api/replay": engine.replay, "/api/window": engine.window}
                action = actions.get(urlsplit(self.path).path)
                if action is None:
                    raise APIError("경로가 없습니다.", 404)
                self.reply(200, action(data))
            except APIError as exc:
                self.reply(exc.status, {"error": str(exc)})

    server = DashboardServer((host, port), Handler)
    server.dashboard = engine
    engine.start_worker()
    return server
