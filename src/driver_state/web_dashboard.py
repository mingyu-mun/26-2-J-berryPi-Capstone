"""Receive validated webboard events and serve a local display.

This transport never creates a waveform, model score, CAN reading, or decision.
The supplied contract and upstream Python StepOut are the display authority.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
import re
import threading
import time
import uuid
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

from jsonschema import Draft202012Validator
from .webboard_library import BaselineCapture, BaselineLibrary, DEFAULT_LIBRARY_DIR, LibraryError

STATIC_DIR = Path(__file__).resolve().parents[2] / "web" / "dashboard"
MAX_BODY = 2 * 1024 * 1024
HISTORY_BYTES = 8 * 1024 * 1024


class APIError(ValueError):
    def __init__(self, message, status=400):
        super().__init__(message)
        self.status = status


def finite_numbers(value):
    if isinstance(value, float) and not math.isfinite(value):
        raise APIError("무한대 또는 NaN 수치는 허용하지 않습니다.")
    if isinstance(value, dict):
        for item in value.values():
            finite_numbers(item)
    elif isinstance(value, list):
        for item in value:
            finite_numbers(item)


def validate_event(event, validator):
    finite_numbers(event)
    error = next(validator.iter_errors(event), None)
    if error is not None:
        # OneOf's top-level error echoes the entire event if str(error) is used.
        raise APIError("입력이 웹보드 규약과 일치하지 않습니다. 필수 필드·자료형·범위를 확인하세요.")
    p = event["payload"]
    if event["type"] == "system.status" and "ecg_reference" in p["bio_baseline"]:
        base = p["bio_baseline"]
        reference = base["ecg_reference"]
        saved = 'profile_id' in reference and 'source_session_id' in reference
        foreign = saved and reference['source_session_id'] != event['session_id']
        if (('profile_id' in reference) != ('source_session_id' in reference)
                or base["status"] != "ready" or reference["start_s"] >= reference["end_s"]
                or not foreign and reference["end_s"] > event["t_rel_s"] + .001):
            raise APIError("심박 기준은 준비 완료된 실제 기준 구간이어야 합니다.")
    if event["type"] == "ai.decision" and p["output"] is not None:
        o, inp = p["output"], p["input"]
        if p["baseline_status"] != "ready":
            raise APIError("기준 준비 전 판정 output은 null이어야 합니다.")
        names = {-1: "측정 불가", 0: "정상", 1: "주의", 2: "졸음"}
        if o["level_name"] != names[o["level"]] or o["candidate"] and o["level"] not in (1, 2):
            raise APIError("상태 판정 필드가 서로 일치하지 않습니다.")
        if inp["t"] is None or abs(inp["t"] - o["t"]) > .001:
            raise APIError("판정 창의 종료 시각이 일치하지 않습니다.")
        if p["window_end_s"] is not None and abs(p["window_end_s"] - o["t"]) > .001:
            raise APIError("판정 창의 종료 시각이 일치하지 않습니다.")
        ch = {"EP": "ecg", "PPG": "ppg"}.get(o["channel"])
        if ch and (inp[f"ok_{ch}"] is not True or inp[f"q_{ch}"] is None or inp["p_ep" if ch == "ecg" else "p_ppg"] is None):
            raise APIError("사용한 판정 채널의 품질 또는 모델 점수가 없습니다.")
    if event["type"] == "can.telemetry":
        keys = (("speed_kmh", "speed"), ("steering_angle_deg", "steering"), ("steering_rate_deg_s", "steering"),
                ("accel_pedal_raw", "accelerator"), ("accelerator_pct", "accelerator"),
                ("brake_pedal_raw", "brake"), ("brake_pct", "brake"))
        for name, key in keys:
            if p[name] is not None and p["signal_times_s"][key] is None:
                raise APIError(f"{name}: 실제 측정 시각이 필요합니다.")
        if any(t is not None and t > event["t_rel_s"] + .001 for t in p["signal_times_s"].values()):
            raise APIError("CAN 측정 시각이 메시지 시각보다 미래입니다.")


class Dashboard:
    """Bounded event buffer shared by displays. It remains empty until ingest()."""
    def __init__(self, clock=time.monotonic, static_dir=STATIC_DIR):
        self.lock, self.clock = threading.RLock(), clock
        schema = json.loads((Path(static_dir) / "contract" / "webboard_event_v1_2.schema.json").read_text(encoding="utf-8"))
        Draft202012Validator.check_schema(schema)
        self.validator = Draft202012Validator(schema)
        self.stream_id, self.session_id = uuid.uuid4().hex, None
        self.seq, self.total_bytes = 0, 0
        self.events, self.retired = deque(), deque(maxlen=64)
        self.latest = {}
        self.baseline_capture = BaselineCapture()
        self.seen, self.seen_order = set(), deque()

    def begin_session(self, session_id):
        """Retire old jobs immediately, including while a new model is loading."""
        with self.lock:
            if self.session_id and self.session_id != session_id:
                self.retired.append(self.session_id)
            self.session_id = session_id
            self.stream_id = uuid.uuid4().hex
            self.events.clear(); self.latest.clear(); self.total_bytes = 0
            self.seen.clear(); self.seen_order.clear()
            self.baseline_capture = BaselineCapture()

    def ingest(self, data):
        batch = data if isinstance(data, list) else [data]
        if not 0 < len(batch) <= 1000:
            raise APIError("1개부터 1,000개까지의 이벤트만 한 번에 받습니다.")
        for event in batch:
            validate_event(event, self.validator)
        accepted = 0
        with self.lock:
            for original in batch:
                event = copy.deepcopy(original)
                sid = event["session_id"]
                if sid in self.retired:
                    continue  # Late packets must not reactivate an old session.
                if sid != self.session_id:
                    if self.session_id is not None:
                        self.retired.append(self.session_id)
                    self.session_id = sid
                    self.events.clear(); self.latest.clear(); self.total_bytes = 0
                    self.seen.clear(); self.seen_order.clear()
                encoded = json.dumps(event, ensure_ascii=False, sort_keys=True).encode("utf-8")
                digest = hashlib.sha256(encoded).digest()
                if digest in self.seen:
                    continue
                self.seen.add(digest); self.seen_order.append(digest)
                if len(self.seen_order) > 1024:
                    self.seen.discard(self.seen_order.popleft())
                self.seq += 1
                size = len(encoded)
                packet = {"seq": self.seq, "event": event, "received": self.clock(), "size": size}
                self.events.append(packet); self.total_bytes += size
                self.baseline_capture.observe(event, size)
                old = self.latest.get(event["type"])
                if old is None or old["event"]["t_rel_s"] <= event["t_rel_s"]:
                    self.latest[event["type"]] = packet
                while len(self.events) > 512 or self.total_bytes > HISTORY_BYTES:
                    self.total_bytes -= self.events.popleft()["size"]
                accepted += 1
            return {"accepted": accepted, "latest_seq": self.seq, "session_id": self.session_id}

    def snapshot(self, after=0):
        with self.lock:
            first = self.events[0]["seq"] if self.events else self.seq + 1
            reset = after > self.seq or after < first - 1
            if reset or after == 0:
                packets = {p["seq"]: p for p in [*self.latest.values(), *self.events]}
            else:
                packets = {p["seq"]: p for p in self.events if p["seq"] > after}
            now = self.clock()
            return {"stream_id": self.stream_id, "session_id": self.session_id, "reset": reset,
                    "events": [{"seq": p["seq"], "event": p["event"], "age_s": max(0, now - p["received"])}
                               for _, p in sorted(packets.items())]}


class DashboardServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True

    def server_close(self):
        super().server_close()
        local = getattr(self, 'local_replay', None)
        if local is not None:
            local.close()
            self.local_replay = None
        library = getattr(self, 'library', None)
        if library is not None:
            library.close()
            self.library = None


def create_server(host="0.0.0.0", port=8000, *, dashboard=None, static_dir=STATIC_DIR,
                  library_dir=DEFAULT_LIBRARY_DIR, recordings_dir=None, model_dir=None, backend_factory=None,
                  can_assist_factory=None):
    engine = dashboard or Dashboard(static_dir=static_dir)
    static = Path(static_dir)
    user_route = re.compile(r'/api/users/([0-9a-f]{32})/baselines(?:/([0-9a-f]{32}|current))?')
    profile_route = re.compile(r'/api/users/([0-9a-f]{32})/model-profiles')
    files = {"/": ("index.html", "text/html"), "/index.html": ("index.html", "text/html"),
             "/app.js": ("app.js", "application/javascript"), "/styles.css": ("styles.css", "text/css"),
             "/contract-validator.js": ("contract-validator.js", "application/javascript"),
             "/webboard-state.js": ("webboard-state.js", "application/javascript"),
             "/contract/webboard_event_v1_2.schema.json": ("contract/webboard_event_v1_2.schema.json", "application/json")}

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):
            pass

        def reply(self, status, data, content_type="application/json"):
            body = data if isinstance(data, bytes) else json.dumps(data, ensure_ascii=False, allow_nan=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", content_type + "; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.end_headers()
            try:
                self.wfile.write(body)
            except (BrokenPipeError, ConnectionResetError):
                pass  # A user switch can cancel an in-flight browser request.

        def do_GET(self):
            parts = urlsplit(self.path)
            try:
                if parts.path == '/api/local/session':
                    return self.reply(200, dict(available=True, **local.status()) if local else
                        dict(available=False, stage='disabled', error='로컬 기록 기능은 run_dashboard.py --recordings-dir로 실행하세요.'))
                if parts.path == '/api/local/recordings':
                    if local is None:
                        raise LibraryError('로컬 기록 기능이 활성화되지 않았습니다.',503)
                    return self.reply(200,local.catalog.refresh())
                profile_match = profile_route.fullmatch(parts.path)
                if profile_match:
                    if local is None:
                        raise LibraryError('모델 기준 저장 기능이 활성화되지 않았습니다.',503)
                    return self.reply(200,dict(profiles=local.profiles.list(profile_match[1])))
                if parts.path == '/api/users':
                    return self.reply(200, {'users': library.users()})
                if parts.path == '/api/baselines/current':
                    with engine.lock:
                        current = engine.baseline_capture.status()
                    return self.reply(200, current)
                match = user_route.fullmatch(parts.path)
                if match:
                    user_id, recording_id = match.groups()
                    if recording_id is None:
                        return self.reply(200, {'user': library.user(user_id), 'baselines': library.list_baselines(user_id)})
                    if recording_id != 'current':
                        return self.reply(200, library.get(user_id, recording_id))
            except LibraryError as exc:
                return self.reply(exc.status, {'error': str(exc)})
            except (OSError,ValueError,TypeError) as exc:
                return self.reply(400,{'error':str(exc) or '로컬 파일을 확인하세요.'})
            if parts.path == "/api/webboard":
                try:
                    after = int(parse_qs(parts.query).get("after", ["0"])[0])
                    if after < 0:
                        raise ValueError()
                except ValueError:
                    return self.reply(400, {"error": "after는 0 이상의 이벤트 번호여야 합니다."})
                result=engine.snapshot(after)
                result['local']=dict(available=True,**local.status()) if local else dict(available=False)
                return self.reply(200,result)
            if parts.path in files:
                name, mime = files[parts.path]
                try:
                    return self.reply(200, (static / name).read_bytes(), mime)
                except FileNotFoundError:
                    return self.reply(404, {"error": "화면 파일을 찾을 수 없습니다."})
            self.reply(404, {"error": "경로가 없습니다."})

        def do_POST(self):
            try:
                origin = self.headers.get("Origin")
                if origin:
                    parsed = urlsplit(origin)
                    if parsed.scheme not in {"http", "https"} or parsed.netloc.lower() != self.headers.get("Host", "").lower():
                        raise APIError("다른 사이트의 요청은 허용하지 않습니다.", 403)
                if self.headers.get("Sec-Fetch-Site") == "cross-site":
                    raise APIError("다른 사이트의 요청은 허용하지 않습니다.", 403)
                path = urlsplit(self.path).path
                route = user_route.fullmatch(path)
                profile_match = profile_route.fullmatch(path)
                local_action = path.removeprefix('/api/local/') if path.startswith('/api/local/') else None
                if path not in {'/api/events', '/api/users'} and not route and not profile_match and local_action not in {'select','apply','start','control'}:
                    raise APIError("경로가 없습니다.", 404)
                if self.headers.get_content_type() != "application/json":
                    raise APIError("application/json 입력이 필요합니다.", 415)
                try:
                    size = int(self.headers.get("Content-Length", "0"))
                except ValueError:
                    raise APIError("Content-Length가 잘못됐습니다.") from None
                limit = MAX_BODY if path == '/api/events' else 8 * 1024 * 1024
                if not 0 < size <= limit:
                    raise APIError(f"{limit // (1024 * 1024)}MB 이하의 JSON 입력이 필요합니다.", 413)
                try:
                    def reject_constant(_):
                        raise ValueError()
                    data = json.loads(self.rfile.read(size).decode("utf-8"), parse_constant=reject_constant)
                except (ValueError, UnicodeDecodeError):
                    raise APIError("JSON을 읽을 수 없습니다.") from None
                if path == '/api/events':
                    if local and local.status()['session_id']:
                        raise APIError('로컬 측정 세션에서는 외부 이벤트를 함께 받을 수 없습니다.',409)
                    return self.reply(200, engine.ingest(data))
                if not isinstance(data, dict):
                    raise APIError('JSON 객체가 필요합니다.')
                if path == '/api/users':
                    return self.reply(201, library.add_user(data.get('nickname')))
                if local_action or profile_match:
                    if local is None:
                        raise LibraryError('로컬 기록 기능이 활성화되지 않았습니다.',503)
                    if profile_match:
                        return self.reply(201,local.save(profile_match[1],data.get('title'),data.get('session_id')))
                    if local_action=='select':
                        result=local.select(data.get('user_id'),data.get('recording_id'))
                    elif local_action=='apply':
                        result=local.apply(data.get('user_id'),data.get('profile_id'))
                    elif local_action=='start':
                        result=local.start(data.get('mode'),data.get('speed',1))
                    else:
                        result=local.control(data.get('action'))
                    return self.reply(200,result)
                user_id, recording_id = route.groups()
                if recording_id not in {None, 'current'}:
                    raise APIError('경로가 없습니다.', 404)
                if recording_id == 'current':
                    with engine.lock:
                        events = engine.baseline_capture.snapshot(data.get('session_id'))
                else:
                    events = data.get('events')
                baseline = library.save(user_id, data.get('title'), events,
                                        lambda event: validate_event(event, engine.validator))
                self.reply(201, baseline)
            except (APIError, LibraryError) as exc:
                self.reply(exc.status, {"error": str(exc)})
            except (OSError, ValueError, TypeError) as exc:
                self.reply(400, {'error': str(exc) or '입력 또는 로컬 파일을 확인하세요.'})

    server = DashboardServer((host, port), Handler)
    try:
        library = BaselineLibrary(library_dir)
        server.library = library
        local = None
        if recordings_dir is not None:
            from .local_replay import LocalReplay
            options = {'model_dir': model_dir} if model_dir is not None else {}
            local = LocalReplay(engine,library,recordings_dir,backend_factory=backend_factory,
                                can_assist_factory=can_assist_factory,**options)
    except Exception:
        server.server_close()
        raise
    server.library = library
    server.local_replay = local
    server.dashboard = engine
    return server
