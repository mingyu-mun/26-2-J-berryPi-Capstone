"""SocketCAN 실시간 수신 → 속도/조향 정렬 → 웹보드 v1.2 이벤트."""
from __future__ import annotations

import argparse
from contextlib import ExitStack
import json
import math
from pathlib import Path
from queue import Empty, Full, Queue
import sys
from threading import Event, Thread
import time
from urllib.error import URLError
from urllib.parse import urlsplit
from urllib.request import Request, urlopen

import can
import cantools

from steering_profile import (Assessment, COMPANY_SPEED_MAX_AGE_US, CONFIG, DATA_DIR, NormalRecorder,
                              STEERING_MAX_AGE_US, load_profile, valid_id, write_line)

OUTPUT_INTERVAL_S = 0.1
CAN_LIST_DBC = Path(__file__).resolve().parent / "mapping" / "can_list.dbc"
CAN_LIST_MAP_ID = CONFIG["signal_map_id"]
CAN_LIST_SIGNALS = {"speed": "CARINFO.Speed", "steering": "SAS_FD_01_10ms.SAS_angleVal"}
UNIT_FACTORS = {
    "speed": {"km/h": 1.0, "kmh": 1.0, "kph": 1.0, "m/s": 3.6},
    "steering": {"deg": 1.0, "°": 1.0, "rad": 180 / math.pi},
}


class Decoder:
    def __init__(self, database, selectors, *, invalid_raw=None):
        self.messages = {}
        self.bad_frames = 0
        for field, selector in selectors.items():
            message_name, dot, signal_name = selector.rpartition(".")
            if not dot or not message_name or not signal_name:
                raise ValueError(f"{selector!r}: expected DBC message.signal")
            try:
                message = database.get_message_by_name(message_name)
                signal = message.get_signal_by_name(signal_name)
            except KeyError as exc:
                raise ValueError(f"DBC signal not found: {selector}") from exc
            factor = UNIT_FACTORS[field].get((signal.unit or "").strip().lower())
            if factor is None:
                raise ValueError(f"{selector}: unsupported {field} unit {signal.unit!r}")
            invalid = (invalid_raw or {}).get(field, signal.raw_invalid)
            if invalid is not None:
                invalid = signal.raw_to_scaled(invalid, decode_choices=False)
            key = (message.frame_id, message.is_extended_frame)
            _, fields = self.messages.setdefault(key, (message, []))
            fields.append((field, signal_name, factor, invalid))

    def filters(self):
        return [
            {"can_id": frame_id, "can_mask": 0x1FFFFFFF if extended else 0x7FF, "extended": extended}
            for frame_id, extended in self.messages
        ]

    def decode(self, frame):
        mapping = self.messages.get((frame.arbitration_id, frame.is_extended_id))
        if mapping is None or frame.is_error_frame or frame.is_remote_frame or not frame.is_rx:
            return {}
        message, fields = mapping
        try:
            values = message.decode(frame.data, decode_choices=False)
        except cantools.database.DecodeError:
            self.bad_frames += 1
            return dict.fromkeys(field for field, *_ in fields)
        updates = {}
        for field, name, factor, invalid in fields:
            if name in values:
                value = values[name]
                scaled = value * factor
                updates[field] = scaled if value != invalid and math.isfinite(scaled) else None
        return updates


def can_list_decoder():
    database = cantools.database.load_file(CAN_LIST_DBC, database_format="dbc")
    return Decoder(database, CAN_LIST_SIGNALS, invalid_raw={"steering": 32767})


def telemetry_event(session_id, origin, rel_us, signals, signal_map_id):
    speed, speed_us = signals.get("speed", (None, None))
    steering, steering_us = signals.get("steering", (None, None))
    return {
        "schema_version": "1.2", "type": "can.telemetry", "session_id": session_id,
        "origin": origin, "t_rel_s": rel_us / 1_000_000,
        "payload": {
            "speed_kmh": speed, "steering_angle_deg": steering, "steering_rate_deg_s": None,
            "road_wheel_angle_deg": None, "accelerator_pct": None, "brake_pct": None,
            "accel_pedal_raw": None, "brake_pedal_raw": None, "baseline_coverage_pct": None,
            "deviation_status": "unavailable", "can_drowsy": None, "rule_id": None,
            "signal_map_id": signal_map_id,
            "signal_times_s": {
                "speed": None if speed_us is None else speed_us / 1_000_000,
                "steering": None if steering_us is None else steering_us / 1_000_000,
                "accelerator": None, "brake": None,
            },
        },
    }


class SessionClock:
    def __init__(self, start_unix_s=None):
        self.start_us = round((time.time() if start_unix_s is None else start_unix_s) * 1_000_000)

    def now_us(self):
        # Compare with CAN receive timestamps on the same UNIX clock. Scheduling uses monotonic time.
        return self.frame_us(time.time())

    def frame_us(self, timestamp):
        return round(timestamp * 1_000_000) - self.start_us


class AlignedSignals:
    def __init__(self, speed_max_age_us):
        self.latest = {}
        self.max_age = {"speed": speed_max_age_us, "steering": STEERING_MAX_AGE_US}

    def observe(self, stamp_us, updates):
        for field, value in updates.items():
            previous = self.latest.get(field)
            if previous is None or stamp_us >= previous[1]:
                self.latest[field] = value, stamp_us

    def snapshot(self, now_us):
        return {
            field: (value, stamp_us)
            for field, (value, stamp_us) in self.latest.items()
            if value is not None and 0 <= now_us - stamp_us <= self.max_age[field]
        }


def receive(bus, decoder, clock, emit, *, session_id, origin, signal_map_id=None, duration_s=None,
            speed_max_age_us=STEERING_MAX_AGE_US, on_steering=None, on_snapshot=None):
    """조향 프레임마다 개인 판정에 전달하고, 웹보드 이벤트는 최대 10 Hz로 내보낸다."""
    aligned = AlignedSignals(speed_max_age_us)
    last_steering_us = -1
    last_output = {}
    last_output_us = -1
    next_emit = time.monotonic()
    stop = math.inf if duration_s is None else next_emit + duration_s
    while (now := time.monotonic()) < stop:
        timeout = min(stop - now, next_emit - now if now < next_emit else OUTPUT_INTERVAL_S)
        frame = bus.recv(timeout=timeout)
        if frame is not None and math.isfinite(frame.timestamp) and frame.timestamp > 0:
            stamp = clock.frame_us(frame.timestamp)
            if stamp >= 0:
                updates = decoder.decode(frame)
                aligned.observe(stamp, updates)
                if on_steering is not None and "steering" in updates and stamp > last_steering_us:
                    paired = aligned.snapshot(stamp)
                    speed, _ = paired.get("speed", (None, None))
                    on_steering(stamp, speed, updates["steering"])
                    last_steering_us = stamp
        now = time.monotonic()
        if now >= next_emit:
            rel_us = clock.now_us()
            snapshot = aligned.snapshot(rel_us)
            if on_snapshot is not None and rel_us >= last_steering_us:
                speed, _ = snapshot.get("speed", (None, None))
                steering, _ = snapshot.get("steering", (None, None))
                on_snapshot(rel_us, speed, steering)
            if rel_us > last_output_us and snapshot != last_output:
                emit(telemetry_event(session_id, origin, rel_us, snapshot, signal_map_id))
                last_output = snapshot
                last_output_us = rel_us
                next_emit = now + OUTPUT_INTERVAL_S


def json_line(event):
    return json.dumps(event, ensure_ascii=False, separators=(",", ":"), allow_nan=False)


class Webboard:
    """HTTP 지연과 실패가 CAN 수신 루프를 정지시키지 않도록 분리한다."""
    def __init__(self, base_url):
        parsed = urlsplit(base_url)
        if parsed.scheme not in {"http", "https"} or not parsed.hostname or parsed.query or parsed.fragment:
            raise ValueError("--webboard requires an HTTP(S) server base URL")
        self.url = base_url.rstrip("/") + "/api/events"
        self.queue = Queue(maxsize=16)
        self.stop = Event()
        self.dropped = self.failed = self.sent = 0
        self.worker = Thread(target=self._run, name="webboard-http", daemon=True)

    def __enter__(self):
        self.worker.start()
        return self

    def offer(self, line):
        try:
            self.queue.put_nowait(line)
        except Full:
            try:
                self.queue.get_nowait()
            except Empty:
                pass
            else:
                self.dropped += 1
            self.queue.put_nowait(line)

    def _run(self):
        while not self.stop.is_set() or not self.queue.empty():
            try:
                line = self.queue.get(timeout=0.1)
            except Empty:
                continue
            try:
                request = Request(self.url, data=line.encode("utf-8"), headers={"Content-Type": "application/json"})
                with urlopen(request, timeout=1.0):
                    self.sent += 1
            except (URLError, OSError, ValueError) as exc:
                self.failed += 1
                if self.failed == 1:
                    print(f"Webboard POST failed: {exc}", file=sys.stderr)

    def __exit__(self, *_):
        self.stop.set()
        self.worker.join(timeout=2.0)
        print(f"Webboard sent={self.sent} failed={self.failed} dropped={self.dropped} "
              f"pending={self.queue.qsize()} worker_active={self.worker.is_alive()}", file=sys.stderr)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--channel", default="can0")
    parser.add_argument("--fd", action="store_true", help="receive CAN FD frames on an FD-configured interface")
    mapping = parser.add_mutually_exclusive_group(required=True)
    mapping.add_argument("--can-list", action="store_true", help="use the supplied Can list.xlsx signal mapping")
    mapping.add_argument("--dbc", help="path to a custom DBC")
    parser.add_argument("--speed", help="DBC message.signal for speed with --dbc")
    parser.add_argument("--steering", help="DBC message.signal for handwheel angle")
    parser.add_argument("--session-id", required=True)
    parser.add_argument("--session-start-unix-s", type=float, help="shared ECG/PPG session start in UNIX seconds")
    parser.add_argument("--signal-map-id")
    parser.add_argument("--webboard", help="optional HTTP(S) base URL for POST /api/events")
    parser.add_argument("--duration-s", type=float, help="stop after this many seconds")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--collect-normal", metavar="DRIVER", help="record this driver's straight-road normal session")
    mode.add_argument("--assess", metavar="DRIVER", help="apply this driver's saved baseline")
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    args = parser.parse_args()
    if args.can_list and (args.speed or args.steering):
        parser.error("--can-list already selects speed and steering")
    if args.dbc and not args.speed:
        parser.error("--dbc requires --speed")
    if (args.collect_normal or args.assess) and not args.can_list:
        parser.error("Personal collection and assessment require --can-list")
    if not args.session_id.strip():
        parser.error("--session-id must be nonempty")
    if args.session_start_unix_s is not None and not math.isfinite(args.session_start_unix_s):
        parser.error("--session-start-unix-s must be finite")
    if args.duration_s is not None and (not math.isfinite(args.duration_s) or args.duration_s <= 0):
        parser.error("--duration-s must be finite and positive")
    if not sys.platform.startswith("linux"):
        parser.exit(2, "Live SocketCAN requires Linux. Run highway_demo.py for a hardware-free check.\n")

    try:
        if args.can_list:
            decoder = can_list_decoder()
            signal_map_id = args.signal_map_id or CAN_LIST_MAP_ID
            speed_max_age_us = COMPANY_SPEED_MAX_AGE_US
        else:
            selection = {"speed": args.speed}
            if args.steering is not None:
                selection["steering"] = args.steering
            decoder = Decoder(cantools.database.load_file(args.dbc, database_format="dbc"), selection)
            signal_map_id = args.signal_map_id
            speed_max_age_us = STEERING_MAX_AGE_US
        profile = load_profile(args.data_dir, args.assess, args.session_id) if args.assess else None
        with ExitStack() as stack:
            bus = stack.enter_context(can.Bus(interface="socketcan", channel=args.channel, fd=args.fd,
                                              can_filters=decoder.filters(), ignore_config=True))
            board = stack.enter_context(Webboard(args.webboard)) if args.webboard else None
            clock = SessionClock(args.session_start_unix_s)
            on_steering = on_snapshot = None
            if args.collect_normal:
                recorder = stack.enter_context(NormalRecorder(args.data_dir, args.collect_normal, args.session_id))
                on_steering = recorder.observe
                print(f"Normal session: {recorder.path}", file=sys.stderr)
            if profile is not None:
                decision_path = args.data_dir / "decisions" / profile.participant / f"{valid_id(args.session_id)}.jsonl"
                decision_path.parent.mkdir(parents=True, exist_ok=True)
                decision_file = stack.enter_context(decision_path.open("x", encoding="utf-8"))

                def emit_decision(decision):
                    write_line(decision_file, decision)
                    decision_file.flush()
                    if decision["deviation_status"] == "deviation":
                        print(f"Personal steering deviation: {json_line(decision)}", file=sys.stderr)

                assessment = Assessment(profile, args.session_id, emit_decision)
                on_steering, on_snapshot = assessment.observe, assessment.expire
                print(f"Personal decisions: {decision_path}", file=sys.stderr)

            def emit(event):
                line = json_line(event)
                print(line, flush=True)
                if board is not None:
                    board.offer(line)

            print(f"Listening on {args.channel} (session {args.session_id})", file=sys.stderr)
            try:
                receive(bus, decoder, clock, emit, session_id=args.session_id, origin="live",
                        signal_map_id=signal_map_id, duration_s=args.duration_s, speed_max_age_us=speed_max_age_us,
                        on_steering=on_steering, on_snapshot=on_snapshot)
            finally:
                print(f"Frames rejected during decode: {decoder.bad_frames}", file=sys.stderr)
    except KeyboardInterrupt:
        return
    except (OSError, ValueError, TypeError, KeyError, can.CanError, cantools.database.Error) as exc:
        parser.exit(2, f"CAN receiver failed: {exc}\n")


if __name__ == "__main__":
    main()
