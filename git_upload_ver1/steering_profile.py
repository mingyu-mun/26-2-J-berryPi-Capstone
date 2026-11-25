"""Collect, build, and apply personal handwheel baselines for the company CAN map."""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import math
from pathlib import Path
import re
from statistics import quantiles
from tempfile import NamedTemporaryFile

DATA_DIR = Path(__file__).resolve().parent / "data"
SPEED_BAND = (95, 100)
ANGLE_BANDS = (1, 2, 5, 10)
STEERING_MAX_AGE_US = 80_000
COMPANY_SPEED_MAX_AGE_US = 120_000
MIN_EPISODES = 5
RULE_ID = "steer_hold_personal_v1"
# Personal decision → webboard can.telemetry deviation_status (schema v1.2).
WEBBOARD_STATUS = {"normal": "within_baseline", "deviation": "outside_baseline"}
CONFIG = {
    "version": 1, "signal_map_id": "can-list-speed-steering-v1",
    "speed_band_kmh": list(SPEED_BAND), "speed_max_age_us": COMPANY_SPEED_MAX_AGE_US,
    "steering_max_age_us": STEERING_MAX_AGE_US,
}


def valid_id(value):
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9_-]{1,48}", value):
        raise ValueError("Use an ID of 1–48 letters, digits, underscores or hyphens")
    return value


def write_line(file, item):
    file.write(json.dumps(item, ensure_ascii=False, allow_nan=False) + "\n")


class NormalRecorder:
    def __init__(self, data_dir, participant, session):
        self.participant, self.session = valid_id(participant), valid_id(session)
        self.path = data_dir / "normal" / self.participant / f"{self.session}.jsonl"
        self.last_stamp = -1

    def __enter__(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.file = self.path.open("x", encoding="utf-8")
        write_line(self.file, dict(CONFIG, participant=self.participant, session=self.session))
        return self

    def observe(self, stamp_us, speed, steer):
        if stamp_us > self.last_stamp:
            write_line(self.file, [stamp_us, speed, steer])
            self.last_stamp = stamp_us

    def __exit__(self, exc_type, *_):
        try:
            if exc_type is None or exc_type is KeyboardInterrupt:
                write_line(self.file, {"complete": True})
        finally:
            self.file.close()


class Holds:
    """Unity hold logic, using integer observation timestamps from the steering stream."""
    def __init__(self):
        self.last_stamp = None
        self.valid = False
        self.active = {}

    def reset(self):
        self.last_stamp = None
        self.valid = False
        self.active.clear()

    def observe(self, stamp_us, speed, steer):
        if self.last_stamp is not None and stamp_us <= self.last_stamp:
            return 0, []
        dt = stamp_us - self.last_stamp if self.last_stamp is not None else 0
        continuous = self.valid and 0 < dt <= STEERING_MAX_AGE_US
        self.last_stamp = stamp_us
        self.valid = speed is not None and steer is not None and SPEED_BAND[0] <= speed <= SPEED_BAND[1]
        if not continuous or not self.valid:
            self.active.clear()
        if not self.valid:
            return 0, []

        ended = []
        direction = 1 if steer > 0 else -1
        for angle in ANGLE_BANDS:
            previous = self.active.get(angle)
            if previous and direction == previous[0] and abs(steer) >= 0.9 * angle:
                continue
            if previous:
                _, started, known_start = self.active.pop(angle)
                duration_us = stamp_us - started
                if known_start and duration_us >= 120_000:
                    ended.append((angle, duration_us / 1_000_000))
            if abs(steer) >= angle:
                self.active[angle] = direction, stamp_us, continuous
        return (dt if continuous else 0), ended


def center_start_time_s(angle_deg, speed_kmh):
    radius = 2.8 / math.tan(math.radians(angle_deg / 15))
    return radius / (speed_kmh / 3.6) * math.acos(1 - 0.85 / radius)


@dataclass(frozen=True)
class Profile:
    participant: str
    sessions: tuple[str, ...]
    limits: dict[int, float]


def build_profile(data_dir, participant, sessions):
    valid_id(participant)
    sessions = [valid_id(session) for session in sessions]
    if len(sessions) < 2 or len(set(sessions)) != len(sessions):
        raise ValueError("Select at least two distinct normal sessions for this driver")
    durations = {angle: [] for angle in ANGLE_BANDS}
    valid_total = 0
    for session in sessions:
        source = data_dir / "normal" / participant / f"{session}.jsonl"
        tracker = Holds()
        valid_session = 0
        complete = False
        with source.open(encoding="utf-8") as file:
            if json.loads(file.readline()) != dict(CONFIG, participant=participant, session=session):
                raise ValueError(f"Incompatible recording: {source.name}")
            for line in file:
                row = json.loads(line)
                if complete:
                    raise ValueError(f"Unexpected data after end of {source.name}")
                if row == {"complete": True}:
                    complete = True
                    continue
                stamp, speed, steer = row
                if (type(stamp) is not int or stamp < 0 or
                        (tracker.last_stamp is not None and stamp <= tracker.last_stamp) or
                        any(value is not None and not math.isfinite(value) for value in (speed, steer))):
                    raise ValueError(f"Invalid observations in {source.name}")
                dt, ended = tracker.observe(stamp, speed, steer)
                valid_session += dt
                for angle, duration in ended:
                    durations[angle].append(duration)
        if not complete or valid_session < 20_000_000:
            raise ValueError(f"{source.name}: need a complete recording with at least 20 s at 95–100 km/h")
        valid_total += valid_session
    if valid_total < 60_000_000:
        raise ValueError("Need at least 60 s of valid data across normal sessions")
    limits = {
        str(angle): round(quantiles(values, n=20, method="inclusive")[-1] + 0.10, 3)
        for angle, values in durations.items() if len(values) >= MIN_EPISODES
    }
    if not limits:
        raise ValueError("Need at least five completed episodes in an angle band")
    payload = dict(CONFIG, participant=participant, sessions=sessions, hold_limits_s=limits,
                   valid_seconds=round(valid_total / 1_000_000, 3),
                   episode_counts={angle: len(values) for angle, values in durations.items()})
    path = data_dir / "profiles" / f"{participant}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent, delete=False) as file:
        temporary = Path(file.name)
        json.dump(payload, file, ensure_ascii=False, indent=2)
        file.write("\n")
    try:
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
    return path


def load_profile(data_dir, participant, session):
    path = data_dir / "profiles" / f"{valid_id(participant)}.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("participant") != participant or any(data.get(key) != value for key, value in CONFIG.items()):
        raise ValueError(f"Incompatible driver profile: {path.name}")
    sessions = tuple(valid_id(value) for value in data["sessions"])
    if valid_id(session) in sessions:
        raise ValueError("Use an assessment session ID distinct from the normal sessions")
    limits = {int(angle): float(seconds) for angle, seconds in data["hold_limits_s"].items()}
    if not limits or any(angle not in ANGLE_BANDS or not math.isfinite(sec) or sec <= 0
                         for angle, sec in limits.items()):
        raise ValueError(f"Invalid hold limits: {path.name}")
    return Profile(participant, sessions, limits)


class Assessment:
    def __init__(self, profile, session, emit):
        self.profile, self.session, self.emit = profile, session, emit
        self.holds = Holds()
        self.last_result = None
        self.expired_through_us = -1

    def publish(self, stamp_us, status, detail=None):
        result = status, None if detail is None else detail["angle_band_deg"]
        if result != self.last_result:
            self.emit(dict(t_rel_s=stamp_us / 1_000_000, session_id=self.session,
                           participant=self.profile.participant, deviation_status=status, detail=detail))
            self.last_result = result

    def expire(self, stamp_us, speed, steer):
        if speed is None or steer is None:
            self.expired_through_us = max(self.expired_through_us, stamp_us)
            self.holds.reset()
            self.publish(stamp_us, "unavailable")

    def observe(self, stamp_us, speed, steer):
        if stamp_us < self.expired_through_us:
            return
        if self.holds.last_stamp is not None and stamp_us <= self.holds.last_stamp:
            return
        self.holds.observe(stamp_us, speed, steer)
        if not self.holds.valid:
            self.publish(stamp_us, "unavailable" if speed is None or steer is None else "out_of_scope")
            return
        for angle in sorted(self.profile.limits, reverse=True):
            active = self.holds.active.get(angle)
            if active is None:
                continue
            direction, start, _ = active
            duration = (stamp_us - start) / 1_000_000
            limit = self.profile.limits[angle]
            reference = center_start_time_s(angle, speed)
            if duration > limit and duration >= 0.5 * reference:
                self.publish(stamp_us, "deviation", dict(angle_band_deg=angle * direction,
                             steering_angle_deg=steer, speed_kmh=speed, hold_s=duration,
                             limit_s=limit, reference_s=reference))
                return
        self.publish(stamp_us, "normal")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--participant", required=True)
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("normal_sessions", nargs="+")
    args = parser.parse_args()
    try:
        print(build_profile(args.data_dir, args.participant, args.normal_sessions))
    except (OSError, ValueError, TypeError, KeyError) as exc:
        parser.exit(2, f"Profile creation failed: {exc}\n")
