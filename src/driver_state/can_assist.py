"""시뮬레이터 CAN 조향 이탈 → 상태 머신 v2 의 ``can_drowsy`` 보조 확정 신호.

- 조향 판정 규칙은 ``git_upload_ver1/steering_profile.py`` (PR #13) 의 개인 기준·``Assessment`` 를 그대로 쓴다.
- 10초 판정 창마다 ``window_flag(end)`` 가 ``True`` (창 안에 개인 기준 이탈) / ``False`` (유효 직선
  주행이었고 이탈 없음) / ``None`` (판정 불가) 를 돌려준다.
- **시뮬레이터 전용.** 실차 데이터셋·실차 기준이 없으므로 ``origin != "simulated"`` 이면 항상 ``None``.
- 상태 머신 v2 에서 ``can_drowsy=True`` 는 생체 '졸음 후보' 일 때만 졸음 확정을 앞당긴다. CAN 단독으로
  경고를 시작하지 않는다.

입력 샘플은 ``(stamp_us, speed_kmh, steering_deg)``. 파일은 NormalRecorder 와 같은 JSONL
(행: ``[stamp_us, speed, steer]``, dict 행은 무시) 또는 회사 매핑 BLF (python-can·cantools 필요).
"""
from __future__ import annotations

from collections import deque
import importlib.util
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory

ROOT = Path(__file__).resolve().parents[2]
CAN_DIR = ROOT / "git_upload_ver1"


def _steering_module():
    name = "steering_profile"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, CAN_DIR / "steering_profile.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


sp = _steering_module()

WINDOW_S = 10.0
MIN_NORMAL_COVERAGE = 0.5        # 창의 절반 이상이 유효 직선 판정이어야 False
SAMPLE_GAP_US = sp.STEERING_MAX_AGE_US
EMIT_INTERVAL_US = 100_000       # can.telemetry 최대 10 Hz


# ---------------------------------------------------------------- 입력 소스
def read_samples(path):
    """JSONL 또는 BLF → (stamp_us, speed, steer) 반복자. stamp 는 첫 프레임 기준."""
    path = Path(path)
    if path.suffix.lower() == ".blf":
        return _read_blf(path)
    return _read_jsonl(path)


def _read_jsonl(path):
    with Path(path).open(encoding="utf-8") as file:
        for line in file:
            row = json.loads(line)
            if isinstance(row, dict):
                continue
            stamp, speed, steer = row
            yield int(stamp), speed, steer


def _read_blf(path):
    import can
    if str(CAN_DIR) not in sys.path:
        sys.path.insert(0, str(CAN_DIR))
    from can_live import can_list_decoder
    decoder = can_list_decoder()
    speed = speed_us = None
    origin = None
    with can.BLFReader(str(path)) as frames:
        for frame in frames:
            if origin is None:
                origin = frame.timestamp
            stamp = round((frame.timestamp - origin) * 1_000_000)
            updates = decoder.decode(frame)
            if "speed" in updates:
                speed, speed_us = updates["speed"], stamp
            if "steering" in updates:
                fresh = speed if speed_us is not None and stamp - speed_us <= sp.COMPANY_SPEED_MAX_AGE_US else None
                yield stamp, fresh, updates["steering"]


def synthetic_highway(duration_s, *, deviation_every_s=30.0, first_deviation_s=20.0,
                      deviation_hold_s=1.2, angle_deg=5.2, normal_hold_s=0.2, speed_kmh=100):
    """Unity 대체 합성 고속도로 주행. 10 ms 조향 표본.

    평소: 1.5초마다 ``angle_deg`` 로 ``normal_hold_s`` 짧은 보정.
    졸음 흉내: ``first_deviation_s`` 부터 ``deviation_every_s`` 마다 같은 방향 조향을 ``deviation_hold_s`` 유지.
    """
    for ms in range(0, int(duration_s * 1000), 10):
        t = ms / 1000
        k = (t - first_deviation_s) / deviation_every_s if deviation_every_s else -1
        in_deviation = k >= 0 and (k - int(k)) * deviation_every_s < deviation_hold_s
        if in_deviation:
            steer = angle_deg
        else:
            steer = angle_deg if 100 <= ms % 1500 < 100 + normal_hold_s * 1000 else 0.0
        yield ms * 1000 + 5000, speed_kmh, steer


def synthetic_profile(participant="SIM", *, normal_hold_s=0.2, data_dir=None):
    """합성 정상 직선 주행 두 번 → PR #13 build_profile 로 개인 기준 생성."""
    def build(root):
        for session in ("sim-normal-01", "sim-normal-02"):
            with sp.NormalRecorder(root, participant, session) as recorder:
                for stamp, speed, steer in synthetic_highway(32, deviation_every_s=0, normal_hold_s=normal_hold_s):
                    recorder.observe(stamp, speed, steer)
        sp.build_profile(root, participant, ("sim-normal-01", "sim-normal-02"))
        return sp.load_profile(root, participant, "sim-assess")
    if data_dir is not None:
        return build(Path(data_dir))
    with TemporaryDirectory() as temp:
        return build(Path(temp))


# ---------------------------------------------------------------- 판정
class CanAssist:
    """런타임 상대 시각(초)으로 CAN 표본을 따라가며 판정과 웹보드 페이로드를 만든다."""

    def __init__(self, source_factory, profile, *, origin="simulated", window_s=WINDOW_S):
        self.source_factory, self.profile = source_factory, profile
        self.origin, self.window_s = origin, float(window_s)
        self.reset()

    @property
    def enabled(self):
        return self.origin == "simulated"

    def reset(self):
        self.source = iter(self.source_factory())
        self.pending = None
        self.assessment = sp.Assessment(self.profile, "sim-assess", lambda _decision: None)
        self.history = deque()           # (stamp_us, status) — 조향 표본마다
        self.latest = None               # (stamp_us, speed, steer)
        self.flag_since_emit = None
        self.last_emit_us = -EMIT_INTERVAL_US
        self.exhausted = False

    def _status(self):
        result = self.assessment.last_result
        return result[0] if result else "unavailable"

    def advance(self, t_rel_s):
        """t_rel_s 까지 표본을 판정하고, 보낼 can.telemetry 페이로드(없으면 None)를 돌려준다."""
        now_us = round(t_rel_s * 1_000_000)
        while True:
            if self.pending is None:
                try:
                    self.pending = next(self.source)
                except StopIteration:
                    self.exhausted = True
                    break
            stamp, speed, steer = self.pending
            if stamp > now_us:
                break
            self.pending = None
            self.assessment.observe(stamp, speed, steer)
            status = self._status()
            self.history.append((stamp, status))
            self.latest = stamp, speed, steer
            if status == "deviation":
                self.flag_since_emit = "deviation"
            elif status == "normal" and self.flag_since_emit is None:
                self.flag_since_emit = "normal"
        if self.latest is None or now_us - self.latest[0] > SAMPLE_GAP_US:
            self.assessment.expire(now_us, None, None)
        while self.history and self.history[0][0] < now_us - 3 * self.window_s * 1_000_000:
            self.history.popleft()
        if now_us - self.last_emit_us < EMIT_INTERVAL_US:
            return None
        self.last_emit_us = now_us
        payload = self.telemetry(now_us)
        self.flag_since_emit = None
        return payload

    def window_flag(self, end_s):
        """(end-window, end] 판정 창의 can_drowsy."""
        if not self.enabled:
            return None
        start_us, end_us = round((end_s - self.window_s) * 1_000_000), round(end_s * 1_000_000)
        inside = [(stamp, status) for stamp, status in self.history if start_us < stamp <= end_us]
        if any(status == "deviation" for _, status in inside):
            return True
        normal_us = 0
        for (stamp, status), (nxt, _) in zip(inside, inside[1:]):
            if status == "normal":
                normal_us += min(nxt - stamp, SAMPLE_GAP_US)
        if normal_us >= MIN_NORMAL_COVERAGE * self.window_s * 1_000_000:
            return False
        return None

    def telemetry(self, now_us):
        fresh = self.latest is not None and now_us - self.latest[0] <= SAMPLE_GAP_US
        stamp, speed, steer = self.latest if fresh else (None, None, None)
        flag = self.flag_since_emit if fresh else None
        status = {"deviation": "outside_baseline", "normal": "within_baseline"}.get(flag, "unavailable")
        drowsy = {"deviation": True, "normal": False}.get(flag) if self.enabled else None
        t = None if stamp is None else stamp / 1_000_000
        return {
            "speed_kmh": speed, "steering_angle_deg": steer, "steering_rate_deg_s": None,
            "road_wheel_angle_deg": None, "accelerator_pct": None, "brake_pct": None,
            "accel_pedal_raw": None, "brake_pedal_raw": None, "baseline_coverage_pct": None,
            "deviation_status": status, "can_drowsy": drowsy,
            "rule_id": sp.RULE_ID if self.enabled else None,
            "signal_map_id": sp.CONFIG["signal_map_id"],
            "signal_times_s": {"speed": t if speed is not None else None,
                               "steering": t if steer is not None else None,
                               "accelerator": None, "brake": None},
        }


def build_can_assist(scenario=None, profile=None, *, deviation_every_s=30.0, first_deviation_s=20.0):
    """서버 옵션 → CanAssist. scenario/profile 이 없으면 내장 합성 주행·합성 기준을 쓴다."""
    if profile is None:
        prof = synthetic_profile()
    else:
        path = Path(profile)
        prof = sp.load_profile(path.parent.parent, path.stem, "sim-assess")
    if scenario is None:
        factory = lambda: synthetic_highway(24 * 3600, deviation_every_s=deviation_every_s,
                                            first_deviation_s=first_deviation_s)
    else:
        factory = lambda: read_samples(scenario)
    return CanAssist(factory, prof, origin="simulated")
