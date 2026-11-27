"""시뮬레이터 차량 경고 → 상태 머신 v2 의 ``can_drowsy`` 보조 확정 신호.

규칙 (vehicle_warning_sim_v1)
- 10초 판정 창 안에서 다음 중 하나가 있으면 ``can_drowsy=True``:
  - 차선안전 표시 ``LaneSafe_indi == 5`` (YellowWarn, 차선 경고). 단 방향지시등이 켜져 있거나
    꺼진 뒤 ``TURN_GRACE_S`` 이내면 정상 차로 변경으로 보고 제외한다.
  - 운전자 부주의 경고 ``Careless_Warn >= 2`` (YellowBlink 이상).
  - (선택) PR #13 개인 조향 기준 이탈. ``use_steering=True`` 일 때만.
- 경고가 없고 경고 상태를 알고 있으며 창의 절반 이상 CAN 을 받았으면 ``False``, 그 밖에는 ``None``.
- **시뮬레이터 전용.** 실차 CAN 데이터셋·검증이 없으므로 ``origin != "simulated"`` 이면 항상 ``None``.
  경고 신호는 Unity 시뮬레이터의 차선 이탈·부주의 경고 기능이 회사 CAN 매핑 형식으로 내보낸다고 가정한다.
- 차량 경고는 졸음 **정답 라벨이 아니라 보조 입력**이다. 상태 머신 v2 에서 생체 판정이 '주의' 이상일 때만
  졸음 격상 근거가 되며, CAN 만으로 경고를 시작하지 않는다.

입력 (모두 첫 표본 기준 마이크로초)
- JSONL 객체 행: ``{"t_us": .., "speed_kmh": .., "steering_deg": .., "LaneSafe_indi": .., "Careless_Warn": ..,
  "LTurnSig_Indi": .., "RTurnSig_Indi": ..}`` — t_us 외에는 바뀐 값만 넣어도 된다 (이벤트 신호는 마지막 값 유지).
- JSONL 배열 행 ``[t_us, speed, steer]`` (``can_live.py --collect-normal`` 기록과 같음).
- 회사 매핑 BLF (python-can·cantools 필요): 0x102 속도, 0x125 조향, 0x100·0x101 경고·방향지시등.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
import importlib.util
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory

ROOT = Path(__file__).resolve().parents[2]
CAN_DIR = ROOT / "git_upload_ver1"
WARNINGS_DBC = CAN_DIR / "mapping" / "can_list_warnings.dbc"


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

RULE_ID = "vehicle_warning_sim_v1"
WINDOW_S = 10.0
MIN_COVERAGE = 0.5               # 창의 절반 이상 (1초 칸 기준) CAN 을 받아야 False
TURN_GRACE_S = 3.0               # 방향지시등이 꺼진 뒤 이 시간까지 차선 경고 제외
LANE_WARN = 5                    # LaneSafe_indi YellowWarn
CARELESS_WARN_MIN = 2            # Careless_Warn YellowBlink 이상
SAMPLE_GAP_US = sp.STEERING_MAX_AGE_US
EMIT_INTERVAL_US = 100_000       # can.telemetry 최대 10 Hz
CODES = ("LaneSafe_indi", "Careless_Warn", "LTurnSig_Indi", "RTurnSig_Indi")


@dataclass
class CanSample:
    t_us: int
    speed: float | None = None
    steer: float | None = None
    codes: dict = field(default_factory=dict)
    has_speed: bool = False
    has_steer: bool = False


# ---------------------------------------------------------------- 입력 소스
def read_samples(path):
    path = Path(path)
    if path.suffix.lower() == ".blf":
        return _read_blf(path)
    return _read_jsonl(path)


def _read_jsonl(path):
    with Path(path).open(encoding="utf-8") as file:
        for line in file:
            row = json.loads(line)
            if isinstance(row, list):
                stamp, speed, steer = row
                yield CanSample(int(stamp), speed, steer, has_speed=True, has_steer=True)
            elif isinstance(row, dict) and "t_us" in row:
                codes = {k: int(row[k]) for k in CODES if row.get(k) is not None}
                yield CanSample(int(row["t_us"]), row.get("speed_kmh"), row.get("steering_deg"), codes,
                                has_speed="speed_kmh" in row, has_steer="steering_deg" in row)


def _read_blf(path):
    import can
    import cantools
    if str(CAN_DIR) not in sys.path:
        sys.path.insert(0, str(CAN_DIR))
    from can_live import can_list_decoder
    decoder = can_list_decoder()
    warnings = cantools.database.load_file(WARNINGS_DBC, database_format="dbc")
    warning_ids = {m.frame_id: m for m in warnings.messages}
    origin = None
    with can.BLFReader(str(path)) as frames:
        for frame in frames:
            if origin is None:
                origin = frame.timestamp
            stamp = round((frame.timestamp - origin) * 1_000_000)
            message = warning_ids.get(frame.arbitration_id)
            if message is not None and not frame.is_extended_id:
                values = message.decode(frame.data, decode_choices=False)
                yield CanSample(stamp, codes={k: int(v) for k, v in values.items() if k in CODES})
                continue
            updates = decoder.decode(frame)
            if updates:
                yield CanSample(stamp, updates.get("speed"), updates.get("steering"),
                                has_speed="speed" in updates, has_steer="steering" in updates)


def synthetic_highway(duration_s, *, deviation_every_s=30.0, first_deviation_s=20.0, lane_change_every_s=60.0,
                      deviation_hold_s=1.2, angle_deg=5.2, normal_hold_s=0.2, speed_kmh=100, steering_step_ms=10):
    """Unity 대체 합성 고속도로 주행 (시연·테스트용, 성능 근거 아님).

    - 10 ms 조향, 100 ms 속도. 시작 시 LaneSafe_indi=4(Green), Careless_Warn=0, 방향지시등 0.
    - ``first_deviation_s`` 부터 ``deviation_every_s`` 마다 이상 상황: 홀수 번째는 깜빡이 없는 차선 경고
      (LaneSafe_indi=5, 1.5초, 같은 방향 조향 유지), 짝수 번째는 부주의 경고(Careless_Warn=2, 3초).
    - ``first_deviation_s + 15`` 부터 ``lane_change_every_s`` 마다 깜빡이 켠 정상 차로 변경
      (좌측 방향지시등 4초, 그 안에서 차선 경고 1초) — 제외돼야 한다.
    """
    state = dict(LaneSafe_indi=4, Careless_Warn=0, LTurnSig_Indi=0, RTurnSig_Indi=0)
    yield CanSample(0, codes=dict(state))

    def target(t):
        lane, careless, turn, steer = 4, 0, 0, None
        if deviation_every_s and t >= first_deviation_s:
            k, phase = divmod(t - first_deviation_s, deviation_every_s)
            if int(k) % 2 == 0:
                if phase < max(1.5, deviation_hold_s):
                    lane = LANE_WARN
                if phase < deviation_hold_s:
                    steer = angle_deg
            elif phase < 3.0:
                careless = CARELESS_WARN_MIN
        if lane_change_every_s and t >= first_deviation_s + 15:
            phase = (t - first_deviation_s - 15) % lane_change_every_s
            if phase < 4.0:
                turn = 1
                steer = -3.0 if phase < 1.5 else 3.0 if phase < 3.0 else 0.0
                if 1.0 <= phase < 2.0:
                    lane = LANE_WARN
        return lane, careless, turn, steer

    for ms in range(0, int(duration_s * 1000), 10):
        t = ms / 1000
        lane, careless, turn, steer = target(t)
        changed = {k: v for k, v in (("LaneSafe_indi", lane), ("Careless_Warn", careless), ("LTurnSig_Indi", turn))
                   if state[k] != v}
        if changed:
            state.update(changed)
            yield CanSample(ms * 1000 + 1000, codes=changed)
        if ms % 100 == 0:
            yield CanSample(ms * 1000 + 2000, speed=speed_kmh, has_speed=True)
        if ms % steering_step_ms:
            continue
        if steer is None:
            steer = angle_deg if 100 <= ms % 1500 < 100 + normal_hold_s * 1000 else 0.0
        yield CanSample(ms * 1000 + 5000, speed=speed_kmh, steer=steer, has_speed=True, has_steer=True)


def synthetic_profile(participant="SIM", *, normal_hold_s=0.2, data_dir=None):
    """합성 정상 직선 주행 두 번 → PR #13 build_profile 로 개인 조향 기준 생성 (조향 규칙 선택 시)."""
    def build(root):
        for session in ("sim-normal-01", "sim-normal-02"):
            with sp.NormalRecorder(root, participant, session) as recorder:
                for s in synthetic_highway(32, deviation_every_s=0, lane_change_every_s=0, normal_hold_s=normal_hold_s):
                    if s.has_steer:
                        recorder.observe(s.t_us, s.speed, s.steer)
        sp.build_profile(root, participant, ("sim-normal-01", "sim-normal-02"))
        return sp.load_profile(root, participant, "sim-assess")
    if data_dir is not None:
        return build(Path(data_dir))
    with TemporaryDirectory() as temp:
        return build(Path(temp))


# ---------------------------------------------------------------- 판정
class CanAssist:
    """런타임 상대 시각(초)으로 CAN 표본을 따라가며 판정과 웹보드 페이로드를 만든다."""

    def __init__(self, source_factory, *, origin="simulated", window_s=WINDOW_S, profile=None):
        self.source_factory, self.profile = source_factory, profile
        self.origin, self.window_s = origin, float(window_s)
        self.reset()

    @property
    def enabled(self):
        return self.origin == "simulated"

    @property
    def use_steering(self):
        return self.profile is not None

    def reset(self):
        self.source = iter(self.source_factory())
        self.pending = None
        self.assessment = sp.Assessment(self.profile, "sim-assess", lambda _d: None) if self.profile else None
        self.codes = {}                  # name -> (value, t_us)
        self.turn_off_us = None          # 마지막으로 방향지시등이 꺼진 시각
        self.speed = self.steer = None   # (value, t_us)
        self.frames = deque()            # 받은 프레임 시각
        self.triggers = deque()          # (t_us, kind)
        self.since_emit = set()
        self.last_emit_us = -EMIT_INTERVAL_US

    # -- state
    def code(self, name):
        item = self.codes.get(name)
        return None if item is None else item[0]

    def _turn_active(self, t_us):
        on = bool(self.code("LTurnSig_Indi")) or bool(self.code("RTurnSig_Indi"))
        recent = self.turn_off_us is not None and t_us - self.turn_off_us <= TURN_GRACE_S * 1_000_000
        return on or recent

    def _apply(self, s):
        was_on = bool(self.code("LTurnSig_Indi")) or bool(self.code("RTurnSig_Indi"))
        for name, value in s.codes.items():
            self.codes[name] = value, s.t_us
        if was_on and not (bool(self.code("LTurnSig_Indi")) or bool(self.code("RTurnSig_Indi"))):
            self.turn_off_us = s.t_us
        if s.has_speed:
            self.speed = s.speed, s.t_us
        if s.has_steer:
            self.steer = s.steer, s.t_us
        self.frames.append(s.t_us)
        kinds = set()
        if self.code("LaneSafe_indi") == LANE_WARN:
            kinds.add("lane_suppressed" if self._turn_active(s.t_us) else "lane")
        if (self.code("Careless_Warn") or 0) >= CARELESS_WARN_MIN:
            kinds.add("careless")
        if self.assessment is not None and s.has_steer:
            speed = self.speed[0] if self.speed and s.t_us - self.speed[1] <= sp.COMPANY_SPEED_MAX_AGE_US else None
            self.assessment.observe(s.t_us, speed, s.steer)
            result = self.assessment.last_result
            if result and result[0] == "deviation":
                kinds.add("steering_suppressed" if self._turn_active(s.t_us) else "steering")
        for kind in kinds:
            self.triggers.append((s.t_us, kind))
        self.since_emit |= kinds

    def advance(self, t_rel_s):
        """t_rel_s 까지 표본을 처리하고, 보낼 can.telemetry 페이로드(없으면 None)를 돌려준다."""
        now_us = round(t_rel_s * 1_000_000)
        while True:
            if self.pending is None:
                try:
                    self.pending = next(self.source)
                except StopIteration:
                    break
            if self.pending.t_us > now_us:
                break
            sample, self.pending = self.pending, None
            self._apply(sample)
        if self.assessment is not None and (self.steer is None or now_us - self.steer[1] > SAMPLE_GAP_US):
            self.assessment.expire(now_us, None, None)
        horizon = now_us - 3 * self.window_s * 1_000_000
        for queue in (self.frames, self.triggers):
            while queue and (queue[0] if queue is self.frames else queue[0][0]) < horizon:
                queue.popleft()
        if now_us - self.last_emit_us < EMIT_INTERVAL_US:
            return None
        self.last_emit_us = now_us
        payload = self.telemetry(now_us)
        self.since_emit = set()
        return payload

    def receiving(self, t_rel_s):
        return bool(self.frames) and round(t_rel_s * 1_000_000) - self.frames[-1] <= 1_000_000

    def window_flag(self, end_s):
        """(end-window, end] 판정 창의 can_drowsy."""
        if not self.enabled:
            return None
        start_us, end_us = round((end_s - self.window_s) * 1_000_000), round(end_s * 1_000_000)
        kinds = {kind for t, kind in self.triggers if start_us < t <= end_us}
        if kinds & {"lane", "careless", "steering"}:
            return True
        slots = {(t - start_us - 1) // 1_000_000 for t in self.frames if start_us < t <= end_us}
        known = self.code("LaneSafe_indi") is not None and self.code("Careless_Warn") is not None
        if known and len(slots) >= MIN_COVERAGE * self.window_s:
            return False
        return None

    def window_reasons(self, end_s):
        start_us, end_us = round((end_s - self.window_s) * 1_000_000), round(end_s * 1_000_000)
        return sorted({kind for t, kind in self.triggers if start_us < t <= end_us})

    def telemetry(self, now_us):
        def fresh(item, max_age):
            return item if item is not None and 0 <= now_us - item[1] <= max_age else (None, None)
        speed, speed_t = fresh(self.speed, sp.COMPANY_SPEED_MAX_AGE_US)
        steer, steer_t = fresh(self.steer, SAMPLE_GAP_US)
        receiving = self.receiving(now_us / 1_000_000)
        warned = bool(self.since_emit & {"lane", "careless", "steering"})
        drowsy = (True if warned else False if receiving else None) if self.enabled else None
        if self.use_steering and "steering" in self.since_emit:
            status = "outside_baseline"
        elif self.use_steering and steer is not None and receiving:
            status = "within_baseline"
        else:
            status = "unavailable"
        codes = {name: self.code(name) for name in CODES}
        return {
            "speed_kmh": speed, "steering_angle_deg": steer, "steering_rate_deg_s": None,
            "road_wheel_angle_deg": None, "accelerator_pct": None, "brake_pct": None,
            "accel_pedal_raw": None, "brake_pedal_raw": None, "baseline_coverage_pct": None,
            "deviation_status": status, "can_drowsy": drowsy,
            "rule_id": RULE_ID if self.enabled else None,
            "signal_map_id": sp.CONFIG["signal_map_id"],
            "signal_times_s": {"speed": None if speed_t is None else speed_t / 1_000_000,
                               "steering": None if steer_t is None else steer_t / 1_000_000,
                               "accelerator": None, "brake": None},
            "status_codes": codes,
            "status_signal_times_s": {name: (None if name not in self.codes else self.codes[name][1] / 1_000_000)
                                      for name in CODES},
        }


def build_can_assist(scenario=None, *, steering_profile=None, use_steering=False,
                     deviation_every_s=30.0, first_deviation_s=20.0, lane_change_every_s=60.0):
    """서버 옵션 → CanAssist. scenario 가 없으면 내장 합성 주행. 조향 규칙은 선택."""
    profile = None
    if use_steering or steering_profile is not None:
        if steering_profile is None:
            profile = synthetic_profile()
        else:
            path = Path(steering_profile)
            profile = sp.load_profile(path.parent.parent, path.stem, "sim-assess")
    if scenario is None:
        factory = lambda: synthetic_highway(24 * 3600, deviation_every_s=deviation_every_s,
                                            first_deviation_s=first_deviation_s,
                                            lane_change_every_s=lane_change_every_s)
    else:
        factory = lambda: read_samples(scenario)
    return CanAssist(factory, origin="simulated", profile=profile)
