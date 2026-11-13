"""상태 머신 v2: 학생 S1-EP 출력 + 잡음 오경보 억제 (docs/state-machine-v2.md).

10초마다 창 하나를 ``StateMachineV2.step`` 에 넣는다. 기존 v0.3 (state_machine.py) 은 그대로 둔다.

흐름: 채널 선택(3장) → 3창 중앙값 → 신뢰도 가중 갱신(4장) → 잡음 동결(5장) → 단계 전이(6장)
      → 졸음 확정(8장: CAN 또는 후보 지속 추가 3분). 기준 수집(2장)은 ``select_baseline``.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import numpy as np

LEVEL_NAMES = {-1: "측정 불가", 0: "정상", 1: "주의", 2: "졸음"}


@dataclass
class V2Config:
    dt: float = 10.0                 # 판정 간격 [s]
    th_on: float = 0.55
    th_off: float = 0.40
    t_up1: float = 60.0              # 정상 → 주의
    t_up2: float = 60.0              # 주의 → 졸음 후보
    t_down: float = 60.0             # 한 단계 하강
    t_hold: float = 60.0             # -1 연속 → 측정 불가
    t_confirm: float = 180.0         # CAN 없을 때 졸음 후보 → 졸음 확정 추가 지속
    med_n: int = 3
    conf_lo: float = 0.05            # q_ecg ≤ 이 값이면 c = 1
    conf_hi: float = 0.10            # q_ecg ≥ 이 값이면 ECG 불량
    ppg_ok_q: float = 0.10
    ppg_conf: float = 0.5            # PPG 대체 판정 신뢰도
    max_hr_diff: float = 10.0
    freeze_q: float = 0.05
    freeze_sec: float = 30.0
    freeze_cap: float = 60.0         # 최근 freeze_span 동안 동결로 멈춘 시간 상한
    freeze_span: float = 300.0
    # 기준 수집 (2장)
    base_wait: float = 60.0
    base_len: float = 300.0
    base_min_clean: int = 20
    base_max_wait: float = 600.0
    base_q: float = 0.05


@dataclass
class WindowIn:
    t: float                           # 창 끝 시각 [s]
    p_ep: float | None                 # 학생 S1-EP (ECG+PPG) 확률
    q_ecg: float = 0.0
    p_ppg: float | None = None         # PPG 대체 경로 확률 (PRV RR)
    q_ppg: float | None = None
    hr_ecg: float | None = None
    hr_ppg: float | None = None
    ok_ecg: bool = True                # 기존 품질 검사 (평탄·포화·멈춤)
    ok_ppg: bool = True
    can_drowsy: bool | None = None     # CAN 졸음 징후 (없으면 None)


@dataclass
class StepOut:
    t: float
    level: int
    level_name: str
    candidate: bool                    # 졸음 후보 (확정 대기)
    score: float | None
    p_used: float | None
    confidence: float
    channel: str                       # "EP" / "PPG" / "none"
    frozen: bool


def ecg_conf(q: float, cfg: V2Config) -> float:
    return float(np.clip((cfg.conf_hi - q) / (cfg.conf_hi - cfg.conf_lo), 0.0, 1.0))


def choose_channel(w: WindowIn, cfg: V2Config) -> tuple[str, float | None, float]:
    """(채널, 확률, 신뢰도)."""
    ecg_bad = (not w.ok_ecg) or w.q_ecg > cfg.conf_hi or w.p_ep is None
    ppg_ok = (w.ok_ppg and w.p_ppg is not None and w.q_ppg is not None and w.q_ppg <= cfg.ppg_ok_q)
    if (w.hr_ecg is not None and w.hr_ppg is not None and np.isfinite(w.hr_ecg) and np.isfinite(w.hr_ppg)
            and abs(w.hr_ecg - w.hr_ppg) > cfg.max_hr_diff):
        # 센서 불일치: 이상 간격이 큰 쪽을 불량으로
        if w.q_ppg is not None and w.q_ppg < w.q_ecg:
            ecg_bad = True
        else:
            ppg_ok = False
    if not ecg_bad:
        return "EP", w.p_ep, ecg_conf(w.q_ecg, cfg)
    if ppg_ok:
        return "PPG", w.p_ppg, cfg.ppg_conf
    return "none", None, 0.0


@dataclass
class StateMachineV2:
    cfg: V2Config = field(default_factory=V2Config)

    def __post_init__(self):
        self.reset()

    def reset(self):
        self.level, self.score = 0, None
        self.up = self.down = self.inv = 0.0
        self.cand, self.cand_t = False, 0.0
        self.hist: deque = deque(maxlen=self.cfg.med_n)
        self.freeze_left = 0.0
        self.freeze_log: deque = deque()          # 동결로 상승 누적을 멈춘 시각들

    def step(self, w: WindowIn) -> StepOut:
        c = self.cfg
        ch, p, conf = choose_channel(w, c)
        if ch == "none":
            self.inv += c.dt
            if self.inv >= c.t_hold:
                self.reset(); self.inv = c.t_hold
                return StepOut(w.t, -1, LEVEL_NAMES[-1], False, None, None, 0.0, ch, False)
            return StepOut(w.t, self.level, LEVEL_NAMES[self.level], self.cand, self.score, None, 0.0, ch, False)
        self.inv = 0.0
        self.hist.append(float(p))
        pm = float(np.median(self.hist))
        self.score = pm if self.score is None else self.score + conf * (pm - self.score)
        # 잡음 동결 (5장): ECG 경로에서 이상 간격이 높으면 30초, 최근 5분 상한
        while self.freeze_log and w.t - self.freeze_log[0] > c.freeze_span:
            self.freeze_log.popleft()
        if ch == "EP" and w.q_ecg > c.freeze_q and len(self.freeze_log) * c.dt < c.freeze_cap:
            self.freeze_left = max(self.freeze_left, c.freeze_sec)
        frozen = self.freeze_left > 0 and len(self.freeze_log) * c.dt < c.freeze_cap
        if self.score > c.th_on:
            self.down = 0.0
            if frozen:
                self.freeze_log.append(w.t)
            else:
                self.up += c.dt
            need = c.t_up1 if self.level == 0 else c.t_up2
            if self.level == 0 and self.up >= need:
                self.level, self.up = 1, 0.0
            elif self.level == 1 and not self.cand and self.up >= need:
                self.cand, self.cand_t, self.up = True, 0.0, 0.0
            elif self.cand and not frozen:
                self.cand_t += c.dt
            if self.cand and (w.can_drowsy or self.cand_t >= c.t_confirm):
                self.level = 2
        elif self.score < c.th_off:
            self.up = 0.0
            self.down += c.dt
            if self.down >= c.t_down and (self.level > 0 or self.cand):
                if self.level == 2:
                    self.level, self.cand = 1, False
                elif self.cand:
                    self.cand = False
                else:
                    self.level = 0
                self.down = 0.0
        self.freeze_left = max(0.0, self.freeze_left - c.dt)
        return StepOut(w.t, self.level, LEVEL_NAMES[self.level], self.cand, self.score, p, conf, ch, frozen)


def select_baseline(t_start: np.ndarray, q: np.ndarray, cfg: V2Config | None = None, win_sec: float = 30.0):
    """기준 수집 (2장). 창 시작 시각·이상 간격 → (성공 여부, 기준 창 인덱스, 구간 [a, b]).

    착석 대기 뒤 base_len 동안 q ≤ base_q 인 창이 base_min_clean 개 이상이면 성공.
    모자라면 구간 끝을 dt 씩 늘리고, 시작 후 base_max_wait 안에 못 채우면 실패 (재측정 안내).
    """
    cfg = cfg or V2Config()
    t_start, q = np.asarray(t_start, float), np.asarray(q, float)
    a = cfg.base_wait
    b = a + cfg.base_len
    while b <= cfg.base_max_wait + 1e-6:
        m = (t_start >= a) & (t_start + win_sec <= b)
        clean = np.flatnonzero(m & (q <= cfg.base_q))
        if len(clean) >= cfg.base_min_clean:
            return True, clean, (a, b)
        b += cfg.dt
    m = (t_start >= a) & (t_start + win_sec <= cfg.base_max_wait)
    return False, np.flatnonzero(m & (q <= cfg.base_q)), (a, cfg.base_max_wait)
