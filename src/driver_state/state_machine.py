"""추론 후단: 윈도우 판정(-1 처리) + 졸음 누적 상태 머신 (정상 → 주의 → 졸음).

분류기는 윈도우마다 P(졸음) 만 낸다. 최종 표시 상태는 아래 규칙으로 정한다 (docs/state-machine.md).

1. 윈도우 판정 ``judge_windows``: 품질 불량, 유효 RR 부족, |HR - PR| 초과, 특징 NaN 중 하나라도
   해당하면 그 윈도우는 -1 (판정 불가). -1 윈도우는 상태 전이 계산에 쓰지 않는다.
2. 누적 점수 s_t = alpha * p_t + (1 - alpha) * s_{t-1}. 첫 유효 윈도우는 s = p.
3. s > on_threshold 인 시간이 escalate_sec 쌓이면 한 단계 상승, s < off_threshold 인 시간이
   recover_sec 쌓이면 한 단계 하강. 그 사이(히스테리시스 구간)에서는 단계·누적을 그대로 둔다.
4. -1 이 invalid_hold_sec 이상 연속되면 '측정 불가'(level -1) 를 표시하고 점수·누적·단계를 초기화한다.
   그보다 짧은 -1 은 건너뛰고 직전 단계를 유지한다.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .features import valid_ibi_count
from .spec import ECG_FEATURES, PPG_FEATURES, InputSpec, StateMachineSpec

UNAVAILABLE = -1   # 측정 불가 단계 / 판정 불가 윈도우


def judge_windows(x_seq: np.ndarray, base_feats: np.ndarray, quality: np.ndarray,
                  channels: tuple[str, ...], spec: InputSpec) -> list[str]:
    """윈도우마다 판정 불가 사유. 빈 문자열이면 유효. 사유가 여러 개면 ', ' 로 잇는다.

    x_seq: (N, C, T) 전처리 파형, base_feats: (N, F) 정규화 전 생체 특징, quality: (N,) bool
    """
    sm = spec.state_machine
    names = ECG_FEATURES + PPG_FEATURES
    i_hr, i_pr = names.index("ecg_hr_mean"), names.index("ppg_pr_mean")
    ecg_ch = channels.index("ecg")
    out = []
    for i in range(len(x_seq)):
        why = []
        if not quality[i]:
            why.append("품질 불량")
        f = base_feats[i, :len(names)]
        if np.isnan(f).any():
            why.append("특징 계산 불가")
        n_ibi = valid_ibi_count(x_seq[i, ecg_ch].astype(np.float64), spec.fs)
        if n_ibi < sm.gate_min_ibi:
            why.append(f"유효 RR {n_ibi}개 < {sm.gate_min_ibi}")
        if not np.isnan(f[i_hr]) and not np.isnan(f[i_pr]) and abs(f[i_hr] - f[i_pr]) > sm.gate_max_hr_ppg_diff:
            why.append(f"|HR-PR| {abs(f[i_hr] - f[i_pr]):.1f} > {sm.gate_max_hr_ppg_diff:g} bpm")
        out.append(", ".join(why))
    return out


@dataclass
class StateStep:
    t: float               # 이 판정이 나온 시각 [s] (윈도우 끝 = 윈도우 시작 + window_sec)
    level: int             # 0 정상 / 1 주의 / 2 졸음 / -1 측정 불가
    level_name: str
    score: float | None    # 누적 점수 s_t (아직 유효 윈도우가 없으면 None)
    p_drowsy: float | None # 이번 윈도우 P(졸음). -1 윈도우면 None
    skipped: bool          # 이번 윈도우가 -1 이라 전이 계산에서 빠졌는지


class DrowsinessStateMachine:
    """윈도우 결과를 시간 순서대로 넣으면 단계를 돌려준다. 실시간(10초마다 1회)과 Replay 공용."""

    def __init__(self, spec: StateMachineSpec, stride_sec: float = 10.0):
        self.spec = spec
        self.stride_sec = stride_sec
        self.reset()

    def reset(self) -> None:
        self.level = 0
        self.score: float | None = None
        self.up_sec = 0.0       # s > on 인 누적 시간
        self.down_sec = 0.0     # s < off 인 누적 시간
        self.invalid_sec = 0.0  # 연속 -1 시간
        self.unavailable = False
        self._last_t: float | None = None

    def _names(self, level: int) -> str:
        return "측정 불가" if level == UNAVAILABLE else self.spec.level_names[level]

    def step(self, t: float, p_drowsy: float | None) -> StateStep:
        """t: 판정 시각 [s], p_drowsy: P(졸음) 또는 None (-1 윈도우)."""
        sm = self.spec
        dt = self.stride_sec if self._last_t is None else max(0.0, t - self._last_t)
        self._last_t = t
        if p_drowsy is None:
            self.invalid_sec += dt
            if self.invalid_sec >= sm.invalid_hold_sec and not self.unavailable:
                lt = self._last_t
                self.reset()
                self._last_t, self.unavailable = lt, True
            lv = UNAVAILABLE if self.unavailable else self.level
            return StateStep(t, lv, self._names(lv), self.score, None, True)

        self.invalid_sec, self.unavailable = 0.0, False
        p = float(p_drowsy)
        self.score = p if self.score is None else sm.alpha * p + (1 - sm.alpha) * self.score
        if self.score > sm.on_threshold:
            self.up_sec, self.down_sec = self.up_sec + dt, 0.0
            if self.up_sec >= sm.escalate_sec and self.level < len(sm.level_names) - 1:
                self.level, self.up_sec = self.level + 1, 0.0
        elif self.score < sm.off_threshold:
            self.down_sec, self.up_sec = self.down_sec + dt, 0.0
            if self.down_sec >= sm.recover_sec and self.level > 0:
                self.level, self.down_sec = self.level - 1, 0.0
        return StateStep(t, self.level, self._names(self.level), self.score, p, False)


def timeline_metrics(times: np.ndarray, levels: np.ndarray, truth: np.ndarray,
                     alarm_level: int = 2) -> dict:
    """상태 머신 타임라인 평가.

    times: (S,) 판정 시각 [s], levels: (S,) 단계, truth: (S,) 그 시각의 정답 (0 정상 / 1 졸음 / -1 미라벨)
    - detection_delay_sec: 정답이 졸음으로 바뀐 시각부터 처음 level >= alarm_level 이 될 때까지 [s].
      졸음 구간이 끝날 때까지 못 잡으면 NaN (missed 로 셈).
    - false_alarms: 정답이 정상인 구간에서 level >= 1 로 올라간 횟수.
    - unavailable_ratio: level == -1 인 판정 비율.
    """
    times, levels, truth = map(np.asarray, (times, levels, truth))
    delays, missed = [], 0
    onsets = [i for i in range(len(truth)) if truth[i] == 1 and (i == 0 or truth[i - 1] != 1)]
    for i in onsets:
        j = i
        while j < len(truth) and truth[j] == 1 and levels[j] < alarm_level:
            j += 1
        if j < len(truth) and truth[j] == 1:
            delays.append(float(times[j] - times[i]))
        else:
            missed += 1
    rising = (levels[1:] >= 1) & (levels[:-1] < 1)
    false_alarms = int(np.sum(rising & (truth[1:] == 0)) + int(len(levels) and levels[0] >= 1 and truth[0] == 0))
    return {"n_drowsy_onsets": len(onsets), "missed": missed,
            "detection_delay_sec": delays, "false_alarms": false_alarms,
            "unavailable_ratio": float(np.mean(levels == UNAVAILABLE)) if len(levels) else 0.0}
