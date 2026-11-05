"""공통 입력 규약 (Input Spec).

5개 후보 모델(MLP, CNN+BiLSTM, Random Forest, XGBoost, LightGBM)이 모두 같은
규약으로 데이터를 받도록, 채널 이름·샘플링 주기·윈도우·특징 순서·라벨 형식을
한 곳에서 정의한다. 값은 ``configs/default.yaml`` 로 덮어쓸 수 있다.

용어
- Recording: 한 사람(subject)의 한 번 측정(session). 채널별 원신호와 샘플링 주기.
- Window: Recording 을 ``window_sec`` 길이로 ``stride_sec`` 간격마다 자른 조각.
  모델 입력 1개 = 윈도우 1개.
- sequence 입력: 윈도우의 리샘플된 파형, shape (N, C, T).
  N = 윈도우 개수, C = 채널 수, T = window_sec * fs.
- features 입력: 윈도우마다 계산한 고정 길이 특징 벡터, shape (N, F).
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

import yaml

# 채널 이름은 아래 문자열로 고정한다. 데이터 담당이 넘겨주는 Recording.signals 의 key.
BIO_CHANNELS: tuple[str, ...] = ("ecg", "ppg")
CAN_CHANNELS: tuple[str, ...] = ("speed", "steering_angle", "accel_pedal", "brake")

# 특징 이름과 순서. 순서가 바뀌면 학습된 모델과 호환되지 않으므로 추가만 하고 순서는 유지한다.
ECG_FEATURES: tuple[str, ...] = (
    "ecg_hr_mean",   # 평균 심박수 [bpm]
    "ecg_rr_mean",   # 평균 RR 간격 [ms]
    "ecg_sdnn",      # RR 간격 표준편차 [ms]
    "ecg_rmssd",     # 연속 RR 차이의 제곱평균제곱근 [ms]
    "ecg_pnn50",     # 연속 RR 차이가 50ms 를 넘는 비율 [0~1]
)
PPG_FEATURES: tuple[str, ...] = (
    "ppg_pr_mean",   # 평균 맥박수 [bpm]
    "ppg_ppi_sdnn",  # 맥파 간격(PPI) 표준편차 [ms]
    "ppg_ppi_rmssd", # PPI 연속 차이 RMS [ms]
)
CAN_FEATURES: tuple[str, ...] = (
    "can_speed_mean",       # 평균 차속 [km/h]
    "can_speed_std",        # 차속 표준편차
    "can_steer_std",        # 조향각 표준편차 [deg]
    "can_steer_rate_mean",  # 조향각 변화율 절댓값 평균 [deg/s]
    "can_accel_mean",       # 가속페달 평균 [%]
    "can_brake_ratio",      # 브레이크 on 비율 [0~1]
)


@dataclass
class LabelSpec:
    """라벨 규약. 4단계 정의가 미확정이라 이름은 임시값이고 정수 id 만 계약으로 쓴다.

    - 모델 출력은 항상 0..n_classes-1 의 정수 id 와 (N, n_classes) 확률.
    - ``binary_map`` 은 다단계 분류가 어려울 때를 대비한 이진 백업 매핑.
    """

    n_classes: int = 4
    names: list[str] = field(default_factory=lambda: ["매우 양호", "양호", "피로", "상태 이상"])
    # 윈도우 라벨 = 윈도우 안 샘플 라벨의 최빈값. 최빈값 비율이 이 값보다 낮으면 버린다.
    min_label_purity: float = 0.8
    binary_map: dict[int, int] = field(default_factory=lambda: {0: 0, 1: 0, 2: 1, 3: 1})


@dataclass
class InputSpec:
    fs: float = 100.0            # 공통 샘플링 주기 [Hz]. 모든 채널을 이 주기로 리샘플.
    window_sec: float = 30.0     # 윈도우 길이 [s]. HRV 계산에 최소 30초 권장.
    stride_sec: float = 10.0     # 윈도우 이동 간격 = 상태 갱신 주기 [s].
    use_can: bool = False        # True 면 CAN 채널·특징도 입력에 포함 (early fusion 실험용).
    # 생체 파형은 센서마다 단위·증폭이 달라 윈도우 단위로 z-score 한다 (fit 불필요).
    # CAN 은 물리 단위가 의미 있으므로 학습셋 통계로 z-score 한다.
    sequence_bio_norm: str = "per_window"   # per_window | global
    feature_norm: str = "global"            # global | none
    label: LabelSpec = field(default_factory=LabelSpec)

    # --- 파생 값 ---
    @property
    def window_len(self) -> int:
        return int(round(self.window_sec * self.fs))

    @property
    def stride_len(self) -> int:
        return int(round(self.stride_sec * self.fs))

    @property
    def channels(self) -> tuple[str, ...]:
        return BIO_CHANNELS + (CAN_CHANNELS if self.use_can else ())

    @property
    def feature_names(self) -> tuple[str, ...]:
        return ECG_FEATURES + PPG_FEATURES + (CAN_FEATURES if self.use_can else ())

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["channels"] = list(self.channels)
        d["feature_names"] = list(self.feature_names)
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "InputSpec":
        d = dict(d)
        d.pop("channels", None)
        d.pop("feature_names", None)
        label = d.pop("label", None) or {}
        if "binary_map" in label:
            label["binary_map"] = {int(k): int(v) for k, v in label["binary_map"].items()}
        return cls(label=LabelSpec(**label), **d)


def load_config(path: str | Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def spec_from_config(cfg: dict[str, Any]) -> InputSpec:
    return InputSpec.from_dict(cfg.get("input", {}))
