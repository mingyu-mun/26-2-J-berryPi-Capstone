"""공통 입력 규약 (Input Spec) v0.2 — 개인 Baseline 반영.

5개 후보 모델(MLP, CNN+BiLSTM, Random Forest, XGBoost, LightGBM)이 모두 같은
규약으로 데이터를 받도록, 채널 이름·샘플링 주기·전처리·윈도우·특징 순서·개인화·라벨
형식을 한 곳에서 정의한다. 값은 ``configs/default.yaml`` 로 덮어쓸 수 있다.

용어
- Recording: 한 사람(subject)의 한 번 측정(session). 채널별 원신호와 샘플링 주기.
- Window: Recording 을 ``window_sec`` 길이로 ``stride_sec`` 간격마다 자른 조각.
  모델 입력 1개 = 윈도우 1개.
- Baseline(프로필): 사람마다 기준 구간(calibration) 특징의 중심값 m_s.
- 모델 입력 (InputPipeline.transform 결과 dict 의 key)
  - sequence: (N, C, T) 전처리된 파형. N=윈도우 수, C=채널 수, T=window_sec*fs
  - features: (N, F+P) 생체 특징 F개 + 개인화 특징 P개
  - personal: (N, P) 개인화 특징만 (CNN+BiLSTM 보조 입력)
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml

# 채널 이름은 아래 문자열로 고정한다. 데이터 담당이 넘겨주는 Recording.signals 의 key.
BIO_CHANNELS: tuple[str, ...] = ("ecg", "ppg")
CAN_CHANNELS: tuple[str, ...] = ("speed", "steering_angle", "accel_pedal", "brake")

# 특징 이름과 순서. 순서가 바뀌면 학습된 모델과 호환되지 않으므로 추가만 하고 순서는 유지한다.
ECG_FEATURES: tuple[str, ...] = (
    "ecg_hr_mean",   # 0 평균 심박수 [bpm]
    "ecg_rr_mean",   # 1 평균 RR 간격 [ms]
    "ecg_sdnn",      # 2 RR 간격 표준편차 [ms]
    "ecg_rmssd",     # 3 연속 RR 차이의 제곱평균제곱근 [ms]
    "ecg_pnn50",     # 4 연속 RR 차이가 50ms 를 넘는 비율 [0~1]
)
PPG_FEATURES: tuple[str, ...] = (
    "ppg_pr_mean",   # 5 평균 맥박수 [bpm]
    "ppg_ppi_sdnn",  # 6 맥파 간격(PPI) 표준편차 [ms]
    "ppg_ppi_rmssd", # 7 PPI 연속 차이 RMS [ms]
)
# 개인화 특징 (Baseline 대비). BIO 특징 뒤에 붙는다.
#   delta = x - m_s,  z = (x - m_s) / max(sigma_s, sigma_pop),  logratio = ln(x / m_s)
PERSONAL_FEATURES: tuple[str, ...] = (
    "pers_hr_delta",        # 8  ECG HR - 기준 HR [bpm]
    "pers_hr_z",            # 9  ECG HR 표준화 편차
    "pers_rmssd_logratio",  # 10 ln(RMSSD / 기준 RMSSD)
    "pers_rmssd_z",         # 11 RMSSD 표준화 편차
    "pers_pr_delta",        # 12 PPG 맥박수 - 기준 맥박수 [bpm]
    "pers_pr_z",            # 13 PPG 맥박수 표준화 편차
    "baseline_valid",       # 14 기준 유효 1 / 무효 0 (정규화하지 않음)
)
# 개인화 특징 계산에 쓰는 기준 특징 (중심값 m_s 를 저장하는 대상)
BASELINE_BASE_FEATURES: tuple[str, ...] = ("ecg_hr_mean", "ecg_rmssd", "ppg_pr_mean")
CAN_FEATURES: tuple[str, ...] = (
    "can_speed_mean", "can_speed_std", "can_steer_std",
    "can_steer_rate_mean", "can_accel_mean", "can_brake_ratio",
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
class PreprocessSpec:
    """윈도우 분할 전 공통 전처리. Recording.prefiltered=True 면 필터 단계는 건너뛴다."""

    notch_hz: float = 60.0               # 전원 잡음 (한국 60 Hz). 0 이면 끔
    antialias_ratio: float = 0.4         # 다운샘플 전 저역통과 차단 = ratio * fs_out
    ecg_band: tuple[float, float] = (0.5, 40.0)
    ppg_band: tuple[float, float] = (0.5, 8.0)
    # 품질 검사 (윈도우 단위, 필터 전 리샘플 신호 기준)
    flat_std: float = 1e-6               # 표준편차가 이 값 미만이면 평탄(접촉 불량)
    max_clip_ratio: float = 0.05         # 윈도우 최솟값/최댓값에 붙은 샘플 비율 상한 (포화)


@dataclass
class BaselineSpec:
    """개인 Baseline 규약 (docs/baseline.md)."""

    calibration_sec: float = 30.0        # 기준 구간 길이 [s]
    min_ibi_count: int = 20              # 기준 구간 유효 박동 간격 최소 개수
    max_hr_ppg_diff: float = 10.0        # ECG HR 과 PPG 맥박수 차이 상한 [bpm]
    min_windows_for_sigma_s: int = 3     # 개인 sigma_s 를 쓰기 위한 최소 기준 윈도우 수
    train_invalid_rate: float = 0.2      # 학습 때 일부러 기준을 무효 처리하는 비율 (미등록 사용자 대비)


@dataclass
class InputSpec:
    fs: float = 100.0            # 공통 샘플링 주기 [Hz]. 모든 채널을 이 주기로 리샘플.
    window_sec: float = 30.0     # 윈도우 길이 [s]. HRV 계산에 최소 30초 권장.
    stride_sec: float = 10.0     # 윈도우 이동 간격 = 상태 갱신 주기 [s].
    use_can: bool = False        # 11/5 결정: 시연 모델은 생체 단독. CAN 은 후순위.
    use_personal: bool = True    # 개인화 특징 P개 사용 여부 (False = 비개인화 비교용)
    # 생체 파형은 센서마다 단위·증폭이 달라 윈도우 단위로 z-score 한다 (fit 불필요).
    sequence_bio_norm: str = "per_window"   # per_window | global
    feature_norm: str = "global"            # global | none
    drop_bad_quality: bool = True           # 품질 불량 윈도우를 학습·평가에서 제외
    label: LabelSpec = field(default_factory=LabelSpec)
    preprocess: PreprocessSpec = field(default_factory=PreprocessSpec)
    baseline: BaselineSpec = field(default_factory=BaselineSpec)

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
    def base_feature_names(self) -> tuple[str, ...]:
        """윈도우 신호에서 직접 계산하는 특징 (개인화 제외)."""
        return ECG_FEATURES + PPG_FEATURES + (CAN_FEATURES if self.use_can else ())

    @property
    def personal_feature_names(self) -> tuple[str, ...]:
        return PERSONAL_FEATURES if self.use_personal else ()

    @property
    def feature_names(self) -> tuple[str, ...]:
        """features 입력의 열 이름. 순서: 생체 → (CAN) → 개인화."""
        return self.base_feature_names + self.personal_feature_names

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
        label = dict(d.pop("label", None) or {})
        if "binary_map" in label:
            label["binary_map"] = {int(k): int(v) for k, v in label["binary_map"].items()}
        pre = dict(d.pop("preprocess", None) or {})
        for k in ("ecg_band", "ppg_band"):
            if k in pre:
                pre[k] = tuple(pre[k])
        base = dict(d.pop("baseline", None) or {})
        return cls(label=LabelSpec(**label), preprocess=PreprocessSpec(**pre),
                   baseline=BaselineSpec(**base), **d)


def load_config(path: str | Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def spec_from_config(cfg: dict[str, Any]) -> InputSpec:
    return InputSpec.from_dict(cfg.get("input", {}))
