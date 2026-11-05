"""교사 규약: 창, 스케일, 특징 이름, Baseline, 품질 기준. 값은 configs/teacher.yaml 로 덮어쓴다."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
from typing import Any

TEACHER_VERSION = "0.1.0"

# 스케일별 HRV 특징. 주파수·비선형은 짧은 창에서 불안정해 2분 이상에서만 계산한다.
TIME_FEATURES = ("hr", "rr_mean", "sdnn", "rmssd", "pnn50", "sd1", "sd2", "sd1_sd2")
LONG_FEATURES = ("lf_log", "hf_log", "lf_hf_log", "lf_nu", "dfa_a1", "sampen")
TREND_FEATURES = ("trend_hr_slope", "trend_rmssd_slope")
# 비율 r = ln(x / m) 을 추가로 만드는 양수 특징 (나머지는 z 만)
RATIO_FEATURES = ("hr", "rr_mean", "sdnn", "rmssd", "sd1", "sd2")
# 입력에 넣지 않는 열 (키·라벨·메타·품질). 지름길 방지용으로 명시해 둔다.
NON_INPUT = ("dataset_id", "subject_id", "session_id", "group", "t_start", "t_rel", "y", "valid",
             "inducer", "coverage_30", "coverage_120", "coverage_300", "baseline_valid_raw", "row_id")


def scale_tag(sec: float) -> str:
    return f"{int(round(sec))}"


@dataclass
class TeacherSpec:
    fs: float = 100.0                    # 공통 전처리 주기 (학생과 동일)
    window_sec: float = 30.0             # 학생과 동일
    stride_sec: float = 10.0             # 학생과 동일
    scales: tuple[float, ...] = (30.0, 120.0, 300.0)
    long_min_sec: float = 120.0          # 주파수·비선형 특징을 계산하는 최소 스케일
    trend_half_sec: float = 300.0        # 추세 = 창 중심 앞뒤 5분
    min_coverage: float = 0.6            # 스케일 구간을 정상 RR 이 이 비율 이상 덮어야 계산
    # 창 유효 조건 (검토안 8장, 규약 3장과 같은 하한)
    min_rr_per_window: int = 20
    max_abnormal_ratio: float = 0.10
    rr_jump: float = 0.20                # 국소 중앙값 대비 변화율이 이보다 크면 이상 RR
    min_label_purity: float = 0.8
    # 개인 Baseline (교사 전용: 5분)
    baseline_sec: float = 300.0
    min_baseline_sec: float = 60.0
    min_windows_for_sigma_s: int = 3
    train_invalid_rate: float = 0.2
    # RR CNN+GRU 입력
    tacho_sec: float = 300.0
    tacho_fs: float = 4.0
    version: str = TEACHER_VERSION

    @property
    def window_len(self) -> int:
        return int(round(self.window_sec * self.fs))

    def scale_features(self, sec: float) -> tuple[str, ...]:
        names = TIME_FEATURES + (LONG_FEATURES if sec >= self.long_min_sec else ())
        return tuple(f"{n}_{scale_tag(sec)}" for n in names)

    @property
    def raw_feature_names(self) -> tuple[str, ...]:
        out: tuple[str, ...] = ()
        for s in self.scales:
            out += self.scale_features(s)
        return out + TREND_FEATURES

    @property
    def baseline_feature_names(self) -> tuple[str, ...]:
        """기준 구간 대비 정규화를 하는 특징 (추세 제외)."""
        return tuple(n for n in self.raw_feature_names if n not in TREND_FEATURES)

    def ratio_feature_names(self) -> tuple[str, ...]:
        return tuple(n for n in self.baseline_feature_names if n.rsplit("_", 1)[0] in RATIO_FEATURES)

    def personal_feature_names(self) -> tuple[str, ...]:
        return (tuple(f"z_{n}" for n in self.baseline_feature_names)
                + tuple(f"r_{n}" for n in self.ratio_feature_names()) + ("baseline_valid",))

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["scales"] = list(self.scales)
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any] | None) -> "TeacherSpec":
        d = dict(d or {})
        known = {f.name for f in fields(cls)}
        d = {k: v for k, v in d.items() if k in known}
        if "scales" in d:
            d["scales"] = tuple(float(x) for x in d["scales"])
        return cls(**d)


@dataclass
class LabelRule:
    """원 라벨 → 이진 라벨 (0 정상, 1 졸음, -1 제외). 어댑터가 labels.csv 를 만들 때 쓴다."""
    kss_drowsy_min: float = 7.0          # KSS >= 7 → 1
    kss_alert_max: float = 5.0           # KSS <= 5 → 0, 사이(6)는 -1
    fatigue_alert: tuple[int, ...] = (0,)        # MPD-DF 피로 단계
    fatigue_drowsy: tuple[int, ...] = (3, 4)
    event_drowsy_sec: float = 60.0       # DD-Database: 이벤트 ±60초 = 졸음
    event_alert_gap_sec: float = 600.0   # 이벤트에서 10분 이상 떨어진 구간 = 정상
    extra: dict = field(default_factory=dict)

    def kss(self, v: float) -> int:
        if v != v:   # NaN
            return -1
        if v >= self.kss_drowsy_min:
            return 1
        if v <= self.kss_alert_max:
            return 0
        return -1

    def fatigue(self, v: int) -> int:
        if v in self.fatigue_drowsy:
            return 1
        if v in self.fatigue_alert:
            return 0
        return -1
