"""개인 Baseline: 프로필 생성, 학습 데이터 구성, 개인화 특징 계산.

수식 (j = 기준 특징: HR, RMSSD, PPG 맥박수)
    m_{s,j}      = 사용자 s 의 기준 구간 특징 중심값 (기준 윈도우들의 중앙값)
    sigma_{s,j}  = 기준 윈도우가 3개 이상일 때 IQR / 1.349 (아니면 없음)
    sigma_pop,j  = 학습 참가자별 '한 사람 안' 특징 표준편차의 중앙값 (fold 마다 학습셋으로만 계산)
    delta_j      = x_j - m_{s,j}
    z_j          = delta_j / max(sigma_{s,j}, sigma_pop,j)
    logratio     = ln(x_rmssd / m_{s,rmssd})

기준 구간 규칙 (학습 데이터와 시연이 같은 규칙을 쓴다)
- 학습: 참가자별 휴식 녹화(is_rest)가 있으면 그 앞 calibration_sec, 없으면 t0 가 가장 이른
  녹화의 앞 calibration_sec. 이 구간과 겹치는 윈도우는 학습·평가 대상에서 뺀다.
  (기준 = 자기 자신이라 delta≈0 이 되어 성능이 부풀려지기 때문)
- 참가자 id 가 없는 데이터: 프로필 무효 → 개인화 특징 0, baseline_valid = 0.
- 시연(Pi): 시연 직전 시연자별 주행이 아닌 상태 30초 → calibrate() → 프로필 JSON.
"""
from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from .data import Recording, WindowSet, prepare_recording, recording_to_windows
from .features import valid_ibi_count, window_features
from .preprocessing import window_quality
from .spec import BASELINE_BASE_FEATURES, PERSONAL_FEATURES, InputSpec

PROFILE_VERSION = 1


@dataclass
class Profile:
    subject: str
    valid: bool
    m: dict[str, float] = field(default_factory=dict)          # 기준 특징 중심값
    sigma_s: dict[str, float] | None = None                     # 개인 변동 범위 (없으면 None)
    n_windows: int = 0
    reason: str = ""                                            # 무효 사유
    created_at: str = ""
    version: int = PROFILE_VERSION

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(asdict(self), ensure_ascii=False, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "Profile":
        return cls(**json.loads(Path(path).read_text(encoding="utf-8")))

    @classmethod
    def invalid(cls, subject: str, reason: str) -> "Profile":
        return cls(subject=subject, valid=False, reason=reason, created_at=_now())


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def profile_from_segment(seg: np.ndarray, raw_seg: np.ndarray, spec: InputSpec, subject: str) -> Profile:
    """전처리된 기준 구간 (C, n) → Profile. 품질 기준을 통과하지 못하면 무효 프로필."""
    b = spec.baseline
    need = int(round(b.calibration_sec * spec.fs))
    if seg.shape[1] < need:
        return Profile.invalid(subject, f"기준 구간 길이 부족 ({seg.shape[1] / spec.fs:.1f}s < {b.calibration_sec}s)")
    seg, raw_seg = seg[:, :need], raw_seg[:, :need]
    if not window_quality(raw_seg, spec, spec.channels):
        return Profile.invalid(subject, "신호 품질 불량 (평탄 또는 포화)")
    ecg = seg[spec.channels.index("ecg")]
    n_ibi = valid_ibi_count(ecg, spec.fs)
    if n_ibi < b.min_ibi_count:
        return Profile.invalid(subject, f"유효 박동 간격 부족 ({n_ibi} < {b.min_ibi_count})")

    # 기준 구간을 윈도우 길이로 나눌 수 있으면 여러 윈도우, 아니면 구간 전체를 윈도우 1개로 본다.
    W, S = spec.window_len, spec.stride_len
    starts = list(range(0, need - W + 1, S)) if need >= W else [0]
    width = W if need >= W else need
    names = spec.base_feature_names
    vals = np.stack([window_features(seg[:, s:s + width], spec.channels, spec) for s in starts])
    cols = {k: vals[:, names.index(k)] for k in BASELINE_BASE_FEATURES}
    m = {k: float(np.nanmedian(v)) if np.isfinite(v).any() else float("nan") for k, v in cols.items()}
    if not all(math.isfinite(x) for x in m.values()) or m["ecg_rmssd"] <= 0:
        return Profile.invalid(subject, "기준 특징 계산 불가")
    if abs(m["ecg_hr_mean"] - m["ppg_pr_mean"]) > b.max_hr_ppg_diff:
        return Profile.invalid(subject, f"ECG·PPG 심박 불일치 ({m['ecg_hr_mean']:.0f} vs {m['ppg_pr_mean']:.0f} bpm)")
    sigma_s = None
    if len(starts) >= b.min_windows_for_sigma_s:
        sigma_s = {k: float((np.nanpercentile(v, 75) - np.nanpercentile(v, 25)) / 1.349) for k, v in cols.items()}
    return Profile(subject, True, m, sigma_s, len(starts), "", _now())


def calibrate(rec: Recording, spec: InputSpec) -> Profile:
    """Recording 의 앞 calibration_sec 으로 프로필 생성 (시연 calibration, 학습 기준 구간 공통)."""
    filt, raw, _ = prepare_recording(rec, spec)
    return profile_from_segment(filt, raw, spec, rec.group)


def build_dataset(recs: list[Recording], spec: InputSpec) -> tuple[WindowSet, dict[str, Profile]]:
    """학습·평가용 윈도우 + 참가자별 프로필. 기준 구간과 겹치는 윈도우는 제외한다."""
    by_group: dict[str, list[Recording]] = {}
    for r in recs:
        by_group.setdefault(r.group, []).append(r)
    parts, profiles = [], {}
    for g, rs in by_group.items():
        base_rec = None
        if rs[0].subject_id is None:
            profiles[g] = Profile.invalid(g, "참가자 id 없음")
        else:
            rest = [r for r in rs if r.is_rest]
            base_rec = rest[0] if rest else min(rs, key=lambda r: r.t0)
        for r in rs:
            prepared = prepare_recording(r, spec)
            if r is base_rec:
                profiles[g] = profile_from_segment(prepared[0], prepared[1], spec, g)
                if r.is_rest:
                    continue   # 휴식 녹화는 기준 전용
                parts.append(recording_to_windows(r, spec, skip_sec=spec.baseline.calibration_sec,
                                                  prepared=prepared))
            else:
                parts.append(recording_to_windows(r, spec, prepared=prepared))
    return WindowSet.concat(parts, spec.channels, spec.window_len), profiles


def fit_sigma_pop(base_feats: np.ndarray, groups: np.ndarray, spec: InputSpec) -> dict[str, float]:
    """학습 참가자별 특징 표준편차의 중앙값. 참가자당 윈도우 3개 이상만 사용."""
    names = spec.base_feature_names
    out = {}
    for k in BASELINE_BASE_FEATURES:
        col = base_feats[:, names.index(k)]
        stds = []
        for g in np.unique(groups):
            v = col[groups == g]
            v = v[np.isfinite(v)]
            if len(v) >= 3:
                stds.append(v.std(ddof=1))
        s = float(np.median(stds)) if stds else 1.0
        out[k] = s if s > 1e-6 else 1.0
    return out


def personal_matrix(base_feats: np.ndarray, groups: np.ndarray, profiles: dict[str, Profile],
                    sigma_pop: dict[str, float], spec: InputSpec,
                    force_invalid: np.ndarray | None = None) -> np.ndarray:
    """생체 특징 (N, F) → 개인화 특징 (N, P). 순서는 spec.PERSONAL_FEATURES.

    프로필이 없거나 무효이거나 force_invalid 인 행은 [0, …, 0, baseline_valid=0].
    현재 윈도우 특징이 NaN 이면 해당 칸은 NaN (정규화 단계에서 중앙값 대체).
    """
    names = spec.base_feature_names
    N = len(base_feats)
    out = np.zeros((N, len(PERSONAL_FEATURES)), dtype=np.float64)
    hr = base_feats[:, names.index("ecg_hr_mean")].astype(np.float64)
    rm = base_feats[:, names.index("ecg_rmssd")].astype(np.float64)
    pr = base_feats[:, names.index("ppg_pr_mean")].astype(np.float64)
    for g in np.unique(groups):
        p = profiles.get(g)
        if p is None or not p.valid:
            continue
        rows = np.where(groups == g)[0]
        if force_invalid is not None:
            rows = rows[~force_invalid[rows]]
        if len(rows) == 0:
            continue

        def scale(k):
            s = p.sigma_s.get(k, 0.0) if p.sigma_s else 0.0
            return max(s, sigma_pop[k])

        d_hr = hr[rows] - p.m["ecg_hr_mean"]
        d_rm = rm[rows] - p.m["ecg_rmssd"]
        d_pr = pr[rows] - p.m["ppg_pr_mean"]
        with np.errstate(divide="ignore", invalid="ignore"):
            lr = np.where(rm[rows] > 0, np.log(rm[rows] / p.m["ecg_rmssd"]), np.nan)
        out[rows] = np.column_stack([
            d_hr, d_hr / scale("ecg_hr_mean"), lr, d_rm / scale("ecg_rmssd"),
            d_pr, d_pr / scale("ppg_pr_mean"), np.ones(len(rows)),
        ])
    return out.astype(np.float32)
