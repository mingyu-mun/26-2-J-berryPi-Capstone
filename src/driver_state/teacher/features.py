"""RR 캐시 → 창 특징표 + 참가자 Baseline 프로필 (+ CNN+GRU 용 RR 타코그램).

창 그리드는 학생(recording_to_windows)과 같다: 세션 시작 + k × stride, 기준 구간과 겹치는 창은 제외,
휴식(is_rest) 세션은 기준 전용이라 창을 만들지 않는다. 키 (dataset_id, subject_id, session_id, t_start).

특징은 창 중심 c = t + window/2 를 기준으로 스케일 L 마다 [c − L/2, c + L/2) 의 정상 RR 로 계산한다.
교사는 오프라인 라벨러라 미래 문맥을 써도 된다 (학생은 쓰지 않음).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import hrv
from .rr import RRSeries
from .spec import TIME_FEATURES, LONG_FEATURES, TeacherSpec


# ---------------------------------------------------------------- 구간 → 특징
def _segment(rr: RRSeries, a: float, b: float):
    """[a, b) 안의 정상 RR: (시각 s, nn ms, 연속 차이 ms, 덮는 비율)."""
    t = rr.rr_t
    i, j = np.searchsorted(t, a), np.searchsorted(t, b)
    good = ~rr.abnormal[i:j]
    nn = rr.rr[i:j] * 1000.0
    # 연속 차이는 이웃 둘 다 정상일 때만 (이상 RR 을 건너뛴 차이는 가짜 큰 값)
    both = good[1:] & good[:-1]
    d = np.diff(nn)[both]
    cov = float(nn[good].sum() / 1000.0 / max(b - a, 1e-9))
    return t[i:j][good], nn[good], d, cov


def scale_values(rr: RRSeries, a: float, b: float, sec: float, spec: TeacherSpec) -> tuple[list[float], float]:
    t, nn, d, cov = _segment(rr, a, b)
    n_feat = len(TIME_FEATURES) + (len(LONG_FEATURES) if sec >= spec.long_min_sec else 0)
    if cov < spec.min_coverage:
        return [np.nan] * n_feat, cov
    vals = hrv.time_domain(nn, d)
    if sec >= spec.long_min_sec:
        vals += hrv.long_features(t, nn)
    return vals, cov


def trend_values(rr: RRSeries, c: float, spec: TeacherSpec) -> list[float]:
    a, b = c - spec.trend_half_sec, c + spec.trend_half_sec
    t, nn, _, cov = _segment(rr, a, b)
    if cov < spec.min_coverage or len(nn) < 20:
        return [np.nan, np.nan]
    hr_slope = hrv.slope_per_min(t, 60000.0 / nn)
    # RMSSD 는 30초 블록 값의 기울기
    edges = np.arange(a, b, 30.0)
    vals = []
    for e in edges:
        _, n2, d2, c2 = _segment(rr, e, e + 30.0)
        vals.append(float(np.sqrt(np.mean(d2 ** 2))) if c2 >= spec.min_coverage and len(d2) >= 5 else np.nan)
    return [hr_slope, hrv.slope_per_min(edges + 15.0, np.asarray(vals))]


# ---------------------------------------------------------------- Baseline
@dataclass
class BaselineInfo:
    session_id: str
    start: float
    end: float


def baseline_plan(sessions: list[RRSeries], spec: TeacherSpec) -> dict[str, BaselineInfo | None]:
    """세션 id → 그 세션 창들이 쓰는 기준 구간. 규약 v0.2 규칙 + v0.3 baseline_range / baseline_per_session."""
    if sessions[0].subject_id is None:
        return {s.session_id: None for s in sessions}

    def rng(s: RRSeries) -> BaselineInfo:
        br = s.meta.get("baseline_range")
        a, b = (float(br[0]), float(br[1])) if br else (0.0, spec.baseline_sec)
        return BaselineInfo(s.session_id, a, min(b, s.duration))

    if any(s.meta.get("baseline_per_session") for s in sessions):
        return {s.session_id: rng(s) for s in sessions}
    rest = [s for s in sessions if s.is_rest]
    base = rest[0] if rest else min(sessions, key=lambda s: (s.t0, s.session_id))
    info = rng(base)
    return {s.session_id: info for s in sessions}


def baseline_profile(rr: RRSeries, info: BaselineInfo, spec: TeacherSpec) -> dict[str, float]:
    """기준 구간의 스케일별 특징 중앙값 m 과 σ_s(IQR/1.349, 창 3개 이상). 무효면 baseline_valid=0."""
    names = spec.baseline_feature_names
    out = {f"m_{n}": np.nan for n in names} | {f"s_{n}": np.nan for n in names}
    length = info.end - info.start
    if length < spec.min_baseline_sec:
        return out | {"baseline_valid": 0.0, "baseline_sec": length}
    ok_any = False
    for sec in spec.scales:
        cols = spec.scale_features(sec)
        L = min(sec, length) if sec == spec.scales[0] else sec
        if L > length + 1e-6:
            continue
        starts = np.arange(info.start, info.end - L + 1e-6, spec.stride_sec)
        vals = np.array([scale_values(rr, s0, s0 + L, sec, spec)[0] for s0 in starts], dtype=np.float64)
        if vals.size == 0:
            continue
        for k, n in enumerate(cols):
            v = vals[:, k]
            v = v[np.isfinite(v)]
            if len(v):
                out[f"m_{n}"] = float(np.median(v))
                ok_any = True
            if len(v) >= spec.min_windows_for_sigma_s:
                out[f"s_{n}"] = float((np.percentile(v, 75) - np.percentile(v, 25)) / 1.349)
    valid = ok_any and np.isfinite(out[f"m_{spec.scale_features(spec.scales[0])[0]}"])
    return out | {"baseline_valid": float(valid), "baseline_sec": length}


# ---------------------------------------------------------------- 창
def _window_label(labels_1hz: np.ndarray, t: float, spec: TeacherSpec) -> int:
    a, W = int(round(t)), int(round(spec.window_sec))
    seg = labels_1hz[a:a + W]
    v = seg[seg >= 0]
    if len(v) == 0:
        return -1
    vals, cnt = np.unique(v, return_counts=True)
    k = int(np.argmax(cnt))
    return int(vals[k]) if cnt[k] / W >= spec.min_label_purity else -1


def window_starts(rr: RRSeries, spec: TeacherSpec, skip: tuple[float, float] | None) -> np.ndarray:
    W, S = spec.window_sec, spec.stride_sec
    starts = np.arange(0.0, rr.duration - W + 1e-6, S)
    if skip is not None:
        a, b = skip
        starts = starts[~((starts < b) & (starts + W > a))]
    return starts


def session_windows(rr: RRSeries, spec: TeacherSpec, skip: tuple[float, float] | None) -> pd.DataFrame:
    W, S = spec.window_sec, spec.stride_sec
    rows = []
    names = spec.raw_feature_names
    t_rr = rr.rr_t
    for t in window_starts(rr, spec, skip):
        c = t + W / 2
        b0 = int(round(t / S))
        blocks = rr.block_ok[b0:b0 + int(round(W / S))]
        i, j = np.searchsorted(t_rr, t), np.searchsorted(t_rr, t + W)
        n_rr, n_bad = j - i, int(rr.abnormal[i:j].sum())
        valid = bool(len(blocks) and blocks.all() and n_rr >= spec.min_rr_per_window
                     and n_bad <= spec.max_abnormal_ratio * max(n_rr, 1))
        vals, covs = [], {}
        for sec in spec.scales:
            v, cov = scale_values(rr, c - sec / 2, c + sec / 2, sec, spec)
            vals += v
            covs[f"coverage_{int(sec)}"] = cov
        vals += trend_values(rr, c, spec)
        row = dict(zip(names, vals))
        row.update(covs)
        row.update(t_rel=float(t), t_start=float(rr.t0 + t), valid=valid,
                   y=_window_label(rr.labels_1hz, t, spec))
        rows.append(row)
    df = pd.DataFrame(rows)
    if len(df) == 0:
        return df
    df["dataset_id"] = rr.dataset_id
    df["subject_id"] = rr.subject_id if rr.subject_id is not None else ""
    df["session_id"] = rr.session_id
    df["group"] = f"{rr.dataset_id}/{rr.group}"
    df["inducer"] = rr.meta.get("drowsiness_inducer", "unknown")
    return df


def group_table(sessions: list[RRSeries], spec: TeacherSpec) -> tuple[pd.DataFrame, pd.DataFrame]:
    """참가자 한 명의 세션들 → (창 특징표, 프로필표). 프로필 키 = profile_key 열."""
    plan = baseline_plan(sessions, spec)
    by_id = {s.session_id: s for s in sessions}
    profiles, tables = {}, []
    for s in sessions:
        info = plan[s.session_id]
        key = f"{s.dataset_id}/{s.group}" + (f"#{info.session_id}" if info else "#none")
        if info is not None and key not in profiles:
            profiles[key] = baseline_profile(by_id[info.session_id], info, spec)
        if s.is_rest:
            continue
        skip = (info.start, info.end) if info is not None and info.session_id == s.session_id else None
        df = session_windows(s, spec, skip)
        if len(df):
            df["profile_key"] = key
            tables.append(df)
    prof = pd.DataFrame([{"profile_key": k, **v} for k, v in profiles.items()])
    return (pd.concat(tables, ignore_index=True) if tables else pd.DataFrame()), prof


def build_tables(rrs: list[RRSeries], spec: TeacherSpec, n_jobs: int = 1) -> tuple[pd.DataFrame, pd.DataFrame]:
    groups: dict[str, list[RRSeries]] = {}
    for r in rrs:
        groups.setdefault(f"{r.dataset_id}/{r.group}", []).append(r)
    items = list(groups.values())
    if n_jobs == 1:
        res = [group_table(g, spec) for g in items]
    else:
        from joblib import Parallel, delayed
        res = Parallel(n_jobs=n_jobs)(delayed(group_table)(g, spec) for g in items)
    win = pd.concat([w for w, _ in res if len(w)], ignore_index=True)
    prof = pd.concat([p for _, p in res if len(p)], ignore_index=True) if any(len(p) for _, p in res) \
        else pd.DataFrame(columns=["profile_key"])
    win["row_id"] = np.arange(len(win))
    return win, prof


# ---------------------------------------------------------------- 타코그램 (모델 B)
def tachogram(rr: RRSeries, c: float, spec: TeacherSpec) -> np.ndarray:
    """창 중심 c 앞뒤 tacho_sec/2 의 정상 RR [ms] 을 tacho_fs 로 선형 보간 → (2, T): [RR ms, 결측 마스크]."""
    T = int(round(spec.tacho_sec * spec.tacho_fs))
    grid = c - spec.tacho_sec / 2 + np.arange(T) / spec.tacho_fs
    t, nn, _, _ = _segment(rr, grid[0] - 3, grid[-1] + 3)
    out = np.zeros((2, T), np.float32)
    if len(nn) < 2:
        out[1] = 1
        return out
    out[0] = np.interp(grid, t, nn)
    # 가장 가까운 정상 박동이 2초 넘게 떨어져 있으면 결측
    k = np.clip(np.searchsorted(t, grid), 1, len(t) - 1)
    gap = np.minimum(np.abs(grid - t[k - 1]), np.abs(t[k] - grid))
    out[1] = (gap > 2.0).astype(np.float32)
    return out


def build_tachograms(rrs: list[RRSeries], win: pd.DataFrame, spec: TeacherSpec) -> np.ndarray:
    """win 행 순서대로 (N, 2, T) float16."""
    idx = {(r.dataset_id, r.subject_id if r.subject_id is not None else "", r.session_id): r for r in rrs}
    T = int(round(spec.tacho_sec * spec.tacho_fs))
    out = np.zeros((len(win), 2, T), np.float16)
    keys = list(zip(win["dataset_id"], win["subject_id"], win["session_id"], win["t_rel"]))
    for n, (d, s, ses, t) in enumerate(keys):
        out[n] = tachogram(idx[(d, s, ses)], t + spec.window_sec / 2, spec)
    return out
