"""Recording → 공통 전처리 → 윈도우 분할.

데이터/전처리 담당이 넘겨줄 형식(Recording)과, 모델이 받는 형식(WindowSet)의 경계.
공개 데이터셋 로더, 실센서·Replay 입력은 모두 Recording 을 만들어 여기로 넘기면 된다.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .preprocessing import prepare_channel, resample, window_quality  # noqa: F401 (resample 재노출)
from .spec import InputSpec


@dataclass
class Recording:
    """한 사람의 한 세션.

    subject_id:  참가자 id. 없으면 None (개인 Baseline 을 만들 수 없는 데이터, baseline_valid=0).
    session_id:  측정 회차. 같은 참가자 안에서 t0 가 가장 이른 세션이 기준 세션 후보.
    signals:     채널 이름 → 1차원 신호. 채널 이름은 spec.BIO_CHANNELS / CAN_CHANNELS.
    fs:          채널 이름 → 원 샘플링 주기 [Hz]. 채널마다 달라도 된다.
    labels:      샘플 단위 라벨(정수, 미라벨 구간은 -1). ``label_fs`` 주기로 샘플링.
    t0:          세션 시작 시각 [s]. 모든 채널은 t0 에서 시작한다고 가정 (동기화는 전처리 단계 책임).
    is_rest:     휴식(기준 측정) 녹화면 True. 있으면 이 녹화가 그 참가자의 기준 세션이 된다.
    prefiltered: 데이터 담당이 노치·대역통과 필터를 이미 적용했으면 True.
    dataset_id:  여러 공개 데이터셋을 합칠 때 출처 구분용.
    """

    subject_id: str | None
    session_id: str
    signals: dict[str, np.ndarray]
    fs: dict[str, float]
    labels: np.ndarray | None = None
    label_fs: float | None = None
    t0: float = 0.0
    is_rest: bool = False
    prefiltered: bool = False
    dataset_id: str = "default"

    @property
    def group(self) -> str:
        """LOSO 분할 키. 참가자 id 가 없으면 세션마다 별도 그룹."""
        return self.subject_id if self.subject_id is not None else f"anon:{self.dataset_id}:{self.session_id}"


@dataclass
class WindowSet:
    x_seq: np.ndarray            # (N, C, T) float32, 전처리된 파형 (정규화 전)
    y: np.ndarray                # (N,) int64, 라벨 없으면 -1
    groups: np.ndarray           # (N,) 참가자 그룹 (LOSO 분할·Baseline 조회 키)
    t_start: np.ndarray          # (N,) 윈도우 시작 시각 [s]
    quality: np.ndarray          # (N,) bool, 품질 검사 통과 여부
    channels: tuple[str, ...]
    meta: dict = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.y)

    def subset(self, idx: np.ndarray) -> "WindowSet":
        return WindowSet(self.x_seq[idx], self.y[idx], self.groups[idx], self.t_start[idx],
                         self.quality[idx], self.channels, self.meta)

    @staticmethod
    def concat(parts: list["WindowSet"], channels: tuple[str, ...], T: int) -> "WindowSet":
        if not parts:
            return WindowSet(np.zeros((0, len(channels), T), np.float32), np.zeros(0, np.int64),
                             np.zeros(0, object), np.zeros(0), np.zeros(0, bool), channels)
        return WindowSet(
            x_seq=np.concatenate([p.x_seq for p in parts]),
            y=np.concatenate([p.y for p in parts]),
            groups=np.concatenate([p.groups for p in parts]),
            t_start=np.concatenate([p.t_start for p in parts]),
            quality=np.concatenate([p.quality for p in parts]),
            channels=channels,
        )


def prepare_recording(rec: Recording, spec: InputSpec) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Recording → (필터된 신호 (C, n), 필터 전 신호 (C, n), 샘플 라벨 (n,)), 모두 spec.fs 주기."""
    chans = spec.channels
    missing = [c for c in chans if c not in rec.signals]
    if missing:
        raise KeyError(f"{rec.group}/{rec.session_id}: 채널 누락 {missing}")
    out = [prepare_channel(rec.signals[c], rec.fs[c], c, spec, rec.prefiltered) for c in chans]
    n = min(len(f) for f, _ in out)
    filt = np.stack([f[:n] for f, _ in out]).astype(np.float32)
    raw = np.stack([r[:n] for _, r in out]).astype(np.float32)
    if rec.labels is not None:
        lab_fs = rec.label_fs or spec.fs
        idx = np.minimum((np.arange(n) * lab_fs / spec.fs).astype(int), len(rec.labels) - 1)
        lab = np.asarray(rec.labels)[idx].astype(np.int64)
    else:
        lab = np.full(n, -1, dtype=np.int64)
    return filt, raw, lab


def recording_to_windows(rec: Recording, spec: InputSpec, skip_sec: float = 0.0,
                         prepared: tuple | None = None) -> WindowSet:
    """윈도우 분할. ``skip_sec`` 이전에 시작하는 윈도우(기준 구간과 겹치는 윈도우)는 만들지 않는다."""
    filt, raw, lab = prepared if prepared is not None else prepare_recording(rec, spec)
    n = filt.shape[1]
    W, S = spec.window_len, spec.stride_len
    first = int(np.ceil(skip_sec * spec.fs))
    xs, ys, ts, qs = [], [], [], []
    for start in range(first, n - W + 1, S):
        y = -1
        if rec.labels is not None:
            seg = lab[start:start + W]
            valid = seg[seg >= 0]
            if len(valid) == 0:
                continue
            vals, cnt = np.unique(valid, return_counts=True)
            k = int(np.argmax(cnt))
            if cnt[k] / W < spec.label.min_label_purity:
                continue   # 라벨이 섞인 경계 윈도우는 버린다
            y = int(vals[k])
        xs.append(filt[:, start:start + W])
        ys.append(y)
        ts.append(rec.t0 + start / spec.fs)
        qs.append(window_quality(raw[:, start:start + W], spec, spec.channels))
    C = len(spec.channels)
    return WindowSet(
        x_seq=np.stack(xs) if xs else np.zeros((0, C, W), np.float32),
        y=np.asarray(ys, dtype=np.int64),
        groups=np.asarray([rec.group] * len(ys), dtype=object),
        t_start=np.asarray(ts, dtype=np.float64),
        quality=np.asarray(qs, dtype=bool),
        channels=spec.channels,
    )


def build_windows(recs: list[Recording], spec: InputSpec) -> WindowSet:
    """Baseline 없이 윈도우만 만든다 (기준 구간 제외 없음). 개인화 학습은 baseline.build_dataset 사용."""
    return WindowSet.concat([recording_to_windows(r, spec) for r in recs], spec.channels, spec.window_len)
