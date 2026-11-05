"""Recording → 리샘플 → 윈도우 분할.

데이터/전처리 담당이 넘겨줄 형식(Recording)과, 모델이 받는 형식(WindowSet)의 경계.
공개 데이터셋 로더나 실센서·Replay 입력은 모두 Recording 을 만들어 여기로 넘기면 된다.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.interpolate import CubicSpline

from .spec import BIO_CHANNELS, InputSpec


@dataclass
class Recording:
    """한 사람의 한 세션.

    signals: 채널 이름 → 1차원 신호. 채널 이름은 spec.BIO_CHANNELS / CAN_CHANNELS.
    fs:      채널 이름 → 그 채널의 원 샘플링 주기 [Hz]. 채널마다 달라도 된다.
    labels:  샘플 단위 라벨(정수, 미라벨 구간은 -1). ``label_fs`` 주기로 샘플링.
    t0:      세션 시작 시각 [s]. 모든 채널은 t0 에서 시작한다고 가정 (동기화는 전처리 단계 책임).
    """

    subject_id: str
    session_id: str
    signals: dict[str, np.ndarray]
    fs: dict[str, float]
    labels: np.ndarray | None = None
    label_fs: float | None = None
    t0: float = 0.0


@dataclass
class WindowSet:
    x_seq: np.ndarray            # (N, C, T) float32, 리샘플된 원파형 (정규화 전)
    y: np.ndarray                # (N,) int64, 라벨 없으면 -1
    groups: np.ndarray           # (N,) subject_id (LOSO 분할 기준)
    t_start: np.ndarray          # (N,) 윈도우 시작 시각 [s]
    channels: tuple[str, ...]
    meta: dict = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.y)

    def subset(self, idx: np.ndarray) -> "WindowSet":
        return WindowSet(self.x_seq[idx], self.y[idx], self.groups[idx], self.t_start[idx],
                         self.channels, self.meta)


def resample(x: np.ndarray, fs_in: float, fs_out: float, kind: str = "cubic") -> np.ndarray:
    """채널 하나를 fs_out 으로 리샘플. 생체신호는 cubic spline, CAN 은 선형 보간."""
    x = np.asarray(x, dtype=np.float64)
    if fs_in == fs_out:
        return x
    n_out = int(np.floor(len(x) * fs_out / fs_in))
    t_in = np.arange(len(x)) / fs_in
    t_out = np.arange(n_out) / fs_out
    if kind == "cubic":
        return CubicSpline(t_in, x)(t_out)
    return np.interp(t_out, t_in, x)


def recording_to_windows(rec: Recording, spec: InputSpec) -> WindowSet:
    chans = spec.channels
    missing = [c for c in chans if c not in rec.signals]
    if missing:
        raise KeyError(f"{rec.subject_id}/{rec.session_id}: 채널 누락 {missing}")

    res = [resample(rec.signals[c], rec.fs[c], spec.fs,
                    "cubic" if c in BIO_CHANNELS else "linear") for c in chans]
    n = min(len(r) for r in res)
    mat = np.stack([r[:n] for r in res]).astype(np.float32)   # (C, n)

    if rec.labels is not None:
        lab_fs = rec.label_fs or spec.fs
        idx = np.minimum((np.arange(n) * lab_fs / spec.fs).astype(int), len(rec.labels) - 1)
        lab = np.asarray(rec.labels)[idx].astype(np.int64)
    else:
        lab = np.full(n, -1, dtype=np.int64)

    W, S = spec.window_len, spec.stride_len
    xs, ys, ts = [], [], []
    for start in range(0, n - W + 1, S):
        seg = lab[start:start + W]
        if rec.labels is not None:
            valid = seg[seg >= 0]
            if len(valid) == 0:
                continue
            vals, cnt = np.unique(valid, return_counts=True)
            k = int(np.argmax(cnt))
            if cnt[k] / W < spec.label.min_label_purity:
                continue   # 라벨이 섞인 경계 윈도우는 버린다
            y = int(vals[k])
        else:
            y = -1
        xs.append(mat[:, start:start + W])
        ys.append(y)
        ts.append(rec.t0 + start / spec.fs)

    C = len(chans)
    return WindowSet(
        x_seq=np.stack(xs) if xs else np.zeros((0, C, W), np.float32),
        y=np.asarray(ys, dtype=np.int64),
        groups=np.asarray([rec.subject_id] * len(ys), dtype=object),
        t_start=np.asarray(ts, dtype=np.float64),
        channels=chans,
    )


def build_windows(recs: list[Recording], spec: InputSpec) -> WindowSet:
    parts = [recording_to_windows(r, spec) for r in recs]
    return WindowSet(
        x_seq=np.concatenate([p.x_seq for p in parts]),
        y=np.concatenate([p.y for p in parts]),
        groups=np.concatenate([p.groups for p in parts]),
        t_start=np.concatenate([p.t_start for p in parts]),
        channels=spec.channels,
    )
