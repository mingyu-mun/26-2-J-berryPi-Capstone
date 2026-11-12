"""원신호 → RR 시계열 캐시. 원 ECG 는 이 단계에서 한 번만 읽는다.

1. 공통 전처리 (학생과 동일: 노치 → anti-aliasing → 100 Hz 리샘플 → 0.5–40 Hz). meta.mains_hz 가 있으면
   노치 주파수를 그 값으로 바꾼다 (유럽 데이터 50 Hz).
2. 기존 detect_ecg_peaks 로 R-peak 검출 → 대역통과 신호의 |x| 에서 ±40 ms 국소 최대로 맞춘 뒤
   포물선 보간으로 sub-sample 시각을 구한다 (100 Hz 의 10 ms 해상도 보완).
3. RR = 연속 peak 간격. 0.3–2.0 s 밖이거나 국소 중앙값(앞뒤 5개) 대비 rr_jump 넘게 벗어나면 이상 RR.
4. 10초 블록마다 필터 전 신호 품질(평탄·포화)을 저장 → 30초 창 품질 = 블록 3개 모두 통과.
5. 라벨은 1 Hz 로 저장 (창 라벨은 windows 단계에서).
"""
from __future__ import annotations

import dataclasses
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from scipy.ndimage import median_filter
from scipy.signal import butter, sosfiltfilt

from ..data import Recording
from ..features import detect_ecg_peaks, detect_ppg_peaks_adaptive
from ..preprocessing import prepare_channel, window_quality
from ..spec import InputSpec
from .spec import TeacherSpec

_MIN_RR, _MAX_RR = 0.3, 2.0


@dataclass
class RRSeries:
    dataset_id: str
    subject_id: str | None
    session_id: str
    group: str
    t0: float
    duration: float                 # 세션 길이 [s]
    beat_t: np.ndarray              # (B,) R-peak 시각 [s, 세션 시작 기준]
    rr: np.ndarray                  # (B-1,) beat_t[i+1]-beat_t[i] [s], 시각은 beat_t[1:]
    abnormal: np.ndarray            # (B-1,) bool
    block_ok: np.ndarray            # (n_blocks,) bool, stride_sec 블록별 신호 품질
    labels_1hz: np.ndarray          # (ceil(duration),) int, -1 = 미라벨
    is_rest: bool = False
    meta: dict = field(default_factory=dict)

    @property
    def rr_t(self) -> np.ndarray:
        return self.beat_t[1:]

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        info = {k: getattr(self, k) for k in ("dataset_id", "subject_id", "session_id", "group", "t0",
                                               "duration", "is_rest", "meta")}
        np.savez_compressed(path, beat_t=self.beat_t, rr=self.rr, abnormal=self.abnormal,
                            block_ok=self.block_ok, labels_1hz=self.labels_1hz,
                            info=np.array(json.dumps(info, ensure_ascii=False)))

    @classmethod
    def load(cls, path: str | Path) -> "RRSeries":
        z = np.load(path, allow_pickle=False)
        info = json.loads(str(z["info"]))
        return cls(beat_t=z["beat_t"], rr=z["rr"], abnormal=z["abnormal"].astype(bool),
                   block_ok=z["block_ok"].astype(bool), labels_1hz=z["labels_1hz"].astype(np.int64), **info)


def _input_spec(spec: TeacherSpec, rec: Recording) -> InputSpec:
    ins = InputSpec(fs=spec.fs, window_sec=spec.window_sec, stride_sec=spec.stride_sec)
    mains = rec.meta.get("mains_hz")
    if mains:
        ins.preprocess = dataclasses.replace(ins.preprocess, notch_hz=float(mains))
    return ins


def refine_peaks(sig: np.ndarray, peaks: np.ndarray, fs: float, search_sec: float = 0.04) -> np.ndarray:
    """peak 을 |sig| 국소 최대로 옮긴 뒤 포물선 보간한 sub-sample 위치(샘플 단위, float)."""
    a = np.abs(sig)
    w = max(1, int(round(search_sec * fs)))
    out = np.empty(len(peaks), dtype=np.float64)
    n = len(a)
    for i, p in enumerate(peaks):
        lo, hi = max(1, p - w), min(n - 1, p + w + 1)
        k = lo + int(np.argmax(a[lo:hi])) if hi > lo else int(p)
        k = min(max(k, 1), n - 2)
        y0, y1, y2 = a[k - 1], a[k], a[k + 1]
        den = y0 - 2 * y1 + y2
        off = 0.5 * (y0 - y2) / den if abs(den) > 1e-12 else 0.0
        out[i] = k + float(np.clip(off, -0.5, 0.5))
    return np.unique(out)


def flag_abnormal(rr: np.ndarray, jump: float) -> np.ndarray:
    """범위 밖이거나 국소 중앙값 대비 jump 넘게 벗어난 RR."""
    bad = (rr < _MIN_RR) | (rr > _MAX_RR)
    if len(rr) >= 3:
        med = median_filter(rr, size=min(11, len(rr) | 1), mode="nearest")
        bad |= np.abs(rr - med) > jump * med
    return bad


def extract_rr(rec: Recording, spec: TeacherSpec, channel: str = "ecg") -> RRSeries:
    """channel="ppg": 맥박 간격(PRV). 적응 최소 간격으로 찾은 수축기 최고점을 ±80 ms 안에서 포물선 보간."""
    ins = _input_spec(spec, rec)
    filt, raw = prepare_channel(rec.signals[channel], rec.fs[channel], channel, ins, rec.prefiltered)
    fs = spec.fs
    if channel == "ppg":
        peaks = detect_ppg_peaks_adaptive(filt, fs)
        sos = butter(3, [0.5, min(8.0, 0.45 * fs)], btype="band", fs=fs, output="sos")
        pw = sosfiltfilt(sos, filt)
        pw = pw - pw.min()                       # refine_peaks 는 |x| 최대를 찾으므로 양수로
        beat_t = refine_peaks(pw, peaks, fs, search_sec=0.08) / fs if len(peaks) else np.zeros(0)
    else:
        peaks = detect_ecg_peaks(filt, fs)
        # 정밀 위치는 QRS 대역(5–25 Hz) 신호로 잡는다 (기저선 영향 제거, 극성 무관)
        sos = butter(3, [5.0, min(25.0, 0.45 * fs)], btype="band", fs=fs, output="sos")
        qrs = sosfiltfilt(sos, filt)
        beat_t = refine_peaks(qrs, peaks, fs) / fs if len(peaks) else np.zeros(0)
    rr = np.diff(beat_t)
    abnormal = flag_abnormal(rr, spec.rr_jump) if len(rr) else np.zeros(0, bool)

    n = len(filt)
    duration = n / fs
    S = int(round(spec.stride_sec * fs))
    n_blocks = n // S
    block_ok = np.array([window_quality(raw[None, b * S:(b + 1) * S], ins, (channel,)) for b in range(n_blocks)],
                        dtype=bool)

    n_sec = int(np.ceil(duration))
    if rec.labels is not None and len(rec.labels):
        lab_fs = rec.label_fs or fs
        idx = np.minimum((np.arange(n_sec) * lab_fs + 0.5 * lab_fs).astype(int), len(rec.labels) - 1)
        labels = np.asarray(rec.labels)[idx].astype(np.int64)
    else:
        labels = np.full(n_sec, -1, np.int64)
    meta = {k: v for k, v in rec.meta.items() if k != "path"}
    meta["rr_channel"] = channel
    return RRSeries(rec.dataset_id, rec.subject_id, rec.session_id, rec.group, rec.t0, duration,
                    beat_t, rr, abnormal, block_ok, labels, rec.is_rest, meta)


def cache_path(root: str | Path, rec_or_rr) -> Path:
    sub = rec_or_rr.subject_id if rec_or_rr.subject_id is not None else "anon"
    return Path(root) / rec_or_rr.dataset_id / str(sub) / f"{rec_or_rr.session_id}.npz"


def load_rr_cache(root: str | Path) -> list[RRSeries]:
    return [RRSeries.load(p) for p in sorted(Path(root).rglob("*.npz"))]
