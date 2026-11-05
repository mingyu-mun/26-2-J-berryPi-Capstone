"""데이터 팀 전달 폴더 → Recording (docs/data-handoff.md 2장, v0.3 선택 필드 포함).

data/<dataset_id>/<subject_id>/<session_id>/{meta.json, ecg.csv, ppg.csv, labels.csv?}
CSV 대신 data.npz (배열 ecg, ppg, labels?) 도 읽는다.

v0.3 선택 필드 (없으면 v0.2 와 똑같이 동작)
- channels: 실제 있는 채널 목록. 없으면 ["ecg", "ppg"]. 교사 데이터처럼 ECG 만 있으면 ["ecg"].
- drowsiness_inducer, baseline_range, baseline_per_session, mains_hz 등은 Recording.meta 로 넘긴다.
- labels_orig.csv 는 로더가 읽지 않는다 (원 라벨 보존용).
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import numpy as np

from .data import Recording

DEFAULT_CHANNELS = ("ecg", "ppg")
_CORE = {"dataset_id", "subject_id", "session_id", "t0", "fs", "label_fs", "is_rest", "prefiltered"}


def _read_csv(path: Path) -> np.ndarray:
    return np.loadtxt(path, delimiter=",", skiprows=1, ndmin=1, dtype=np.float64)


def _t0(v) -> float:
    if isinstance(v, (int, float)):
        return float(v)
    return datetime.fromisoformat(str(v)).timestamp()


def load_session(session_dir: str | Path, channels: tuple[str, ...] | None = None) -> Recording:
    """세션 폴더 하나를 읽는다. channels 를 주면 그 채널만 읽는다 (없으면 meta 의 channels)."""
    d = Path(session_dir)
    meta = json.loads((d / "meta.json").read_text(encoding="utf-8"))
    present = tuple(meta.get("channels", DEFAULT_CHANNELS))
    want = tuple(channels) if channels is not None else present
    missing = [c for c in want if c not in present]
    if missing:
        raise KeyError(f"{d}: 채널 {missing} 이 없습니다 (meta.channels={list(present)}).")
    if (d / "data.npz").exists():
        z = np.load(d / "data.npz")
        signals = {k: np.asarray(z[k], dtype=np.float64) for k in want}
        labels = np.asarray(z["labels"]).astype(np.int64) if "labels" in z else None
    else:
        signals = {k: _read_csv(d / f"{k}.csv") for k in want}
        labels = _read_csv(d / "labels.csv").astype(np.int64) if (d / "labels.csv").exists() else None
    for k, x in signals.items():
        if not np.isfinite(x).all():
            raise ValueError(f"{d}: {k} 에 NaN/inf 가 있습니다. 전달 규약 4장대로 처리해 주세요.")
    return Recording(
        subject_id=meta.get("subject_id"), session_id=meta["session_id"], signals=signals,
        fs={k: float(meta["fs"][k]) for k in want}, labels=labels,
        label_fs=float(meta["label_fs"]) if meta.get("label_fs") else None,
        t0=_t0(meta.get("t0", 0.0)), is_rest=bool(meta.get("is_rest", False)),
        prefiltered=bool(meta.get("prefiltered", False)), dataset_id=meta.get("dataset_id", "default"),
        meta={k: v for k, v in meta.items() if k not in _CORE} | {"path": str(d)},
    )


def iter_session_dirs(root: str | Path) -> list[Path]:
    return [p.parent for p in sorted(Path(root).rglob("meta.json"))]


def load_tree(root: str | Path, channels: tuple[str, ...] | None = None) -> list[Recording]:
    """root 아래 meta.json 이 있는 모든 세션 폴더를 읽는다."""
    return [load_session(p, channels) for p in iter_session_dirs(root)]


def write_session(session_dir: str | Path, meta: dict, signals: dict[str, np.ndarray],
                  labels: np.ndarray | None = None, labels_orig: "object | None" = None) -> Path:
    """규약 폴더 하나를 쓴다 (어댑터·합성 데이터 공용). 신호는 data.npz(float32) 로 저장.

    labels_orig: pandas.DataFrame 이면 labels_orig.csv 로 저장 (원 라벨 + 측정 시각).
    """
    d = Path(session_dir)
    d.mkdir(parents=True, exist_ok=True)
    meta = dict(meta)
    meta.setdefault("channels", list(signals))
    arrays = {k: np.asarray(v, dtype=np.float32) for k, v in signals.items()}
    if labels is not None:
        arrays["labels"] = np.asarray(labels, dtype=np.int8)
    np.savez_compressed(d / "data.npz", **arrays)
    (d / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    if labels_orig is not None:
        labels_orig.to_csv(d / "labels_orig.csv", index=False)
    return d
