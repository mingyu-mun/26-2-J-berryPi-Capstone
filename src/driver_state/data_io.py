"""데이터 팀 전달 폴더 → Recording (docs/data-handoff.md 2장).

data/<dataset_id>/<subject_id>/<session_id>/{meta.json, ecg.csv, ppg.csv, labels.csv?}
CSV 대신 data.npz (배열 ecg, ppg, labels?) 도 읽는다.
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import numpy as np

from .data import Recording


def _read_csv(path: Path) -> np.ndarray:
    return np.loadtxt(path, delimiter=",", skiprows=1, ndmin=1, dtype=np.float64)


def _t0(v) -> float:
    if isinstance(v, (int, float)):
        return float(v)
    return datetime.fromisoformat(str(v)).timestamp()


def load_session(session_dir: str | Path) -> Recording:
    d = Path(session_dir)
    meta = json.loads((d / "meta.json").read_text(encoding="utf-8"))
    if (d / "data.npz").exists():
        z = np.load(d / "data.npz")
        signals = {k: z[k] for k in ("ecg", "ppg")}
        labels = z["labels"] if "labels" in z else None
    else:
        signals = {k: _read_csv(d / f"{k}.csv") for k in ("ecg", "ppg")}
        labels = _read_csv(d / "labels.csv").astype(np.int64) if (d / "labels.csv").exists() else None
    for k, x in signals.items():
        if not np.isfinite(x).all():
            raise ValueError(f"{d}: {k} 에 NaN/inf 가 있습니다. 전달 규약 4장대로 처리해 주세요.")
    return Recording(
        subject_id=meta.get("subject_id"), session_id=meta["session_id"], signals=signals,
        fs={k: float(v) for k, v in meta["fs"].items()}, labels=labels,
        label_fs=float(meta["label_fs"]) if meta.get("label_fs") else None,
        t0=_t0(meta.get("t0", 0.0)), is_rest=bool(meta.get("is_rest", False)),
        prefiltered=bool(meta.get("prefiltered", False)), dataset_id=meta.get("dataset_id", "default"),
    )


def load_tree(root: str | Path) -> list[Recording]:
    """root 아래 meta.json 이 있는 모든 세션 폴더를 읽는다."""
    return [load_session(p.parent) for p in sorted(Path(root).rglob("meta.json"))]
