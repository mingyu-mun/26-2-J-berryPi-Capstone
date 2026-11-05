"""시연(Pi) 추론: 학습 산출물 묶음 저장/로드, calibration, Replay 추론.

시연 흐름
1. (사전) 학습 PC 에서 save_bundle() → pipeline.json + model/ (+ model.onnx)
2. 시연자 30초 calibration (주행 아님) → baseline.calibrate() → profile.json
3. CAN 시나리오 Replay 와 함께 실시간 ECG·PPG 를 Recording 으로 받아 predict_recording()
   (CAN 은 모델 입력이 아니다. 11/5 결정)
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .baseline import Profile
from .data import Recording, WindowSet, recording_to_windows
from .models import REGISTRY, CandidateModel, model_input
from .normalization import InputPipeline


def save_bundle(path: str | Path, pipe: InputPipeline, model: CandidateModel, onnx: bool = False) -> None:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    pipe.save(path / "pipeline.json")
    model.save(path / "model")
    (path / "bundle.json").write_text(json.dumps({"model": model.name}, indent=2))
    if onnx and hasattr(model, "export_onnx"):
        model.export_onnx(path / "model.onnx")


@dataclass
class WindowResult:
    t_start: float
    state: int
    state_name: str
    proba: list[float]
    quality_ok: bool
    baseline_valid: bool


class StatePredictor:
    def __init__(self, bundle_dir: str | Path):
        bundle_dir = Path(bundle_dir)
        self.pipe = InputPipeline.load(bundle_dir / "pipeline.json")
        name = json.loads((bundle_dir / "bundle.json").read_text())["model"]
        self.model = REGISTRY[name].load(bundle_dir / "model")
        self.spec = self.pipe.spec
        self.profile: Profile | None = None

    def set_profile(self, profile: Profile | None) -> None:
        """시연자 프로필 지정. None 또는 무효면 baseline_valid=0 으로 추론 (미등록 사용자)."""
        self.profile = profile

    def predict_windows(self, ws: WindowSet) -> list[WindowResult]:
        if len(ws) == 0:
            return []
        groups = np.asarray(["user"] * len(ws), dtype=object)
        ws = WindowSet(ws.x_seq, ws.y, groups, ws.t_start, ws.quality, ws.channels)
        profiles = {"user": self.profile} if self.profile is not None else {}
        inp = self.pipe.transform(ws, profiles=profiles)
        proba = self.model.predict_proba(model_input(inp, self.model.input_kind))
        names = self.spec.label.names
        valid = bool(self.profile and self.profile.valid and self.spec.use_personal)
        return [WindowResult(float(t), int(p.argmax()), names[int(p.argmax())] if int(p.argmax()) < len(names) else "",
                             [round(float(v), 4) for v in p], bool(q), valid)
                for t, p, q in zip(ws.t_start, proba, ws.quality)]

    def predict_recording(self, rec: Recording) -> list[WindowResult]:
        """Replay 또는 버퍼에 쌓인 실시간 신호 → 윈도우마다 상태. 라벨은 쓰지 않는다."""
        rec = Recording(rec.subject_id, rec.session_id, rec.signals, rec.fs, None, None, rec.t0,
                        prefiltered=rec.prefiltered)
        return self.predict_windows(recording_to_windows(rec, self.spec))
