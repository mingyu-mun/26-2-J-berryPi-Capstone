"""시연(Pi) 추론: 학습 산출물 묶음 저장/로드, calibration, Replay 추론.

시연 흐름
1. (사전) 학습 PC 에서 save_bundle() → pipeline.json + model/ (+ model.onnx)
2. 시연자 30초 calibration (주행 아님) → baseline.calibrate() → profile.json
3. CAN 시나리오 Replay 와 함께 실시간 ECG·PPG 를 Recording 으로 받아 predict_recording()
   (CAN 은 모델 입력이 아니다. 11/5 결정)
4. 윈도우마다: 판정(-1 여부) → 분류기 P(졸음) → 상태 머신 → 정상 / 주의 / 졸음 / 측정 불가
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .baseline import Profile
from .data import Recording, WindowSet, recording_to_windows
from .models import REGISTRY, CandidateModel, model_input
from .features import extract_features
from .normalization import InputPipeline
from .state_machine import UNAVAILABLE, DrowsinessStateMachine, judge_windows


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
    t_start: float          # 윈도우 시작 [s]
    t_end: float            # 판정 시각 = 윈도우 끝 [s]
    state: int              # 분류기 출력 0 정상 / 1 졸음, 판정 불가 윈도우는 -1
    state_name: str
    proba: list[float]      # 분류기 확률 (판정 불가 윈도우는 빈 리스트)
    valid: bool             # 윈도우 판정 통과 여부
    invalid_reason: str     # 판정 불가 사유 (통과면 빈 문자열)
    level: int              # 상태 머신 단계 0 정상 / 1 주의 / 2 졸음 / -1 측정 불가
    level_name: str
    score: float | None     # 누적 점수 s_t
    quality_ok: bool
    baseline_valid: bool


class StatePredictor:
    """시연 추론기. predict_windows 를 시간 순서대로 여러 번 불러도 상태 머신이 이어진다.
    새 시연자·새 Replay 를 시작할 때는 reset() (set_profile 도 자동으로 reset)."""

    def __init__(self, bundle_dir: str | Path):
        bundle_dir = Path(bundle_dir)
        self.pipe = InputPipeline.load(bundle_dir / "pipeline.json")
        name = json.loads((bundle_dir / "bundle.json").read_text())["model"]
        self.model = REGISTRY[name].load(bundle_dir / "model")
        self.spec = self.pipe.spec
        self.profile: Profile | None = None
        self.machine = DrowsinessStateMachine(self.spec.state_machine, self.spec.stride_sec)

    def reset(self) -> None:
        self.machine.reset()

    def set_profile(self, profile: Profile | None) -> None:
        """시연자 프로필 지정. None 또는 무효면 baseline_valid=0 으로 추론 (미등록 사용자)."""
        self.profile = profile
        self.reset()

    def predict_windows(self, ws: WindowSet, t_offset: float = 0.0) -> list[WindowResult]:
        if len(ws) == 0:
            return []
        groups = np.asarray(["user"] * len(ws), dtype=object)
        ws = WindowSet(ws.x_seq, ws.y, groups, ws.t_start, ws.quality, ws.channels)
        profiles = {"user": self.profile} if self.profile is not None else {}
        base = extract_features(ws, self.spec)
        reasons = judge_windows(ws.x_seq, base, ws.quality, ws.channels, self.spec)
        inp = self.pipe.transform(ws, base, profiles=profiles)
        proba = self.model.predict_proba(model_input(inp, self.model.input_kind))
        names = self.spec.label.names
        bvalid = bool(self.profile and self.profile.valid and self.spec.use_personal)
        out = []
        for t, p, q, why in zip(ws.t_start, proba, ws.quality, reasons):
            t_end = float(t) + t_offset + self.spec.window_sec
            ok = why == ""
            st = self.machine.step(t_end, float(p[1]) if ok else None)
            k = int(p.argmax()) if ok else UNAVAILABLE
            out.append(WindowResult(
                float(t) + t_offset, t_end, k, names[k] if ok else "판정 불가",
                [round(float(v), 4) for v in p] if ok else [], ok, why,
                st.level, st.level_name, None if st.score is None else round(st.score, 4),
                bool(q), bvalid))
        return out

    def predict_recording(self, rec: Recording) -> list[WindowResult]:
        """Replay 또는 버퍼에 쌓인 실시간 신호 → 윈도우마다 상태. 라벨은 쓰지 않는다.
        상태 머신을 처음부터 다시 시작한다."""
        self.reset()
        rec = Recording(rec.subject_id, rec.session_id, rec.signals, rec.fs, None, None, rec.t0,
                        prefiltered=rec.prefiltered)
        return self.predict_windows(recording_to_windows(rec, self.spec))
