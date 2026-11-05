"""시연 흐름 리허설 (합성 데이터).

    python scripts/demo_pipeline.py                 # MLP
    python scripts/demo_pipeline.py --model xgboost --onnx

1. 학습: 합성 참가자로 개인화 입력 모델 학습 → outputs/demo/bundle/
2. calibration: 처음 보는 '시연자'의 30초 휴식 신호 → outputs/demo/profile.json
3. Replay: 같은 시연자의 시나리오 신호 → 10초마다 상태 추론 (CAN 시나리오는 화면 연출용, 입력 아님)
실제 시연에서는 2·3의 신호가 센서에서 들어온다. 합성 데이터라 출력은 흐름 확인용이다.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from driver_state.baseline import build_dataset, calibrate  # noqa: E402
from driver_state.inference import StatePredictor, save_bundle  # noqa: E402
from driver_state.models import build_model, model_input  # noqa: E402
from driver_state.normalization import InputPipeline  # noqa: E402
from driver_state.spec import load_config, spec_from_config  # noqa: E402
from driver_state.synthetic import make_dataset, make_recording  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "configs/default.yaml"))
    ap.add_argument("--model", default="mlp")
    ap.add_argument("--onnx", action="store_true")
    ap.add_argument("--out", default=str(ROOT / "outputs/demo"))
    args = ap.parse_args()
    cfg = load_config(args.config)
    spec = spec_from_config(cfg)
    out = Path(args.out)

    # 1. 학습 (전체 합성 참가자)
    ws, profiles = build_dataset(make_dataset(6, 180.0, spec.baseline.calibration_sec), spec)
    keep = np.where((ws.y >= 0) & ws.quality)[0]
    ws = ws.subset(keep)
    pipe = InputPipeline(spec).fit(ws, None, profiles)
    inp = pipe.transform(ws, None, profiles, invalid_rate=spec.baseline.train_invalid_rate)
    model = build_model(args.model, spec.label.n_classes, **(cfg["models"].get(args.model) or {}))
    model.fit(model_input(inp, model.input_kind), ws.y)
    save_bundle(out / "bundle", pipe, model, onnx=args.onnx)
    print(f"[1] 학습 완료: {args.model}, 윈도우 {len(ws)}개, 특징 {len(spec.feature_names)}개 → {out / 'bundle'}")

    # 2. 시연자 calibration (처음 보는 사람, 평소 심박 85 bpm)
    rng = np.random.default_rng(123)
    calib = make_recording("presenter", [(None, spec.baseline.calibration_sec)], 85.0, 1.0, rng)
    profile = calibrate(calib, spec)
    profile.save(out / "profile.json")
    print(f"[2] calibration: valid={profile.valid} m={ {k: round(v, 1) for k, v in profile.m.items()} } "
          f"{profile.reason} → {out / 'profile.json'}")

    # 3. 시나리오 Replay 추론
    scenario = [(0, 60.0), (3, 60.0), (1, 60.0)]
    replay = make_recording("presenter", scenario, 85.0, 1.0, rng, session_id="replay")
    predictor = StatePredictor(out / "bundle")
    predictor.set_profile(profile)
    truth = np.concatenate([np.full(int(s), k) for k, s in scenario])
    print("[3] Replay 추론 (윈도우 30초, 10초마다 갱신)")
    print("    시작(s) | 예측        | 시나리오 구간 | 품질 | 기준 사용")
    for r in predictor.predict_recording(replay):
        mid = int(min(r.t_start + spec.window_sec / 2, len(truth) - 1))
        print(f"    {r.t_start:7.0f} | {r.state} {r.state_name:<8} | {truth[mid]:13d} | "
              f"{'OK' if r.quality_ok else '불량':<4} | {r.baseline_valid}")


if __name__ == "__main__":
    main()
