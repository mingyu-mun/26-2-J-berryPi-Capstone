"""시연 흐름 리허설 (합성 데이터).

    python scripts/demo_pipeline.py                 # MLP
    python scripts/demo_pipeline.py --model xgboost --onnx

1. 학습: 합성 참가자로 개인화 입력 모델 학습 → outputs/demo/bundle/
2. calibration: 처음 보는 '시연자'의 30초 휴식 신호 → outputs/demo/profile.json
3. Replay: 같은 시연자의 시나리오 신호 → 10초마다 윈도우 판정 → P(졸음) → 상태 머신(정상/주의/졸음)
   (CAN 시나리오는 화면 연출용, 입력 아님). 중간에 PPG 접촉 불량 구간을 넣어 -1 처리도 보여준다.
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
from driver_state.state_machine import timeline_metrics  # noqa: E402
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

    # 3. 시나리오 Replay 추론: 정상 60초 → 졸음 120초 → 정상 120초, 140~148초 PPG 접촉 불량
    scenario = [(0, 60.0), (1, 120.0), (0, 120.0)]
    replay = make_recording("presenter", scenario, 85.0, 1.0, rng, session_id="replay")
    fs_ppg = replay.fs["ppg"]
    replay.signals["ppg"][int(140 * fs_ppg):int(148 * fs_ppg)] = 0.0
    predictor = StatePredictor(out / "bundle")
    predictor.set_profile(profile)
    truth = np.concatenate([np.full(int(s), k) for k, s in scenario])
    res = predictor.predict_recording(replay)
    print("[3] Replay 추론 (윈도우 30초, 10초마다 갱신, 시각 = 윈도우 끝)")
    print("    시각(s) | 정답 | 분류기     | P(졸음) | 누적 s | 최종 단계  | 판정 불가 사유")
    for r in res:
        tt = truth[min(int(r.t_end) - 1, len(truth) - 1)]
        p = f"{r.proba[1]:.2f}" if r.valid else "  - "
        sc = f"{r.score:.2f}" if r.score is not None else "  - "
        print(f"    {r.t_end:7.0f} | {tt:4d} | {r.state:2d} {r.state_name:<7} | {p:>7} | {sc:>6} | "
              f"{r.level:2d} {r.level_name:<6} | {r.invalid_reason}")
    m = timeline_metrics(np.array([r.t_end for r in res]), np.array([r.level for r in res]),
                         np.array([truth[min(int(r.t_end) - 1, len(truth) - 1)] for r in res]))
    print(f"[4] 타임라인 평가: 졸음 감지 지연 {m['detection_delay_sec']} s, 놓침 {m['missed']}회, "
          f"오경보 {m['false_alarms']}회, 측정 불가 비율 {m['unavailable_ratio']:.2f}")

if __name__ == "__main__":
    main()
