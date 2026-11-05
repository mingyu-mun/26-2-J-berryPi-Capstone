"""후보 모델 5종 비교 실행.

    python scripts/run_benchmark.py --synthetic            # 합성 데이터로 동작 확인
    python scripts/run_benchmark.py --synthetic --binary   # 이진 백업 라벨
    python scripts/run_benchmark.py --models mlp xgboost   # 일부 모델만

공개 데이터셋 로더가 생기면 load_recordings() 에서 Recording 리스트를 만들어 넘기면 된다.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from driver_state.data import build_windows  # noqa: E402
from driver_state.evaluate import run_benchmark, summary_markdown  # noqa: E402
from driver_state.spec import load_config, spec_from_config  # noqa: E402
from driver_state.synthetic import make_dataset  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "configs/default.yaml"))
    ap.add_argument("--synthetic", action="store_true", help="합성 데이터 사용")
    ap.add_argument("--subjects", type=int, default=6)
    ap.add_argument("--seconds-per-class", type=float, default=180.0)
    ap.add_argument("--models", nargs="*", default=None)
    ap.add_argument("--binary", action="store_true", help="이진 백업 라벨로 평가")
    ap.add_argument("--use-can", action="store_true", help="CAN 채널·특징 포함 (early fusion 실험)")
    ap.add_argument("--out", default=str(ROOT / "outputs/benchmark"))
    args = ap.parse_args()

    cfg = load_config(args.config)
    spec = spec_from_config(cfg)
    if args.use_can:
        spec.use_can = True
    if not args.synthetic:
        sys.exit("아직 공개 데이터셋 로더가 없습니다. --synthetic 으로 실행하세요.")
    recs = make_dataset(args.subjects, args.seconds_per_class, seed=cfg["evaluation"]["seed"])
    ws = build_windows(recs, spec)
    print(f"windows={len(ws)} shape={ws.x_seq.shape} channels={ws.channels}")

    model_cfgs = {k: v for k, v in cfg["models"].items() if not args.models or k in args.models}
    res = run_benchmark(ws, spec, model_cfgs, n_folds=cfg["evaluation"]["n_folds"],
                        seed=cfg["evaluation"]["seed"], label_mode="binary" if args.binary else "multi")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "results.json").write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
    md = summary_markdown(res)
    (out / "summary.md").write_text(md + "\n", encoding="utf-8")
    print("\n" + md)


if __name__ == "__main__":
    main()
