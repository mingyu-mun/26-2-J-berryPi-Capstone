"""후보 모델 5종 비교 실행 (LOSO).

    python scripts/run_benchmark.py --synthetic                     # 개인화 입력으로 비교
    python scripts/run_benchmark.py --synthetic --compare-personal  # 개인화 없음 vs 있음
    python scripts/run_benchmark.py --synthetic --models mlp xgboost

공개 데이터셋 로더가 생기면 Recording 리스트를 만들어 build_dataset() 에 넘기면 된다.
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from driver_state.baseline import build_dataset  # noqa: E402
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
    ap.add_argument("--no-personal", action="store_true", help="개인화 특징 없이 평가")
    ap.add_argument("--compare-personal", action="store_true", help="개인화 없음/있음 둘 다 평가")
    ap.add_argument("--out", default=str(ROOT / "outputs/benchmark"))
    args = ap.parse_args()

    cfg = load_config(args.config)
    spec = spec_from_config(cfg)
    if not args.synthetic:
        sys.exit("아직 공개 데이터셋 로더가 없습니다. --synthetic 으로 실행하세요.")
    seed = cfg["evaluation"]["seed"]
    recs = make_dataset(args.subjects, args.seconds_per_class, spec.baseline.calibration_sec, seed=seed)
    ws, profiles = build_dataset(recs, spec)
    print(f"windows={len(ws)} shape={ws.x_seq.shape} channels={ws.channels} "
          f"quality_ok={int(ws.quality.sum())}/{len(ws)}")
    for g, p in profiles.items():
        print(f"  profile {g}: valid={p.valid} m={ {k: round(v, 1) for k, v in p.m.items()} } {p.reason}")

    modes = [False, True] if args.compare_personal else [not args.no_personal]
    model_cfgs = {k: v for k, v in cfg["models"].items() if not args.models or k in args.models}
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    md_all = []
    for use_personal in modes:
        s = copy.deepcopy(spec)
        s.use_personal = use_personal
        tag = "personal" if use_personal else "no_personal"
        print(f"\n=== {tag} (features {len(s.feature_names)}개) ===")
        res = run_benchmark(ws, s, model_cfgs, profiles, n_folds=cfg["evaluation"]["n_folds"], seed=seed)
        (out / f"results_{tag}.json").write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
        md_all.append(f"### {tag}\n\n" + summary_markdown(res))
    md = "\n\n".join(md_all)
    (out / "summary.md").write_text(md + "\n", encoding="utf-8")
    print("\n" + md)


if __name__ == "__main__":
    main()
