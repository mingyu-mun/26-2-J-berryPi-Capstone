"""교사 모델 실행 (노트북 CPU 용). 단계마다 결과를 파일로 남긴다.

    python scripts/teacher.py synth    --out data_syn                    # 합성 데이터 (동작 확인용)
    python scripts/teacher.py inspect  --src D:\\raw\\AdVitam\\Exp4         # 원 데이터 구조 출력
    python scripts/teacher.py rr       --data data --out cache/rr --jobs 4
    python scripts/teacher.py features --rr cache/rr --out cache --jobs 4 [--no-tacho]
    python scripts/teacher.py evaluate --cache cache --out outputs/teacher [--models rule lgbm ...] [--mode lodo|subject]
    python scripts/teacher.py tau      --oof outputs/teacher/oof_lodo.parquet [--model lgbm]
    python scripts/teacher.py fit      --cache cache --oof outputs/teacher/oof_lodo.parquet --out teacher_bundle
    python scripts/teacher.py label    --bundle teacher_bundle --data data/maus data/clas [--out labels_out]
    python scripts/teacher.py demo     # 합성 데이터로 synth → … → label 전 과정
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import yaml  # noqa: E402

from driver_state.data_io import iter_session_dirs, load_session  # noqa: E402
from driver_state.teacher.bundle import TeacherBundle, fit_final, label_report, label_tree  # noqa: E402
from driver_state.teacher.evaluate import choose_best, metrics, report_markdown, run_cv  # noqa: E402
from driver_state.teacher.features import build_tables, build_tachograms  # noqa: E402
from driver_state.teacher.rr import RRSeries, cache_path, extract_rr, load_rr_cache  # noqa: E402
from driver_state.teacher.spec import TeacherSpec  # noqa: E402
from driver_state.teacher.tau import choose_tau  # noqa: E402


def load_cfg(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def _spec(cfg) -> TeacherSpec:
    return TeacherSpec.from_dict(cfg.get("teacher"))


# ------------------------------------------------------------------ 단계
def cmd_synth(a, cfg):
    from driver_state.teacher.synthetic import write_synthetic
    write_synthetic(a.out, a.subjects, a.minutes, a.seed)
    print(f"합성 데이터 → {a.out}")


def cmd_inspect(a, cfg):
    """원 데이터 폴더의 파일 구성과 텍스트 파일 앞부분을 출력한다 (어댑터 작성용)."""
    src = Path(a.src)
    files = sorted(p for p in src.rglob("*") if p.is_file())
    by_ext: dict[str, list[Path]] = {}
    for p in files:
        by_ext.setdefault(p.suffix.lower(), []).append(p)
    print(f"{src}: 파일 {len(files)}개, {sum(p.stat().st_size for p in files) / 1e9:.2f} GB")
    for ext, ps in sorted(by_ext.items(), key=lambda kv: -len(kv[1])):
        print(f"  {ext or '(없음)'}: {len(ps)}개, 예: {ps[0].relative_to(src)}")
    print("\n디렉터리 (깊이 3):")
    for d in sorted({p.parent.relative_to(src) for p in files if len(p.relative_to(src).parts) <= 4})[:60]:
        print(f"  {d}")
    for ext in (".csv", ".txt", ".tsv", ".json", ".hea", ".xlsx"):
        for p in by_ext.get(ext, [])[:3]:
            print(f"\n--- {p.relative_to(src)} ---")
            if ext == ".xlsx":
                print(pd.read_excel(p, nrows=5).to_string())
                continue
            with open(p, encoding="utf-8", errors="replace") as f:
                for _ in range(8):
                    line = f.readline()
                    if not line:
                        break
                    print("  " + line.rstrip()[:200])
    for ext, how in ((".mat", "scipy.io.loadmat"), (".edf", "pyedflib/mne"), (".acq", "bioread")):
        if ext in by_ext:
            print(f"\n{ext} 파일 {len(by_ext[ext])}개: {how} 로 열어 채널 이름을 확인해야 합니다.")
            if ext == ".mat":
                try:
                    from scipy.io import whosmat
                    print("  " + str(whosmat(by_ext[ext][0])[:10]))
                except Exception as e:  # noqa: BLE001
                    print(f"  열기 실패: {e}")


def _rr_one(d: str, out: str, spec: TeacherSpec, overwrite: bool) -> str:
    rec = load_session(d, channels=("ecg",))
    path = cache_path(out, rec)
    if path.exists() and not overwrite:
        return f"skip {path}"
    extract_rr(rec, spec).save(path)
    return f"ok   {path}"


def cmd_rr(a, cfg):
    spec = _spec(cfg)
    dirs = [str(d) for root in a.data for d in iter_session_dirs(root)]
    print(f"세션 {len(dirs)}개 → {a.out}")
    t = time.perf_counter()
    if a.jobs == 1:
        msgs = [_rr_one(d, a.out, spec, a.overwrite) for d in dirs]
    else:
        from joblib import Parallel, delayed
        msgs = Parallel(n_jobs=a.jobs, verbose=5)(delayed(_rr_one)(d, a.out, spec, a.overwrite) for d in dirs)
    print(f"{sum(m.startswith('ok') for m in msgs)}개 처리, {sum(m.startswith('skip') for m in msgs)}개 건너뜀, "
          f"{time.perf_counter() - t:.0f}s")


def cmd_features(a, cfg):
    spec = _spec(cfg)
    rrs = load_rr_cache(a.rr)
    t = time.perf_counter()
    win, prof = build_tables(rrs, spec, a.jobs)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    win.to_parquet(out / "windows.parquet")
    prof.to_parquet(out / "profiles.parquet")
    print(f"창 {len(win)}개 (유효 {int(win['valid'].sum())}, 라벨 {(win['y'] >= 0).sum()}), "
          f"프로필 {len(prof)}개 (유효 {int(prof['baseline_valid'].sum())}), {time.perf_counter() - t:.0f}s")
    if not a.no_tacho:
        tc = build_tachograms(rrs, win, spec)
        np.save(out / "tacho.npy", tc)
        print(f"타코그램 {tc.shape} → {out / 'tacho.npy'} ({tc.nbytes / 1e6:.0f} MB)")
    (out / "spec.json").write_text(json.dumps(spec.to_dict(), indent=2), encoding="utf-8")


def _load_cache(path: str, need_tacho: bool):
    c = Path(path)
    win = pd.read_parquet(c / "windows.parquet")
    prof = pd.read_parquet(c / "profiles.parquet")
    tacho = np.load(c / "tacho.npy", mmap_mode="r") if need_tacho and (c / "tacho.npy").exists() else None
    return win, prof, tacho


def cmd_evaluate(a, cfg):
    spec = _spec(cfg)
    ev = cfg.get("evaluation", {})
    models = a.models or ev.get("models")
    win, prof, tacho = _load_cache(a.cache, any(m in ("rrnet", "ensemble") for m in models))
    if tacho is None:
        models = [m for m in models if m not in ("rrnet", "ensemble")]
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    modes = ["lodo", "subject"] if a.mode == "both" else [a.mode]
    for mode in modes:
        if mode == "lodo" and win["dataset_id"].nunique() < 2:
            print("데이터셋이 하나뿐이라 LODO 를 건너뜁니다")
            continue
        print(f"\n=== {mode.upper()} ===")
        oof, res = run_cv(win, prof, spec, list(models), tacho, mode, ev.get("subject_folds", 5),
                          ev.get("seed", 0), cfg.get("models"), ev.get("val_frac", 0.2))
        summ = metrics(oof, list(models))
        best = choose_best(summ)
        oof.to_parquet(out / f"oof_{mode}.parquet")
        md = report_markdown(summ, best, mode, res)
        (out / f"report_{mode}.md").write_text(md, encoding="utf-8")
        (out / f"report_{mode}.json").write_text(json.dumps(
            {"best": best, "summary": summ, "temperatures": {r.name: r.temperatures for r in res},
             "fit_sec": {r.name: r.fit_sec for r in res}}, ensure_ascii=False, indent=2, default=float), encoding="utf-8")
        print("\n" + md)


def cmd_tau(a, cfg):
    oof = pd.read_parquet(a.oof)
    if a.exclude:
        oof = oof[~oof["dataset_id"].isin(a.exclude)]
        print(f"τ 계산에서 제외: {a.exclude}")
    model = a.model
    if model is None:
        rep = Path(a.oof).with_name(Path(a.oof).name.replace("oof_", "report_").replace(".parquet", ".json"))
        model = json.loads(rep.read_text(encoding="utf-8"))["best"] if rep.exists() else "lgbm"
    t = cfg.get("tau", {})
    res = choose_tau(oof, f"p_{model}", a.precision or t.get("drowsy_precision", 0.85),
                     a.npv or t.get("alert_precision", 0.90))
    out = Path(a.out or Path(a.oof).with_name("tau.json"))
    out.write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
    tab = pd.DataFrame(res["table"])
    print(tab.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print(f"\n{model}: τ_hi={res['tau_hi']:.2f} (졸음 정밀도 최저 {res['drowsy_precision_min']:.3f}, "
          f"졸음 라벨 {res['frac_labeled_drowsy']:.1%}), τ_lo={res['tau_lo']:.2f} "
          f"(정상 정밀도 최저 {res['alert_precision_min']:.3f}, 정상 라벨 {res['frac_labeled_alert']:.1%})")
    for n in res["notes"]:
        print("  주의: " + n)
    print(f"→ {out}")


def cmd_fit(a, cfg):
    spec = _spec(cfg)
    oof = pd.read_parquet(a.oof) if a.oof else None
    model = a.model
    if model is None and a.oof:
        rep = Path(a.oof).with_name(Path(a.oof).name.replace("oof_", "report_").replace(".parquet", ".json"))
        model = json.loads(rep.read_text(encoding="utf-8"))["best"] if rep.exists() else None
    model = model or "lgbm"
    win, prof, tacho = _load_cache(a.cache, model in ("rrnet", "ensemble"))
    print(f"최종 교사 학습: {model}")
    b = fit_final(win, prof, spec, model, tacho, oof, cfg.get("evaluation", {}).get("seed", 0), cfg.get("models"))
    tau_path = Path(a.tau) if a.tau else (Path(a.oof).with_name("tau.json") if a.oof else None)
    if tau_path and tau_path.exists():
        t = json.loads(tau_path.read_text(encoding="utf-8"))
        if t.get("model") == model:
            b.tau = {k: t[k] for k in ("tau_hi", "tau_lo", "target_drowsy_precision", "target_alert_precision")}
    b.save(a.out)
    print(f"→ {a.out} ({b.version}, τ={b.tau})")


def cmd_label(a, cfg):
    b = TeacherBundle.load(a.bundle)
    win = label_tree(b, a.data, a.out, a.tau_hi, a.tau_lo, a.jobs)
    md = label_report(win)
    rep = Path(a.out or a.bundle) / "label_report.md"
    rep.write_text(md, encoding="utf-8")
    print(md + f"\n→ {rep}")


def cmd_demo(a, cfg):
    base = Path(a.out)
    ns = argparse.Namespace
    cmd_synth(ns(out=str(base / "data"), subjects=a.subjects, minutes=a.minutes, seed=0), cfg)
    cmd_rr(ns(data=[str(base / "data")], out=str(base / "cache/rr"), jobs=a.jobs, overwrite=True), cfg)
    cmd_features(ns(rr=str(base / "cache/rr"), out=str(base / "cache"), jobs=a.jobs, no_tacho=False), cfg)
    cmd_evaluate(ns(cache=str(base / "cache"), out=str(base / "outputs"), models=a.models, mode="lodo"), cfg)
    oof = str(base / "outputs/oof_lodo.parquet")
    cmd_tau(ns(oof=oof, model=None, precision=None, npv=None, out=None, exclude=None), cfg)
    cmd_fit(ns(cache=str(base / "cache"), oof=oof, model=None, tau=None, out=str(base / "teacher_bundle")), cfg)
    cmd_label(ns(bundle=str(base / "teacher_bundle"), data=[str(base / "data/syn_dddb")],
                 out=str(base / "labels"), tau_hi=None, tau_lo=None, jobs=a.jobs), cfg)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=str(ROOT / "configs/teacher.yaml"))
    sp = ap.add_subparsers(dest="cmd", required=True)
    p = sp.add_parser("synth"); p.add_argument("--out", required=True); p.add_argument("--subjects", type=int, default=6)
    p.add_argument("--minutes", type=float, default=30.0); p.add_argument("--seed", type=int, default=0)
    p = sp.add_parser("inspect"); p.add_argument("--src", required=True)
    p = sp.add_parser("rr"); p.add_argument("--data", nargs="+", required=True); p.add_argument("--out", required=True)
    p.add_argument("--jobs", type=int, default=1); p.add_argument("--overwrite", action="store_true")
    p = sp.add_parser("features"); p.add_argument("--rr", required=True); p.add_argument("--out", required=True)
    p.add_argument("--jobs", type=int, default=1); p.add_argument("--no-tacho", action="store_true")
    p = sp.add_parser("evaluate"); p.add_argument("--cache", required=True); p.add_argument("--out", required=True)
    p.add_argument("--models", nargs="*"); p.add_argument("--mode", choices=["lodo", "subject", "both"], default="lodo")
    p = sp.add_parser("tau"); p.add_argument("--oof", required=True); p.add_argument("--model")
    p.add_argument("--precision", type=float); p.add_argument("--npv", type=float); p.add_argument("--out")
    p.add_argument("--exclude", nargs="*", help="τ 계산에서 뺄 데이터셋 (라벨이 거친 데이터 등)")
    p = sp.add_parser("fit"); p.add_argument("--cache", required=True); p.add_argument("--oof")
    p.add_argument("--model"); p.add_argument("--tau"); p.add_argument("--out", required=True)
    p = sp.add_parser("label"); p.add_argument("--bundle", required=True); p.add_argument("--data", nargs="+", required=True)
    p.add_argument("--out"); p.add_argument("--tau-hi", dest="tau_hi", type=float); p.add_argument("--tau-lo", dest="tau_lo", type=float)
    p.add_argument("--jobs", type=int, default=1)
    p = sp.add_parser("demo"); p.add_argument("--out", default="outputs/teacher_demo")
    p.add_argument("--subjects", type=int, default=6); p.add_argument("--minutes", type=float, default=30.0)
    p.add_argument("--jobs", type=int, default=4); p.add_argument("--models", nargs="*")
    a = ap.parse_args()
    cfg = load_cfg(a.config)
    globals()[f"cmd_{a.cmd}"](a, cfg)


if __name__ == "__main__":
    main()
