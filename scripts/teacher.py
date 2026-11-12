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


def _rr_one(d: str, out: str, spec: TeacherSpec, overwrite: bool, channel: str = "ecg") -> str:
    try:
        rec = load_session(d, channels=(channel,))
    except Exception as e:  # PPG 없는 세션
        return f"skip {d} ({type(e).__name__})"
    if channel not in rec.signals:
        return f"skip {d} (no {channel})"
    path = cache_path(out, rec)
    if path.exists() and not overwrite:
        return f"skip {path}"
    extract_rr(rec, spec, channel).save(path)
    return f"ok   {path}"


def cmd_rr(a, cfg):
    spec = _spec(cfg)
    dirs = [str(d) for root in a.data for d in iter_session_dirs(root)]
    print(f"세션 {len(dirs)}개 → {a.out}")
    t = time.perf_counter()
    if a.jobs == 1:
        msgs = [_rr_one(d, a.out, spec, a.overwrite, a.channel) for d in dirs]
    else:
        from joblib import Parallel, delayed
        msgs = Parallel(n_jobs=a.jobs, verbose=5)(delayed(_rr_one)(d, a.out, spec, a.overwrite, a.channel) for d in dirs)
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
    if getattr(a, "datasets", None):
        sel = win["dataset_id"].isin(a.datasets).to_numpy()
        print(f"데이터셋 {a.datasets} 만 사용: 창 {sel.sum()} / {len(sel)}")
        win = win[sel].reset_index(drop=True)
        tacho = np.asarray(tacho[np.flatnonzero(sel)]) if tacho is not None else None
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


def cmd_pair_stats(a, cfg):
    """쌍 통계 표(4.5), CLAS 과제 대응(10장), 기준이 졸음이던 참가자(3.3)."""
    from driver_state.teacher.pair_eval import load_all
    from driver_state.teacher.pairs import PairIndex, STUDENT_TRAIN, clas_task_intervals, _stim_name
    import re
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    ann, prof, *_ = load_all(a.cache, a.student, a.data, a.tbin_sec)
    sel = ann[ann["valid"] & (ann["src"].eq("teacher") | ann["dataset_id"].isin(STUDENT_TRAIN))]
    stats = PairIndex(sel).stats()
    # CLAS 자극 이름 패턴과 과제 구간
    data = Path(a.data)
    pat, iv_cnt = {}, {}
    for p in sorted((data / "clas").rglob("labels_orig.csv")):
        o = pd.read_csv(p)
        for s in o["stimulus"]:
            k = re.sub(r"[0-9]+", "#", _stim_name(s))
            pat[k] = pat.get(k, 0) + 1
        for _, _, kind in clas_task_intervals(o):
            iv_cnt[kind] = iv_cnt.get(kind, 0) + 1
    cond_cnt = ann[ann["dataset_id"].isin(["clas", "maus", "wesad"])].groupby(["dataset_id", "cond"]).size()
    # 3.3
    adv_kss0 = drozy_base = 0
    drozy_not_test1 = []
    for p in sorted((data / "advitam").glob("*/block1/labels_orig.csv")):
        o = pd.read_csv(p)
        r = o[o["item"] == "KSS_0"]
        adv_kss0 += int(len(r) and float(r["kss"].iloc[0]) >= 7)
    for key in prof.loc[prof["profile_key"].str.startswith("drozy/"), "profile_key"]:
        g, _, ses = key.partition("#")
        subj = g.split("/")[1]
        if ses != "test1":
            drozy_not_test1.append(f"{subj}:{ses}")
        p = data / "drozy" / subj / ses / "labels_orig.csv"
        if p.exists():
            drozy_base += int(float(pd.read_csv(p)["kss"].iloc[0]) >= 7)
    res = {"pair_stats": stats.to_dict(orient="index"), "clas_stimulus_patterns": pat,
           "clas_task_intervals": iv_cnt, "cond_windows": {f"{d}/{c}": int(n) for (d, c), n in cond_cnt.items()},
           "advitam_kss0_ge7": adv_kss0, "drozy_base_kss_ge7": drozy_base, "drozy_base_not_test1": drozy_not_test1,
           "base_kind": ann.groupby(["dataset_id", "base_kind"])["group"].nunique().rename("subjects")
                          .reset_index().to_dict(orient="records")}
    (out / "pair_stats.json").write_text(json.dumps(res, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(stats.to_string())
    print(json.dumps({k: v for k, v in res.items() if k != "pair_stats"}, ensure_ascii=False, indent=1, default=str))


def cmd_pair_eval(a, cfg):
    from driver_state.teacher.pair_eval import (README_B, README_M3_RRNET, README_TIME, evaluate, load_all,
                                                report_markdown, run_lodo, run_m7)
    from driver_state.teacher.metrics_time import m2_dataset, m3_dataset
    from driver_state.teacher.evaluate import _auc
    spec = _spec(cfg)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    ann, prof, wt, pt, ws, ps = load_all(a.cache, a.student, a.data, a.tbin_sec, getattr(a, "nw_dir", None))
    from driver_state.teacher.pairs import feature_names as _fn
    from driver_state.teacher.nw_features import NW_COLS
    names = {"rr": _fn(spec), "nw": NW_COLS, "rrnw": _fn(spec) + NW_COLS}[getattr(a, "features", "rr")]
    t0 = time.time()
    if (out / "oof_lodo.parquet").exists() and (out / "wesad_scores.parquet").exists() and a.resume:
        oof, wes = pd.read_parquet(out / "oof_lodo.parquet"), pd.read_parquet(out / "wesad_scores.parquet")
        print("LODO 결과를 이어서 씀 (--resume)")
    else:
        oof, wes = run_lodo(ann, prof, wt, pt, ws, ps, a.b0_oof, a.conds, a.seeds, spec, names=names)
        oof.to_parquet(out / "oof_lodo.parquet")          # M7 전에 먼저 저장 (중단 대비)
        wes.to_parquet(out / "wesad_scores.parquet")
    # 재현 확인 (5.1, 7장): 행 선택 전 전체 행
    repro = {"B0_raw": {}, "B1_raw": {}, "BTt_within_session": {}, "B0_M3": {}}
    for d, g in oof.groupby("dataset_id"):
        if "d_B0" in g:
            gg = g[np.isfinite(g["d_B0"])]
            repro["B0_raw"][d] = (round(_auc(gg["y"].to_numpy(), gg["d_B0"].to_numpy()), 3), README_B["rrnet"][d])
            if d in README_M3_RRNET:
                repro["B0_M3"][d] = (round(m3_dataset(gg, "d_B0"), 3), README_M3_RRNET[d])
        gg = g[np.isfinite(g["d_B1"])]
        repro["B1_raw"][d] = (round(_auc(gg["y"].to_numpy(), gg["d_B1"].to_numpy()), 3), README_B["dev_core"][d])
        if "d_BTt" in g and d in README_TIME:
            num = den = 0.0
            for _, s in g.groupby(["group", "session_id"]):
                if s["y"].nunique() == 2:
                    n = (s["y"] == 1).sum() * (s["y"] == 0).sum()
                    num += _auc(s["y"].to_numpy(), s["d_BTt"].to_numpy()) * n; den += n
            repro["BTt_within_session"][d] = (round(num / den, 3) if den else None, README_TIME[d])
    print("재현 확인:", json.dumps(repro, ensure_ascii=False))
    m7_path = out / "m7_drozy_5fold.parquet"
    if a.skip_m7:
        m7 = None
    elif a.resume and m7_path.exists():
        m7 = pd.read_parquet(m7_path)
    else:
        m7 = run_m7(ann, prof, a.conds, a.seeds, spec, names=names)
        m7.to_parquet(m7_path)                              # 지표 계산 전에 저장 (중단 대비)
    print("학습 끝, 지표 계산", flush=True)
    cond_cols = (["BTt", "BTfit"] if "BT" in a.conds else []) + [c for c in ("P0", "P1", "P2") if c in a.conds] \
        + (["P0raw"] if "P0" in a.conds else []) + (["B0"] if "d_B0" in oof else []) + ["B1"]
    rep = evaluate(oof, wes, m7, cond_cols, a.seeds)
    rep["repro"] = repro
    rep["elapsed_sec"] = round(time.time() - t0)
    oof.to_parquet(out / "oof_lodo.parquet")
    wes.to_parquet(out / "wesad_scores.parquet")
    if m7 is not None:
        m7.to_parquet(out / "m7_drozy_5fold.parquet")
    (out / "report_pair.json").write_text(json.dumps(rep, ensure_ascii=False, indent=2, default=float), encoding="utf-8")
    md = report_markdown(rep, repro)
    (out / "report_pair.md").write_text(md, encoding="utf-8")
    print(md)


def cmd_anomaly_eval(a, cfg):
    """docs/teacher-anomaly-stage1.md: 방향 없는 학습 거리 A1~A3 vs B0·B1."""
    from driver_state.teacher.anomaly import evaluate as an_eval, report_markdown as an_md, run as an_run
    from driver_state.teacher.pair_eval import load_all
    from driver_state.teacher.metrics_time import m1_dataset
    spec = _spec(cfg)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    ann, prof, wt, pt, ws, ps = load_all(a.cache, a.student, a.data, a.tbin_sec, a.nw_dir)
    ann_t = ann[ann["src"] == "teacher"].reset_index(drop=True)
    ann_s = ann[ann["src"] == "student"].reset_index(drop=True)
    rr_ctx = None
    if any(c.startswith("R") for c in a.conds):
        from driver_state.teacher.anomaly_rr import base_rr_of, reference_tachograms
        from driver_state.teacher.features import build_tachograms
        rr_t = load_rr_cache(Path(a.cache) / "rr")
        rr_w = [r for r in load_rr_cache(Path(a.student) / "rr") if r.dataset_id == "wesad"]
        refs_t, short_t = reference_tachograms(rr_t, spec)
        refs_w, short_w = reference_tachograms(rr_w, spec)
        T_t = np.load(Path(a.cache) / "tacho.npy", mmap_mode="r")
        wmask = (ws["dataset_id"] == "wesad").to_numpy()
        T_s = np.zeros((len(ws), 2, T_t.shape[2]), np.float16)
        T_s[wmask] = build_tachograms(rr_w, ws[wmask], spec)
        base = {**base_rr_of(pt).to_dict(), **base_rr_of(ps).to_dict()}
        rr_ctx = {"tacho_t": np.asarray(T_t), "tacho_s": T_s, "refs": {**refs_t, **refs_w}, "base": base,
                  "device": "cuda" if __import__("torch").cuda.is_available() else "cpu"}
        info = {"profiles_with_ref": {d: sum(k.startswith(d + "/") for k in rr_ctx["refs"]) for d in
                                      ("advitam", "dddb", "drozy", "mpddf", "wesad")},
                "profiles_total": {d: int(pd.concat([pt, ps])["profile_key"].str.startswith(d + "/").sum()) for d in
                                   ("advitam", "dddb", "drozy", "mpddf", "wesad")},
                "short_baseline": [k for k, v in {**short_t, **short_w}.items() if v],
                "ref_tachos_per_profile_median": float(np.median([len(v) for v in rr_ctx["refs"].values()]))}
        print("기준 타코그램:", json.dumps(info, ensure_ascii=False), flush=True)
        (out / "reference_info.json").write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")
    oof_p, wes_p = out / "oof_lodo.parquet", out / "wesad_scores.parquet"
    if a.resume and oof_p.exists() and wes_p.exists():
        oof, wes = pd.read_parquet(oof_p), pd.read_parquet(wes_p)
    else:
        oof, wes = an_run(wt, pt, ann_t, ws, ps, ann_s, a.b0_oof, a.conds, a.seeds, spec, rr_ctx=rr_ctx, features=a.features)
        oof.to_parquet(oof_p); wes.to_parquet(wes_p)
    repro = {}
    for c in ("B0", "B1"):
        col = f"s_{c}"
        if col in oof:
            r = oof[np.isfinite(oof[col])]
            repro[c] = round(float(np.mean([m1_dataset(g, col)[0] for _, g in r.groupby("dataset_id")])), 3)
    print("재현 확인 (M1 평균):", repro, flush=True)
    rep = an_eval(oof, wes, (["B0"] if "s_B0" in oof else []) + ["B1"] + a.conds, a.seeds)
    rep["repro"] = repro
    (out / "report_anomaly.json").write_text(json.dumps(rep, ensure_ascii=False, indent=2, default=float), encoding="utf-8")
    md = an_md(rep, repro)
    (out / "report_anomaly.md").write_text(md, encoding="utf-8")
    print(md)


def cmd_f2c_label(a, cfg):
    """docs/student-f2c-stage1.md 1장: 교사 F2C 로 창별 라벨 표."""
    from driver_state.teacher.f2c_label import build, summary
    from driver_state.teacher.pair_eval import load_all
    spec = _spec(cfg)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    ann, prof, wt, pt, ws, ps = load_all(a.cache, a.student, a.data, 150.0, a.nw_dir)
    ann_t = ann[ann["src"] == "teacher"].reset_index(drop=True)
    ann_s = ann[ann["src"] == "student"].reset_index(drop=True)
    assert len(ann_t) == len(wt) and len(ann_s) == len(ws), "창 표와 주석 표 길이가 다름"
    lab, params = build(wt, pt, ann_t, ws, ps, ann_s, pd.read_parquet(a.oof), spec, a.seeds)
    lab.to_parquet(out / "f2c_labels.parquet")
    arrs = {}
    for tag, lst in params.items():
        for i, p in enumerate(lst):
            arrs[f"{tag}/s{i}/W"] = p["W"]; arrs[f"{tag}/s{i}/alpha"] = np.float32(p["alpha"]); arrs[f"{tag}/s{i}/b"] = np.float32(p["b"])
    np.savez(out / "f2c_params.npz", **arrs)
    sm = summary(lab)
    (out / "label_summary.txt").write_text(sm.to_string(), encoding="utf-8")
    print(sm.to_string(), flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=str(ROOT / "configs/teacher.yaml"))
    sp = ap.add_subparsers(dest="cmd", required=True)
    p = sp.add_parser("synth"); p.add_argument("--out", required=True); p.add_argument("--subjects", type=int, default=6)
    p.add_argument("--minutes", type=float, default=30.0); p.add_argument("--seed", type=int, default=0)
    p = sp.add_parser("inspect"); p.add_argument("--src", required=True)
    p = sp.add_parser("rr"); p.add_argument("--data", nargs="+", required=True); p.add_argument("--out", required=True)
    p.add_argument("--channel", choices=["ecg", "ppg"], default="ecg", help="ppg: 맥박 간격 (PRV)")
    p.add_argument("--jobs", type=int, default=1); p.add_argument("--overwrite", action="store_true")
    p = sp.add_parser("features"); p.add_argument("--rr", required=True); p.add_argument("--out", required=True)
    p.add_argument("--jobs", type=int, default=1); p.add_argument("--no-tacho", action="store_true")
    p = sp.add_parser("evaluate"); p.add_argument("--cache", required=True); p.add_argument("--out", required=True)
    p.add_argument("--models", nargs="*"); p.add_argument("--mode", choices=["lodo", "subject", "both"], default="lodo")
    p.add_argument("--datasets", nargs="*", help="이 데이터셋만 써서 평가 (예: mpddf advitam)")
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
    p = sp.add_parser("anomaly-eval"); p.add_argument("--cache", required=True); p.add_argument("--student", required=True)
    p.add_argument("--data", required=True); p.add_argument("--out", required=True); p.add_argument("--b0-oof", dest="b0_oof")
    p.add_argument("--conds", nargs="+", default=["F2", "R1", "R2", "R3"]); p.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    p.add_argument("--tbin-sec", dest="tbin_sec", type=float, default=150.0); p.add_argument("--resume", action="store_true")
    p.add_argument("--features", choices=["rr", "nw", "rrnw"], default="rr"); p.add_argument("--nw-dir", dest="nw_dir")
    p = sp.add_parser("f2c-label"); p.add_argument("--cache", required=True); p.add_argument("--student", required=True)
    p.add_argument("--data", required=True); p.add_argument("--out", required=True); p.add_argument("--nw-dir", dest="nw_dir", required=True)
    p.add_argument("--oof", required=True, help="anomaly-eval --features rrnw 의 oof_lodo.parquet")
    p.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    for name in ("pair-stats", "pair-eval"):
        p = sp.add_parser(name); p.add_argument("--cache", required=True); p.add_argument("--student", required=True)
        p.add_argument("--data", required=True, help="규약 폴더 루트 (labels_orig.csv)"); p.add_argument("--out", required=True)
        p.add_argument("--tbin-sec", dest="tbin_sec", type=float, default=300.0, help="시간 층 폭 (규격 8.3)")
        if name == "pair-eval":
            p.add_argument("--b0-oof", dest="b0_oof"); p.add_argument("--conds", nargs="+", default=["BT", "P0", "P1", "P2"])
            p.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
            p.add_argument("--skip-m7", dest="skip_m7", action="store_true")
            p.add_argument("--resume", action="store_true", help="저장된 oof_lodo·wesad_scores 가 있으면 LODO 를 건너뜀")
            p.add_argument("--features", choices=["rr", "nw", "rrnw"], default="rr"); p.add_argument("--nw-dir", dest="nw_dir")
    a = ap.parse_args()
    cfg = load_cfg(a.config)
    globals()[f"cmd_{a.cmd.replace('-', '_')}"](a, cfg)


if __name__ == "__main__":
    main()
