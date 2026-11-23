"""모델 번들(models/)만으로 실측 세션을 추론한다. GPU·학습 데이터 없이 CPU 에서 돈다.

    python scripts/infer_bundle.py --session <규약 세션 폴더> [<같은 사람의 다른 세션> ...] --out out/infer
    # 교사 F2C 도 함께 (NormWear 코드·체크포인트 필요, models/README.md 참고):
    python scripts/infer_bundle.py --session ... --out ... --teacher --normwear-dir <NormWear 상위 폴더> --ckpt <model.safetensors>

세션 폴더 형식은 docs/data-handoff.md (meta.json + ecg.csv + ppg.csv). 첫 세션의 앞부분에서 기준을 수집한다.
결과: <out>/windows.csv (창별 학생·교사 점수, 품질, 상태 머신 단계), <out>/summary.json
"""
from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from safetensors.torch import load_file

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
warnings.filterwarnings("ignore")

from driver_state.data_io import load_session  # noqa: E402
from driver_state.state_machine_v2 import StateMachineV2, V2Config, WindowIn, select_baseline  # noqa: E402
from driver_state.student.distill import Student  # noqa: E402
from driver_state.teacher.anomaly import z_matrix  # noqa: E402
from driver_state.teacher.features import baseline_plan, build_tables  # noqa: E402
from driver_state.teacher.personal import PersonalNormalizer  # noqa: E402
from driver_state.teacher.rr import extract_rr  # noqa: E402
from driver_state.teacher.spec import TeacherSpec  # noqa: E402
from student_cache import cut, prep  # noqa: E402
from window_quality import abn_ratio  # noqa: E402


def sig(x):
    return 1 / (1 + np.exp(-np.asarray(x, float)))


def load_students(mdir: Path, cond: str, dev):
    ms = []
    for p in sorted((mdir / "student").glob(f"{cond}_s*.safetensors")):
        sd = load_file(str(p))
        c_in = sd["g.conv.0.weight"].shape[1]
        m = Student(c_in, "dist", sd["W"].numpy(), 0.0, 0.0, 36).to(dev)
        m.load_state_dict(sd)
        m.eval()
        ms.append(m)
    return ms


@torch.no_grad()
def student_logits(ms, X, R, zr, mask, dev):
    """X (N,C,L), R (K,C,L), mask (C,) 채널 켜기/끄기. 시드 평균 logit."""
    M = torch.as_tensor(mask, dtype=torch.float32)[None, :, None]
    x, r = torch.as_tensor(X) * M, torch.as_tensor(R) * M
    out = []
    for m in ms:
        gr = m.g(r.to(dev)).median(0).values
        out.append(m.logit(torch.as_tensor(zr).to(dev), m.g(x.to(dev)) - gr).cpu().numpy())
    return np.mean(out, 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--session", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--models", default=str(ROOT / "models"))
    ap.add_argument("--force", action="store_true", help="기준 수집이 실패해도 0~300초 기준으로 계속")
    ap.add_argument("--baseline", nargs=2, type=float, help="기준 구간 직접 지정 [초] (자동 수집 대신)")
    ap.add_argument("--teacher", action="store_true")
    ap.add_argument("--ppg-noshort", dest="ppg_noshort", action="store_true",
                    help="PPG 대체 판정에서 짧은 변동 HRV 를 끔 (상태 머신이 이 점수를 씀)")
    ap.add_argument("--normwear-dir"); ap.add_argument("--ckpt")
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    mdir = Path(a.models)
    B = json.loads((mdir / "bundle.json").read_text(encoding="utf-8"))
    spec = TeacherSpec.from_dict(B["spec"])
    names = B["rr_feature_names"]
    norm = PersonalNormalizer.from_dict(spec, B["normalizer"])
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    # 1. 세션·RR, 기준 수집 (상태 머신 v2 2장)
    recs = [load_session(d, channels=("ecg", "ppg")) for d in a.session]
    for r_ in recs:
        r_.prefiltered = False                     # 파이프라인 필터를 항상 적용
    rrs = [extract_rr(r_, spec, "ecg") for r_ in recs]
    rr = rrs[0]
    if rr.subject_id is None:
        for x in rrs:
            x.subject_id = "self"
    rr.t0 = min(x.t0 for x in rrs) - 1.0
    grid = np.arange(0.0, rr.duration - 30 + 1e-6, 10.0)
    ok_b, idx_b, (ba, bb) = select_baseline(grid, abn_ratio(rr, grid))
    base = {"ok": bool(ok_b), "clean_windows": int(len(idx_b)), "range": [float(ba), float(bb)]}
    if a.baseline:
        ok_b, (ba, bb) = True, tuple(a.baseline)
    if not ok_b:
        msg = f"기준 수집 실패 (깨끗한 창 {len(idx_b)}개 < 20): 센서를 다시 붙이고 재측정하세요."
        if not a.force:
            (out / "summary.json").write_text(json.dumps({"baseline_check": base, "message": msg}, ensure_ascii=False, indent=1), encoding="utf-8")
            print(msg); return
        print(msg + " (--force: 0~300초 기준으로 계속)")
        ba, bb = 0.0, spec.baseline_sec
    rr.meta["baseline_range"] = [ba, bb]
    win, prof = build_tables(rrs, spec)
    info = next(v for v in baseline_plan(rrs, spec).values() if v is not None)
    zr, dev_core = z_matrix(norm, win, prof, names)
    by = {x.session_id: (x, rc) for x, rc in zip(rrs, recs)}
    q = np.ones(len(win))
    for sid, g in win.groupby("session_id"):
        q[g.index.to_numpy()] = abn_ratio(by[sid][0], g["t_rel"].to_numpy(float))
    print(f"창 {len(win)} (유효 {int(win['valid'].sum())}), 기준 {info.start:.0f}~{info.end:.0f}초", flush=True)

    # 2. 파형 (64 Hz)
    W = B["student"]["wave"]
    rs = np.arange(info.start, info.end - 30 + 1e-6, spec.stride_sec)[: W["max_ref_windows"]]
    L = int(W["fs"] * W["win_sec"])
    X = np.zeros((len(win), 2, L), np.float32); R = np.zeros((len(rs), 2, L), np.float32)
    for sid, g in win.groupby("session_id"):
        rc = by[sid][1]
        for c, ch in enumerate(("ecg", "ppg")):
            if ch in rc.signals:
                X[g.index.to_numpy(), c] = cut(prep(rc.signals[ch], rc.fs[ch], ch, spec.causal), g["t_rel"].to_numpy(float))[0]
    rc0 = by[info.session_id][1]
    for c, ch in enumerate(("ecg", "ppg")):
        if ch in rc0.signals:
            R[:, c] = cut(prep(rc0.signals[ch], rc0.fs[ch], ch, spec.causal), rs)[0]

    # 3. PPG 대체 경로: PRV RR 특징
    prrs = [extract_rr(rc, spec, "ppg") for rc in recs if "ppg" in rc.signals]
    zr_prv = np.zeros_like(zr); q_ppg = np.ones(len(win)); hr_ppg = np.full(len(win), np.nan)
    valid_ppg = np.zeros(len(win), bool)
    if len(prrs) == len(recs):
        for x, y in zip(prrs, rrs):
            x.subject_id, x.t0 = y.subject_id, y.t0
        prrs[0].meta["baseline_range"] = [ba, bb]
        pwin, pprof = build_tables(prrs, spec)
        pz, _ = z_matrix(norm, pwin, pprof, names)
        k = ["session_id", "t_rel"]
        pdf = pd.concat([pwin[k + ["hr_30", "valid"]].rename(columns={"valid": "valid_ppg"}), pd.DataFrame(pz, columns=[f"pz{i}" for i in range(pz.shape[1])])], axis=1)
        mm = win[k].merge(pdf, on=k, how="left")
        zr_prv = np.nan_to_num(mm[[f"pz{i}" for i in range(pz.shape[1])]].to_numpy(np.float32), nan=0.0)
        hr_ppg = mm["hr_30"].to_numpy(float)
        valid_ppg = mm["valid_ppg"].fillna(False).to_numpy(bool)
        pby = {x.session_id: x for x in prrs}
        for sid, g in win.groupby("session_id"):
            q_ppg[g.index.to_numpy()] = abn_ratio(pby[sid], g["t_rel"].to_numpy(float))

    # 4. 학생
    res = win[["session_id", "t_rel", "valid"]].copy()
    res["hr_ecg"], res["hr_ppg"], res["q_ecg"], res["q_ppg"], res["dev_core"] = win["hr_30"].to_numpy(), hr_ppg, q, q_ppg, dev_core
    res["valid_ppg"] = valid_ppg
    ep = load_students(mdir, "S1-EP", dev)
    res["student_EP_logit"] = student_logits(ep, X, R, zr, [1, 1], dev)
    res["student_PPG_logit"] = student_logits(ep, X, R, zr_prv, [0, 1], dev)
    short = ("rmssd", "sd1", "pnn50", "sd1_sd2", "hf_log", "lf_hf_log", "lf_nu", "sampen", "dfa_a1")
    keep = np.array([0.0 if n.rsplit("_", 1)[0] in short else 1.0 for n in names], np.float32)
    res["student_PPG_ns_logit"] = student_logits(ep, X, R, (zr_prv * keep).astype(np.float32), [0, 1], dev)
    ppg_col = "student_PPG_ns_logit" if a.ppg_noshort else "student_PPG_logit"
    e = load_students(mdir, "S1-E", dev)
    if e:
        res["student_E_logit"] = student_logits(e, X[:, :1], R[:, :1], zr, [1], dev)
    th = B["thresholds"]["student_logit"]["S1-EP"]

    # 5. 교사 (선택)
    if a.teacher:
        from driver_state.teacher.nw_features import NW_COLS
        from normwear_embed import Embedder, prep_ecg, segments
        comps = np.load(mdir / "teacher" / "nw_pca.npz")["comps"]
        emb = Embedder(a.normwear_dir, a.ckpt, dev)
        E = np.zeros((len(win), len(NW_COLS)), np.float32)
        for sid, g in win.groupby("session_id"):
            rc = by[sid][1]
            E[g.index.to_numpy()] = emb(segments(prep_ecg(rc.signals["ecg"], rc.fs["ecg"]), g["t_rel"].to_numpy(float))).astype(np.float32) @ comps.T
        Er = emb(segments(prep_ecg(rc0.signals["ecg"], rc0.fs["ecg"]), rs)).astype(np.float32) @ comps.T
        zn = np.clip(np.nan_to_num((E - np.median(Er, 0)) / np.asarray(B["nw_sigma_pop"]), nan=0.0), -5, 5)
        Z = np.concatenate([zr, zn], 1)
        P = dict(np.load(mdir / "teacher" / "f2c_final.npz"))
        lg, i = [], 0
        while f"final__s{i}__W" in P:
            Wt, al, b = P[f"final__s{i}__W"], float(P[f"final__s{i}__alpha"]), float(P[f"final__s{i}__b"])
            lg.append(np.log1p(np.exp(al)) * (np.sqrt(((Z @ Wt.T) ** 2).sum(1) + 1e-8) - 1e-4) + b)
            i += 1
        res["teacher_logit"] = np.mean(lg, 0)

    # 6. 상태 머신 v2 (세션마다 처음부터)
    lv, cand, ch, fr = [], [], [], []
    for sid, g in res.groupby("session_id", sort=False):
        sm = StateMachineV2(V2Config())
        g = g.sort_values("t_rel")
        for r in g.to_dict("records"):
            o = sm.step(WindowIn(t=r["t_rel"] + 30, p_ep=float(sig(r["student_EP_logit"])) if r["valid"] else None, q_ecg=float(r["q_ecg"]),
                                 p_ppg=float(sig(r[ppg_col])) if r["valid_ppg"] else None, q_ppg=float(r["q_ppg"]),
                                 hr_ecg=r["hr_ecg"] if np.isfinite(r["hr_ecg"]) else None,
                                 hr_ppg=r["hr_ppg"] if np.isfinite(r["hr_ppg"]) else None))
            lv.append((r["session_id"], r["t_rel"], o.level, o.candidate, o.channel, o.frozen))
    S = pd.DataFrame(lv, columns=["session_id", "t_rel", "level", "candidate", "channel", "frozen"])
    res = res.merge(S, on=["session_id", "t_rel"], how="left")
    res["student_alarm"] = res["student_EP_logit"] >= th
    res.to_csv(out / "windows.csv", index=False)
    summ = {"baseline_check": base, "baseline_used": [info.session_id, float(info.start), float(info.end)],
            "per_session": {sid: {"windows": int(len(g)), "valid": int(g["valid"].sum()),
                                  "student_alarm_rate": float(g.loc[g["valid"], "student_alarm"].mean()),
                                  "time_in_warning_or_above": float((g["level"] >= 1).mean()),
                                  "drowsy_confirmed": int((g["level"] == 2).sum()),
                                  "ppg_fallback_windows": int((g["channel"] == "PPG").sum())}
                            for sid, g in res.groupby("session_id")}}
    (out / "summary.json").write_text(json.dumps(summ, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(summ, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
