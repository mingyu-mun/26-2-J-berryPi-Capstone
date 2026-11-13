"""교사 F2C 로 창별 라벨 표 만들기 (docs/student-f2c-stage1.md 1장).

- 교사 4종: LODO oof (anomaly-eval --features rrnw --conds F2 결과) 의 s_F2 를 그대로 쓴다.
- CLAS·MAUS: 최종 F2C (교사 4종 전체 + WESAD 스트레스 전체) 로 라벨.
- WESAD: 묶음 k 의 창은 묶음 k 스트레스를 뺀 최종 F2C 로 라벨 (4 묶음 교차).
- 학생 학습 목표용 z (RR 36 + NW 32) 는 교사 4종 전체로 맞춘 정규화로 모든 창에 계산.
- 학생 LODO 초기값용으로 fold 별 교사 W·α·b 도 저장 (시드 0).
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd

from .anomaly import predict, train_model, wesad_folds, z_matrix
from .evaluate import usable
from .models.base import sample_weights
from .nw_features import NW_COLS, z_of
from .personal import PersonalNormalizer
from .pairs import TEACHER_DATASETS

STUDENT_SETS = ("clas", "maus", "wesad")


def _has_nw(win, prof):
    okp = prof.set_index("profile_key")[[f"m_{c}" for c in NW_COLS]].notna().all(1)
    return win["profile_key"].map(okp).fillna(False).to_numpy(bool) & win[NW_COLS].notna().all(1).to_numpy()


def _params(m) -> dict:
    return {"W": m.W.detach().cpu().numpy().astype(np.float32),
            "alpha": float(m.alpha.detach()), "b": float(m.b.detach())}


def _z(norm, win_fit, win, prof, names):
    zr, dev = z_matrix(norm, win, prof, names)
    return np.concatenate([zr, z_of(win_fit, win, prof)], 1), dev


def build(win_t, prof_t, ann_t, win_s, prof_s, ann_s, oof: pd.DataFrame, spec, seeds=(0, 1, 2, 3, 4), log=print):
    names = list(spec.baseline_feature_names)
    ds = win_t["dataset_id"].to_numpy()
    use_t = usable(win_t) & (ann_t["baseline_valid"].to_numpy() > 0.5) & _has_nw(win_t, prof_t)
    lab_t = win_t["valid"].to_numpy(bool) & (ann_t["baseline_valid"].to_numpy() > 0.5) & _has_nw(win_t, prof_t)
    sds = win_s["dataset_id"].to_numpy()
    lab_s = (np.isin(sds, STUDENT_SETS) & win_s["valid"].to_numpy(bool)
             & (ann_s["baseline_valid"].to_numpy() > 0.5) & _has_nw(win_s, prof_s))
    cond = ann_s["cond"].to_numpy()
    wes = sds == "wesad"
    wf = wesad_folds(ann_s.loc[wes, "group"])
    wfold = ann_s["group"].map(wf).fillna(-1).astype(int).to_numpy()
    stress = lab_s & wes & (cond == "stress")

    params: dict[str, list] = {}

    def fit(tr_t, tr_s, tag):
        norm = PersonalNormalizer(spec).fit(win_t[tr_t])
        wfit = win_t[tr_t]
        Zt, _ = _z(norm, wfit, win_t[tr_t], prof_t, names)
        Zs, _ = _z(norm, wfit, win_s[tr_s], prof_s, names)
        y_tr = win_t.loc[tr_t, "y"].to_numpy(int)
        w_tr = sample_weights(ds[tr_t], win_t.loc[tr_t, "group"].to_numpy(), y_tr)
        gs = win_s.loc[tr_s, "group"].to_numpy()
        w_s = 1.0 / pd.Series(gs).map(pd.Series(gs).value_counts()).to_numpy(float)
        w_s = w_s / w_s.sum() * (w_tr.sum() / len(np.unique(ds[tr_t])))
        Z = np.concatenate([Zt, Zs]); y = np.r_[y_tr, np.zeros(len(Zs), int)]; w = np.r_[w_tr, w_s]
        ms = []
        for seed in seeds:
            t0 = time.perf_counter()
            ms.append(train_model("lin", Z, y, w, seed))
            log(f"    {tag} seed {seed} ({time.perf_counter() - t0:.0f}s)")
        params[tag] = [_params(m) for m in ms]
        return norm, wfit, ms

    # 학생 LODO 용 fold 교사 (anomaly.run 과 같은 학습 집합: 시험 데이터셋 제외, WESAD 묶음 fi 제외)
    for fi, test in enumerate(TEACHER_DATASETS):
        log(f"[fold 교사: 시험 {test}]")
        fit(use_t & (ds != test), stress & (wfold != fi), f"lodo_{test}")

    # 최종 교사: 전체, 그리고 WESAD 묶음 k 제외판
    log("[최종 교사]")
    norm, wfit, m_all = fit(use_t, stress, "final")
    finals = {}
    for k in sorted(set(wfold[wes & (wfold >= 0)])):
        finals[k] = fit(use_t, stress & (wfold != k), f"final_wes{k}")[2]

    # 학생 목표 z (모든 라벨 행, 최종 정규화)
    Zt_all, dev_t = _z(norm, wfit, win_t[lab_t], prof_t, names)
    Zs_all, dev_s = _z(norm, wfit, win_s[lab_s], prof_s, names)

    def score(ms, Z):
        L = np.stack([predict(m, Z)[1] for m in ms])
        return L.mean(0), L.std(0)

    # 학생 데이터 라벨
    sub = win_s.loc[lab_s, ["row_id", "dataset_id", "group", "session_id", "t_rel"]].reset_index(drop=True)
    sub["cond"] = cond[lab_s]
    sub["y"] = -1
    lg = np.full(len(sub), np.nan); sd = np.full(len(sub), np.nan)
    wk = wfold[lab_s]
    nonw = sub["dataset_id"].to_numpy() != "wesad"
    lg[nonw], sd[nonw] = score(m_all, Zs_all[nonw])
    for k, ms in finals.items():
        r = (~nonw) & (wk == k)
        if r.any():
            lg[r], sd[r] = score(ms, Zs_all[r])
    sub["logit"], sub["logit_sd"], sub["dev_core"] = lg, sd, dev_s
    sub["label_src"] = np.where(nonw, "final", "final_wes" + pd.Series(wk).astype(str))

    # 교사 데이터 라벨: LODO oof (정답 없는 창(y<0)도 같은 fold 모델로 매겨진 값이 없으므로 최종 모델 값으로 채움)
    tt = win_t.loc[lab_t, ["row_id", "dataset_id", "group", "session_id", "t_rel", "y"]].reset_index(drop=True)
    tt["cond"] = "drive"
    seed_cols = [c for c in oof.columns if c.startswith("s_F2_s")]
    o = oof[["row_id", "s_F2"] + seed_cols].copy()
    o["sd"] = o[seed_cols].std(1, ddof=0)
    tt = tt.merge(o[["row_id", "s_F2", "sd"]], on="row_id", how="left")
    fl, fsd = score(m_all, Zt_all)
    miss = ~np.isfinite(tt["s_F2"].to_numpy(float))
    tt["logit"] = np.where(miss, fl, tt["s_F2"])
    tt["logit_sd"] = np.where(miss, fsd, tt["sd"])
    tt["label_src"] = np.where(miss, "final", "lodo_oof")
    tt["dev_core"] = dev_t
    tt = tt.drop(columns=["s_F2", "sd"])

    lab = pd.concat([tt, sub], ignore_index=True)
    lab["p"] = 1 / (1 + np.exp(-lab["logit"].to_numpy()))
    Z = np.concatenate([Zt_all, Zs_all])
    for j, n in enumerate(names):
        lab[f"zr_{n}"] = Z[:, j]
    for j, c in enumerate(NW_COLS):
        lab[f"zn_{c[3:]}"] = Z[:, len(names) + j]
    return lab, params


def summary(lab: pd.DataFrame) -> pd.DataFrame:
    g = lab.groupby(["dataset_id", "cond"])
    return pd.DataFrame({"n": g.size(), "p_mean": g["p"].mean(), "p_ge_0.5": g["p"].apply(lambda x: float((x >= 0.5).mean())),
                         "p_median": g["p"].median(), "logit_sd_median": g["logit_sd"].median()}).round(3)
