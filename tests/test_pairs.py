"""쌍 비교 교사 1단계 테스트 (docs/teacher-pair-stage1.md 9.3, 합성 데이터)."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from driver_state.data_io import load_tree  # noqa: E402
from driver_state.teacher.features import build_tables  # noqa: E402
from driver_state.teacher.metrics_time import m1_dataset  # noqa: E402
from driver_state.teacher.models.pairnet import PairNet, n_params  # noqa: E402
from driver_state.teacher.pair_eval import train_masks  # noqa: E402
from driver_state.teacher.pairs import (FeatureScaler, PairIndex, PairSampler, _r_candidates, annotate,  # noqa: E402
                                        feature_names, read_session_info, scale_groups)
from driver_state.teacher.rr import extract_rr  # noqa: E402
from driver_state.teacher.spec import TeacherSpec  # noqa: E402
from driver_state.teacher.synthetic import write_synthetic  # noqa: E402

RENAME = {"syn_advitam": "advitam", "syn_mpddf": "mpddf", "syn_dddb": "dddb", "syn_drozy": "drozy"}


def _rename(df):
    df = df.copy()
    for a, b in RENAME.items():
        for c in ("dataset_id", "group", "profile_key"):
            if c in df:
                df[c] = df[c].str.replace(a, b, regex=False)
    return df


@pytest.fixture(scope="module")
def ann(tmp_path_factory):
    root = tmp_path_factory.mktemp("pair")
    tree = write_synthetic(root / "syn", n_subjects=3, minutes=14, seed=2, fs_override=250.0)
    spec = TeacherSpec()
    rrs = [extract_rr(r, spec) for r in load_tree(tree, channels=("ecg",))]
    for r in rrs:
        r.save(root / "rr" / r.dataset_id / str(r.subject_id) / f"{r.session_id}.npz")
    win, prof = build_tables(rrs, spec)
    info = _rename(read_session_info([root / "rr"]))
    a = annotate(_rename(win), _rename(prof), info, None)
    # 학생용 S 쌍 확인용: advitam 한 명을 maus 과제 창처럼 복제
    st = a[a["dataset_id"] == "advitam"].copy()
    st["dataset_id"] = "maus"
    st["group"] = "maus/" + st["group"]
    st["profile_key"] = "maus/" + st["profile_key"]
    st["cond"], st["y"], st["base_kind"] = "task", -1, "rest_session"
    w = st.copy(); w["dataset_id"] = "wesad"; w["group"] = "wesad/" + w["group"]; w["profile_key"] = "wesad/" + w["profile_key"]
    a = pd.concat([a, st, w], ignore_index=True)
    a["row_id"] = np.arange(len(a))
    return spec, a, _rename(prof)


def _sampler(a, seed=0, fold=0, exclude=None):
    ds = a["dataset_id"].to_numpy()
    train, _ = train_masks(ds, exclude or "none")
    sub = a[train & a["valid"].to_numpy()].reset_index(drop=True)
    return sub, PairSampler(PairIndex(sub), seed, fold)


def test_param_counts():
    assert n_params(1) == 2273 and n_params(2) == 2306


def test_pairs_same_profile(ann):
    _, a, _ = ann
    sub, s = _sampler(a)
    for _ in range(5):
        ra, rb, _ = s.sample_r()
        ta, tb = s.sample_t()
        assert (sub["profile_key"].to_numpy()[ra] == sub["profile_key"].to_numpy()[rb]).all()
        assert (sub["profile_key"].to_numpy()[ta] == sub["profile_key"].to_numpy()[tb]).all()


def test_pairs_gap(ann):
    _, a, _ = ann
    sub, s = _sampler(a)
    for _ in range(5):
        for x, y in (s.sample_r()[:2], s.sample_t()):
            same = sub["session_id"].to_numpy()[x] == sub["session_id"].to_numpy()[y]
            dt = np.abs(sub["t_rel"].to_numpy()[x] - sub["t_rel"].to_numpy()[y])
            assert (dt[same] >= 300).all()


def test_r_labels(ann):
    _, a, _ = ann
    sub, s = _sampler(a)
    ra, rb, _ = s.sample_r()
    ta, tb = s.sample_t()
    y = sub["y"].to_numpy()
    assert len(ra) and (y[ra] == 1).all() and (y[rb] == 0).all()
    assert len(ta) and (y[ta] == y[tb]).all() and (y[ta] >= 0).all()


def test_tm_flag(ann):
    _, a, _ = ann
    sub, _ = _sampler(a)
    ix = PairIndex(sub)
    for g in ix.all:
        for i in g.r_ok:
            cand, tm = _r_candidates(g, i)
            assert (g.ses[cand[tm]] != g.ses[i]).all() and (g.tbin[cand[tm]] == g.tbin[i]).all()


def test_lodo_exclusion(ann):
    spec, a, _ = ann
    ds = a["dataset_id"].to_numpy()
    train, teacher_train = train_masks(ds, "mpddf")
    assert not (ds[train] == "mpddf").any() and not (ds[teacher_train] == "mpddf").any()
    sub, s = _sampler(a, exclude="mpddf")
    assert "mpddf" not in set(sub["dataset_id"])
    names = feature_names(spec)
    f1 = FeatureScaler(names).fit(a[train & a["valid"].to_numpy()])
    f2 = FeatureScaler(names).fit(a[(ds != "mpddf") & (ds != "wesad") & a["valid"].to_numpy()])
    np.testing.assert_allclose(f1.mean, f2.mean)


def test_wesad_never_trained(ann):
    _, a, _ = ann
    ds = a["dataset_id"].to_numpy()
    for test in ("advitam", "dddb", "drozy", "mpddf"):
        train, teacher_train = train_masks(ds, test)
        assert not (ds[train] == "wesad").any() and not (ds[teacher_train] == "wesad").any()
    sub, s = _sampler(a)
    assert "wesad" not in s.ix.by_ds["S"] and "maus" in s.ix.by_ds["S"]


def test_pair_mask(ann):
    spec, *_ = ann
    names = feature_names(spec)
    net = PairNet("P1", 0, scale_groups(names), {"drop120": 0, "drop300": 0, "noise": 0})
    A = np.ones((4, 36)); B = np.ones((4, 36))
    A[0, 3] = np.nan; B[1, 5] = np.nan
    xa, xb = net._pair_inputs(A, B)
    assert xa[0, 3] == 0 and xb[0, 3] == 0 and xa[1, 5] == 0 and xb[1, 5] == 0 and xa[2, 3] == 1


def test_ref_score(ann):
    spec, *_ = ann
    net = PairNet("P2", 0, scale_groups(feature_names(spec)))
    X = np.random.default_rng(0).normal(size=(5, 36))
    X[0, :10] = np.nan
    np.testing.assert_allclose(net.score_diff(X, X.copy()), 0.0, atol=1e-6)


def test_m1_time_only(ann):
    _, a, _ = ann
    d = a[(a["y"] >= 0) & a["valid"] & (a["dataset_id"] == "mpddf")].copy()
    d["t_score"] = d["t_rel"]
    d["y_score"] = d["y"]
    m_t, _ = m1_dataset(d, "t_score")
    m_y, _ = m1_dataset(d, "y_score")
    assert 0.40 <= m_t <= 0.60 or np.isnan(m_t)
    assert m_y == pytest.approx(1.0)


def test_determinism(ann):
    spec, a, prof = ann
    sub, s1 = _sampler(a, seed=3, fold=1)
    _, s2 = _sampler(a, seed=3, fold=1)
    for _ in range(3):
        np.testing.assert_array_equal(s1.sample_r()[0], s2.sample_r()[0])
    names = feature_names(spec)
    sc = FeatureScaler(names).fit(sub)
    Z = sc.transform(sub[names].to_numpy())
    ref = np.zeros(len(sub), int)
    Zref = Z[:1]
    outs = []
    for _ in range(2):
        _, smp = _sampler(a, seed=1, fold=0)
        net = PairNet("P1", 1, scale_groups(names), {"steps": 15})
        net.fit_pairs(smp, Z, ref, Zref, sub["dataset_id"].to_numpy())
        outs.append(net.score(np.nan_to_num(Z[:20]))[0])
    np.testing.assert_allclose(outs[0], outs[1])
