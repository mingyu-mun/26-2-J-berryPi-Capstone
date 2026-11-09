"""교사 파이프라인 테스트 (합성 데이터, 몇십 초)."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from driver_state.data_io import load_session, load_tree, write_session  # noqa: E402
from driver_state.teacher.bundle import TeacherBundle, fit_final, label_tree  # noqa: E402
from driver_state.teacher.calibrate import ece, fit_temperature  # noqa: E402
from driver_state.teacher.evaluate import choose_best, make_input, metrics, run_cv  # noqa: E402
from driver_state.teacher.features import build_tables, build_tachograms  # noqa: E402
from driver_state.teacher.models import sample_weights  # noqa: E402
from driver_state.teacher.personal import PersonalNormalizer  # noqa: E402
from driver_state.teacher.rr import RRSeries, extract_rr  # noqa: E402
from driver_state.teacher.spec import TREND_FEATURES, LabelRule, TeacherSpec  # noqa: E402
from driver_state.teacher.synthetic import write_synthetic  # noqa: E402
from driver_state.teacher.tau import apply_tau, choose_tau  # noqa: E402

FAST = {"rrnet": {"epochs": 1, "batch_size": 128}}


@pytest.fixture(scope="module")
def tree(tmp_path_factory):
    return write_synthetic(tmp_path_factory.mktemp("syn"), n_subjects=3, minutes=12, seed=1, fs_override=250.0)


@pytest.fixture(scope="module")
def tables(tree):
    spec = TeacherSpec()
    rrs = [extract_rr(r, spec) for r in load_tree(tree, channels=("ecg",))]
    win, prof = build_tables(rrs, spec)
    return spec, rrs, win, prof, build_tachograms(rrs, win, spec)


# ---------------- 로더 v0.3
def test_loader_ecg_only_and_meta(tmp_path):
    d = write_session(tmp_path / "ds/S1/a", {"dataset_id": "ds", "subject_id": "S1", "session_id": "a", "t0": 0,
                                             "fs": {"ecg": 250}, "drowsiness_inducer": "none", "baseline_range": [0, 60]},
                      {"ecg": np.random.default_rng(0).standard_normal(2500)})
    rec = load_session(d)
    assert set(rec.signals) == {"ecg"} and rec.meta["drowsiness_inducer"] == "none"
    assert rec.meta["baseline_range"] == [0, 60]
    with pytest.raises(KeyError):
        load_session(d, channels=("ecg", "ppg"))


def test_v02_folder_still_loads(tmp_path):
    d = tmp_path / "s"
    d.mkdir()
    (d / "meta.json").write_text(json.dumps({"dataset_id": "x", "subject_id": "U", "session_id": "s", "t0": 0,
                                             "fs": {"ecg": 100, "ppg": 50}}))
    pd.DataFrame({"value": np.zeros(100)}).to_csv(d / "ecg.csv", index=False)
    pd.DataFrame({"value": np.zeros(50)}).to_csv(d / "ppg.csv", index=False)
    assert set(load_session(d).signals) == {"ecg", "ppg"}


# ---------------- RR · 창 · 특징
def test_rr_detection_matches_synthetic_rate(tables):
    _, rrs, *_ = tables
    for r in rrs:
        assert r.abnormal.mean() < 0.02
        assert 0.6 < np.median(r.rr) < 1.15        # 합성 심박 52–100 bpm


def test_rr_cache_roundtrip(tables, tmp_path):
    r = tables[1][0]
    r.save(tmp_path / "x.npz")
    r2 = RRSeries.load(tmp_path / "x.npz")
    np.testing.assert_allclose(r.beat_t, r2.beat_t)
    assert r2.meta == r.meta and r2.group == r.group


def test_window_grid_and_baseline_exclusion(tables):
    spec, rrs, win, prof, _ = tables
    assert np.allclose(win["t_rel"] % spec.stride_sec, 0)              # 학생과 같은 10초 격자
    assert not (win["session_id"] == "rest").any()                      # 휴식 세션은 기준 전용
    m = win[win["dataset_id"] == "syn_mpddf"]
    assert not ((m["t_rel"] < 360) & (m["t_rel"] + spec.window_sec > 60)).any()   # baseline_range 제외
    a = win[win["dataset_id"] == "syn_dddb"]
    for _, g in a.groupby("session_id"):                                 # baseline_per_session: 녹화마다 앞 5분 제외
        assert g["t_rel"].min() >= spec.baseline_sec
    assert set(prof["baseline_valid"]) == {1.0}
    assert win[win["dataset_id"] == "syn_dddb"]["profile_key"].nunique() == 2 * 3   # 3명 × 녹화 2회


def test_feature_columns_and_no_shortcuts(tables):
    spec, _, win, *_ = tables
    for n in spec.raw_feature_names:
        assert n in win
    forbidden = {"t_rel", "t_start", "dataset_id", "inducer", "coverage_30"}
    assert not forbidden & set(spec.raw_feature_names)
    assert "lf_hf_log_30" not in spec.raw_feature_names and "lf_hf_log_120" in spec.raw_feature_names
    assert win["hr_30"].notna().mean() > 0.95


def test_personal_z_and_invalid(tables):
    spec, _, win, prof, _ = tables
    norm = PersonalNormalizer(spec).fit(win)
    force = np.zeros(len(win), bool)
    force[:5] = True
    P = norm.transform(win, prof, force)
    assert P["baseline_valid"].iloc[:5].eq(0).all() and P["baseline_valid"].iloc[5:].eq(1).all()
    assert P.iloc[:5].drop(columns="baseline_valid").isna().all().all()
    from driver_state.teacher.personal import pers_columns
    assert pers_columns(P.columns, "signed") == list(spec.personal_feature_names())
    assert {"dev_30", "dev_120", "dev_300", "dev_core", "az_hr_120"} <= set(P.columns)
    assert (P["az_hr_120"].dropna() >= 0).all()
    assert not set(TREND_FEATURES) & {c[2:] for c in P.columns}


# ---------------- 가중치 · 보정 · τ
def test_sample_weights_balance():
    d = np.array(["a"] * 90 + ["b"] * 10)
    g = np.array(["a1"] * 60 + ["a2"] * 30 + ["b1"] * 10)
    y = np.array(([0, 1] * 50))
    w = sample_weights(d, g, y)
    assert np.isclose(w[y == 1].sum(), w[y == 0].sum())
    assert abs(w[d == "a"].sum() - w[d == "b"].sum()) / w.sum() < 0.1


def test_temperature_recovers_scale():
    rng = np.random.default_rng(0)
    z = rng.standard_normal(5000) * 2
    y = (rng.random(5000) < 1 / (1 + np.exp(-z))).astype(int)
    assert abs(fit_temperature(3 * z, y) - 3) < 0.4
    assert ece(np.full(100, 0.5), np.r_[np.zeros(50), np.ones(50)]) < 1e-9


def test_tau_choice_and_apply():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 4000)
    p = np.clip(y * 0.6 + rng.random(4000) * 0.4, 0, 1)
    oof = pd.DataFrame({"dataset_id": np.repeat(["a", "b"], 2000), "y": y, "p_m": p})
    t = choose_tau(oof, "p_m", 0.9, 0.9)
    assert t["achieved"] and t["tau_lo"] < t["tau_hi"]
    lab = apply_tau(np.array([0.0, 0.5, 0.99]), t["tau_hi"], t["tau_lo"], np.array([True, True, False]))
    assert lab[0] == 0 and lab[2] == -1


def test_label_rule():
    r = LabelRule()
    assert [r.kss(v) for v in (3, 5, 6, 7, 9)] == [0, 0, -1, 1, 1]
    assert [r.fatigue(v) for v in range(5)] == [0, -1, 1, 1, 1]


# ---------------- end-to-end
def test_lodo_end_to_end_and_bundle(tables, tree, tmp_path):
    spec, rrs, win, prof, tacho = tables
    models = ["rule", "lgbm", "lgbm_nopers", "rrnet", "ensemble"]
    oof, res = run_cv(win, prof, spec, models, tacho, "lodo", seed=0, params=FAST, log=lambda *_: None)
    assert set(oof["fold"]) <= set(win["dataset_id"]) and len(oof)
    for m in models:
        assert oof[f"p_{m}"].between(0, 1).all()
    summ = metrics(oof, models)
    assert choose_best(summ) in {"lgbm", "lgbm_nopers", "rrnet", "ensemble"}
    # 테스트 참가자는 학습에 없어야 한다
    assert oof.groupby("fold")["dataset_id"].nunique().eq(1).all()

    b = fit_final(win, prof, spec, "ensemble", tacho, oof, params=FAST, log=lambda *_: None)
    b.tau = {"tau_hi": 0.8, "tau_lo": 0.2}
    b.save(tmp_path / "bundle")
    b2 = TeacherBundle.load(tmp_path / "bundle")
    X = make_input(win.head(50), prof, spec, b.norm, tacho[:50])
    np.testing.assert_allclose(b.model.predict_proba(X), b2.model.predict_proba(X), rtol=1e-5, atol=1e-6)

    out = tmp_path / "labels"
    lw = label_tree(b2, [tree / "syn_dddb"], out, log=lambda *_: None)
    f = out / "syn_dddb" / "S00" / "rec1"
    tp, lab = pd.read_csv(f / "teacher_p.csv"), pd.read_csv(f / "labels_window.csv")
    assert len(tp) == len(lab) and set(lab["label"]) <= {-1, 0, 1}
    assert json.loads((f / "teacher_label.json").read_text())["tau_hi"] == 0.8
    assert lw["p_drowsy"].between(0, 1).all()
