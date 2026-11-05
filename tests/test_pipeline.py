import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from driver_state.baseline import Profile, build_dataset, calibrate, personal_matrix  # noqa: E402
from driver_state.data import Recording, build_windows  # noqa: E402
from driver_state.features import detect_ecg_peaks, extract_features  # noqa: E402
from driver_state.inference import StatePredictor, save_bundle  # noqa: E402
from driver_state.models import REGISTRY, build_model, model_input  # noqa: E402
from driver_state.normalization import InputPipeline  # noqa: E402
from driver_state.preprocessing import prepare_channel  # noqa: E402
from driver_state.spec import PERSONAL_FEATURES, InputSpec, StateMachineSpec  # noqa: E402
from driver_state.state_machine import DrowsinessStateMachine, judge_windows, timeline_metrics  # noqa: E402
from driver_state.synthetic import make_dataset, make_recording, make_subject  # noqa: E402

FAST = {"mlp": dict(epochs=3), "cnn_bilstm": dict(epochs=2), "random_forest": dict(n_estimators=10),
        "xgboost": dict(n_estimators=10), "lightgbm": dict(n_estimators=10)}


@pytest.fixture(scope="module")
def data():
    spec = InputSpec()
    ws, profiles = build_dataset(make_dataset(3, 60.0), spec)
    return spec, ws, profiles


# ---------- 입력 규약 ----------

def test_window_shape_and_baseline_exclusion(data):
    spec, ws, profiles = data
    assert ws.x_seq.shape[1:] == (len(spec.channels), spec.window_len)
    assert set(np.unique(ws.y)) <= set(range(spec.label.n_classes))
    # 기준 구간(앞 30초)과 겹치는 윈도우는 없어야 한다
    assert ws.t_start.min() >= spec.baseline.calibration_sec
    assert ws.quality.all()


def test_feature_layout(data):
    spec, ws, profiles = data
    assert spec.feature_names[8:] == PERSONAL_FEATURES
    assert len(spec.feature_names) == 15
    f = extract_features(ws, spec)
    assert f.shape == (len(ws), 8)
    hr = f[:, spec.feature_names.index("ecg_hr_mean")]
    assert np.nanmedian(hr) == pytest.approx(70, abs=15)


def test_ecg_peak_detection():
    from driver_state.synthetic import _beat_times, _ecg
    rng = np.random.default_rng(0)
    beats = _beat_times(30, 60, 1.0, 0.0, rng)
    peaks = detect_ecg_peaks(_ecg(beats, 3000, 100.0, rng), 100.0)
    assert abs(len(peaks) - len(beats)) <= 1


# ---------- 전처리 ----------

def test_antialias_removes_mains_noise():
    """250 Hz ECG 의 60 Hz 전원 잡음이 100 Hz 로 줄일 때 40 Hz 로 접혀 들어오지 않아야 한다."""
    spec = InputSpec()
    fs = 250.0
    t = np.arange(int(30 * fs)) / fs
    x = np.sin(2 * np.pi * 60 * t)                     # 순수 60 Hz 잡음
    filt, _ = prepare_channel(x, fs, "ecg", spec)
    naive, _ = prepare_channel(x, fs, "ecg", spec, prefiltered=True)  # 필터 없이 리샘플만
    assert np.std(filt) < 0.02 * np.std(naive)


def test_quality_flags_flat_signal():
    spec = InputSpec()
    rec = make_subject("Q", 60.0, rest_sec=0, seed=1)
    rec.signals["ppg"] = np.full_like(rec.signals["ppg"], 512.0)   # 접촉 불량
    ws = build_windows([rec], spec)
    assert not ws.quality.any()


# ---------- Baseline ----------

def test_profiles_and_personal_features(data):
    spec, ws, profiles = data
    assert all(p.valid for p in profiles.values())
    pipe = InputPipeline(spec).fit(ws, None, profiles)
    assert set(pipe.sigma_pop) == {"ecg_hr_mean", "ecg_rmssd", "ppg_pr_mean"}
    base = extract_features(ws, spec)
    pers = personal_matrix(base, ws.groups, profiles, pipe.sigma_pop, spec)
    assert pers.shape == (len(ws), 7) and (pers[:, -1] == 1).all()
    g = ws.groups[0]
    i = np.where(ws.groups == g)[0][0]
    assert pers[i, 0] == pytest.approx(base[i, 0] - profiles[g].m["ecg_hr_mean"], abs=1e-3)


def test_missing_subject_id_gives_invalid_baseline():
    spec = InputSpec()
    rec = make_subject(None, 60.0, seed=2)
    ws, profiles = build_dataset([rec], spec)
    (p,) = profiles.values()
    assert not p.valid
    pipe = InputPipeline(spec).fit(ws, None, profiles)
    out = pipe.transform(ws, None, profiles)
    assert (out["personal"][:, -1] == 0).all()


def test_calibration_quality_rejects_bad_signal():
    spec = InputSpec()
    rng = np.random.default_rng(0)
    rec = make_recording("u", [(None, 30.0)], 70, 1.0, rng)
    assert calibrate(rec, spec).valid
    rec.signals["ecg"] = rng.standard_normal(len(rec.signals["ecg"])) * 0.01   # 박동 없음
    p = calibrate(rec, spec)
    assert not p.valid and p.reason
    short = make_recording("u", [(None, 10.0)], 70, 1.0, rng)
    assert not calibrate(short, spec).valid


def test_invalid_rate_and_mask_not_normalized(data, tmp_path):
    spec, ws, profiles = data
    pipe = InputPipeline(spec).fit(ws, None, profiles)
    out = pipe.transform(ws, None, profiles, invalid_rate=0.5, seed=0)
    mask = out["features"][:, -1]
    assert set(np.unique(mask)) == {0.0, 1.0}
    assert 0.2 < mask.mean() < 0.8
    pipe.save(tmp_path / "pipe.json")
    pipe2 = InputPipeline.load(tmp_path / "pipe.json")
    assert pipe2.sigma_pop == pytest.approx(pipe.sigma_pop)
    assert np.allclose(pipe2.transform(ws, None, profiles)["features"], pipe.transform(ws, None, profiles)["features"])


def test_profile_json_roundtrip(tmp_path):
    p = Profile("u", True, {"ecg_hr_mean": 70.0, "ecg_rmssd": 40.0, "ppg_pr_mean": 70.5}, None, 1, "", "t")
    p.save(tmp_path / "p.json")
    assert Profile.load(tmp_path / "p.json") == p


def test_no_personal_mode(data):
    spec, ws, profiles = data
    spec2 = InputSpec(use_personal=False)
    out = InputPipeline(spec2).fit(ws, None, profiles).transform(ws, None, profiles)
    assert out["features"].shape[1] == 8 and out["personal"].shape[1] == 0
    m = build_model("cnn_bilstm", 2, epochs=1).fit(model_input(out, "sequence+personal"), ws.y)
    assert m.net.personal_dim == 0


# ---------- 모델 계약 ----------

@pytest.mark.parametrize("name", sorted(REGISTRY))
def test_model_contract(name, data, tmp_path):
    spec, ws, profiles = data
    out = InputPipeline(spec).fit(ws, None, profiles).transform(ws, None, profiles)
    K = spec.label.n_classes
    m = build_model(name, K, **FAST[name])
    X = model_input(out, m.input_kind)
    Xv = tuple(a[:20] for a in X) if isinstance(X, tuple) else X[:20]
    m.fit(X, ws.y, Xv, ws.y[:20])
    p = m.predict_proba(X)
    assert p.shape == (len(ws), K)
    assert np.allclose(p.sum(axis=1), 1, atol=1e-5)
    m.save(tmp_path / name)
    m2 = type(m).load(tmp_path / name)
    assert np.allclose(m2.predict_proba(X), p, atol=1e-5)
    assert m.size_bytes() > 0


def test_missing_class_in_train(data):
    spec, ws, profiles = data
    out = InputPipeline(spec).fit(ws, None, profiles).transform(ws, None, profiles)["features"]
    keep = ws.y != 1
    for name in ("random_forest", "xgboost", "lightgbm"):
        m = build_model(name, 2, **FAST[name]).fit(out[keep], ws.y[keep])
        p = m.predict_proba(out)
        assert p.shape == (len(out), 2) and np.allclose(p[:, 1], 0)


@pytest.mark.parametrize("name,shapes", [("mlp", [(15,)]), ("cnn_bilstm", [(2, 3000), (7,)])])
def test_onnx_export(name, shapes, tmp_path):
    ort = pytest.importorskip("onnxruntime")
    pytest.importorskip("onnx")
    import torch
    rng = np.random.default_rng(0)
    X = tuple(rng.standard_normal((16, *s)).astype(np.float32) for s in shapes)
    X = X if len(X) > 1 else X[0]
    m = build_model(name, 2, epochs=1).fit(X, np.arange(16) % 2)
    m.export_onnx(tmp_path / "m.onnx")
    sess = ort.InferenceSession(str(tmp_path / "m.onnx"))
    names = [i.name for i in sess.get_inputs()]
    assert names == ["x", "p"][:len(shapes)]
    Xs = X if isinstance(X, tuple) else (X,)
    for b in (1, 4):
        feed = {n: a[:b] for n, a in zip(names, Xs)}
        ref = m.net(*[torch.from_numpy(a[:b]) for a in Xs]).detach().numpy()
        assert np.allclose(sess.run(None, feed)[0], ref, atol=1e-4)


# ---------- 시연 추론 ----------

def test_demo_inference_flow(data, tmp_path):
    spec, ws, profiles = data
    pipe = InputPipeline(spec).fit(ws, None, profiles)
    out = pipe.transform(ws, None, profiles, invalid_rate=0.2)
    m = build_model("lightgbm", 2, n_estimators=10).fit(out["features"], ws.y)
    save_bundle(tmp_path / "b", pipe, m)
    pred = StatePredictor(tmp_path / "b")
    rng = np.random.default_rng(5)
    prof = calibrate(make_recording("p", [(None, 30.0)], 80, 1.0, rng), spec)
    assert prof.valid
    replay = make_recording("p", [(0, 60.0)], 80, 1.0, rng)
    pred.set_profile(prof)
    res = pred.predict_recording(replay)
    assert len(res) == 4 and all(r.baseline_valid for r in res)
    assert all(r.valid and r.level in (0, 1, 2) and len(r.proba) == 2 for r in res)
    pred.set_profile(None)   # 미등록 사용자
    assert not any(r.baseline_valid for r in pred.predict_recording(replay))


def test_handoff_folder_loader(tmp_path):
    import json
    from driver_state.data_io import load_tree
    rng = np.random.default_rng(0)
    rec = make_recording("User1", [(None, 30.0), (0, 60.0)], 72, 1.0, rng)
    d = tmp_path / "jberry_sensor" / "User1" / "s1"
    d.mkdir(parents=True)
    for k in ("ecg", "ppg"):
        np.savetxt(d / f"{k}.csv", rec.signals[k], header="value", comments="", delimiter=",")
    np.savetxt(d / "labels.csv", rec.labels, header="label", comments="", fmt="%d")
    (d / "meta.json").write_text(json.dumps({
        "dataset_id": "jberry_sensor", "subject_id": "User1", "session_id": "s1",
        "t0": "2026-11-20T14:03:12+09:00", "fs": {"ecg": 250, "ppg": 50}, "label_fs": 20,
        "is_rest": False, "prefiltered": False}))
    (loaded,) = load_tree(tmp_path)
    assert loaded.subject_id == "User1" and loaded.fs["ecg"] == 250
    ws, profiles = build_dataset([loaded], InputSpec())
    assert profiles["User1"].valid and len(ws) > 0


# ---------- 윈도우 판정 · 상태 머신 ----------

def test_window_gate_marks_bad_windows(data):
    spec, ws, profiles = data
    base = extract_features(ws, spec)
    ok = judge_windows(ws.x_seq, base, ws.quality, ws.channels, spec)
    assert sum(r == "" for r in ok) >= 0.9 * len(ws)          # 합성 정상 신호는 대부분 통과
    q = ws.quality.copy(); q[0] = False
    b = base.copy(); b[1, 5] = np.nan; b[2, 5] = b[2, 0] + 30   # NaN, |HR-PR| 30 bpm
    flat = ws.x_seq.copy(); flat[3, 0] = 0.0                      # ECG 박동 없음
    r = judge_windows(flat, b, q, ws.channels, spec)
    assert "품질" in r[0] and "특징" in r[1] and "HR-PR" in r[2] and "유효 RR" in r[3]


def test_state_machine_escalates_and_recovers():
    sm = DrowsinessStateMachine(StateMachineSpec(), stride_sec=10.0)
    levels = [sm.step(10.0 * (i + 1), p).level for i, p in enumerate([0.1, 0.9, 0.9, 0.9, 0.5, 0.1, 0.1, 0.1])]
    # s: .1 .5 .7 .8 .65 .375 .24 .17 → 정상, 정상(히스테리시스), 주의, 졸음, 졸음(유지), 주의, 정상, 정상
    assert levels == [0, 0, 1, 2, 2, 1, 0, 0]


def test_state_machine_skips_short_invalid_and_resets_long():
    sm = DrowsinessStateMachine(StateMachineSpec(invalid_hold_sec=30.0), stride_sec=10.0)
    for i, p in enumerate([0.9, 0.9, 0.9]):
        sm.step(10.0 * (i + 1), p)
    assert sm.level == 2
    s1 = sm.step(40.0, None); s2 = sm.step(50.0, None)        # -1 20초: 직전 단계 유지
    assert s1.skipped and s1.level == 2 and s2.level == 2
    s3 = sm.step(60.0, None)                                    # 30초 연속 → 측정 불가 + 초기화
    assert s3.level == -1 and s3.level_name == "측정 불가"
    s4 = sm.step(70.0, 0.1)
    assert s4.level == 0 and s4.score == pytest.approx(0.1)


def test_timeline_metrics():
    t = np.arange(1, 9) * 10.0
    truth = np.array([0, 0, 1, 1, 1, 0, 0, 0])
    lv = np.array([0, 1, 1, 2, 2, 1, 0, -1])
    m = timeline_metrics(t, lv, truth)
    assert m["detection_delay_sec"] == [10.0] and m["missed"] == 0
    assert m["false_alarms"] == 1 and m["unavailable_ratio"] == pytest.approx(1 / 8)


def test_predictor_marks_contact_loss(data, tmp_path):
    spec, ws, profiles = data
    pipe = InputPipeline(spec).fit(ws, None, profiles)
    m = build_model("lightgbm", 2, n_estimators=10).fit(pipe.transform(ws, None, profiles)["features"], ws.y)
    save_bundle(tmp_path / "b", pipe, m)
    pred = StatePredictor(tmp_path / "b")
    rng = np.random.default_rng(7)
    rec = make_recording("p", [(0, 120.0)], 75, 1.0, rng)
    rec.signals["ppg"][: int(50 * 50)] = 512.0                  # 앞 50초 PPG 접촉 불량
    res = pred.predict_recording(rec)
    assert not res[0].valid and res[0].state == -1 and res[0].proba == []
    assert res[-1].valid


def test_quality_flags_partial_dropout():
    """윈도우 일부(8초)만 값이 멈춰도 품질 불량. 업샘플 overshoot 때문에 포화 검사로는 못 잡는 경우."""
    spec = InputSpec()
    rec = make_subject("Q", 60.0, rest_sec=0, seed=3)
    fs = rec.fs["ppg"]
    rec.signals["ppg"][int(5 * fs):int(13 * fs)] = 0.0
    ws = build_windows([rec], spec)
    assert not ws.quality[ws.t_start < 13].any() and ws.quality[ws.t_start >= 13].all()
