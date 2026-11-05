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
from driver_state.spec import PERSONAL_FEATURES, InputSpec  # noqa: E402
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
    m = build_model("cnn_bilstm", 4, epochs=1).fit(model_input(out, "sequence+personal"), ws.y)
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
    keep = ws.y != 3
    for name in ("random_forest", "xgboost", "lightgbm"):
        m = build_model(name, 4, **FAST[name]).fit(out[keep], ws.y[keep])
        p = m.predict_proba(out)
        assert p.shape == (len(out), 4) and np.allclose(p[:, 3], 0)


@pytest.mark.parametrize("name,shapes", [("mlp", [(15,)]), ("cnn_bilstm", [(2, 3000), (7,)])])
def test_onnx_export(name, shapes, tmp_path):
    ort = pytest.importorskip("onnxruntime")
    pytest.importorskip("onnx")
    import torch
    rng = np.random.default_rng(0)
    X = tuple(rng.standard_normal((16, *s)).astype(np.float32) for s in shapes)
    X = X if len(X) > 1 else X[0]
    m = build_model(name, 4, epochs=1).fit(X, np.arange(16) % 4)
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
    m = build_model("lightgbm", 4, n_estimators=10).fit(out["features"], ws.y)
    save_bundle(tmp_path / "b", pipe, m)
    pred = StatePredictor(tmp_path / "b")
    rng = np.random.default_rng(5)
    prof = calibrate(make_recording("p", [(None, 30.0)], 80, 1.0, rng), spec)
    assert prof.valid
    replay = make_recording("p", [(0, 60.0)], 80, 1.0, rng)
    pred.set_profile(prof)
    res = pred.predict_recording(replay)
    assert len(res) == 4 and all(r.baseline_valid for r in res)
    pred.set_profile(None)   # 미등록 사용자
    assert not any(r.baseline_valid for r in pred.predict_recording(replay))
