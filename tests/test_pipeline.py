import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from driver_state.data import build_windows  # noqa: E402
from driver_state.features import detect_ecg_peaks, extract_features  # noqa: E402
from driver_state.models import REGISTRY, build_model  # noqa: E402
from driver_state.normalization import InputPipeline  # noqa: E402
from driver_state.spec import InputSpec  # noqa: E402
from driver_state.synthetic import make_dataset  # noqa: E402

FAST = {"mlp": dict(epochs=3), "cnn_bilstm": dict(epochs=2), "random_forest": dict(n_estimators=10),
        "xgboost": dict(n_estimators=10), "lightgbm": dict(n_estimators=10)}


@pytest.fixture(scope="module")
def data():
    spec = InputSpec()
    ws = build_windows(make_dataset(3, 60.0), spec)
    return spec, ws


def test_window_shape(data):
    spec, ws = data
    assert ws.x_seq.shape[1:] == (len(spec.channels), spec.window_len)
    assert set(np.unique(ws.y)) <= set(range(spec.label.n_classes))
    assert len(np.unique(ws.groups)) == 3


def test_feature_order_and_values(data):
    spec, ws = data
    f = extract_features(ws, spec)
    assert f.shape == (len(ws), len(spec.feature_names))
    hr = f[:, spec.feature_names.index("ecg_hr_mean")]
    assert np.nanmedian(hr) == pytest.approx(70, abs=15)   # 합성 심박수 60~80+α


def test_ecg_peak_detection():
    from driver_state.synthetic import _beat_times, _ecg
    rng = np.random.default_rng(0)
    beats = _beat_times(30, 60, 1.0, 0.0, rng)
    peaks = detect_ecg_peaks(_ecg(beats, 3000, 100.0, rng), 100.0)
    assert abs(len(peaks) - len(beats)) <= 1


def test_normalization_roundtrip(data, tmp_path):
    spec, ws = data
    pipe = InputPipeline(spec).fit(ws)
    out = pipe.transform(ws)
    seq = out["sequence"]
    assert np.allclose(seq.mean(axis=2), 0, atol=1e-3)          # per-window z-score
    assert np.allclose(out["features"].mean(axis=0), 0, atol=1e-3)
    assert not np.isnan(out["features"]).any()
    pipe.save(tmp_path / "pipe.json")
    pipe2 = InputPipeline.load(tmp_path / "pipe.json")
    assert np.allclose(pipe2.transform(ws)["features"], out["features"])


def test_can_spec():
    spec = InputSpec(use_can=True)
    ws = build_windows(make_dataset(2, 60.0), spec)
    assert ws.x_seq.shape[1] == 6
    out = InputPipeline(spec).fit(ws).transform(ws)
    assert out["features"].shape[1] == len(spec.feature_names) == 14


@pytest.mark.parametrize("name", sorted(REGISTRY))
def test_model_contract(name, data, tmp_path):
    spec, ws = data
    out = InputPipeline(spec).fit(ws).transform(ws)
    K = spec.label.n_classes
    m = build_model(name, K, **FAST[name])
    X = out[m.input_kind]
    m.fit(X, ws.y, X[:20], ws.y[:20])
    p = m.predict_proba(X)
    assert p.shape == (len(X), K)
    assert np.allclose(p.sum(axis=1), 1, atol=1e-5)
    m.save(tmp_path / name)
    m2 = type(m).load(tmp_path / name)
    assert np.allclose(m2.predict_proba(X), p, atol=1e-5)
    assert m.size_bytes() > 0


def test_missing_class_in_train(data):
    """학습셋에 없는 클래스가 있어도 출력 shape 은 (N, n_classes)."""
    spec, ws = data
    out = InputPipeline(spec).fit(ws).transform(ws)["features"]
    keep = ws.y != 3
    for name in ("random_forest", "xgboost", "lightgbm"):
        m = build_model(name, 4, **FAST[name]).fit(out[keep], ws.y[keep])
        p = m.predict_proba(out)
        assert p.shape == (len(out), 4) and np.allclose(p[:, 3], 0)


@pytest.mark.parametrize("name,shape", [("mlp", (8,)), ("cnn_bilstm", (2, 3000))])
def test_onnx_export(name, shape, tmp_path):
    ort = pytest.importorskip("onnxruntime")
    pytest.importorskip("onnx")
    import torch
    X = np.random.default_rng(0).standard_normal((16, *shape)).astype(np.float32)
    m = build_model(name, 4, epochs=1).fit(X, np.arange(16) % 4)
    m.export_onnx(tmp_path / "m.onnx")
    sess = ort.InferenceSession(str(tmp_path / "m.onnx"))
    for b in (1, 4):
        ref = m.net(torch.from_numpy(X[:b])).detach().numpy()
        assert np.allclose(sess.run(None, {"x": X[:b]})[0], ref, atol=1e-4)
