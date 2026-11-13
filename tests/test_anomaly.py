"""생체 이상 점수 1단계 테스트 (docs/teacher-anomaly-stage1.md 7장)."""
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from driver_state.teacher.anomaly import _Score, predict, train_model, wesad_folds  # noqa: E402


def test_score_nonnegative_and_zero_at_baseline():
    for kind in ("lin", "mlp"):
        torch.manual_seed(0)
        m = _Score(kind).eval()
        Z = np.random.default_rng(0).normal(size=(50, 36)).astype(np.float32)
        s, _ = predict(m, Z)
        assert (s >= -1e-6).all()
        s0, _ = predict(m, np.zeros((3, 36), np.float32))
        np.testing.assert_allclose(s0, 0.0, atol=1e-3)


def test_wesad_folds_disjoint():
    f = wesad_folds([f"wesad/W{i:02d}" for i in range(15)])
    assert set(f.values()) == {0, 1, 2, 3}
    assert len(f) == 15


def test_distance_learns_deviation_both_directions():
    # 졸음 = 기준에서 양쪽 어느 방향으로든 멀어짐 → 방향 없는 거리가 둘 다 잡아야 한다
    rng = np.random.default_rng(1)
    n = 2000
    Z = rng.normal(0, 0.3, size=(n, 36)).astype(np.float32)
    y = rng.integers(0, 2, n)
    Z[y == 1, 0] += np.where(rng.random((y == 1).sum()) < 0.5, 2.0, -2.0)
    m = train_model("lin", Z, y, np.ones(n), seed=0, hp={"steps": 300})
    s, _ = predict(m, Z)
    up = (y == 1) & (Z[:, 0] > 1)
    down = (y == 1) & (Z[:, 0] < -1)
    assert s[up].mean() > s[y == 0].mean() and s[down].mean() > s[y == 0].mean()


def test_determinism():
    rng = np.random.default_rng(2)
    Z = rng.normal(size=(300, 36)).astype(np.float32)
    y = rng.integers(0, 2, 300)
    a = predict(train_model("mlp", Z, y, np.ones(300), 3, {"steps": 30}), Z)[1]
    b = predict(train_model("mlp", Z, y, np.ones(300), 3, {"steps": 30}), Z)[1]
    np.testing.assert_allclose(a, b)


def test_rr_distance_zero_when_window_equals_reference():
    from driver_state.teacher.anomaly_rr import Trainer
    rng = np.random.default_rng(0)
    T = np.stack([np.stack([800 + 30 * rng.standard_normal(1200), np.zeros(1200)]) for _ in range(3)]).astype(np.float16)
    refs = {"p": T[:1]}
    tr = Trainer("R2", 0)
    lg = tr.predict(T[:1], ["p"], refs, {"p": 800.0}, np.arange(1))
    d = (lg - float(tr.m.b)) / float(torch.nn.functional.softplus(tr.m.alpha))
    assert abs(d[0]) < 1e-2
