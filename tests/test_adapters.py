"""어댑터 공용 함수 테스트 (원본 데이터 없이)."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from driver_state.adapters.common import finite_segments  # noqa: E402
from driver_state.adapters.mand import dehold  # noqa: E402


def test_finite_segments_interpolates_short_and_splits_long():
    fs = 100.0
    x = np.arange(20000, dtype=float)
    y = x.copy()
    x[:50] = np.nan                    # 앞 결측: 잘라냄
    x[5000:5010] = np.nan              # 0.1 s: 보간
    x[10000:10100] = np.nan            # 1 s: 분리
    y[12000] = np.nan                  # 다른 채널 결측도 함께 본다
    segs = finite_segments({"a": x, "b": y}, fs, min_seg_sec=10)
    assert [a for a, _ in segs] == [50, 10100]
    s0 = segs[0][1]["a"]
    assert len(s0) == 10000 - 50 and np.isfinite(s0).all()
    assert np.allclose(s0[5000 - 50:5010 - 50], np.arange(5000, 5010))
    assert np.isfinite(segs[1][1]["b"]).all()


def test_finite_segments_drops_short():
    x = np.ones(500)
    assert finite_segments({"a": x}, 100.0, min_seg_sec=10) == []


def test_dehold_linear_between_changes():
    x = np.array([0, 0, 0, 3, 3, 3, 6, np.nan])
    out = dehold(x)
    assert np.allclose(out[:7], [0, 1, 2, 3, 4, 5, 6])
    assert np.isnan(out[7])
