"""어댑터 공용 함수 테스트 (원본 데이터 없이)."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from driver_state.adapters.common import finite_segments  # noqa: E402
from driver_state.adapters.dddb import event_labels  # noqa: E402
from driver_state.adapters.mand import dehold  # noqa: E402
from driver_state.adapters.mpddf import per_second_labels  # noqa: E402
from driver_state.teacher.spec import LabelRule  # noqa: E402


def test_mpddf_change_points_hold_until_next():
    import pandas as pd
    ann = pd.DataFrame({"t": [10.0, 40.0, 70.0, 90.0], "value": ["0", "3", "Severe Artifacts", "1"]})
    lab = per_second_labels(ann, 100, LabelRule())
    assert (lab[:10] == -1).all() and (lab[10:40] == 0).all() and (lab[40:70] == 1).all()
    assert (lab[70:] == -1).all()


def test_dddb_event_labels():
    rule = LabelRule(event_drowsy_sec=60, event_alert_gap_sec=600)
    lab = event_labels(np.array([1000.0]), 2000, rule)
    assert lab[1000] == 1 and lab[941] == 1 and lab[900] == -1 and lab[300] == 0 and lab[1700] == 0


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
