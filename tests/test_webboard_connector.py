"""웹보드 커넥터가 만든 이벤트가 대시보드 규약(v1.2)을 통과하는지."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
pytest.importorskip("jsonschema")

from driver_state.state_machine_v2 import StateMachineV2, V2Config, WindowIn  # noqa: E402
from driver_state.web_dashboard import Dashboard  # noqa: E402
from driver_state.webboard_connector import ai_decision, bio_wave, system_status, wave_channel  # noqa: E402


def test_events_pass_contract():
    d = Dashboard()
    sm = StateMachineV2(V2Config())
    ev = [system_status("s1", 0.0, bio_status="collecting", clean_windows=3, bio_elapsed_s=0.0),
          bio_wave("s1", 1.0, ecg=wave_channel(0.0, 100.0, [0.1] * 100, "a.u.", "common_preprocessed", True), ppg=None)]
    for i in range(12):
        w = WindowIn(t=330.0 + 10 * i, p_ep=0.8, q_ecg=0.01, p_ppg=None, q_ppg=None, hr_ecg=70.0, hr_ppg=None,
                     ok_ecg=True, ok_ppg=False)
        ev.append(ai_decision("s1", w, sm.step(w), baseline_status="ready", bio_origin="recorded", origin="recorded"))
    r = d.ingest(ev)
    assert r["accepted"] == len(ev)
    levels = [e["payload"]["output"]["level"] for e in ev if e["type"] == "ai.decision"]
    assert max(levels) >= 1


def test_output_null_before_baseline_and_nan_cleaned():
    w = WindowIn(t=40.0, p_ep=float("nan"), q_ecg=0.0)
    e = ai_decision("s2", w, StateMachineV2().step(w), baseline_status="collecting")
    assert e["payload"]["output"] is None and e["payload"]["input"]["p_ep"] is None
    Dashboard().ingest(e)
    assert wave_channel(0.0, 100.0, [1.0, float("inf")], "a.u.", "common_preprocessed", None) is None
