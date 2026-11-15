"""Dashboard transport: preserve state semantics, reject unsafe/missing inputs."""
import json
import sys
import threading
from dataclasses import asdict
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from driver_state.state_machine_v2 import StateMachineV2, WindowIn
from driver_state.web_dashboard import APIError, Dashboard, create_server, parse_window


def live_window(t, **kw):
    return {"t": t, "p_ep": .82, "q_ecg": .02, "ok_ecg": True, **kw}


def replay_csv(times=(0, 10, 20), sessions=None):
    head = "session_id,t_rel,student_S1-EP_p,q_abn,valid\n"
    return head + "\n".join(f"{sessions[i] if sessions else 'a'},{t},0.82,0.02,true" for i, t in enumerate(times))


def test_dashboard_uses_original_python_state_machine():
    dash, ref = Dashboard(), StateMachineV2()
    dash.control({"action": "settings", "scenario": "drowsy", "can_scenario": "confirm"})
    for _ in range(50):
        status = dash.control({"action": "step"})
        original = asdict(ref.step(WindowIn(**status["input"])))
        assert status["output"] == original
    assert status["output"]["level"] == 2
    assert len(status["trend"]) == 36


def test_missing_quality_or_status_never_becomes_valid():
    dash = Dashboard("live")
    initial = dash.status()
    assert initial["output"] is None and initial["connection"]["stale"]
    for t in range(10, 70, 10):
        out = dash.window({"t": t, "p_ep": .2})
        assert out["output"]["channel"] == "none"
    assert out["output"]["level"] == -1
    assert not parse_window({"t": 1, "p_ep": .2, "q_ecg": .02}).ok_ecg


def test_live_stale_clears_display_and_restarts_evidence():
    now = [0.0]
    dash = Dashboard("live", clock=lambda: now[0])
    for t in (10, 20, 30):
        dash.window(live_window(t))
    now[0] = 21
    stale = dash.status()
    assert stale["connection"]["stale"]
    assert stale["output"] is None and stale["input"] is None
    resumed = dash.window(live_window(40))
    assert resumed["output"]["level"] == 0
    assert len(resumed["trend"]) == 1


def test_live_gaps_reset_and_fast_duplicate_nonfinite_windows_reject():
    dash = Dashboard("live")
    for t in (10, 20, 30):
        dash.window(live_window(t))
    assert dash.window(live_window(50))["output"]["level"] == 0
    for t in (50, 49, 55, 61):
        with pytest.raises(APIError):
            dash.window(live_window(t))
    for changes in ({"p_ep": float("nan")}, {"q_ecg": -1}, {"q_ecg": .5, "ok_ecg": "true"},
                    {"t": True}, {"hr_ecg": 0}, {"can_drowsy": 1}, {"p_ep": 1.01}):
        with pytest.raises(APIError):
            dash.window({**live_window(60), **changes})


def test_csv_keeps_time_and_never_bridges_session_or_missing_window():
    dash = Dashboard()
    dash.replay({"name": "sample.csv", "csv": replay_csv((100, 110, 120, 0), ("a", "a", "a", "b"))})
    first = dash.control({"action": "step"})
    assert first["output"]["t"] == 130
    dash.control({"action": "step"}); dash.control({"action": "step"})
    end = dash.control({"action": "step"})
    assert end["output"]["t"] == 30 and end["output"]["level"] == 0
    assert len(end["trend"]) == 1
    with pytest.raises(APIError):
        dash.control({"action": "step"})
    assert dash.status()["csv"]["position"] == 4
    dash.replay({"csv": replay_csv((0, 10, 20, 40))})
    for _ in range(4):
        out = dash.control({"action": "step"})
    assert out["output"]["level"] == 0 and len(out["trend"]) == 1


def test_csv_missing_quality_and_loop_do_not_invent_evidence():
    dash = Dashboard()
    dash.replay({"csv": "t_rel,student_S1-EP_p\n0,.9\n"})
    assert dash.control({"action": "step"})["output"]["channel"] == "none"
    dash.replay({"csv": replay_csv()})
    dash.control({"action": "settings", "loop_csv": True})
    for _ in range(12):
        assert dash.control({"action": "step"})["output"]["level"] == 0
    before = dash.status()
    with pytest.raises(APIError):
        dash.replay({"csv": "t_rel,student_S1-EP_p\n0,NaN\n-10,.9\n"})
    assert dash.status()["csv"] == before["csv"]


def test_mode_switch_clears_previous_source_and_settings_atomic():
    dash = Dashboard()
    dash.control({"action": "step"})
    live = dash.control({"action": "source", "mode": "live"})
    assert live["output"] is None and not live["running"]
    with pytest.raises(APIError):
        dash.control({"action": "start"})
    with pytest.raises(APIError):
        dash.control({"action": "settings", "scenario": "drowsy", "manual_p": -1})
    assert dash.status()["scenario"] == "normal"
    dash.control({"action": "source", "mode": "demo"})
    with pytest.raises(APIError):
        dash.window(live_window(10))


@pytest.fixture
def http_server(tmp_path):
    for name in ("index.html", "styles.css", "app.js"):
        (tmp_path / name).write_text("test " + name)
    server = create_server("127.0.0.1", 0, static_dir=tmp_path)
    worker = threading.Thread(target=server.serve_forever, daemon=True)
    worker.start()
    yield f"http://127.0.0.1:{server.server_port}"
    server.shutdown(); server.server_close(); worker.join(timeout=2)
    assert not server.dashboard.worker.is_alive()


def test_http_static_status_and_write_validation(http_server):
    with urlopen(http_server + "/") as response:
        assert response.status == 200 and b"index.html" in response.read()
    for asset in ("app.js", "styles.css"):
        with urlopen(http_server + "/" + asset) as response:
            assert response.status == 200 and asset.encode() in response.read()
    with urlopen(http_server + "/api/status") as response:
        assert json.load(response)["output"] is None
    request = Request(http_server + "/api/control", data=b'{"action":"step"}',
                      headers={"Content-Type": "application/json"})
    with urlopen(request) as response:
        assert json.load(response)["output"]["t"] == 10
    for origin, body, mime, expected in (("http://elsewhere.test", b'{}', "application/json", 403),
                                         (None, b'{', "application/json", 400),
                                         (None, b'{}', "text/plain", 415)):
        headers = {"Content-Type": mime}
        if origin:
            headers["Origin"] = origin
        request = Request(http_server + "/api/control", data=body, headers=headers)
        with pytest.raises(HTTPError) as exc:
            urlopen(request)
        assert exc.value.code == expected
    with pytest.raises(HTTPError) as exc:
        urlopen(http_server + "/../../README.md")
    assert exc.value.code == 404


def test_nested_malformed_payloads_are_readable_errors():
    dash = Dashboard()
    cases = ([], {"action": {}}, {"action": "settings", "scenario": []},
             {"action": "settings", "can_scenario": {}}, {"action": "source", "mode": []})
    for data in cases:
        with pytest.raises(APIError):
            dash.control(data)
    for text in ([], {}, None):
        with pytest.raises(APIError):
            dash.replay({"csv": text})
    with pytest.raises(APIError):
        parse_window({"t": 10, "p_ep": {"bad": True}})
