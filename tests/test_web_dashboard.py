"""Transport tests with explicit fixtures; the application never creates measurements."""
import copy
import json
import sys
import threading
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from driver_state.web_dashboard import APIError, Dashboard, create_server

CONTRACT = Path(__file__).resolve().parents[1] / "web/dashboard/contract"


def status(session="a", t=0):
    event = json.loads((CONTRACT / "empty_status_template.json").read_text())
    event.update(session_id=session, t_rel_s=t)
    return event


def decision(t=30):
    return {"schema_version": "1.2", "type": "ai.decision", "session_id": "a", "origin": "live", "t_rel_s": t,
            "payload": {"model_id": "S1-EP", "state_machine_id": "v2", "window_start_s": t - 30,
                        "window_end_s": t, "baseline_status": "ready", "bio_origin": "live", "can_origin": "unavailable",
                        "input": {"t": t, "p_ep": .99, "q_ecg": .02, "p_ppg": None, "q_ppg": None,
                                  "hr_ecg": 73, "hr_ppg": None, "ok_ecg": True, "ok_ppg": None, "can_drowsy": None},
                        "output": {"t": t, "level": 0, "level_name": "정상", "candidate": False, "score": .99,
                                   "p_used": .99, "confidence": 1, "channel": "EP", "frozen": False},
                        "confirmation_source": None}}


def test_empty_until_real_input_and_waiting_template_has_no_measurements():
    dash = Dashboard()
    assert dash.snapshot()["events"] == []
    dash.ingest(status())
    events = dash.snapshot()["events"]
    assert len(events) == 1 and events[0]["event"] == status()
    assert not any(e["event"]["type"] in {"ai.decision", "bio.wave", "can.telemetry"} for e in events)


def test_ingestion_never_infers_a_state_from_a_model_score():
    dash = Dashboard()
    data = decision()
    dash.ingest(data)
    # The upstream output is retained even though p=.99. This is a display relay.
    assert dash.snapshot()["events"][0]["event"]["payload"]["output"]["level"] == 0
    data["payload"]["output"]["level"] = 2
    assert dash.snapshot()["events"][0]["event"]["payload"]["output"]["level"] == 0


def test_schema_and_semantic_errors_reject_atomically():
    dash = Dashboard()
    bad = decision(); bad["payload"]["baseline_status"] = "collecting"
    no_quality = decision(); no_quality["payload"]["input"]["q_ecg"] = None
    wrong_state = decision(); wrong_state["payload"]["output"]["level_name"] = "졸음"
    wrong_time = decision(); wrong_time["payload"]["output"]["t"] = 0
    infinite = decision(); infinite["payload"]["input"]["p_ep"] = float("nan")
    for event in ({}, None, [], bad, no_quality, wrong_state, wrong_time, infinite):
        with pytest.raises(APIError):
            dash.ingest([status(), event])
        assert dash.snapshot()["events"] == []


def test_duplicate_does_not_refresh_measurement_age():
    now = [0.0]
    dash = Dashboard(clock=lambda: now[0])
    event = decision()
    assert dash.ingest(event)["accepted"] == 1
    now[0] = 45
    assert dash.ingest(event)["accepted"] == 0
    assert dash.snapshot()["events"][0]["age_s"] == 45


def test_optional_hr_reference_accepts_old_events_and_rejects_invalid_references():
    event=status(t=370)
    event['origin']='live'
    base=event['payload']['bio_baseline']
    base.update(status='ready',ecg_reference=dict(mean_bpm=75.,clean_windows=21,start_s=60.,end_s=360.))
    dash=Dashboard()
    assert dash.ingest(event)['accepted']==1
    assert dash.snapshot()['events'][0]['event']['payload']['bio_baseline']['ecg_reference']['mean_bpm']==75.
    for key,value in [('mean_bpm',0),('clean_windows',0),('end_s',371.),('start_s',360.)]:
        bad=copy.deepcopy(event)
        bad['payload']['bio_baseline']['ecg_reference'][key]=value
        with pytest.raises(APIError):Dashboard().ingest(bad)
    bad=copy.deepcopy(event);bad['payload']['bio_baseline']['status']='collecting'
    with pytest.raises(APIError):Dashboard().ingest(bad)


def test_new_session_resets_stream_and_late_old_session_is_ignored():
    dash = Dashboard()
    dash.ingest(status("a", 100))
    dash.ingest(status("b", 0))
    assert dash.ingest(status("a", 110))["accepted"] == 0
    result = dash.snapshot()
    assert result["session_id"] == "b"
    assert [p["event"]["session_id"] for p in result["events"]] == ["b"]


def test_cursor_overflow_recovers_latest_streams_and_is_bounded():
    dash = Dashboard()
    dash.ingest(decision())
    for i in range(520):
        dash.ingest(status(t=i))
    result = dash.snapshot(after=1)
    assert result["reset"] is True and len(dash.events) <= 512
    assert any(p["event"]["type"] == "ai.decision" for p in result["events"])
    last = dash.seq
    assert dash.snapshot(after=last)["events"] == []


@pytest.fixture
def server(tmp_path):
    instance = create_server("127.0.0.1", 0, library_dir=tmp_path)
    worker = threading.Thread(target=instance.serve_forever, daemon=True); worker.start()
    yield f"http://127.0.0.1:{instance.server_port}"
    instance.shutdown(); instance.server_close(); worker.join(timeout=2)


def post(url, data, **headers):
    body = json.dumps(data).encode()
    return urlopen(Request(url, data=body, headers={"Content-Type": "application/json", **headers}), timeout=3)


def test_http_actual_assets_schema_event_roundtrip_and_removed_demo_routes(server):
    for path in ("/", "/styles.css", "/app.js", "/webboard-state.js", "/contract-validator.js", "/contract/webboard_event_v1_2.schema.json"):
        with urlopen(server + path, timeout=3) as response:
            assert response.status == 200 and len(response.read()) > 0
    with urlopen(server + "/api/webboard") as response:
        assert json.load(response)["events"] == []
    with post(server + "/api/events", decision()) as response:
        assert json.load(response)["accepted"] == 1
    with urlopen(server + "/api/webboard") as response:
        assert json.load(response)["events"][0]["event"] == decision()
    with pytest.raises(HTTPError) as exc:
        post(server + "/api/control", {"action": "step"})
    assert exc.value.code == 404


def test_http_bad_origin_unknown_file_invalid_query_and_bad_json(server):
    with pytest.raises(HTTPError) as exc:
        post(server + "/api/events", status(), Origin="http://other-site.test")
    assert exc.value.code == 403
    for path in ("/../../README.md", "/contract/types.ts", "/api/webboard?after=oops"):
        with pytest.raises(HTTPError) as exc:
            urlopen(server + path)
        assert exc.value.code in (400, 404)
    for data in (b'{', b'NaN', b'1e999'):
        with pytest.raises(HTTPError) as exc:
            urlopen(Request(server + "/api/events", data=data, headers={"Content-Type": "application/json"}))
        assert exc.value.code == 400
