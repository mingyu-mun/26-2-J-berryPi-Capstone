"""Explicit fixtures exercise storage and capture, never a production data source."""
import copy
import json
from pathlib import Path
import sys
import threading
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from driver_state.web_dashboard import APIError, Dashboard, create_server, validate_event
from driver_state.webboard_library import BaselineLibrary, LibraryError


def status(t, state, session='capture'):
    event = json.loads((ROOT / 'web/dashboard/contract/empty_status_template.json').read_text())
    event.update(session_id=session, t_rel_s=t, origin='recorded')
    event['payload']['bio_baseline'].update(status=state, clean_windows=21 if state == 'ready' else 1)
    return event


def recording(session='capture'):
    channel = dict(t_start_rel_s=0, fs_hz=100, samples=[.1, .2, .1], unit='a.u.',
                   stage='common_preprocessed', quality_ok=True)
    wave = dict(schema_version='1.2', type='bio.wave', session_id=session, origin='recorded', t_rel_s=.1,
                payload=dict(ecg=channel, ppg=copy.deepcopy(channel)))
    return [status(0, 'collecting', session), wave, status(2, 'ready', session)]


def test_library_roundtrip_persists_and_scopes_records_by_user(tmp_path):
    library = BaselineLibrary(tmp_path)
    validator = Dashboard().validator
    a = library.add_user('사용자 A'); b = library.add_user('사용자 B')
    original = recording()
    saved = library.save(a['id'], '아침 기준', original, lambda e: validate_event(e, validator))
    assert library.get(a['id'], saved['id'])['events'] == original
    assert library.list_baselines(b['id']) == []
    with pytest.raises(LibraryError) as error:
        library.get(b['id'], saved['id'])
    assert error.value.status == 404
    library.close()
    restored = BaselineLibrary(tmp_path)
    assert len(restored.users()) == 2
    assert restored.get(a['id'], saved['id'])['events'] == original
    assert restored.list_baselines(a['id'])[0]['clean_windows'] == 21
    restored.close()


@pytest.mark.parametrize('nickname', ['', '  ', 'a' * 33, 'bad\x00name', None])
def test_invalid_nicknames_are_not_stored(tmp_path, nickname):
    library = BaselineLibrary(tmp_path)
    with pytest.raises(LibraryError):
        library.add_user(nickname)
    assert library.users() == []
    library.close()


def test_nickname_normalization_and_duplicates(tmp_path):
    library = BaselineLibrary(tmp_path)
    library.add_user('  Cafe\u0301  ')
    with pytest.raises(LibraryError) as error:
        library.add_user('CAFÉ')
    assert error.value.status == 409 and len(library.users()) == 1
    library.close()


def test_bad_or_unfinished_recording_is_rejected_atomically(tmp_path):
    library = BaselineLibrary(tmp_path); user = library.add_user('검사')
    validator = Dashboard().validator
    mixed = recording(); mixed[1]['session_id'] = 'other'
    no_wave = [status(0, 'collecting'), status(2, 'ready')]
    no_collection = recording()[1:]
    invalid = recording(); invalid[-1]['payload']['bio_baseline']['status'] = 'unknown'
    for events in ([], recording()[:-1], mixed, no_wave, no_collection, invalid):
        with pytest.raises((APIError, LibraryError)):
            library.save(user['id'], '실패', events, lambda e: validate_event(e, validator))
        assert library.list_baselines(user['id']) == []
    library.close()


def test_capture_freezes_at_ready_and_resets_on_new_session():
    engine = Dashboard()
    assert not engine.baseline_capture.status()['ready']
    engine.ingest(recording())
    assert engine.baseline_capture.status()['ready']
    engine.ingest(status(20, 'retry_required'))
    assert len(engine.baseline_capture.snapshot('capture')) == 3
    engine.ingest(status(0, 'collecting', 'next'))
    assert not engine.baseline_capture.status()['ready']
    with pytest.raises(LibraryError) as error:
        engine.baseline_capture.snapshot('capture')
    assert error.value.status == 409
    assert all(e['session_id'] == 'next' for e in engine.baseline_capture.events)
    engine.ingest(status(1, 'retry_required', 'next'))
    assert engine.baseline_capture.status()['state'] == 'retry_required'
    assert not engine.baseline_capture.status()['ready']


def test_baseline_capture_survives_display_history_eviction():
    engine = Dashboard()
    engine.ingest(recording()[:2])
    for second in range(1, 601):
        engine.ingest(status(second, 'collecting'))
    engine.ingest(status(601, 'ready'))
    assert len(engine.events) <= 512
    snapshot = engine.baseline_capture.snapshot('capture')
    assert len(snapshot) > 512
    assert snapshot[0]['t_rel_s'] == 0
    assert any(e['type'] == 'bio.wave' for e in snapshot)


def test_size_limit_does_not_leave_partial_baseline(tmp_path, monkeypatch):
    import driver_state.webboard_library as module
    library = BaselineLibrary(tmp_path); user = library.add_user('크기 검사')
    validator = Dashboard().validator
    monkeypatch.setattr(module, 'MAX_RECORDING_BYTES', 1)
    with pytest.raises(LibraryError) as error:
        library.save(user['id'], '너무 큼', recording(), lambda e: validate_event(e, validator))
    assert error.value.status == 413 and library.list_baselines(user['id']) == []
    engine = Dashboard(); engine.ingest(recording())
    assert engine.baseline_capture.status()['overflow']
    assert not engine.baseline_capture.status()['ready']
    library.close()


@pytest.fixture
def api(tmp_path):
    server = create_server('127.0.0.1', 0, library_dir=tmp_path)
    worker = threading.Thread(target=server.serve_forever, daemon=True); worker.start()
    base = f'http://127.0.0.1:{server.server_port}'
    def request(path, data=None, origin=None):
        headers = {'Content-Type': 'application/json'}
        if origin: headers['Origin'] = origin
        body = json.dumps(data).encode() if data is not None else None
        with urlopen(Request(base + path, data=body, headers=headers), timeout=5) as response:
            return json.load(response)
    yield request
    server.shutdown(); server.server_close(); worker.join(timeout=2)


def test_http_import_capture_lookup_and_owner_isolation(api):
    assert api('/api/users')['users'] == []
    assert not api('/api/baselines/current')['ready']
    a = api('/api/users', {'nickname': '가'}); b = api('/api/users', {'nickname': '나'})
    imported = api(f"/api/users/{a['id']}/baselines", {'title': '파일 기준', 'events': recording()})
    assert api('/api/webboard')['events'] == []  # Importing does not publish into a live session.
    assert api(f"/api/users/{a['id']}/baselines/{imported['id']}")['events'] == recording()
    assert api(f"/api/users/{b['id']}/baselines")['baselines'] == []
    with pytest.raises(HTTPError) as error:
        api(f"/api/users/{b['id']}/baselines/{imported['id']}")
    assert error.value.code == 404
    api('/api/events', recording('live-session'))
    assert api('/api/baselines/current')['ready']
    with pytest.raises(HTTPError) as error:
        api(f"/api/users/{a['id']}/baselines/current", {'title': '과거 세션', 'session_id': 'capture'})
    assert error.value.code == 409
    saved = api(f"/api/users/{a['id']}/baselines/current", {'title': 'Pi 기준', 'session_id': 'live-session'})
    assert saved['session_id'] == 'live-session'
    assert len(api(f"/api/users/{a['id']}/baselines")['baselines']) == 2
    with pytest.raises(HTTPError) as error:
        api('/api/users', {'nickname': 'cross-site'}, 'http://other-site.test')
    assert error.value.code == 403
