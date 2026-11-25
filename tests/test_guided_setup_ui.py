"""Browser flows against the real HTTP/storage/replay APIs.

Raw samples and model outputs here are explicit test fixtures. Production never
uses this backend. Real-weight/raw-recording verification is run separately.
"""
from dataclasses import asdict
from pathlib import Path
from threading import Event, Thread
from types import SimpleNamespace
import copy
import json
import shutil
import sys

import numpy as np
import pytest

playwright = pytest.importorskip('playwright.sync_api')
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from driver_state.model_profiles import model_identity
from driver_state.state_machine_v2 import StateMachineV2, WindowIn
from driver_state.web_dashboard import create_server, validate_event


class FixtureBackend:
    def __init__(self):
        self.gate = Event(); self.gate.set()
        self.entered = Event()
        self.exports = self.calls = 0
        self.reset()

    def reset(self):
        self.loaded = False; self.clean = 0; self.baseline_range = None
        self.ecg_reference = None; self.sm = StateMachineV2()

    def consume(self, signals, fs, start, end):
        self.calls += 1; self.entered.set()
        assert self.gate.wait(30), 'test must release its model gate'
        ready = self.loaded or end >= 360
        if ready:
            self.clean = 21; self.baseline_range = [60., 360.]
            if not self.ecg_reference:
                self.ecg_reference = dict(mean_bpm=75., clean_windows=21, start_s=60., end_s=360.)
        input_ready = ready and end >= (300 if self.loaded else 390)
        window = WindowIn(t=end, p_ep=.2 if input_ready else None, p_ppg=.2 if input_ready else None,
                          hr_ecg=75., hr_ppg=75., q_ecg=0., q_ppg=0.)
        return dict(status='ready' if ready else 'collecting', clean_windows=self.clean,
                    baseline_range=self.baseline_range, ecg_reference=self.ecg_reference,
                    profile_loaded=self.loaded, input_ready=input_ready, input=asdict(window),
                    output=asdict(self.sm.step(window)) if input_ready else None)

    def export_baseline(self, *, source_session_id, processing):
        self.exports += 1
        return profile(source_session_id, processing)

    def restore_baseline(self, snapshot):
        self.reset(); self.loaded = True; self.clean = 21; self.baseline_range = [60., 360.]
        meta = snapshot['metadata']
        self.ecg_reference = dict(meta['ecg_reference'], source_session_id=meta['source_session_id'],
                                  profile_id=meta['profile_id'])


def profile(session, processing):
    names = json.loads((ROOT/'models/bundle.json').read_text())['rr_feature_names']
    return dict(metadata=dict(format_version=1, identity=model_identity(ROOT/'models'), processing=processing,
                feature_names=names, clean_windows=21, start_s=60., end_s=360., source_session_id=session,
                ecg_reference=dict(mean_bpm=75., clean_windows=21, start_s=60., end_s=360.)),
                statistics=np.ones((2, 2, 36), np.float64), references=np.zeros((3, 2, 32), np.float32))


def raw_folder(root, name, title, *, invalid=False):
    directory = root/name; directory.mkdir(parents=True)
    (directory/'ecg.csv').write_text('value\n1\n' if invalid else 'time,adc\n'+''.join(f'{i/100},512\n' for i in range(48000)))
    (directory/'ppg.csv').write_text('time,red,ir\n'+''.join(f'{i/100},50000,60000\n' for i in range(48000)))
    (directory/'meta.json').write_text('{"session_id":"fixture","fs":{"ecg":100,"ppg":100}}')
    (directory/'recording.json').write_text(json.dumps(dict(title=title, ecg='ecg.csv', ppg='ppg.csv', meta='meta.json')))


@pytest.fixture
def ui(tmp_path):
    chromium = shutil.which('chromium') or shutil.which('chromium-browser')
    if not chromium:
        pytest.skip('UI checks require Chromium and the optional playwright package')
    raw = tmp_path/'raw'
    raw_folder(raw, 'baseline', '기준 원본')
    raw_folder(raw, 'input', '새 판정 원본')
    raw_folder(raw, 'invalid', '잘못된 원본', invalid=True)
    backend = FixtureBackend()
    server = create_server('127.0.0.1', 0, recordings_dir=raw, library_dir=tmp_path/'library', backend_factory=lambda:backend)
    local = server.local_replay
    entries = local.catalog.refresh()['recordings']
    ids = {r['title']:r['id'] for r in entries}
    a = server.library.add_user('기존 사용자'); b = server.library.add_user('다른 사용자')
    snapshot = profile('previous-session', local.processing(local.catalog.get(ids['기준 원본'])))
    snapshot['metadata']['recording'] = local.catalog.public(local.catalog.get(ids['기준 원본']))
    saved = local.profiles.save(a['id'], '보관된 기준', snapshot)
    bad = copy.deepcopy(snapshot); bad['metadata']['identity']['model_sha256'] = '0'*64
    incompatible = local.profiles.save(a['id'], '호환 안 되는 기준', bad)
    template = json.loads((ROOT/'web/dashboard/contract/empty_status_template.json').read_text())
    first = copy.deepcopy(template); first.update(origin='recorded', session_id='legacy', t_rel_s=0.)
    first['payload']['bio_baseline']['status'] = 'collecting'
    last = copy.deepcopy(first); last['t_rel_s'] = 2.; last['payload']['bio_baseline'].update(status='ready', clean_windows=21)
    channel = dict(t_start_rel_s=0., fs_hz=100., samples=[.1, .2, .1], unit='a.u.', stage='common_preprocessed', quality_ok=True)
    wave = dict(schema_version='1.2', type='bio.wave', origin='recorded', session_id='legacy', t_rel_s=.1,
                payload=dict(ecg=channel, ppg=copy.deepcopy(channel)))
    events = [first, wave, last]
    legacy = server.library.save(a['id'], '예전 화면 기록', events, lambda event:validate_event(event, server.dashboard.validator))
    thread = Thread(target=server.serve_forever, daemon=True); thread.start()
    url = f'http://127.0.0.1:{server.server_port}'
    with playwright.sync_playwright() as p:
        browser = p.chromium.launch(executable_path=chromium, headless=True, args=['--no-sandbox'])
        page = browser.new_page(viewport={'width':1280, 'height':900})
        errors = []; page.on('pageerror', lambda error:errors.append(str(error)))
        page.goto(url); page.wait_for_function('localStatus.available && recordings.length===3')
        try:
            yield SimpleNamespace(page=page, backend=backend, server=server, url=url, ids=ids, a=a, b=b,
                                  saved=saved, incompatible=incompatible, legacy=legacy, events=events)
            assert not errors, errors
        finally:
            backend.gate.set(); browser.close(); server.shutdown(); server.server_close(); thread.join(5)


def open_setup(page, step=None):
    page.locator('#setupButton').click(); page.wait_for_function('!busy')
    if step is not None:
        assert page.evaluate('setup.step') == step


def next_step(page, step):
    page.locator('#setupNext').click()
    page.wait_for_function(f'setup.step==={step} && !busy')


@pytest.mark.parametrize('width', [1280, 390])
def test_new_user_guided_flow_progress_restore_drafts_and_folded_playback(ui, width):
    page, backend = ui.page, ui.backend
    page.set_viewport_size({'width':width, 'height':850})
    open_setup(page, 0)
    assert page.locator('#setupNext').is_disabled()
    assert '사용자를 선택' in page.locator('#setupNextHint').inner_text()
    page.locator('#nicknameInput').fill('새 사용자')
    page.locator('#registerUser').click(); page.wait_for_function('setup.step===1 && !busy')
    assert page.locator('#setupNext').is_disabled()
    assert '원본 기록' in page.locator('#setupNextHint').inner_text()
    page.locator('#setupBack').click()
    assert page.locator('#nicknameInput').input_value() == '새 사용자'
    next_step(page, 1)
    page.locator('#baselineRecordSelect').select_option(ui.ids['기준 원본']); next_step(page, 2)
    page.locator('#baselineSpeedSelect').select_option('0')
    backend.gate.clear(); backend.entered.clear()
    page.locator('#setupNext').click(); assert backend.entered.wait(5)
    page.wait_for_function('!busy')
    sid = ui.server.local_replay.status()['session_id']
    page.locator('#closeSetup').click(); open_setup(page, 2)
    assert ui.server.local_replay.status()['session_id'] == sid
    page.reload(); page.wait_for_function('localStatus.user && !busy'); open_setup(page, 2)
    backend.gate.set(); page.wait_for_function('setup.step===3', timeout=15000)
    assert page.locator('#setupNext').is_disabled()
    page.locator('#profileTitle').fill('오전 안정 기준')
    page.locator('#setupBack').click(); next_step(page, 3)
    assert page.locator('#profileTitle').input_value() == '오전 안정 기준'
    next_step(page, 4)
    assert len(ui.server.local_replay.profiles.list(ui.server.local_replay.user['id'])) == 1
    assert page.locator('#replayRecordSelect').input_value() == ''
    assert page.locator('#setupNext').is_disabled()
    assert '기준 원본' in page.locator('#reviewBaselineRecord').inner_text()
    page.locator('#replayRecordSelect').select_option(ui.ids['새 판정 원본'])
    page.locator('#speedSelect').select_option('0')
    backend.gate.clear(); backend.entered.clear()
    page.locator('#setupNext').click(); assert backend.entered.wait(5)
    page.wait_for_function('localStatus.profile_applied && !busy')
    assert not page.evaluate('localStatus.input_ready')
    assert page.locator('#driverState').inner_text() == '판정 대기'
    generation = ui.server.local_replay.generation
    page.locator('#replayPanel>summary').click()
    assert not page.locator('#replayPanel').evaluate('(panel)=>panel.open')
    assert ui.server.local_replay.generation == generation
    assert '새 판정 원본' in page.locator('#summaryRecording').inner_text()
    assert page.locator('#stateModule').is_visible() and page.locator('#ecgWave').is_visible()
    page.reload(); page.wait_for_function('localStatus.profile_applied && recordings.length===3')
    assert not page.locator('#replayPanel').evaluate('(panel)=>panel.open')
    assert ui.server.local_replay.generation == generation
    backend.gate.set(); page.wait_for_function("localStatus.stage==='completed'", timeout=15000)
    assert '기록 끝' in page.locator('#summaryPlayback').inner_text()
    assert page.locator('#driverState').inner_text() == '정상'
    page.locator('#replayPanel>summary').click()
    page.locator('#localRewind').click(); page.wait_for_function("localStatus.stage==='paused' && !busy")
    assert page.evaluate('localStatus.position_s') == 0
    assert page.locator('#ecgHr').inner_text() == '··'
    assert page.locator('#driverState').inner_text() == '판정 대기'
    page.locator('#localPlay').click(); page.wait_for_function("localStatus.stage==='completed'", timeout=15000)
    page.locator('#localStop').click(); page.wait_for_function("localStatus.stage==='prepared' && !busy")
    assert page.locator('#ecgHr').inner_text() == '··'
    open_setup(page, 4); page.locator('#setupBack').click()
    assert page.evaluate('setup.step') == 1
    assert page.locator('#baselineModeSaved').is_checked()
    next_step(page, 4); page.locator('#closeSetup').click()
    page.reload(); page.wait_for_function('localStatus.profile && recordings.length===3')
    open_setup(page, 4)
    assert '오전 안정 기준' in page.locator('#reviewProfile').inner_text()
    assert page.evaluate('document.documentElement.scrollWidth<=innerWidth')
    box = page.locator('#setupNext').bounding_box()
    assert box['y'] >= 0 and box['y']+box['height'] <= 850


def test_saved_profile_skip_incompatible_error_retry_and_user_isolation(ui):
    page, backend = ui.page, ui.backend
    open_setup(page, 0)
    page.locator('#userSelect').select_option(ui.a['id']); page.wait_for_function('!busy'); next_step(page, 1)
    assert page.locator('#baselineModeSaved').is_checked()
    page.locator('#profileSelect').select_option(ui.incompatible['id'])
    page.locator('#setupNext').click(); page.wait_for_function('!busy')
    assert page.evaluate('setup.step') == 1
    assert '버전' in page.locator('#setupNotice').inner_text()
    page.locator('#profileSelect').select_option(ui.saved['id']); next_step(page, 4)
    assert backend.exports == backend.calls == 0
    assert '불러오기 완료' in page.locator('#reviewProfile').inner_text()
    page.locator('#replayRecordSelect').select_option(ui.ids['잘못된 원본'])
    page.locator('#speedSelect').select_option('0'); page.locator('#setupNext').click()
    page.wait_for_function("localStatus.stage==='error' && !busy", timeout=10000)
    assert page.locator('#setupDialog').is_visible()
    assert page.evaluate('setup.step') == 4
    assert 'ECG' in page.locator('#replayReady').inner_text()
    page.locator('#replayRecordSelect').select_option(ui.ids['새 판정 원본'])
    backend.gate.clear(); backend.entered.clear()
    page.locator('#setupNext').click(); assert backend.entered.wait(5)
    page.wait_for_function('!busy'); old_future = ui.server.local_replay.future
    open_setup(page, 4); page.locator('#setupBack').click(); page.locator('#setupBack').click()
    page.locator('#userSelect').select_option(ui.b['id']); page.wait_for_function('!busy')
    assert ui.server.local_replay.user['id'] == ui.b['id']
    assert ui.server.local_replay.profile is None
    assert ui.server.dashboard.snapshot()['events'] == []
    assert page.locator('#ecgHr').inner_text() == '··'
    backend.gate.set(); old_future.result(timeout=5)
    assert ui.server.dashboard.snapshot()['events'] == []
    assert page.locator('#driverState').inner_text() == '판정 대기'


def test_auxiliary_json_and_legacy_results_remain_view_only(ui):
    page = ui.page
    page.set_viewport_size({'width':320, 'height':750})
    page.locator('#toolsMenu>summary').click(); page.locator('#resultsMenuButton').click()
    page.wait_for_function('!busy')
    page.locator('#legacyUserSelect').select_option(ui.a['id']); page.wait_for_function('!busy')
    page.locator('#legacySelect').select_option(ui.legacy['id']); page.locator('#loadLegacy').click()
    page.wait_for_function("source==='file' && !busy")
    page.locator('#playFile').click(); page.wait_for_function('position===3')
    assert page.locator('#driverState').inner_text() == '판정 대기'
    assert '모델 실행 없음' in page.locator('#activeProfile').inner_text()
    page.locator('#rewindFile').click(); assert page.evaluate('position') == 0
    page.locator('#stepFile').click(); assert page.evaluate('position') == 1
    page.locator('#toolsMenu>summary').click(); page.locator('#resultsMenuButton').click()
    page.locator('#eventFile').set_input_files(dict(name='results.jsonl', mimeType='application/x-ndjson',
        buffer='\n'.join(json.dumps(event) for event in ui.events).encode()))
    page.wait_for_function("resultContext?.title==='results.jsonl' && !busy")
    assert ui.backend.calls == 0
    assert ui.server.local_replay.user is None and ui.server.local_replay.profile is None
    page.locator('#backToLocal').click()
    assert page.locator('#driverState').inner_text() == '판정 대기'
    assert page.locator('#ecgHr').inner_text() == '··'
    assert page.evaluate('document.documentElement.scrollWidth<=innerWidth')


def test_calibration_failure_can_change_source_and_retry_in_the_same_dialog(ui):
    page = ui.page
    open_setup(page, 0)
    page.locator('#nicknameInput').fill('재시도 사용자')
    page.locator('#registerUser').click(); page.wait_for_function('setup.step===1 && !busy')
    page.locator('#baselineRecordSelect').select_option(ui.ids['잘못된 원본']); next_step(page, 2)
    page.locator('#baselineSpeedSelect').select_option('0'); page.locator('#setupNext').click()
    page.wait_for_function("localStatus.stage==='error' && !busy", timeout=10000)
    assert page.locator('#setupDialog').is_visible()
    assert 'ECG' in page.locator('#calibrationState').inner_text()
    assert page.locator('#setupNext').inner_text() == '기준 계산 다시 시도'
    page.locator('#setupBack').click()
    assert page.locator('#baselineRecordSelect').input_value() == ui.ids['잘못된 원본']
    page.locator('#baselineRecordSelect').select_option(ui.ids['기준 원본']); next_step(page, 2)
    assert page.locator('#baselineSpeedSelect').input_value() == '0'
    page.locator('#setupNext').click(); page.wait_for_function('setup.step===3', timeout=15000)
    page.locator('#profileTitle').fill('재시도 후 기준'); next_step(page, 4)
    assert '재시도 후 기준' in page.locator('#reviewProfile').inner_text()
