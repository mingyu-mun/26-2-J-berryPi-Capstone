"""Team v1.2 HTTP producer -> external-only dashboard -> browser.

The connector and state-machine sources are unchanged from the team repository.
Values below are explicit protocol fixtures, never a production input source.
"""
from pathlib import Path
from queue import Queue
from threading import Thread
from urllib.error import HTTPError
from urllib.request import Request, urlopen
import json
import os
import re
import shutil
import signal
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from driver_state.state_machine_v2 import StateMachineV2, WindowIn
from driver_state.webboard_connector import WebboardPublisher, ai_decision, bio_wave, envelope, system_status, wave_channel


@pytest.fixture
def receiver(tmp_path):
    process = subprocess.Popen([sys.executable, '-u', str(ROOT/'scripts/run_dashboard.py'),
        '--external-events', '--host', '127.0.0.1', '--port', '0', '--data-dir', str(tmp_path/'library'),
        '--recordings-dir', str(tmp_path/'must-not-be-opened'), '--models', str(tmp_path/'no-models-needed')],
        cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding='utf-8')
    lines = Queue()
    def read_output():
        for line in process.stdout:
            lines.put(line)
    reader = Thread(target=read_output, daemon=True); reader.start()
    try:
        line = lines.get(timeout=15)
        match = re.search(r'http://127\.0\.0\.1:\d+', line)
        assert match, line
        yield match.group(0)
        assert process.poll() is None
        assert not (tmp_path/'must-not-be-opened').exists()
    finally:
        if process.poll() is None:
            if os.name == 'posix': process.send_signal(signal.SIGINT)
            else: process.terminate()
            try: process.wait(timeout=10)
            except subprocess.TimeoutExpired: process.kill(); process.wait(timeout=5)
        process.stdout.close(); reader.join(2)


def get(url, path):
    with urlopen(url+path, timeout=5) as response:
        return json.load(response)


def team_frame(machine, index=1, session='team-live'):
    t = 360.+index*10
    window = WindowIn(t=t, p_ep=.99, q_ecg=0., p_ppg=.98, q_ppg=0., hr_ecg=75., hr_ppg=76., ok_ecg=True, ok_ppg=True)
    output = machine.step(window)
    can = dict(speed_kmh=0., steering_angle_deg=12., steering_rate_deg_s=4., accelerator_pct=None,
        brake_pct=None, baseline_coverage_pct=None, deviation_status='unavailable', can_drowsy=None,
        rule_id=None, signal_map_id=None, road_wheel_angle_deg=None, accel_pedal_raw=0, brake_pedal_raw=12,
        signal_times_s=dict(speed=t, steering=t, accelerator=t, brake=t))
    return [system_status(session,t,bio_status='ready',clean_windows=21,bio_elapsed_s=300.,
                connections=dict(ecg='connected',ppg='connected',can='connected',model='connected')),
        bio_wave(session,t,ecg=wave_channel(t-1,100.,[.1,.3,.2,.1]*25,'mV','common_preprocessed',True),
                 ppg=wave_channel(t-1,100.,[.2,.5,.3,.2]*25,'a.u.','common_preprocessed',True)),
        ai_decision(session,window,output,baseline_status='ready',bio_origin='live'),
        envelope('can.telemetry',session,'live',t,can)]


def test_team_http_four_event_types_and_session_reset(receiver):
    assert not get(receiver,'/api/local/session')['available']
    assert get(receiver,'/api/webboard')['events']==[]
    publisher=WebboardPublisher(receiver); machine=StateMachineV2()
    for i in range(1,29):
        result=publisher.send(team_frame(machine,i))
        assert result[0]['accepted']==4
    snapshot=get(receiver,'/api/webboard')
    assert {p['event']['type'] for p in snapshot['events']}=={'bio.wave','ai.decision','can.telemetry','system.status'}
    assert next(p['event'] for p in reversed(snapshot['events']) if p['event']['type']=='ai.decision')['payload']['output']['level']==2
    invalid=team_frame(StateMachineV2())[0]; invalid['schema_version']='1.1'
    with pytest.raises(HTTPError) as error:
        publisher.send(invalid)
    assert error.value.code==400
    next_session=system_status('next-live',0.,bio_status='collecting',clean_windows=0)
    assert publisher.send(next_session)[0]['accepted']==1
    assert get(receiver,'/api/webboard')['session_id']=='next-live'
    assert publisher.send(team_frame(StateMachineV2()))[0]['accepted']==0
    # A local session cannot claim ownership and interrupt an external producer.
    request=Request(receiver+'/api/local/select',data=b'{}',headers={'Content-Type':'application/json'})
    with pytest.raises(HTTPError) as error:
        urlopen(request,timeout=5)
    assert error.value.code==503
    assert publisher.send(system_status('next-live',1.,bio_status='collecting',clean_windows=1))[0]['accepted']==1


def test_team_frames_display_in_new_browser_and_footer_is_collapsed(receiver):
    playwright=pytest.importorskip('playwright.sync_api')
    chromium=shutil.which('chromium') or shutil.which('chromium-browser')
    if not chromium: pytest.skip('Browser compatibility check requires Chromium')
    with playwright.sync_playwright() as p:
        browser=p.chromium.launch(executable_path=chromium,headless=True,args=['--no-sandbox'])
        page=browser.new_page(viewport={'width':1280,'height':900}); errors=[]
        page.on('pageerror',lambda error:errors.append(str(error)))
        try:
            page.goto(receiver); page.wait_for_function('!pollBusy && transportState===\'connected\'')
            assert page.locator('#ecgHr').inner_text()=='··'
            assert page.locator('#driverState').inner_text()=='판정 대기'
            assert page.locator('#inputState').inner_text()=='외부 이벤트 대기'
            assert page.locator('#setupButton').is_hidden()
            publisher=WebboardPublisher(receiver); machine=StateMachineV2()
            publisher.send(team_frame(machine))
            page.wait_for_function("document.getElementById('driverState').textContent==='정상'")
            assert page.locator('#ecgHr').inner_text()=='75' and page.locator('#ppgHr').inner_text()=='76'
            assert page.evaluate('board.view().waves.ecg.points.length')==100
            assert page.evaluate('board.view().waves.ppg.points.length')==100
            assert '송신측 기준 준비 완료' in page.locator('#activeProfile').inner_text()
            assert page.locator('.wave-footer').count()==0
            assert '품질' not in page.locator('.signal-section').inner_text()
            assert page.locator('#ecgQuality').is_hidden()
            page.locator('#detailPanel>summary').click()
            assert page.locator('#ecgQuality').inner_text()=='품질 양호'
            page.locator('#detailPanel>summary').click()
            for i in range(2,11): publisher.send(team_frame(machine,i))
            page.wait_for_function("board.view().state.candidate===true")
            assert page.locator('#driverState').inner_text()=='주의'
            for i in range(11,29): publisher.send(team_frame(machine,i))
            page.wait_for_function("document.getElementById('driverState').textContent==='졸음'")
            assert 'CAN' not in page.locator('body').inner_text()
            page.locator('#replayPanel>summary').click()
            page.set_viewport_size({'width':390,'height':850})
            assert page.evaluate('document.documentElement.scrollWidth<=innerWidth')
            assert page.locator('#ecgWave').is_visible()
            publisher.send(system_status('new-input',0.,bio_status='collecting',clean_windows=0))
            page.wait_for_function("board.session==='new-input'")
            assert page.locator('#ecgHr').inner_text()=='··'
            assert page.locator('#driverState').inner_text()=='판정 대기'
            publisher.send(team_frame(StateMachineV2(),session='new-input'))
            page.wait_for_function("document.getElementById('driverState').textContent==='정상'")
            # Advance only the test view clock to exercise missing-live-data handling.
            page.evaluate('board.clock=()=>performance.now()/1000+40')
            page.wait_for_function("document.getElementById('ecgHr').textContent==='··'")
            assert page.locator('#driverState').inner_text()=='판정 대기'
            assert not get(receiver,'/api/local/session')['available']
            assert not errors, errors
        finally:
            browser.close()
