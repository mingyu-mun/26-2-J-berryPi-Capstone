"""Explicit raw-recording fixtures; no hardware and no automatic demo source."""
from dataclasses import asdict
from pathlib import Path
from threading import Event
import json
import sys
import time

import pytest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from driver_state.local_recordings import RecordingCatalog, RawRecording
from driver_state.local_replay import LocalReplay
from driver_state.state_machine_v2 import WindowIn
from driver_state.web_dashboard import Dashboard
from driver_state.webboard_library import BaselineLibrary, LibraryError


def raw_record(root,seconds=50):
    root.mkdir(exist_ok=True)
    (root/'ecg.csv').write_text('time,adc\n'+''.join(f'{i/100},512\n' for i in range(seconds*100)))
    (root/'ppg.csv').write_text('time,red,ir\n'+''.join(f'{i/100},50000,60000\n' for i in range(seconds*100)))
    (root/'meta.json').write_text(json.dumps(dict(fs={'ecg':100,'ppg':100},session_id='fixture-record')))
    (root/'recording.json').write_text(json.dumps(dict(title='명시적 시험 기록',ecg='ecg.csv',ppg='ppg.csv',meta='meta.json')))


def test_catalog_requires_explicit_pairing_and_rejects_escape(tmp_path):
    root=tmp_path/'records';raw_record(root)
    catalog=RecordingCatalog(root);data=catalog.refresh();assert len(data['recordings'])==1
    entry=catalog.get(data['recordings'][0]['id'])
    with RawRecording(entry) as source:assert source.duration==50.
    outside=tmp_path/'outside.csv';outside.write_text('secret')
    manifest=json.loads((root/'recording.json').read_text());manifest['ecg']='../outside.csv'
    (root/'recording.json').write_text(json.dumps(manifest))
    result=catalog.refresh();assert result['recordings']==[] and result['errors']
    with pytest.raises(LibraryError):catalog.get(entry['id'])


def test_raw_reader_rejects_derived_csv_and_bad_metadata(tmp_path):
    root=tmp_path/'records';raw_record(root)
    (root/'ecg.csv').write_text('value\n1\n2\n')
    catalog=RecordingCatalog(root);item=catalog.refresh()['recordings'][0]
    with pytest.raises(LibraryError,match='가공된'):
        with RawRecording(catalog.get(item['id'])):pass
    (root/'meta.json').write_text('{"fs":{"ecg":0,"ppg":100}}')
    assert catalog.refresh()['recordings']==[]


class BlockingBackend:
    def __init__(self):self.entered=Event();self.release=Event()
    def reset(self):pass
    def consume(self,signals,fs,start,end):
        self.entered.set();assert self.release.wait(5)
        return dict(status='collecting',clean_windows=0,baseline_range=None,
            input=asdict(WindowIn(t=end,p_ep=None,ok_ecg=False,ok_ppg=False)),output=None)


def test_user_switch_discards_inflight_events_and_old_profiles(tmp_path):
    root=tmp_path/'records';raw_record(root)
    library=BaselineLibrary(tmp_path/'library');a=library.add_user('A');b=library.add_user('B')
    engine=Dashboard();backend=BlockingBackend()
    local=LocalReplay(engine,library,root,backend_factory=lambda:backend)
    try:
        record=local.catalog.refresh()['recordings'][0]['id']
        local.select(a['id'],record);local.start('calibrate',0)
        assert backend.entered.wait(5)
        old=local.status()['session_id'];fresh=local.select(b['id'],record)
        assert fresh['session_id']!=old and fresh['user']['id']==b['id'] and fresh['profile'] is None
        assert engine.snapshot()['events']==[]
        backend.release.set();time.sleep(.1)
        assert engine.snapshot()['events']==[] and local.status()['user']['id']==b['id']
        with pytest.raises(LibraryError):local.save(a['id'],'old',old)
    finally:
        backend.release.set();local.close();library.close()


def test_replay_requires_user_recording_and_applied_model_profile(tmp_path):
    root=tmp_path/'records';raw_record(root)
    library=BaselineLibrary(tmp_path/'library');local=LocalReplay(Dashboard(),library,root)
    try:
        assert local.status()['stage']=='idle' and local.engine.snapshot()['events']==[]
        with pytest.raises(LibraryError):local.start('replay',0)
        user=library.add_user('empty');record=local.catalog.refresh()['recordings'][0]['id']
        local.select(user['id'],record)
        with pytest.raises(LibraryError,match='기준'):local.start('replay',0)
        with pytest.raises(LibraryError):local.start('calibrate','fast')
    finally:local.close();library.close()
