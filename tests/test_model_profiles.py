"""Numeric profile fixtures test persistence, compatibility and real-weight equivalence."""
import copy
import json
from pathlib import Path
import sys
from threading import Thread
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import numpy as np
import pytest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from driver_state.model_profiles import ModelProfileStore, model_identity, validate_profile
from driver_state.webboard_library import BaselineLibrary, LibraryError


@pytest.fixture
def profile():
    names=json.loads((ROOT/'models/bundle.json').read_text())['rr_feature_names']
    stats=np.ones((2,2,36),np.float64)
    stats[:,1,-1]=np.nan  # Insufficient baseline windows for a spread is legitimate.
    return dict(metadata=dict(format_version=1,identity=model_identity(ROOT/'models'),processing={'test':1},
        feature_names=names,clean_windows=21,start_s=60.,end_s=360.,source_session_id='fixture-source',
        ecg_reference=dict(mean_bpm=75.,clean_windows=21,start_s=60.,end_s=360.)),
        statistics=stats,references=np.linspace(-1,1,192,dtype=np.float32).reshape(3,2,32))


def test_profiles_persist_and_preserve_missing_statistics(tmp_path,profile):
    lib=BaselineLibrary(tmp_path);store=ModelProfileStore(lib)
    user=lib.add_user('사용자1');other=lib.add_user('사용자2')
    saved=store.save(user['id'],'첫 기준',profile)
    restored=store.load(user['id'],saved['id'],profile['metadata']['identity'],{'test':1})
    np.testing.assert_array_equal(restored['statistics'],profile['statistics'])
    np.testing.assert_array_equal(restored['references'],profile['references'])
    assert store.list(other['id'])==[]
    with pytest.raises(LibraryError,match='찾을 수'):store.load(other['id'],saved['id'])
    lib.close()
    lib=BaselineLibrary(tmp_path);store=ModelProfileStore(lib)
    assert store.load(user['id'],saved['id'])['metadata']['user_id']==user['id']
    lib.close()


def test_incomplete_corrupt_or_incompatible_profiles_are_rejected(tmp_path,profile):
    lib=BaselineLibrary(tmp_path);store=ModelProfileStore(lib);user=lib.add_user('검사')
    for change in ('missing','shape','infinite','low_quality','order','dtype'):
        bad=copy.deepcopy(profile)
        if change=='missing':del bad['metadata']['feature_names']
        elif change=='shape':bad['references']=bad['references'][:2]
        elif change=='infinite':bad['references'][0,0,0]=np.inf
        elif change=='low_quality':bad['metadata']['clean_windows']=19
        elif change=='order':bad['metadata']['feature_names']=list(reversed(bad['metadata']['feature_names']))
        else:bad['references']=bad['references'].astype(np.float64)
        with pytest.raises(LibraryError):store.save(user['id'],'실패',bad)
    assert store.list(user['id'])==[] and not list(store.directory.glob('*.npz'))
    saved=store.save(user['id'],'유효',profile)
    with pytest.raises(LibraryError,match='버전'):store.load(user['id'],saved['id'],{'changed':True})
    with pytest.raises(LibraryError,match='설정'):store.load(user['id'],saved['id'],processing={'test':2})
    path=store.directory/(saved['id']+'.npz');path.write_bytes(b'incomplete')
    with pytest.raises(LibraryError,match='손상'):store.load(user['id'],saved['id'])
    lib.close()


def test_local_api_applies_owned_profiles_and_keeps_external_packets_separate(tmp_path,profile):
    from driver_state.web_dashboard import create_server
    root=tmp_path/'raw';root.mkdir()
    (root/'ecg.csv').write_text('time,adc\n0,512\n')
    (root/'ppg.csv').write_text('time,red,ir\n0,50000,60000\n')
    (root/'meta.json').write_text('{"fs":{"ecg":100,"ppg":100}}')
    (root/'recording.json').write_text('{"ecg":"ecg.csv","ppg":"ppg.csv","meta":"meta.json"}')
    server=create_server('127.0.0.1',0,recordings_dir=root,library_dir=tmp_path/'library')
    thread=Thread(target=server.serve_forever,daemon=True);thread.start()
    def api(path,data=None,origin=None):
        headers={'Content-Type':'application/json'}
        if origin:headers['Origin']=origin
        req=Request(f'http://127.0.0.1:{server.server_port}'+path,
                    data=None if data is None else json.dumps(data).encode(),headers=headers)
        with urlopen(req,timeout=5) as result:return json.load(result)
    try:
        assert api('/api/local/session')['stage']=='idle'
        assert api('/api/webboard')['events']==[]
        a=api('/api/users',{'nickname':'A'});b=api('/api/users',{'nickname':'B'})
        record=api('/api/local/recordings')['recordings'][0]['id']
        state=api('/api/local/select',{'user_id':a['id'],'recording_id':record})
        profile['metadata']['processing']=server.local_replay.processing(server.local_replay.recording)
        saved=server.local_replay.profiles.save(a['id'],'모델 기준',profile)
        assert len(api(f"/api/users/{a['id']}/model-profiles")['profiles'])==1
        applied=api('/api/local/apply',{'user_id':a['id'],'profile_id':saved['id']})
        assert applied['profile']['profile_id']==saved['id'] and not applied['profile_applied']
        with pytest.raises(HTTPError) as error:api('/api/local/select',{},'https://other-site.invalid')
        assert error.value.code==403
        event=json.loads((ROOT/'web/dashboard/contract/empty_status_template.json').read_text())
        with pytest.raises(HTTPError) as error:api('/api/events',event)
        assert error.value.code==409
        changed=api('/api/local/select',{'user_id':b['id'],'recording_id':record})
        assert changed['profile'] is None and api('/api/webboard')['events']==[]
        with pytest.raises(HTTPError) as error:api('/api/local/apply',{'user_id':b['id'],'profile_id':saved['id']})
        assert error.value.code==404
    finally:
        server.shutdown();server.server_close();thread.join(2)


def test_restored_actual_weights_produce_identical_scores_and_wait_for_current_history(tmp_path,monkeypatch):
    from driver_state.streaming_model import CausalStudent
    import driver_state.streaming_model as module
    from driver_state.teacher.rr import RRSeries
    # Synthetic waveforms/beat times are isolated test fixtures, never an app source.
    fs=100.;t=np.arange(43000)/fs
    signals={'ecg':np.sin(2*np.pi*1.25*t)+.2*np.sin(2*np.pi*9*t),
             'ppg':np.sin(2*np.pi*1.25*t-.3)+.1*np.sin(2*np.pi*2.5*t)}
    beats=np.r_[0.,np.cumsum(.8+.02*np.sin(np.arange(600)*.29))]
    def extract(rec,spec,channel):
        duration=len(rec.signals[channel])/rec.fs[channel]
        b=beats[beats<duration]
        return RRSeries('test','test','test','test',0.,duration,b,np.diff(b),
                        np.zeros(len(b)-1,bool),np.ones(int(duration/10),bool),np.full(int(duration),-1))
    monkeypatch.setattr(module,'extract_rr',extract)
    first=CausalStudent(ROOT/'models',feature_mode='causal_trailing_v1',experimental=True)
    expected=first.consume(signals,{'ecg':fs,'ppg':fs},0.,430.)
    assert expected['status']=='ready' and expected['input']['p_ep'] is not None
    snapshot=first.export_baseline(source_session_id='actual-weight-test',processing={'fixture':True})
    lib=BaselineLibrary(tmp_path);store=ModelProfileStore(lib);user=lib.add_user('동등성')
    saved=store.save(user['id'],'수치 검증',snapshot)
    restored=store.load(user['id'],saved['id'],first.identity,{'fixture':True})
    second=CausalStudent(ROOT/'models',feature_mode='causal_trailing_v1',experimental=True)
    second.restore_baseline(restored)
    early=second.consume(signals,{'ecg':fs,'ppg':fs},0.,40.)
    assert early['status']=='ready' and not early['input_ready'] and early['output'] is None
    assert early['input']['p_ep'] is None
    actual=second.consume(signals,{'ecg':fs,'ppg':fs},0.,430.)
    assert actual['input_ready']
    assert actual['input']['p_ep']==expected['input']['p_ep']
    assert actual['input']['p_ppg']==expected['input']['p_ppg']
    assert actual['output']==expected['output']
    assert second.zero==0. and second.sm.up==first.sm.up
    lib.close()
