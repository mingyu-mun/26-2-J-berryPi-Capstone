"""Fake hardware checks only; application sensor mode has no simulated source."""
from dataclasses import asdict
from pathlib import Path
from threading import Event
import io
import json
import shutil
import subprocess
import sys

import numpy as np
import pytest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from driver_state.pi_sensors import MAX30102, Sample, SampleGap, SerialDecoder, serial_reader
from driver_state.pi_runtime import PPGCorrection, Runtime, load_config
from driver_state.streaming_io import Publisher
from driver_state.state_machine_v2 import WindowIn
from driver_state.web_dashboard import Dashboard, validate_event


@pytest.fixture
def config():
    cfg=load_config(ROOT/'tests/sensors.example.json')
    cfg['serial']['reset_wait_s']=0.
    return cfg


def test_default_serial_format_matches_uploaded_receiver(config):
    decoder=SerialDecoder(config['serial'])
    assert decoder.parse('512,0',10.).value==512
    assert decoder.parse('500,1',10.01).contact_ok  # second field has no confirmed meaning
    assert not decoder.parse('-1,0',10.02).contact_ok
    with pytest.raises(ValueError,match='field count'):decoder.parse('512',10.03)


@pytest.mark.parametrize('line',['nan,0','-0.5,0','1024,0','1,2,3','text,0'])
def test_bad_serial_data_is_not_a_measurement(config,line):
    with pytest.raises(ValueError):SerialDecoder(config['serial']).parse(line,0.)


def test_device_time_and_sequence_loss_are_explicit(config):
    cfg=config['serial']|dict(format='json',clock='device')
    d=SerialDecoder(cfg)
    assert d.parse('{"t_us":1000,"adc":512,"seq":1}',10.).t==10.
    assert d.parse('{"t_us":5000,"adc":500,"seq":2}',10.01).t==pytest.approx(10.004)
    with pytest.raises(SampleGap,match='sequence'):
        d.parse('{"t_us":9000,"adc":501,"seq":4}',10.02)


def test_reader_opens_configured_port_and_closes(config):
    calls=[]; samples=[];stop=Event()
    class Serial:
        def __init__(self,*args,**kwargs):calls.append(args)
        def reset_input_buffer(self):calls.append('drain')
        def readline(self,*_):return b'512,0\n'
        def close(self):calls.append('close')
    def emit(c,s):samples.append((c,s));stop.set()
    serial_reader(config['serial']|dict(port='/dev/test'),emit,lambda *_:None,stop,
                  clock=lambda:10.,serial_factory=Serial)
    assert calls[0]==('/dev/test',115200) and calls[-1]=='close'
    assert samples[0][1].value==512


class Bus:
    def __init__(self):
        self.registers={0xFF:0x15,0x09:0,0x04:0,0x06:0,0x05:0}
        self.data=[];self.requests=[]
    def read_byte_data(self,a,r):return self.registers.get(r,0)
    def write_byte_data(self,a,r,v):self.registers[r]=0 if r==0x09 and v==0x40 else v
    def read_i2c_block_data(self,a,r,n):
        self.requests.append(n)
        result=self.data[:n];self.data=self.data[n:]
        self.registers[0x06]=(self.registers[0x06]+n//6)&31
        return result
    def add(self,red,ir):
        for v in (red,ir):self.data.extend([(v>>16)&255,(v>>8)&255,v&255])
        self.registers[0x04]=(self.registers[0x04]+1)&31


def test_max30102_settings_fifo_and_loss(config):
    bus=Bus();sensor=MAX30102(bus,config['ppg']);sensor.configure()
    assert bus.registers[0x0A]==0x27 and bus.registers[0x0C]==0x24
    for i in range(7):bus.add(50000,60000 if i else 100)
    result=sensor.poll(100.)
    assert len(result)==7 and result[-1].t==pytest.approx(100.)
    assert not result[0].contact_ok and result[1].contact_ok
    assert bus.requests==[30,12]  # SMBus limit stays below 32 bytes
    bus.registers[0x05]=1
    with pytest.raises(SampleGap,match='overflow'):sensor.poll(100.1)


def test_max30102_contact_is_separate_from_clipping(config):
    bus=Bus();sensor=MAX30102(bus,config['ppg']);sensor.configure()
    bus.add(262143,262143)
    sample=sensor.poll(1.)[0]
    assert sample.value==262143 and sample.contact_ok


def test_max30102_clock_stays_monotonic_during_small_drift(config):
    bus=Bus();sensor=MAX30102(bus,config['ppg']);sensor.configure()
    previous=-1.
    for i in range(200):
        bus.add(50000,60000)
        now=i/101.
        sample=sensor.poll(now)[0]
        assert sample.t>previous and abs(sample.t-now)<.05
        previous=sample.t


def test_ppg_warmup_delay_and_loss_reset():
    correction=PPGCorrection()
    results=[correction.process(Sample(i/100.,60000.)) for i in range(500)]
    assert not results[0].contact_ok and results[99].contact_ok
    assert results[-1].t==pytest.approx(4.99-.11)
    assert abs(results[-1].value)<=2
    assert not correction.process(Sample(5.,100.,False)).contact_ok
    assert not correction.process(Sample(5.01,60000.)).contact_ok


class Backend:
    def reset(self):pass
    def consume(self,signals,fs,start,end):
        return dict(status='collecting',clean_windows=0,baseline_range=None,
                    input=asdict(WindowIn(t=end,p_ep=None,ok_ecg=False,ok_ppg=False)),output=None)


def test_no_sensor_creates_no_measurements_and_reports_waiting(config):
    now=[0.];pub=Publisher('http://test',clock=lambda:now[0])
    runtime=Runtime(config,Backend(),pub,clock=lambda:now[0],synchronous=True)
    validator=Dashboard().validator
    for t in (1.,2.,3.):now[0]=t;runtime.tick()
    for _,e in pub.pending:
        validate_event(e,validator)
        if e['type']=='bio.wave':assert e['payload']==dict(ecg=None,ppg=None)
        if e['type']=='system.status':assert e['payload']['bio_baseline']['status']=='waiting'
    assert not any(e['type']=='ai.decision' for _,e in pub.pending)


def test_ecg_voltage_and_disconnect_reset(config):
    now=[0.];pub=Publisher('http://test',clock=lambda:now[0])
    runtime=Runtime(config,Backend(),pub,clock=lambda:now[0],synchronous=True)
    runtime.state('ecg',True,'USB')
    runtime.emit('ecg',Sample(.1,512))
    assert runtime.buffers['ecg'].samples[-1].value==pytest.approx(512*5000/1023)
    old=runtime.session;now[0]=2.;runtime.state('ecg',False,'unplugged')
    assert runtime.session!=old and not runtime.buffers['ecg'].samples


def test_complete_one_second_wave_intervals(config):
    now=[0.];pub=Publisher('http://test',clock=lambda:now[0])
    cfg=json.loads(json.dumps(config));cfg['serial']['fs_hz']=100.
    runtime=Runtime(cfg,Backend(),pub,clock=lambda:now[0],synchronous=True)
    for c in ('ecg','ppg'):runtime.state(c,True,'test fixture')
    for i in range(250):
        t=i/100.
        runtime.emit('ecg',Sample(t,512.+50*np.sin(2*np.pi*t)))
        runtime.emit('ppg',Sample(t,60000.+1000*np.sin(2*np.pi*t)))
    for t in (2.2,3.2):now[0]=t;runtime.tick()
    waves=[e for _,e in pub.pending if e['type']=='bio.wave']
    # The second event cannot extrapolate input that has not been received.
    assert waves[0]['payload']['ecg'] and len(waves[0]['payload']['ecg']['samples'])==200
    assert waves[1]['payload']['ecg'] is None
    validator=Dashboard().validator
    for _,event in pub.pending:validate_event(event,validator)


def test_model_does_not_filter_future_samples(monkeypatch):
    import driver_state.streaming_model as module
    from driver_state.teacher.rr import RRSeries
    from driver_state.teacher.spec import TeacherSpec
    seen=[]
    def extract(rec,spec,channel):
        seen.append(rec.signals[channel].copy())
        beats=np.arange(0.,60.,.8)
        return RRSeries('test','test','test','test',0.,60.,beats,np.diff(beats),
                        np.zeros(len(beats)-1,bool),np.ones(6,bool),np.full(60,-1))
    monkeypatch.setattr(module,'extract_rr',extract)
    model=module.CausalStudent.__new__(module.CausalStudent)
    model.spec=TeacherSpec();model.reset()
    result=model.consume({'ecg':np.r_[np.ones(6000),np.full(1000,np.nan)]},{'ecg':100},0.,60.)
    assert result['output'] is None and len(seen[0])==6000 and np.isfinite(seen[0]).all()


def test_long_features_do_not_access_future_rr():
    from driver_state.streaming_model import trailing_features
    from driver_state.teacher.rr import RRSeries
    from driver_state.teacher.spec import TeacherSpec
    beats=np.arange(0.,700.,.8)
    rr=RRSeries('test','test','test','test',0.,700.,beats,np.diff(beats),
                np.zeros(len(beats)-1,bool),np.ones(70,bool),np.full(700,-1))
    before=trailing_features(rr,360.,TeacherSpec())
    rr.rr[rr.rr_t>=360]=1.5
    np.testing.assert_allclose(before,trailing_features(rr,360.,TeacherSpec()),equal_nan=True)


def test_baseline_hr_uses_only_accepted_windows_with_valid_nn_coverage():
    from driver_state.streaming_model import baseline_heart_rate
    from driver_state.teacher.rr import RRSeries
    from driver_state.teacher.spec import TeacherSpec
    beats=np.arange(0.,400.,.8)
    rr=RRSeries('test','test','test','test',0.,400.,beats,np.diff(beats),
                np.zeros(len(beats)-1,bool),np.ones(40,bool),np.full(400,-1))
    assert baseline_heart_rate(rr,[60.,70.,80.],TeacherSpec())==pytest.approx(dict(mean_bpm=75.,clean_windows=3))
    rr.abnormal[(rr.rr_t>=60)&(rr.rr_t<90)]=True
    reference=baseline_heart_rate(rr,[60.,100.],TeacherSpec())
    assert reference==pytest.approx(dict(mean_bpm=75.,clean_windows=1))
    rr.rr[rr.rr_t>=130]=1.5
    assert baseline_heart_rate(rr,[60.,100.],TeacherSpec())==reference
    assert baseline_heart_rate(rr,[60.],TeacherSpec()) is None


def test_runtime_publishes_ready_reference_and_clears_it_on_reset(config):
    now=[0.];pub=Publisher('http://test',clock=lambda:now[0])
    runtime=Runtime(config,Backend(),pub,clock=lambda:now[0],synchronous=True)
    now[0]=370.
    for channel,value in [('ecg',512.),('ppg',60000.)]:
        runtime.state(channel,True,'test fixture')
        runtime.emit(channel,Sample(369.9,value))
    reference=dict(mean_bpm=75.,clean_windows=21,start_s=60.,end_s=360.)
    runtime.result.update(status='ready',baseline_range=[60.,360.],clean_windows=21,ecg_reference=reference)
    event=runtime._status(370.)
    validate_event(event,Dashboard().validator)
    assert event['payload']['bio_baseline']['ecg_reference']==reference
    runtime.reset('new measurement')
    assert 'ecg_reference' not in runtime._status(0.)['payload']['bio_baseline']


def test_publisher_accepts_idempotent_retries():
    event=dict(session_id='a',type='system.status',t_rel_s=1.,payload={})
    def opener(*args,**kwargs):
        return io.StringIO(json.dumps(dict(accepted=0,session_id='a',latest_seq=1)))
    publisher=Publisher('http://test',opener=opener)
    publisher.put(event)
    assert publisher.flush() and publisher.last_error is None
    publisher.put(event)
    assert not publisher.pending


def test_publisher_rejects_retired_sessions():
    def opener(*args,**kwargs):
        return io.StringIO(json.dumps(dict(accepted=0,session_id='different',latest_seq=1)))
    publisher=Publisher('http://test',opener=opener)
    publisher.put(dict(session_id='old',type='system.status',t_rel_s=1.,payload={}))
    assert not publisher.flush() and publisher.last_error


@pytest.mark.skipif(not sys.platform.startswith('linux') or not shutil.which('bash'),
                    reason='Pi launcher requires Linux and bash')
@pytest.mark.parametrize('action', ['check', 'run'])
@pytest.mark.parametrize('new_install', [False, True])
def test_moved_launcher_keeps_existing_installation_and_prefers_new_paths(tmp_path, action, new_install):
    project=tmp_path/'project with spaces'
    current=project/'tests';current.mkdir(parents=True)
    previous=project/'pi-integration-test';previous.mkdir()
    shutil.copyfile(ROOT/'tests/pi_test.sh',current/'pi_test.sh')
    original='{"serial":{"port":"/dev/serial/by-id/existing"}}\n'
    old_config=previous/'sensors.local.json';old_config.write_text(original)
    for directory in (previous,current) if new_install else (previous,):
        python=directory/'.venv/bin/python';python.parent.mkdir(parents=True)
        python.write_text('#!/usr/bin/env bash\nprintf "%s\\n" "$0" "$@"\n')
        python.chmod(0o755)
    if new_install:
        (current/'sensors.local.json').write_text('{"new":true}\n')
    expected=current if new_install else previous
    completed=subprocess.run(['bash',str(current/'pi_test.sh'),action,'--port','8002'],
                             cwd=tmp_path,capture_output=True,text=True,check=True)
    assert completed.stdout.splitlines()==[
        str(expected/'.venv/bin/python'),str(current/'run.py'),
        '--config',str(expected/'sensors.local.json'),
        *(['--check'] if action=='check' else []),'--port','8002']
    assert old_config.read_text()==original
    assert not (previous/'results').exists()
