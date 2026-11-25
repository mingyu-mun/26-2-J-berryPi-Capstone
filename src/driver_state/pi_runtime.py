"""Sensor collection -> bounded buffers -> CPU inference -> webboard events."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path
from threading import Event, RLock, Thread
import json
import logging
import math
import time
import uuid

import numpy as np

from .streaming_io import ChannelBuffer, DataUnavailable, Publisher
from .pi_sensors import Sample, ppg_reader, serial_reader
from .preprocessing import prepare_channel
from .spec import InputSpec

LOG=logging.getLogger(__name__)


class PPGCorrection:
    """Uploaded pipeline's causal EMA + symmetric 23-tap FIR, not a BPM model."""
    coeffs=(172,321,579,927,1360,1858,2390,2916,3391,3768,4012,4096)

    def __init__(self):
        self.reset()

    def reset(self):
        self.dc=0
        self.values=[0]*32
        self.offset=0
        self.count=0

    def process(self,sample):
        if not sample.contact_ok:
            self.reset()
            return replace(sample,t=sample.t-.11,contact_ok=False)
        ir=int(sample.value)
        self.dc+=((ir<<15)-self.dc)>>4
        self.values[self.offset]=ir-(self.dc>>15)
        y=self.coeffs[11]*self.values[(self.offset-11)&31]
        for i in range(11):
            y+=self.coeffs[i]*(self.values[(self.offset-i)&31]+self.values[(self.offset-22+i)&31])
        self.offset=(self.offset+1)&31
        self.count+=1
        return replace(sample,t=sample.t-.11,value=float(y>>15),contact_ok=self.count>=100)


def load_config(path):
    config=json.loads(Path(path).read_text(encoding='utf-8'))
    serial=config['serial']
    if serial['format'] not in ('adc','csv','json') or serial['clock'] not in ('device','sample_count'):
        raise ValueError('unsupported serial format/clock')
    for name in ('fs_hz','baudrate','adc_max','gap_s','vref_mv'):
        if not isinstance(serial[name],(int,float)) or not math.isfinite(serial[name]) or serial[name]<=0:
            raise ValueError(f'invalid serial.{name}')
    if serial['fs_hz']<100:
        raise ValueError('ECG sample rate must be at least 100 Hz; >=250 Hz recommended')
    if serial['fs_hz']>2000 or not 0<=serial.get('reset_wait_s',2.)<=10:
        raise ValueError('invalid ECG rate/reset wait')
    if serial['format']=='csv' and (not isinstance(serial.get('expected_fields'),int) or serial['expected_fields']<1):
        raise ValueError('serial.expected_fields must match the Arduino CSV line')
    ppg=config['ppg']
    if ppg['fs_hz']!=100 or not 0<=ppg['led_current']<=255 or not 0<=ppg['address']<=127:
        raise ValueError('MAX30102 adapter requires configured 100 Hz and valid registers')
    if not 0<ppg['finger_threshold']<262143 or ppg['bus']<0:
        raise ValueError('invalid PPG settings')
    if config['inference']['feature_mode']!='causal_trailing_v1' or config['inference']['experimental'] is not True:
        raise ValueError('explicit experimental causal feature configuration is required')
    if config['inference'].get('threads',1) not in (1,2,3,4):
        raise ValueError('invalid torch thread count')
    return config


def empty_input():
    return {key:None for key in ('t','p_ep','q_ecg','p_ppg','q_ppg','hr_ecg','hr_ppg','ok_ecg','ok_ppg','can_drowsy')}


class Runtime:
    def __init__(self,config,backend,publisher,*,clock=time.monotonic,origin='live',synchronous=False,baseline_profile=None,
                 can_assist=None):
        self.config,self.backend,self.publisher=config,backend,publisher
        # Optional simulator-only CAN steering assist (can_assist.CanAssist); None keeps CAN unavailable.
        self.can_assist=can_assist
        self.clock,self.origin=clock,origin
        self.baseline_profile=baseline_profile
        self.lock=RLock()
        self.stop=Event()
        self.pool=None if synchronous else ThreadPoolExecutor(max_workers=1,thread_name_prefix='jberry-model')
        self.future=None
        self.connections={'ecg':False,'ppg':False}
        self.last_data={'ecg':None,'ppg':None}
        self.sensor_errors={'ecg':None,'ppg':None}
        self.generation=0
        self.backend_generation=-1
        self.retry_at=None
        self.model_error=None
        self.ppg_filter=PPGCorrection()
        self.reset('startup')
        if baseline_profile is not None:
            self.backend.restore_baseline(baseline_profile)
            self.backend_generation=self.generation
            self.result.update(status='ready',clean_windows=self.backend.clean,
                baseline_range=self.backend.baseline_range,ecg_reference=self.backend.ecg_reference,
                profile_loaded=True,input_ready=False)

    def reset(self,reason):
        # Caller holds RLock; generation rejects in-flight old model results.
        with self.lock:
            self.generation+=1
            self.epoch=self.clock()
            prefix='실시간입력방식실험' if self.origin=='live' else '기록입력_실시간방식실험'
            self.session=f'{prefix}_{uuid.uuid4().hex[:12]}'
            self.buffers={c:ChannelBuffer(fs) for c,fs in self.rates.items()}
            self.publisher.clear()
            self.result=dict(status='collecting',clean_windows=0,baseline_range=None,input=None,output=None)
            self.data_start=None
            self.last_wave=-1.
            self.last_status=-1.
            self.last_job=-1.
            self.counts={c:0 for c in self.rates}
            self.report_counts=self.counts.copy()
            self.last_report=0.
            self.retry_at=None
            self.ppg_filter.reset()
            self.confirmation=None
            if self.can_assist is not None:
                self.can_assist.reset()
            LOG.info('session_reset reason=%s session=%s',reason,self.session)

    @property
    def rates(self):
        return {'ecg':float(self.config['serial']['fs_hz']),'ppg':float(self.config['ppg']['fs_hz'])}

    def state(self,channel,connected,reason):
        with self.lock:
            old=self.connections[channel]
            self.connections[channel]=connected
            if old and not connected:
                self.reset(f'{channel}: {reason}')
            if old!=connected or (not connected and self.sensor_errors[channel]!=reason):
                LOG.info('sensor=%s connected=%s reason=%s',channel,connected,reason)
            self.sensor_errors[channel]=None if connected else reason

    def emit(self,channel,sample):
        with self.lock:
            if channel=='ecg':
                sample=replace(sample,value=sample.value*self.config['serial']['vref_mv']/self.config['serial']['adc_max'])
            else:
                sample=self.ppg_filter.process(sample)
            relative=sample.t-self.epoch
            if relative<0:
                return
            try:
                self.buffers[channel].append(replace(sample,t=relative))
                self.counts[channel]+=1
                self.last_data[channel]=self.clock()
            except DataUnavailable as exc:
                self.reset(f'{channel}: {exc}')

    def event(self,kind,t,payload,origin=None):
        return dict(schema_version='1.2',type=kind,session_id=self.session,
                    origin=origin or self.origin,t_rel_s=float(max(0,t)),payload=payload)

    def _status(self,t):
        r=self.result
        baseline=r.get('baseline_range')
        elapsed=max(0,min(t,baseline[1])-baseline[0]) if baseline else max(0,t-(self.data_start or 0)-60)
        if baseline and r.get('profile_loaded'):
            elapsed=baseline[1]-baseline[0]
        baseline_status=r['status'] if all(self.connections.values()) and all(b.samples for b in self.buffers.values()) else 'waiting'
        states={c:('unknown' if self.origin=='recorded' else 'connected' if ok else 'disconnected')
                for c,ok in self.connections.items()}
        states['can']='unknown' if self.can_assist is None else 'connected' if self.can_assist.latest else 'disconnected'
        states['model']='disconnected' if self.model_error else 'connected'
        bio_baseline=dict(status=baseline_status,clean_windows=int(r['clean_windows']),
            required_clean_windows=20,elapsed_s=float(elapsed))
        if baseline_status=='ready' and r.get('ecg_reference') is not None:
            bio_baseline['ecg_reference']=r['ecg_reference']
        return self.event('system.status',t,dict(
            bio_baseline=bio_baseline,
            can_baseline=dict(status='ready' if self.can_assist is not None else 'unavailable',coverage_pct=None,elapsed_s=None),
            connections=states))

    def _wave(self,end):
        channels={}
        spec=InputSpec()
        for c,buffer in self.buffers.items():
            try:
                first=buffer.samples[0].t
                start=max(math.ceil(first*100)/100.,end-5.)
                if end-start<.5:
                    raise DataUnavailable('wave filter warmup')
                raw=buffer.uniform(start,end,require_contact=False)
                filtered,_=prepare_channel(raw,buffer.fs,c,spec,False)
                emit_start=max(start,self.last_wave if self.last_wave>=0 else start)
                a=max(0,int(round((emit_start-start)*100)))
                b=int(round((end-start)*100))
                values=filtered[a:b]
                if not len(values):
                    raise DataUnavailable('no wave samples')
                contact=True
                for sample in reversed(buffer.samples):
                    if sample.t<emit_start:
                        break
                    contact=contact and sample.contact_ok
                channels[c]=dict(t_start_rel_s=float(start+a/100.),fs_hz=100.,
                    samples=np.round(values,6).tolist(),unit='mV' if c=='ecg' else 'a.u.',
                    stage='common_preprocessed',quality_ok=None) if contact else None
            except (DataUnavailable,ValueError,IndexError):
                channels[c]=None
        return self.event('bio.wave',end,channels)

    def _snapshot(self,end):
        if self.data_start is None:
            if not all(b.samples for b in self.buffers.values()):
                return None
            self.data_start=float(math.ceil(max(b.samples[0].t for b in self.buffers.values())))
        start=self.data_start if self.result['status']!='ready' else max(self.data_start,end-330.)
        if end-start<30:
            return None
        signals={}
        for c,buffer in self.buffers.items():
            try:
                contact_start=self.data_start+60 if self.result['status']!='ready' else start
                signals[c]=buffer.uniform(start,end,contact_start=contact_start)
            except DataUnavailable:
                pass
        return signals,self.rates,start,end

    @property
    def can_origin(self):
        return 'unavailable' if self.can_assist is None else self.can_assist.origin

    def _confirmation(self,output,can_flag):
        # Level 2 reached in a window whose CAN flag was True -> CAN confirmed; otherwise persistence.
        if not output or output['level']!=2:
            self.confirmation=None
        elif self.confirmation is None:
            self.confirmation='can' if can_flag is True else 'persistence'
        return self.confirmation

    def _work(self,args,generation,can_flag=None):
        began=time.perf_counter()
        try:
            if self.backend_generation!=generation:
                self.backend.reset()
                if self.baseline_profile is not None:
                    self.backend.restore_baseline(self.baseline_profile)
                self.backend_generation=generation
            result=self.backend.consume(*args,can_drowsy=can_flag) if self.can_assist is not None else self.backend.consume(*args)
            with self.lock:
                if generation!=self.generation:
                    return
                self.model_error=None
                self.result=result
                if result['status']=='retry_required' and self.retry_at is None:
                    self.retry_at=self.clock()+30
                inp=result['input']
                end=args[-1]
                if inp is not None and self.clock()-self.epoch-end<25:
                    output=result['output']
                    self.publisher.put(self.event('ai.decision',end,dict(
                        model_id='S1-EP',state_machine_id='v2',window_start_s=end-30,window_end_s=end,
                        baseline_status=result['status'],bio_origin=self.origin,can_origin=self.can_origin,
                        input=inp,output=output,
                        confirmation_source=self._confirmation(output,inp.get('can_drowsy')))))
                LOG.info('model elapsed_s=%.3f window_end_s=%.1f baseline=%s',time.perf_counter()-began,end,result['status'])
        except Exception as exc:
            with self.lock:
                if generation==self.generation:
                    self.model_error=str(exc)
                    LOG.exception('model_failed')

    def tick(self):
        with self.lock:
            if self.retry_at is not None and self.clock()>=self.retry_at:
                self.reset('automatic baseline retry after 30 seconds')
            now=self.clock()-self.epoch
            if now-self.last_report>=10:
                duration=now-self.last_report
                rates={c:(n-self.report_counts[c])/duration for c,n in self.counts.items()}
                LOG.info('acquisition ecg_rx_hz=%.2f ppg_rx_hz=%.2f ecg_connected=%s ppg_connected=%s',
                         rates['ecg'],rates['ppg'],self.connections['ecg'],self.connections['ppg'])
                self.report_counts=self.counts.copy()
                self.last_report=now
            # Include the 110 ms PPG FIR delay and FIFO delivery margin.
            wave_end=math.floor(max(0,now-.15))
            if wave_end>=1. and wave_end>self.last_wave:
                self.publisher.put(self._wave(wave_end))
                self.last_wave=wave_end
            if self.can_assist is not None:
                payload=self.can_assist.advance(now)
                if payload is not None:
                    self.publisher.put(self.event('can.telemetry',now,payload,origin=self.can_origin))
            if now-self.last_status>=1:
                self.publisher.put(self._status(now))
                self.last_status=now
            end=math.floor(max(0,now-.15)/10)*10
            if end>=30 and end>self.last_job and (self.future is None or self.future.done()):
                args=self._snapshot(end)
                if args is not None:
                    generation=self.generation
                    self.last_job=end
                    can_flag=self.can_assist.window_flag(end) if self.can_assist is not None else None
                    if self.pool:
                        self.future=self.pool.submit(self._work,args,generation,can_flag)
                    else:
                        self._work(args,generation,can_flag)

    def run(self):
        threads=[Thread(target=serial_reader,args=(self.config['serial'],self.emit,self.state,self.stop),daemon=True),
                 Thread(target=ppg_reader,args=(self.config['ppg'],self.emit,self.state,self.stop),daemon=True)]
        for thread in threads:
            thread.start()
        # HTTP must not block the sensor polling or model scheduling thread.
        def publish_loop():
            while not self.stop.wait(.1):
                self.publisher.flush()
        sender=Thread(target=publish_loop,daemon=True);sender.start()
        try:
            while not self.stop.wait(.05):
                self.tick()
        finally:
            self.stop.set()
            for thread in threads:
                thread.join(3)
            sender.join(3)
            self.close()

    def close(self):
        self.stop.set()
        if self.pool:
            self.pool.shutdown(wait=True,cancel_futures=True)
        with self.lock:
            t=self.clock()-self.epoch
            self.result=dict(status='waiting',clean_windows=0,baseline_range=None,input=None,output=None)
            self.connections={'ecg':False,'ppg':False}
            self.model_error='stopped'
            self.publisher.put(self.event('bio.wave',t,dict(ecg=None,ppg=None)))
            self.publisher.put(self.event('ai.decision',t,dict(model_id='S1-EP',state_machine_id='v2',
                window_start_s=None,window_end_s=None,baseline_status='waiting',bio_origin=self.origin,
                can_origin=self.can_origin,input=empty_input(),output=None,confirmation_source=None)))
            self.publisher.put(self._status(t))
        self.publisher.flush()
        self.publisher.close()
