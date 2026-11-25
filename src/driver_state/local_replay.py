"""One server-owned local replay session; model work is serialized off HTTP threads."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Condition, RLock
import logging
import time
import uuid

from .local_recordings import RawRecording, RecordingCatalog
from .model_profiles import ModelProfileStore, model_identity, validate_profile
from .webboard_library import LibraryError

ROOT = Path(__file__).resolve().parents[2]
LOG = logging.getLogger(__name__)


class LocalReplay:
    def __init__(self, engine, library, recordings_dir, model_dir=ROOT/'models', backend_factory=None):
        self.engine, self.library = engine, library
        self.catalog = RecordingCatalog(recordings_dir)
        self.profiles = ModelProfileStore(library)
        self.model_dir = Path(model_dir)
        self.backend_factory = backend_factory
        self.backend = None
        self.lock = RLock()
        self.condition = Condition(self.lock)
        self.pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix='local-bio-replay')
        self.future = None
        self.generation = 0
        self.revision = 0
        self.controller_id = uuid.uuid4().hex
        self.closed = False
        self.user = self.recording = self.profile = self.candidate = None
        self.state = dict(stage='idle', mode=None, session_id=None, position_s=0., duration_s=None,
                          input_ready=False, baseline_status='waiting', clean_windows=0,
                          speed=1., error=None, paused=False,profile_applied=False)

    def status(self):
        with self.lock:
            return dict(self.state, user=self.user,revision=self.revision,controller_id=self.controller_id,
                recording=self.catalog.public(self.recording) if self.recording else None,
                profile=({k:self.profile['metadata'][k] for k in ('profile_id','title','user_id','created_at')}
                         if self.profile else None), can_save=self.candidate is not None)

    def _clear(self):
        self.generation += 1
        self.revision += 1
        self.candidate = None
        if self.future:
            self.future.cancel()
        self.condition.notify_all()
        self.state.update(session_id='local_'+uuid.uuid4().hex, position_s=0., duration_s=None,
                          input_ready=False, baseline_status='waiting', clean_windows=0,
                          paused=False, error=None,profile_applied=False)
        self.engine.begin_session(self.state['session_id'])

    def config(self, recording):
        from .pi_runtime import load_config
        config = load_config(ROOT/'tests/sensors.example.json')
        config['serial']['fs_hz'] = recording['rates']['ecg']
        return config

    def processing(self, recording):
        cfg = self.config(recording)
        # Source rates are provenance, not a personal identity. The same code
        # resamples each source using its own metadata to common/model rates.
        return dict(common_fs_hz=100., serial={k:cfg['serial'][k] for k in ('adc_max','vref_mv')},
                    ppg={k:cfg['ppg'][k] for k in ('fs_hz','finger_threshold')}, clock='sample_index')

    def select(self, user_id, recording_id):
        user = self.library.user(user_id) if user_id else None
        recording = self.catalog.get(recording_id) if recording_id else None
        with self.condition:
            if self.user is None or user is None or self.user['id'] != user['id']:
                self.profile = None
            self._clear()
            self.user, self.recording = user, recording
            self.state.update(stage='prepared', mode=None)
            if self.profile and recording:
                try:
                    validate_profile(self.profile, model_identity(self.model_dir), self.processing(recording))
                except LibraryError as exc:
                    self.profile = None
                    self.state['error'] = str(exc)
            return self.status()

    def apply(self, user_id, profile_id):
        with self.condition:
            if not self.user or user_id != self.user['id']:
                raise LibraryError('먼저 측정할 사용자를 선택하세요.', 409)
            snapshot = self.profiles.load(user_id, profile_id, model_identity(self.model_dir),
                self.processing(self.recording) if self.recording else None)
            self._clear()
            self.profile = snapshot
            self.state.update(stage='prepared',mode=None,baseline_status='ready')
            return self.status()

    def save(self, user_id, title, session_id):
        with self.lock:
            if not self.user or user_id != self.user['id'] or session_id != self.state['session_id'] or self.candidate is None:
                raise LibraryError('현재 사용자의 기준 측정이 완료된 뒤 저장하세요.', 409)
            result = self.profiles.save(user_id,title,self.candidate)
            self.profile = self.profiles.load(user_id,result['id'])
            self.candidate = None
            self.revision += 1
            return result

    def start(self, mode, speed=1., paused=False):
        if mode not in ('calibrate','replay') or isinstance(speed,bool) or speed not in (0,1,5,10):
            raise LibraryError('기준 측정 또는 기록 재생 모드와 공급 속도를 확인하세요.')
        with self.condition:
            if self.closed or not self.user or not self.recording:
                raise LibraryError('사용자와 로컬 원본 기록을 먼저 선택하세요.',409)
            recording = self.catalog.get(self.recording['id'])
            profile = self.profile if mode=='replay' else None
            if mode=='replay':
                if profile is None:
                    raise LibraryError('모델용 기준을 먼저 불러오세요.',409)
                validate_profile(profile,model_identity(self.model_dir),self.processing(recording))
            self._clear()
            self.recording = recording
            self.state.update(stage='paused' if paused else 'loading',mode=mode,speed=float(speed),paused=paused,
                              baseline_status='ready' if profile else 'collecting')
            generation = self.generation
            self.future = self.pool.submit(self._run,generation,recording,profile)
            return self.status()

    def control(self, action):
        if action=='rewind':
            with self.lock:
                mode,speed=self.state['mode'],self.state['speed']
            return self.start(mode,speed,paused=True)
        with self.condition:
            self.revision += 1
            if action=='stop':
                self._clear()
                self.state.update(stage='prepared',mode=None,baseline_status='ready' if self.profile else 'waiting')
            elif action=='pause' and self.state['stage'] in ('loading','playing'):
                self.state.update(paused=True,stage='paused')
            elif action=='resume' and self.state['stage']=='paused':
                self.state.update(paused=False,stage='playing')
                self.condition.notify_all()
            else:
                raise LibraryError('현재 재생 상태에서 사용할 수 없는 제어입니다.',409)
            return self.status()

    def _run(self, generation, recording, profile):
        owner = self
        class Sink:
            def put(self,event):
                with owner.lock:
                    if generation==owner.generation and not owner.closed:
                        owner.engine.ingest(event)
            def clear(self): pass
            def flush(self): return True
        try:
            from .pi_runtime import Runtime
            with self.lock:
                if generation!=self.generation or self.closed:
                    return
            if self.backend is None:
                if self.backend_factory:
                    self.backend=self.backend_factory()
                else:
                    from .streaming_model import CausalStudent
                    self.backend=CausalStudent(self.model_dir,feature_mode='causal_trailing_v1',experimental=True)
            clock=[0.]
            runtime=Runtime(self.config(recording),self.backend,Sink(),clock=lambda:clock[0],
                            origin='recorded',synchronous=True,baseline_profile=profile)
            # Stable controller session ID also labels profile provenance and events.
            with self.lock:
                if generation!=self.generation or self.closed:
                    return
                runtime.session=self.state['session_id']
                self.state['profile_applied']=profile is not None
                self.revision+=1
            for c in ('ecg','ppg'):
                runtime.state(c,True,'local raw recording; hardware not opened')
            with RawRecording(recording) as source:
                # This synchronous Runtime starts no sensor/model threads.
                # Keep its last recorded frame instead of emitting live shutdown events.
                with self.lock:
                    if generation!=self.generation or self.closed:
                        return
                    self.state['duration_s']=source.duration
                    self.revision+=1
                previous=time.monotonic()
                for second in range(1,int(source.duration)+1):
                    with self.condition:
                        while generation==self.generation and not self.closed and self.state['paused']:
                            self.condition.wait()
                            previous=time.monotonic()
                        if generation!=self.generation or self.closed:
                            return
                        speed=self.state['speed']
                        remaining=max(0.,1./speed-(time.monotonic()-previous)) if speed else 0.
                        if remaining:
                            self.condition.wait(timeout=remaining)
                            if generation!=self.generation or self.closed:
                                return
                            if self.state['paused']:
                                while self.state['paused'] and generation==self.generation and not self.closed:
                                    self.condition.wait()
                            if generation!=self.generation or self.closed:
                                return
                        self.state['stage']='playing'
                    clock[0]=float(second)
                    if not source.emit_until(runtime,second):
                        raise LibraryError('기록이 처리 도중 변경됐거나 원본 표본이 부족합니다.')
                    runtime.tick()
                    if runtime.session!=self.state['session_id']:
                        with self.lock:
                            if generation!=self.generation:
                                return
                        raise LibraryError('원본 입력이 측정 세션을 초기화했습니다. 기록의 연속성을 확인하세요.')
                    if runtime.model_error:
                        raise LibraryError(runtime.model_error)
                    with self.lock:
                        if generation!=self.generation or self.closed:
                            return
                        self.state.update(position_s=float(second),input_ready=runtime.result.get('input_ready',False),
                                          baseline_status=runtime.result['status'],clean_windows=runtime.result['clean_windows'])
                        self.revision+=1
                        if runtime.result['status']=='retry_required':
                            raise LibraryError('유효한 기준 구간이 부족합니다. 다른 기준 기록을 선택하세요.')
                        if self.state['mode']=='calibrate' and runtime.result['status']=='ready':
                            # tick emits periodic status before the model job.
                            # Publish the newly computed ready state before stopping.
                            Sink().put(runtime._status(clock[0]))
                            self.candidate=self.backend.export_baseline(source_session_id=runtime.session,
                                                                       processing=self.processing(recording))
                            self.candidate['metadata']['recording']=self.catalog.public(recording)
                            self.state['stage']='baseline_ready'
                            return
                    previous=time.monotonic()
                with self.lock:
                    if generation==self.generation:
                        if self.state['mode']=='calibrate':
                            raise LibraryError('기록이 기준 준비 전에 끝났습니다. 충분한 길이와 품질의 원본이 필요합니다.')
                        if not runtime.result.get('input_ready') or runtime.result.get('output') is None:
                            raise LibraryError('현재 입력 준비 또는 유효한 판정 전에 기록이 끝났습니다. 더 길고 양호한 원본을 선택하세요.')
                        self.state['stage']='completed'
                        self.revision+=1
        except Exception as exc:
            LOG.exception('local_replay_failed')
            with self.lock:
                if generation==self.generation and not self.closed:
                    self.candidate=None
                    self.state.update(stage='error',error=str(exc),paused=False)
                    self.revision+=1

    def close(self):
        with self.condition:
            self.closed=True
            self.generation+=1
            self.condition.notify_all()
        self.pool.shutdown(wait=True,cancel_futures=True)
