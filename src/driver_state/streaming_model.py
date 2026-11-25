"""Experimental causal adapter for the existing S1-EP weights.

Long RR scales end at the current window end, unlike the offline centered
features. The weights are real; equivalence/accuracy must be re-evaluated.
"""
from __future__ import annotations

from dataclasses import asdict
from fractions import Fraction
from pathlib import Path
import json
import math

import numpy as np
import pandas as pd
from scipy.signal import butter, resample_poly, sosfiltfilt
from scipy.special import expit

from .data import Recording
from .state_machine_v2 import StateMachineV2, V2Config, WindowIn, select_baseline
from .teacher.features import BaselineInfo, baseline_profile, scale_values
from .teacher.personal import PersonalNormalizer
from .teacher.rr import extract_rr
from .teacher.spec import TeacherSpec


def trailing_features(rr, end, spec):
    """Every accessed RR is <= end. Never recenter a long scale on a 30s window."""
    values=[]
    for seconds in spec.scales:
        v,_=scale_values(rr,end-seconds,end,seconds,spec)
        values.extend(v)
    return values


def baseline_heart_rate(rr, starts, spec):
    """Mean HR of accepted clean 30s baseline windows, for display only.

    Use the same NN/coverage calculation as the displayed 30s HR. This
    statistic is not a model threshold and never changes inference.
    """
    values = [scale_values(rr, float(t), float(t) + 30, 30, spec)[0][0] for t in starts]
    values = [float(v) for v in values if np.isfinite(v) and v > 0]
    return dict(mean_bpm=float(np.mean(values)), clean_windows=len(values)) if values else None


def quality(rr,start,end,spec):
    i,j=np.searchsorted(rr.rr_t,[start,end])
    n=int(j-i)
    q=float(rr.abnormal[i:j].mean()) if n else 1.
    b0=max(0,int(math.floor(start/spec.stride_sec)))
    blocks=rr.block_ok[b0:b0+3]
    ok=bool(len(blocks)==3 and blocks.all() and n>=spec.min_rr_per_window and q<=spec.max_abnormal_ratio)
    return q,ok


def wave_windows(signal,fs,ch,starts,bundle):
    config=bundle['student']['wave']
    lo,hi=config['bands'][ch]
    sos=butter(3,[lo,min(hi,.45*fs)],btype='band',fs=fs,output='sos')
    filtered=sosfiltfilt(sos,np.asarray(signal,float))
    ratio=Fraction(config['fs']/fs).limit_denominator(1_000_000)
    resampled=resample_poly(filtered,ratio.numerator,ratio.denominator)
    length=int(config['fs']*config['win_sec'])
    result=[]
    for start in starts:
        a=int(round(start*config['fs']))
        part=resampled[a:a+length]
        if a<0 or len(part)!=length or part.std()<1e-6:
            raise ValueError('incomplete/flat model waveform window')
        # Match original student's float16 cache, then float32 inference input.
        result.append(((part-part.mean())/part.std()).clip(-config['clip'],config['clip']).astype(np.float16).astype(np.float32))
    return np.stack(result)


class CausalStudent:
    def __init__(self,model_dir,*,feature_mode,experimental,threads=1):
        if feature_mode!='causal_trailing_v1' or experimental is not True:
            raise ValueError('streaming S1-EP requires explicit experimental causal_trailing_v1 configuration')
        import torch
        from safetensors.torch import load_file
        from .student.distill import Student
        self.torch=torch
        torch.set_num_threads(threads)
        model_dir=Path(model_dir)
        from .model_profiles import model_identity
        self.identity=model_identity(model_dir)
        self.bundle=json.loads((model_dir/'bundle.json').read_text(encoding='utf-8'))
        if self.bundle['student']['wave']['fs']!=64 or self.bundle['student']['wave']['win_sec']!=30:
            raise ValueError('this adapter requires the 64 Hz / 30 s S1-EP waveform contract')
        self.spec=TeacherSpec.from_dict(self.bundle['spec'])
        self.names=self.bundle['rr_feature_names']
        if self.names!=list(self.spec.baseline_feature_names):
            raise ValueError('bundle RR feature order mismatch')
        self.norm=PersonalNormalizer.from_dict(self.spec,self.bundle['normalizer'])
        self.models=[]
        paths=sorted((model_dir/'student').glob('S1-EP_s*.safetensors'))
        if not paths:
            raise FileNotFoundError('S1-EP safetensors weights missing')
        for path in paths:
            weights=load_file(str(path),device='cpu')
            model=Student(2,'dist',weights['W'].numpy(),0.,0.,len(self.names))
            model.load_state_dict(weights)
            model.eval()
            self.models.append(model)
        self.reset()

    def reset(self):
        self.profiles={}
        self.references=[]
        self.status='collecting'
        self.clean=0
        self.baseline_range=None
        self.ecg_reference=None
        self.zero=None
        self.last_end=None
        self.has_prediction=False
        self.restored=False
        self.profile_metadata=None
        self.sm=StateMachineV2(V2Config())

    def export_baseline(self, *, source_session_id, processing):
        from .model_profiles import validate_profile
        if self.status!='ready' or self.ecg_reference is None:
            raise ValueError('기준 준비 완료 후 저장할 수 있습니다.')
        statistics=np.array([[[self.profiles[c][f'{prefix}_{n}'] for n in self.names]
                              for prefix in ('m','s')] for c in ('ecg','ppg')],dtype=np.float64)
        references=np.stack([[item[mask].detach().cpu().numpy() for mask in ((1,1),(0,1))]
                              for item in self.references]).astype(np.float32)
        snapshot=dict(metadata=dict(format_version=1,identity=self.identity,processing=processing,
            feature_names=self.names,clean_windows=self.clean,start_s=self.baseline_range[0],
            end_s=self.baseline_range[1],source_session_id=source_session_id,
            ecg_reference=dict(self.ecg_reference)),statistics=statistics,references=references)
        return validate_profile(snapshot,self.identity,processing)

    def restore_baseline(self, snapshot):
        from .model_profiles import validate_profile
        validate_profile(snapshot,self.identity)
        meta=snapshot['metadata']
        if meta['feature_names']!=self.names:
            raise ValueError('기준 특징 순서가 모델과 다릅니다.')
        self.reset()
        for i,c in enumerate(('ecg','ppg')):
            self.profiles[c]={f'{prefix}_{n}':float(snapshot['statistics'][i,j,k])
                for j,prefix in enumerate(('m','s')) for k,n in enumerate(self.names)}
            self.profiles[c].update(baseline_valid=1.,baseline_sec=meta['end_s']-meta['start_s'])
        self.references=[{mask:self.torch.from_numpy(snapshot['references'][i,j].copy())
            for j,mask in enumerate(((1,1),(0,1)))} for i in range(len(self.models))]
        self.baseline_range=[meta['start_s'],meta['end_s']]
        self.ecg_reference=dict(meta['ecg_reference'],source_session_id=meta['source_session_id'],
                                profile_id=meta['profile_id'])
        self.clean=meta['clean_windows']
        self.profile_metadata=meta
        self.restored=True
        self.status='ready'

    def _record(self,signals,fs):
        return Recording('streaming', 'stream',signals,fs,prefiltered=False)

    def _normalized(self,values,ch):
        win=pd.DataFrame([dict(zip(self.names,values))|{'profile_key':ch}])
        prof=pd.DataFrame([self.profiles[ch]|{'profile_key':ch}])
        p=self.norm.transform(win,prof)
        z=p[[f'z_{n}' for n in self.names]].to_numpy(float)
        return np.clip(np.nan_to_num(z,nan=0.),-5,5).astype(np.float32)

    def _calibrate(self,rec,rrs,offset,end):
        spec=self.spec
        rr=rrs.get('ecg')
        if rr is None:
            return
        grid=np.arange(0,rr.duration-30+1e-6,10.)
        qs=np.array([quality(rr,t,t+30,spec)[0] for t in grid])
        mask=(grid>=60)&(grid+30<=min(end-self.zero,600))
        self.clean=int(np.sum(mask&(qs<=.05)))
        # select_baseline itself assumes its caller has observed the whole
        # proposed five-minute reference interval; enforce that in streaming.
        if end-self.zero<360:
            return
        good,idx,(a,b)=select_baseline(grid,qs)
        if good and len(rrs)==2 and rr.duration>=b:
            starts=np.arange(a,b-30+1e-6,10.)[:self.bundle['student']['wave']['max_ref_windows']]
            refs=np.stack([wave_windows(rec.signals[c],rec.fs[c],c,starts,self.bundle) for c in ('ecg','ppg')],axis=1)
            for c in ('ecg','ppg'):
                profile=baseline_profile(rrs[c],BaselineInfo('live',a,b),spec)
                if not profile['baseline_valid']:
                    return
                self.profiles[c]=profile
            with self.torch.inference_mode():
                self.references=[]
                for model in self.models:
                    by_mask={}
                    for mask in ((1,1),(0,1)):
                        r=self.torch.tensor(refs,dtype=self.torch.float32)*self.torch.tensor(mask)[None,:,None]
                        by_mask[mask]=model.g(r).median(0).values
                    self.references.append(by_mask)
            self.clean=len(idx)
            self.baseline_range=[offset+a,offset+b]
            reference=baseline_heart_rate(rr,grid[idx],spec)
            self.ecg_reference=(reference | dict(start_s=offset+a,end_s=offset+b)) if reference else None
            self.status='ready'
        elif end-self.zero>=600:
            self.status='retry_required'

    def consume(self,signals,fs,offset,end,can_drowsy=None):
        if end<=offset:
            raise ValueError('invalid inference time interval')
        bounded={}
        for channel,values in signals.items():
            count=int(np.ceil((end-offset)*fs[channel]-1e-9))
            values=np.asarray(values)[:count]
            if len(values)!=count or not np.isfinite(values).all():
                raise ValueError('incomplete/nonfinite inference prefix')
            bounded[channel]=values
        signals=bounded
        if self.zero is None:
            self.zero=offset
        rec=self._record(signals,fs)
        rrs={c:extract_rr(rec,self.spec,c) for c in signals if len(signals[c])/fs[c]>=30}
        if self.status!='ready':
            self._calibrate(rec,rrs,offset,end)
            if end-self.zero>=600 and self.status!='ready':
                self.status='retry_required'
        result=dict(status=self.status,clean_windows=self.clean,baseline_range=self.baseline_range,ecg_reference=self.ecg_reference,
            input=None,output=None,feature_mode='causal_trailing_v1',experimental=True,
            profile_loaded=self.restored,input_ready=False)
        input_ready=(end-self.zero>=max(self.spec.scales)) if self.restored else (
            self.status=='ready' and end>=self.baseline_range[1]+30)
        result['input_ready']=bool(input_ready)
        local_end=end-offset
        inputs={}
        for c in ('ecg','ppg'):
            if c in rrs and local_end>=30:
                q,ok=quality(rrs[c],local_end-30,local_end,self.spec)
                features=trailing_features(rrs[c],local_end,self.spec)
                hr=features[0]
                inputs[c]=(q,ok,float(hr) if np.isfinite(hr) else None,features)
            else:
                inputs[c]=(1.,False,None,None)
        p_ep,p_ppg=None,None
        if self.status=='ready' and input_ready:
            logits={}
            with self.torch.inference_mode():
                for name,mask,ch in (('ep',(1,1),'ecg'),('ppg',(0,1),'ppg')):
                    required=('ecg','ppg') if name=='ep' else ('ppg',)
                    if not all(c in signals for c in required) or not inputs[ch][1]:
                        continue
                    try:
                        wave=np.zeros((1,2,1920),np.float32)
                        for i,c in enumerate(('ecg','ppg')):
                            if mask[i]:
                                wave[:,i]=wave_windows(signals[c],fs[c],c,[local_end-30],self.bundle)
                    except ValueError:
                        continue
                    z=self.torch.from_numpy(self._normalized(inputs[ch][3],ch))
                    x=self.torch.from_numpy(wave)
                    out=[]
                    for i,model in enumerate(self.models):
                        delta=model.g(x)-self.references[i][mask]
                        out.append(float(model.logit(z,delta).item()))
                    if not np.isfinite(out).all():
                        raise ValueError('nonfinite model output')
                    logits[name]=float(np.mean(out))
            p_ep=float(expit(logits['ep'])) if 'ep' in logits else None
            p_ppg=float(expit(logits['ppg'])) if 'ppg' in logits else None
        wi=WindowIn(t=end,p_ep=p_ep,q_ecg=inputs['ecg'][0],p_ppg=p_ppg,q_ppg=inputs['ppg'][0],
            hr_ecg=inputs['ecg'][2],hr_ppg=inputs['ppg'][2],ok_ecg=inputs['ecg'][1],ok_ppg=inputs['ppg'][1],
            can_drowsy=can_drowsy)
        result['input']=asdict(wi)
        if self.status=='ready' and input_ready:
            if self.last_end is not None and end-self.last_end>10.001:
                self.sm.reset()  # Never accumulate persistence across skipped windows.
                self.has_prediction=False
            output=self.sm.step(wi)
            if output.channel!='none':
                self.has_prediction=True
            if self.has_prediction or output.level==-1:
                result['output']=asdict(output)
            self.last_end=end
        return result
