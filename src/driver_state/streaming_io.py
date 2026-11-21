"""Bounded timestamp buffers and an HTTP publisher; no sensor synthesis."""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from urllib.request import Request, urlopen
import json
import logging
import math
import time

import numpy as np


@dataclass(frozen=True)
class Sample:
    t: float
    value: float
    contact_ok: bool = True


class DataUnavailable(ValueError):
    pass


class ChannelBuffer:
    def __init__(self, fs, keep_s=900., max_gap=.2):
        self.fs,self.keep_s,self.max_gap=fs,keep_s,max_gap
        self.samples=deque(maxlen=int(math.ceil(fs*keep_s*2)))

    def append(self,sample):
        if not math.isfinite(sample.t) or not math.isfinite(sample.value):
            raise DataUnavailable('nonfinite sample')
        if self.samples and sample.t<=self.samples[-1].t:
            raise DataUnavailable('sample clock must increase')
        self.samples.append(sample)
        cutoff=sample.t-self.keep_s
        while self.samples and self.samples[0].t<cutoff:
            self.samples.popleft()

    def uniform(self,start,end,require_contact=True,contact_start=None):
        if end<=start or len(self.samples)<2:
            raise DataUnavailable('not enough samples')
        values=[]
        for value in reversed(self.samples):
            values.append(value)
            if value.t<start:
                break
        values.reverse()
        # Include one neighboring sample on each side, never extrapolate.
        t=np.fromiter((v.t for v in values),float)
        lo=max(0,int(np.searchsorted(t,start))-1)
        hi=min(len(t),int(np.searchsorted(t,end))+1)
        values=values[lo:hi];t=t[lo:hi]
        grid=start+np.arange(int(np.ceil((end-start)*self.fs-1e-9)))/self.fs
        if not len(grid) or grid[0]<t[0]-1e-7 or grid[-1]>t[-1]+1e-7:
            raise DataUnavailable('incomplete coverage')
        if np.any(np.diff(t)>self.max_gap):
            raise DataUnavailable('gap exceeds interpolation limit')
        if require_contact and not all(v.contact_ok for v in values if contact_start is None or v.t>=contact_start):
            raise DataUnavailable('lead off / finger absent / saturated')
        y=np.array([v.value for v in values],float)
        return np.interp(grid,t,y)

    @property
    def latest(self):
        return self.samples[-1].t if self.samples else None


class Publisher:
    def __init__(self,url,clock=time.monotonic,log_path=None,opener=urlopen):
        self.url,self.clock,self.opener=url,clock,opener
        self.pending=deque(maxlen=64)
        self.lock=Lock()
        self.dropped=0
        self.last_error=None
        self.version=0
        self.last_encoded={}
        self.log=None
        if log_path:
            p=Path(log_path);p.parent.mkdir(parents=True,exist_ok=True)
            self.log=p.open('w',encoding='utf-8')

    def put(self,event):
        # Prove strict JSON before buffering. Never replace NaN by a fake zero.
        encoded=json.dumps(event,ensure_ascii=False,allow_nan=False,separators=(',',':'))
        with self.lock:
            key=(event['session_id'],event['type'])
            if self.last_encoded.get(key)==encoded:
                return
            self.last_encoded[key]=encoded
            if self.log:
                self.log.write(encoded+'\n');self.log.flush()
            if len(self.pending)==self.pending.maxlen:
                self.dropped+=1
            self.pending.append((self.clock(),event))

    def clear(self):
        with self.lock:
            self.pending.clear()
            self.last_encoded.clear()
            self.version+=1

    def flush(self):
        with self.lock:
            now=self.clock()
            version=self.version
            fresh=[]
            while self.pending:
                at,event=self.pending.popleft()
                limit=25. if event['type']=='ai.decision' else 2.
                if now-at<=limit:
                    fresh.append((at,event))
                else:
                    self.dropped+=1
        if not fresh:
            return True
        body=json.dumps([v for _,v in fresh],ensure_ascii=False,allow_nan=False).encode('utf-8')
        request=Request(self.url,data=body,headers={'Content-Type':'application/json'},method='POST')
        try:
            with self.opener(request,timeout=2) as response:
                result=json.load(response)
                accepted=result.get('accepted')
                same_session=all(v['session_id']==result.get('session_id') for _,v in fresh)
                if accepted!=len(fresh) and not (isinstance(accepted,int) and 0<=accepted<len(fresh) and same_session):
                    raise RuntimeError('webboard rejected/retired some events')
            self.last_error=None
            return True
        except Exception as exc:
            if self.last_error!=str(exc):
                logging.getLogger(__name__).warning('publish_failed: %s',exc)
            self.last_error=str(exc)
            with self.lock:
                if version!=self.version:
                    return False
                # Keep latest packet of each type for retry, not old wave backlog.
                current=list(self.pending)
                newest={}
                for at,event in fresh+current:
                    newest[(event['session_id'],event['type'])]=(at,event)
                self.pending.clear()
                self.pending.extend(sorted(newest.values(),key=lambda v:v[1]['t_rel_s']))
            return False

    def close(self):
        if self.log:
            self.log.close()
