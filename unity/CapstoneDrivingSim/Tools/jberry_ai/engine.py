"""
engine.py — J_Berry PI 저장소 모델(S1-EP + 상태 머신 v2)을 '스트리밍 입력'으로 돌리는 실시간 추론 엔진.

저장소(repo/)의 코드와 가중치를 그대로 쓴다. 새로 만든 것은 '들어오는 표본을 쌓고, 10초마다 새 창 하나를 판정'하는
얼개뿐이며, 창 하나의 계산은 scripts/infer_bundle.py 의 batch 경로와 같은 함수(extract_rr, scale_values,
PersonalNormalizer, z_matrix, student_cache.prep/cut, Student.logit, StateMachineV2.step)를 호출한다.

  입력: ECG·PPG 원신호 (각자 원 샘플링 주기), 세션 시작 기준 경과 시간
  출력: 창마다 WindowIn, StepOut, 기준 수집 상태 → 웹보드 규약 v1.2 이벤트로 변환 (server.py)

[모드 — 학습 때와 실시간의 차이]
  저장소 교사 특징(RR 36개)은 창 중심 c = t+15 를 기준으로 스케일 L(30/120/300 s) 구간 [c−L/2, c+L/2) 에서 계산한다.
  즉 300초 스케일은 창이 끝난 뒤 135초의 '미래' RR 까지 쓴다 (교사는 오프라인 라벨러라 허용, features.py 주석).
    exact  : 학습·batch 와 똑같이 계산. 대신 판정이 창 종료 후 135초 늦게 나온다.
    causal : 실시간. 각 스케일을 창 끝에서 끝나는 구간 [t_end−L, t_end) 로 계산 (과거만). 지연 없음, 입력 분포는 약간 다름.
"""
from __future__ import annotations

import json
import math
import sys
import time
import warnings
from dataclasses import dataclass, replace as _dc_replace
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE / "repo"
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))
warnings.filterwarnings("ignore")

import torch  # noqa: E402
from safetensors.torch import load_file  # noqa: E402

from driver_state.data import Recording  # noqa: E402
from driver_state.preprocessing import prepare_channel  # noqa: E402
from driver_state.state_machine_v2 import StateMachineV2, V2Config, WindowIn, select_baseline  # noqa: E402
from driver_state.student.distill import Student  # noqa: E402
from driver_state.teacher.anomaly import z_matrix  # noqa: E402
from driver_state.teacher.features import BaselineInfo, baseline_profile, scale_values  # noqa: E402
from driver_state.teacher.personal import PersonalNormalizer  # noqa: E402
from driver_state.teacher.rr import extract_rr, _input_spec  # noqa: E402
from driver_state.teacher.spec import TeacherSpec  # noqa: E402
from student_cache import cut, prep  # noqa: E402
from window_quality import abn_ratio  # noqa: E402

torch.set_num_threads(2)

# ---------------------------------------------------------------- 실시간용 검출 임계 (저장소와 유일하게 다른 부분)
# 저장소 detect_ecg_peaks 는 높이 임계를 '세션 전체' 에너지의 99 백분위 × 0.3 으로 잡는다.
# 실시간에는 그때까지 받은 신호만 있으므로, 앞부분에 움직임 잡음이 몇 초만 있어도 백분위가 튀어 박동을 대량으로 놓친다
# (rec4 첫 460초: 618 → 247 박동). 그래서 같은 식을 '조각별 값의 중앙값' 으로 바꿨다. 조각 길이 = 버퍼/6 (10~60초): 긴 버퍼에서는 60초 조각(batch 와 거의 같음), 실측 시작 직후 짧은 버퍼에서도 조각이 여러 개라 잡음에 끌려가지 않는다.
# rec4 전체에서 batch 박동 2726개 모두 ±20 ms 안에 다시 잡히고 12개만 더 잡힌다 (PPG 는 2652/2652 동일).
import driver_state.features as _F  # noqa: E402
import driver_state.teacher.rr as _RRm  # noqa: E402
from scipy.signal import find_peaks as _find_peaks, welch as _welch  # noqa: E402


def _chunk_median(x, fs, fn, sec=None):
    """조각별 fn 값의 중앙값. 조각 길이 = 버퍼 길이/6 을 10~60초로 제한 (짧은 버퍼에서도 조각이 6개 이상)."""
    if sec is None:
        sec = min(60.0, max(10.0, len(x) / fs / 6.0))
    n = int(sec * fs)
    v = [fn(x[i:i + n]) for i in range(0, max(1, len(x) - n + 1), n)] if len(x) >= n else [fn(x)]
    return float(np.median(v))


def _ecg_peaks_rt(ecg, fs):
    e = _F._bandpass(ecg, fs, 5.0, 15.0) ** 2
    h = 0.3 * _chunk_median(e, fs, lambda s: np.percentile(s, 99))
    return _find_peaks(e, distance=int(_F._MIN_IBI * fs), height=h)[0]


def _ppg_peaks_rt(ppg, fs):
    f = _F._bandpass(ppg, fs, 0.5, 8.0)
    n = int(10 * fs)
    frs = []
    for i in range(0, len(f) - n, n):
        fr, P = _welch(f[i:i + n], fs, nperseg=n)
        m = (fr > 0.7) & (fr < 3.0)
        if m.any():
            frs.append(fr[m][np.argmax(P[m])])
    per = 1.0 / np.median(frs) if frs else 0.6
    return _find_peaks(f, distance=max(1, int(0.6 * per * fs)), prominence=0.3 * _chunk_median(f, fs, np.std))[0]


def use_realtime_detectors(on: bool = True):
    _RRm.detect_ecg_peaks = _ecg_peaks_rt if on else _F.detect_ecg_peaks
    _RRm.detect_ppg_peaks_adaptive = _ppg_peaks_rt if on else _F.detect_ppg_peaks_adaptive


use_realtime_detectors(True)

# ---------------------------------------------------------------- 화면용 기준 파형 (판정에는 쓰지 않음)
BASE_WAVE_SEC = 9.0     # 기준 구간 가운데에서 잘라 둘 길이 [s] (6초 화면 + 위상 맞춤 여유 3초)


def rr_stats(rr, a: float, b: float):
    """[a, b) 안 정상 RR 의 중앙값과 견고한 산포 (사분위 범위/1.349) [ms] — 몇 개 남은 잘못 잡힌 박동에 끌려가지 않게"""
    t = rr.rr_t
    m = (t >= a) & (t < b) & ~rr.abnormal
    v = rr.rr[m] * 1000.0
    if len(v) < 5:
        return None, None
    q1, q3 = np.percentile(v, [25, 75])
    return float(np.median(v)), float((q3 - q1) / 1.349)


def sig(x):
    return float(1.0 / (1.0 + math.exp(-float(x))))


def _f(x):
    """numpy/NaN → 파이썬 float 또는 None"""
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


@dataclass
class WindowResult:
    t_start: float
    t_end: float
    win_in: WindowIn
    step: object                 # StepOut
    logit_ep: float | None
    logit_ppg: float | None
    valid: bool
    valid_ppg: bool
    proc_ms: float
    zr: np.ndarray | None = None


class StreamEngine:
    def __init__(self, fs_ecg: float, fs_ppg: float, mode: str = "causal", models: Path | None = None,
                 subject_id: str = "self", session_id: str = "live", force_baseline: bool = True,
                 baseline_override: tuple | None = None, edge_margin: float = 5.0):
        self.mode = mode
        self.fs = {"ecg": float(fs_ecg), "ppg": float(fs_ppg)}
        mdir = Path(models) if models else REPO / "models"
        self.B = json.loads((mdir / "bundle.json").read_text(encoding="utf-8"))
        self.spec = TeacherSpec.from_dict(self.B["spec"])
        self.names = self.B["rr_feature_names"]
        self.norm = PersonalNormalizer.from_dict(self.spec, self.B["normalizer"])
        self.th_logit = float(self.B["thresholds"]["student_logit"]["S1-EP"])
        self.ep = self._load_students(mdir, "S1-EP")
        self.subject_id, self.session_id = subject_id, session_id
        self.force_baseline = force_baseline
        self.baseline_override = baseline_override
        self.edge_margin = edge_margin            # 버퍼 끝 몇 초는 필터 가장자리라 판정에 안 씀
        self.cfg = V2Config()
        self.sm = StateMachineV2(self.cfg)
        # 창 끝 시각 → can_drowsy (True/False/None). server.py 가 차량 경고 규칙(vehicle_warning_sim_v1)을 연결한다.
        # 2026-11-26: CAN 은 상태 머신 단계 전이에 쓰지 않는다 (생체만으로 정상·주의·졸음).
        #             졸음 중 CAN 경고가 있으면 server.py 가 '졸음 확정'을 표시한다. WindowIn.can_drowsy 는 기록용으로 남긴다.
        self.can_flag = None
        self.buf = {"ecg": [], "ppg": []}         # 받은 원신호 조각 목록
        self.n = {"ecg": 0, "ppg": 0}
        self._cat_cache = {"ecg": (0, np.zeros(0)), "ppg": (0, np.zeros(0))}
        # 기준 수집 상태 (상태 머신 v2 2장)
        self.base_status = "waiting"              # waiting | collecting | ready | retry_required
        self.base_clean = 0
        self.base_range = None
        self.base_forced = False
        self.base_source = "auto"               # auto | file:<이름>
        self.prof = self.pprof = None
        self.R = None
        self.next_start = None                    # 다음에 판정할 창 시작 시각
        self.lookahead = 0.0 if mode == "causal" else max(self.spec.scales) / 2 - self.spec.window_sec / 2
        self.last_q_block = {"ecg": None, "ppg": None}

    # ------------------------------------------------------------ 모델
    @staticmethod
    def _load_students(mdir: Path, cond: str):
        ms = []
        for p in sorted((mdir / "student").glob(f"{cond}_s*.safetensors")):
            sd = load_file(str(p))
            m = Student(sd["g.conv.0.weight"].shape[1], "dist", sd["W"].numpy(), 0.0, 0.0, 36)
            m.load_state_dict(sd)
            m.eval()
            ms.append(m)
        if not ms:
            raise FileNotFoundError(f"{mdir}/student/{cond}_s*.safetensors 없음")
        return ms

    @torch.no_grad()
    def _logit(self, X, zr, mask):
        M = torch.as_tensor(mask, dtype=torch.float32)[None, :, None]
        x, r = torch.as_tensor(X) * M, torch.as_tensor(self.R) * M
        out = []
        for m in self.ep:
            gr = m.g(r).median(0).values
            out.append(m.logit(torch.as_tensor(zr), m.g(x) - gr).numpy())
        return float(np.mean(out, 0)[0])

    # ------------------------------------------------------------ 입력
    def push(self, ch: str, samples) -> None:
        a = np.asarray(samples, np.float64)
        if a.size:
            self.buf[ch].append(a)
            self.n[ch] += a.size

    def duration(self) -> float:
        return min(self.n["ecg"] / self.fs["ecg"], self.n["ppg"] / self.fs["ppg"])

    def _signal(self, ch: str) -> np.ndarray:
        n_cached, arr = self._cat_cache[ch]
        if n_cached != self.n[ch]:
            arr = np.concatenate(self.buf[ch]) if self.buf[ch] else np.zeros(0)
            self.buf[ch] = [arr]
            self._cat_cache[ch] = (self.n[ch], arr)
        return arr

    def _recording(self) -> Recording:
        d = self.duration()
        sig_ = {ch: self._signal(ch)[: int(d * self.fs[ch])] for ch in ("ecg", "ppg")}
        return Recording(subject_id=self.subject_id, session_id=self.session_id, signals=sig_, fs=dict(self.fs),
                         t0=0.0, prefiltered=False, dataset_id="jberry_sensor", meta={})

    # ------------------------------------------------------------ 창 특징 (features.session_windows 의 한 행)
    def _window_row(self, rr, t: float) -> tuple[dict, bool]:
        spec = self.spec
        W, S = spec.window_sec, spec.stride_sec
        b0 = int(round(t / S))
        blocks = rr.block_ok[b0:b0 + int(round(W / S))]
        t_rr = rr.rr_t
        i, j = np.searchsorted(t_rr, t), np.searchsorted(t_rr, t + W)
        n_rr, n_bad = j - i, int(rr.abnormal[i:j].sum())
        valid = bool(len(blocks) and blocks.all() and n_rr >= spec.min_rr_per_window
                     and n_bad <= spec.max_abnormal_ratio * max(n_rr, 1))
        vals = []
        for sec in spec.scales:
            if self.mode == "causal":
                a, b = t + W - sec, t + W              # 창 끝에서 끝나는 과거 구간
            else:
                c = t + W / 2
                a, b = c - sec / 2, c + sec / 2        # 학습과 같은 중심 정렬
            v, _ = scale_values(rr, a, b, sec, spec)
            vals += v
        row = dict(zip(spec.raw_feature_names, vals + [np.nan] * (len(spec.raw_feature_names) - len(vals))))
        row.update(t_rel=float(t), valid=valid, dataset_id="jberry_sensor", subject_id=self.subject_id,
                   session_id=self.session_id, profile_key="live", baseline_valid=1.0)
        return row, valid

    def _z(self, row: dict, prof: pd.DataFrame) -> np.ndarray:
        df = pd.DataFrame([row])
        zr, _ = z_matrix(self.norm, df, prof, self.names)
        return zr

    # ------------------------------------------------------------ 기준 수집
    def _try_baseline(self, rec, rr, prr) -> None:
        d = self.duration()
        if d < self.cfg.base_wait + 30 + self.edge_margin:
            self.base_status = "waiting" if d < self.cfg.base_wait else "collecting"
            self.base_clean = 0
            return
        grid = np.arange(0.0, d - 30 - self.edge_margin + 1e-6, self.spec.stride_sec)
        ok, idx, (a, b) = select_baseline(grid, abn_ratio(rr, grid), self.cfg)
        if self.baseline_override is not None:
            a, b = self.baseline_override
            ok = d >= b + self.edge_margin
            if not ok:
                self.base_status = "collecting"
                return
        self.base_clean = int(len(idx))
        if not ok:
            if d >= self.cfg.base_max_wait:
                self.base_status = "retry_required"
                if not self.force_baseline:
                    return
                # Replay 에서는 재측정을 할 수 없으므로 infer_bundle --force 와 같이 기본 구간으로 진행 (표시는 유지)
                a, b = self.cfg.base_wait, self.cfg.base_wait + self.cfg.base_len
                self.base_forced = True
            else:
                self.base_status = "collecting"
                return
        self._set_profile(rec, rr, prr, a, b)
        self.next_start = float(b)                 # 기준 구간과 겹치지 않는 첫 창부터 판정
        print(f"[engine] 기준 {'강제 ' if self.base_forced else ''}확정 {a:.0f}~{b:.0f}s, 깨끗한 창 {self.base_clean}", flush=True)

    def _set_profile(self, rec, rr, prr, a: float, b: float) -> None:
        """기준 구간 [a, b) 로 개인 프로필(RR 특징 중앙값·산포)과 학생 기준 파형 R 을 만든다."""
        self.base_range = (float(a), float(b))
        info = BaselineInfo(self.session_id, a, b)
        self.prof = pd.DataFrame([{"profile_key": "live", **baseline_profile(rr, info, self.spec)}])
        self.pprof = pd.DataFrame([{"profile_key": "live", **baseline_profile(prr, info, self.spec)}])
        # 학생 기준 파형 (infer_bundle 2절: 기준 구간 창들, 최대 max_ref_windows 개)
        Wv = self.B["student"]["wave"]
        rs = np.arange(a, b - 30 + 1e-6, self.spec.stride_sec)[: Wv["max_ref_windows"]]
        L = int(Wv["fs"] * Wv["win_sec"])
        self.R = np.zeros((len(rs), 2, L), np.float32)
        for c, ch in enumerate(("ecg", "ppg")):
            self.R[:, c] = cut(prep(rec.signals[ch], rec.fs[ch], ch), rs)[0]
        if self.base_status != "retry_required":
            self.base_status = "ready"
        # 화면용: 기준 구간 가운데 9초 파형(공통 전처리 100 Hz)과 박동 간격 중앙값
        self.base_wave, self.base_rr = {}, {}
        ins = _input_spec(self.spec, rec)
        fs100 = float(self.spec.fs)
        mid = 0.5 * (a + b)
        for ch, r_ in (("ecg", rr), ("ppg", prr)):
            self.base_rr[ch] = rr_stats(r_, a, b)
            try:
                f = rec.fs[ch]
                s0 = max(0.0, mid - 10.0)
                raw = rec.signals[ch][int(s0 * f): int((mid + 10.0) * f)]
                filt, _ = prepare_channel(raw, f, ch, ins, False)
                i0 = int((mid - BASE_WAVE_SEC / 2 - s0) * fs100)
                seg = filt[max(0, i0): max(0, i0) + int(BASE_WAVE_SEC * fs100)]
                if len(seg) >= int(BASE_WAVE_SEC * fs100) * 0.9:
                    self.base_wave[ch] = np.asarray(seg, dtype=np.float32)
            except Exception:
                pass

    def use_baseline_recording(self, ecg, ppg, fs_ecg: float, fs_ppg: float, name: str = "file") -> tuple:
        """별도 기준(baseline) 측정 파일로 개인 기준을 정한다. 이후 들어오는 입력은 처음(0 s)부터 바로 판정.
        파일 안에서도 상태 머신 v2 규칙(60 s 대기, 깨끗한 창 20개)으로 기준 구간을 고른다.
        규칙을 못 채우면 [min(60, 길이×0.2), 끝] 을 강제로 쓰고 base_forced 로 표시한다."""
        rec = Recording(subject_id=self.subject_id, session_id=f"{self.session_id}-baseline",
                        signals={"ecg": np.asarray(ecg, np.float64), "ppg": np.asarray(ppg, np.float64)},
                        fs={"ecg": float(fs_ecg), "ppg": float(fs_ppg)}, t0=0.0, prefiltered=False,
                        dataset_id="jberry_sensor", meta={})
        dur = min(len(ecg) / fs_ecg, len(ppg) / fs_ppg)
        rr = extract_rr(rec, self.spec, "ecg")
        prr = extract_rr(rec, self.spec, "ppg")
        grid = np.arange(0.0, dur - 30 + 1e-6, self.spec.stride_sec)
        ok, idx, (a, b) = select_baseline(grid, abn_ratio(rr, grid), self.cfg)
        self.base_clean = int(len(idx))
        if not ok or b > dur:
            a, b = min(self.cfg.base_wait, dur * 0.2), dur
            self.base_forced = True
        self.base_status = "ready"
        self._set_profile(rec, rr, prr, a, b)
        self.base_source = f"file:{name}"
        self.next_start = 0.0
        print(f"[engine] 기준 파일 {name}: {'강제 ' if self.base_forced else ''}{a:.0f}~{b:.0f}s "
              f"(파일 길이 {dur:.0f}s, 깨끗한 창 {self.base_clean})", flush=True)
        return a, b

    @property
    def baseline_ready(self) -> bool:
        return self.prof is not None

    # ------------------------------------------------------------ 판정
    def poll(self) -> list[WindowResult]:
        """판정할 수 있는 새 창이 있으면 처리해서 돌려준다 (10초마다 1개)."""
        d = self.duration()
        out = []
        if d < 40:
            self.base_status = "waiting"
            return out
        lim = d - self.edge_margin
        if not self.baseline_ready:
            if d < getattr(self, "_next_base_try", 0.0):
                return out
            self._next_base_try = d + self.spec.stride_sec / 2     # 기준 확인은 5초 데이터마다
        elif self.next_start + 30 + self.lookahead > lim:
            return out
        rec = self._recording()
        rr = extract_rr(rec, self.spec, "ecg")
        prr = extract_rr(rec, self.spec, "ppg")
        for ch, r_ in (("ecg", rr), ("ppg", prr)):
            self.last_q_block[ch] = bool(r_.block_ok[-1]) if len(r_.block_ok) else None
        if not self.baseline_ready:
            self._try_baseline(rec, rr, prr)
            if not self.baseline_ready:
                return out
        while self.next_start + 30 + self.lookahead <= lim:
            out.append(self._step_window(rec, rr, prr, self.next_start))
            self.next_start += self.spec.stride_sec
        return out

    def _prep_cache(self, rec, ch):
        key = (ch, len(rec.signals[ch]))
        if getattr(self, "_pc_key", {}).get(ch) != key:
            self._pc_key = getattr(self, "_pc_key", {})
            self._pc_val = getattr(self, "_pc_val", {})
            self._pc_key[ch], self._pc_val[ch] = key, prep(rec.signals[ch], self.fs[ch], ch)
        return self._pc_val[ch]

    def _step_window(self, rec, rr, prr, t: float) -> WindowResult:
        t0 = time.perf_counter()
        row, valid = self._window_row(rr, t)
        zr = self._z(row, self.prof)
        prow, valid_ppg = self._window_row(prr, t)
        zr_prv = self._z(prow, self.pprof)
        q_ecg = float(abn_ratio(rr, np.array([t]))[0])
        q_ppg = float(abn_ratio(prr, np.array([t]))[0])
        # 학생 입력 파형: batch 와 같이 받은 신호 전체를 prep(대역통과+64 Hz) 한 뒤 30초를 자른다 (전체 0.03 s 수준)
        X = np.zeros((1, 2, self.R.shape[2]), np.float32)
        for c, ch in enumerate(("ecg", "ppg")):
            X[0, c] = cut(self._prep_cache(rec, ch), np.array([t]))[0][0]
        lg_ep = self._logit(X, zr, [1, 1])
        lg_ppg = self._logit(X, zr_prv, [0, 1])
        hr_e, hr_p = _f(row.get("hr_30")), _f(prow.get("hr_30"))
        w = WindowIn(t=t + 30, p_ep=sig(lg_ep) if valid else None, q_ecg=q_ecg,
                     p_ppg=sig(lg_ppg) if valid_ppg else None, q_ppg=q_ppg,
                     hr_ecg=hr_e, hr_ppg=hr_p, can_drowsy=self.can_flag(t + 30) if self.can_flag else None)
        st = self.sm.step(_dc_replace(w, can_drowsy=None))
        return WindowResult(t, t + 30, w, st, lg_ep, lg_ppg, valid, valid_ppg, (time.perf_counter() - t0) * 1000, zr[0])

    def preview(self, ch: str, sec: float = 8.0, tail: float = 0.0):
        """최근 sec 초를 저장소 공통 전처리(prepare_channel: 노치·대역통과·100 Hz)로 처리. (fs, 파형, 시작 시각)"""
        x = self._signal(ch)
        n = int(sec * self.fs[ch])
        if len(x) < n // 2:
            return None
        seg = x[-n:]
        if getattr(self, "_ins", None) is None:
            self._ins = _input_spec(self.spec, self._recording())
        ins = self._ins
        filt, _ = prepare_channel(seg, self.fs[ch], ch, ins, False)
        t_end = len(x) / self.fs[ch]
        fs = float(self.spec.fs)
        return fs, filt, t_end - len(filt) / fs
