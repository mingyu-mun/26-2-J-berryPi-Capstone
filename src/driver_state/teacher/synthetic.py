"""교사 파이프라인 동작 확인용 합성 데이터셋 4개 (공개 데이터 구조를 흉내 냄).

**실제 졸음 생리를 재현한 데이터가 아니다.** 잠재 졸음도 d(t) 가 오르면 심박이 조금 내려가고
호흡성 변동(HF)이 커지도록 만든 장난감 신호다. 사람마다 평소 심박·HRV 를 크게 다르게 두어
개인 정규화가 필요한 상황을 만들고, 데이터셋마다 기기(fs, 극성, 잡음, 전원 주파수)를 다르게 했다.
여기서 나온 성능 수치는 코드가 끝까지 돌고 평가가 올바르게 계산되는지만 보여 준다.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ..data_io import write_session
from .spec import LabelRule

DATASETS = {
    # 이름: (fs, 극성, 잡음, 전원 Hz, 라벨 방식, 세션 구조, 유발 요인)
    "syn_advitam": (1000.0, 1, 0.02, 50.0, "kss", "rest_session", "sleep_deprivation"),
    "syn_mpddf": (1024.0, 1, 0.04, 50.0, "fatigue", "baseline_range", "prolonged_driving"),
    "syn_dddb": (256.0, -1, 0.03, 60.0, "events", "per_session", "prolonged_driving"),
    "syn_drozy": (512.0, 1, 0.03, 50.0, "kss", "multi_session", "sleep_deprivation"),
}


def _beats(dur: float, d_of_t, hr0: float, rsa: float, rng) -> np.ndarray:
    t, out = 0.0, []
    lf_ph, hf_ph = rng.uniform(0, 2 * np.pi, 2)
    while t < dur:
        d = d_of_t(t)
        base = 60.0 / (hr0 - 7.0 * d)
        resp = 0.27 - 0.06 * d
        ibi = base * (1 + rsa * 0.035 * (1 + 1.2 * d) * np.sin(2 * np.pi * resp * t + hf_ph)
                      + 0.02 * np.sin(2 * np.pi * 0.09 * t + lf_ph) + 0.012 * rng.standard_normal())
        t += float(np.clip(ibi, 0.35, 1.8))
        out.append(t)
    return np.asarray(out[:-1])


def _ecg(beats: np.ndarray, dur: float, fs: float, pol: int, noise: float, mains: float, rng) -> np.ndarray:
    n = int(dur * fs)
    tt = np.arange(-0.3, 0.45, 1 / fs)
    tpl = sum(a * np.exp(-0.5 * ((tt - o) / w) ** 2) for o, a, w in ((-0.2, 0.1, 0.025), (0.0, 1.0, 0.011), (0.25, 0.3, 0.04)))
    x = np.zeros(n + len(tpl))
    k0 = int(round(0.3 * fs))
    for b in beats:
        i = int(round(b * fs)) - k0
        if 0 <= i < n:
            x[i:i + len(tpl)] += tpl
    x = pol * x[:n]
    t = np.arange(n) / fs
    x += 0.15 * np.sin(2 * np.pi * 0.2 * t) + noise * rng.standard_normal(n) + 0.05 * np.sin(2 * np.pi * mains * t)
    return x


def _drowsy_curve(dur: float, start: float, end: float, rest: float):
    def f(t):
        if t < rest:
            return start * 0.5
        return float(np.clip(start + (end - start) * (t - rest) / max(dur - rest, 1), 0, 1))
    return f


def _labels(kind: str, d_of_t, dur: float, rule: LabelRule, rng, t_from: float):
    n = int(np.ceil(dur))
    lab = np.full(n, -1, np.int64)
    rows = []
    if kind == "kss":
        for tk in np.arange(t_from + 300, dur + 1, 300):        # 5분마다 KSS (앞 5분 상태)
            kss = int(np.clip(round(2.5 + 6.5 * d_of_t(tk) + 0.8 * rng.standard_normal()), 1, 9))
            rows.append({"t": float(tk), "kss": kss})
            lab[int(tk - 300):int(tk)] = rule.kss(kss)
    elif kind == "fatigue":
        for tk in np.arange(t_from, dur - 29, 30):                # 30초마다 피로 0–4
            v = int(np.clip(round(4 * d_of_t(tk + 15) + 0.5 * rng.standard_normal()), 0, 4))
            rows.append({"t": float(tk), "fatigue": v})
            lab[int(tk):int(tk + 30)] = rule.fatigue(v)
    else:                                                          # 졸음 이벤트 시각
        ev = [t for t in np.arange(t_from + 60, dur - 60, 20) if d_of_t(t) > 0.7 and rng.random() < 0.15]
        lab[int(t_from):] = 0
        for e in ev:
            rows.append({"t": float(e), "event": 1})
        sec = np.arange(n)
        if ev:
            dist = np.min(np.abs(sec[:, None] - np.asarray(ev)[None, :]), axis=1)
            lab[dist <= rule.event_drowsy_sec] = 1
            lab[(dist > rule.event_drowsy_sec) & (dist < rule.event_alert_gap_sec)] = -1
        lab[:int(t_from)] = -1
    return lab, pd.DataFrame(rows)


def write_synthetic(root: str | Path, n_subjects: int = 6, minutes: float = 30.0, seed: int = 0,
                    datasets: tuple[str, ...] | None = None, fs_override: float | None = None) -> Path:
    root = Path(root)
    rule = LabelRule()
    for di, name in enumerate(datasets or DATASETS):
        fs, pol, noise, mains, kind, layout, inducer = DATASETS[name]
        fs = fs_override or fs
        rng = np.random.default_rng(seed + 100 * di)
        hr_bias = rng.uniform(-3, 3)
        for si in range(n_subjects):
            sid = f"S{si:02d}"
            hr0 = rng.uniform(58, 85) + hr_bias
            rsa = rng.uniform(0.5, 1.6)
            sleepy = si % 2 == 0           # 절반은 졸음이 크게 오는 참가자
            dur = minutes * 60.0
            sessions = []
            if layout == "rest_session":
                sessions = [("rest", 300.0, True, 0.0, 0.0, None), ("drive", dur, False,
                            0.35 if sleepy else 0.0, 0.95 if sleepy else 0.35, inducer if sleepy else "none")]
            elif layout == "baseline_range":
                sessions = [("drive", dur + 360, False, 0.0, 0.95 if sleepy else 0.4, inducer)]
            elif layout == "per_session":
                sessions = [(f"rec{k}", dur, False, 0.0, 0.95 if sleepy else 0.5, inducer) for k in (1, 2)]
            else:
                sessions = [("pvt1", 600.0, False, 0.0, 0.15, "none")] + \
                    [(f"pvt{k}", 600.0, False, 0.5 + 0.2 * k if sleepy else 0.2, 0.7 + 0.15 * k if sleepy else 0.35, inducer)
                     for k in (2, 3)]
            for k, (ses, sdur, is_rest, d0, d1, ind) in enumerate(sessions):
                rest = 360.0 if layout == "baseline_range" else (0.0 if is_rest else 300.0 if layout != "multi_session" else 0.0)
                d_of_t = _drowsy_curve(sdur, d0, d1, rest) if not is_rest else (lambda t: 0.0)
                beats = _beats(sdur, d_of_t, hr0, rsa, rng)
                ecg = _ecg(beats, sdur, fs, pol, noise, mains, rng)
                meta = {"dataset_id": name, "subject_id": sid, "session_id": ses, "t0": float(k * 86400 if layout == "per_session" else k * 3600),
                        "fs": {"ecg": fs}, "channels": ["ecg"], "label_fs": 1, "is_rest": is_rest, "prefiltered": False,
                        "mains_hz": mains, "drowsiness_inducer": ind or "none", "notes": "synthetic"}
                labels, orig = None, None
                if not is_rest:
                    t_from = 360.0 if layout == "baseline_range" else (300.0 if layout in ("per_session",) else 0.0)
                    if layout == "rest_session":
                        t_from = 0.0
                    labels, orig = _labels(kind, d_of_t, sdur, rule, rng, t_from)
                if layout == "baseline_range":
                    meta["baseline_range"] = [60.0, 360.0]
                if layout == "per_session":
                    meta["baseline_per_session"] = True
                write_session(root / name / sid / ses, meta, {"ecg": ecg}, labels, orig)
    return root
