"""파이프라인 동작 확인용 합성 데이터.

**실제 생리 현상을 흉내 낸 데이터가 아니다.** 클래스가 바뀌면 심박수·HRV·조향 패턴이
조금씩 달라지도록 만든 장난감 신호로, 코드가 끝까지 도는지와 입력 규약이 맞는지만 확인한다.
여기서 나온 정확도는 모델 성능 근거로 쓰면 안 된다.
"""
from __future__ import annotations

import numpy as np

from .data import Recording

# 클래스별 (심박수 변화 bpm, HRV 배율, 불규칙 박동 확률, 조향 흔들림 배율). 임의값.
# 0 = 정상, 1 = 졸음 (심박 감소·HRV 증가 방향으로 가정)
_CLASS_EFFECT = {
    0: (2.0, 0.9, 0.00, 1.0),
    1: (-7.0, 1.6, 0.00, 1.8),
}


def _beat_times(dur: float, hr: float, hrv: float, irregular: float, rng) -> np.ndarray:
    t, out = 0.0, []
    base = 60.0 / hr
    while t < dur:
        ibi = base * (1 + 0.05 * hrv * rng.standard_normal())
        if rng.random() < irregular:
            ibi *= rng.choice([0.6, 1.5])
        t += float(np.clip(ibi, 0.3, 2.0))
        out.append(t)
    return np.asarray(out[:-1])


def _ecg(beats: np.ndarray, n: int, fs: float, rng) -> np.ndarray:
    t = np.arange(n) / fs
    x = np.zeros(n)
    for b in beats:   # P, QRS, T 를 가우시안 펄스로 근사
        for off, amp, w in ((-0.2, 0.1, 0.025), (0.0, 1.0, 0.012), (0.25, 0.3, 0.04)):
            m = np.abs(t - (b + off)) < 5 * w
            x[m] += amp * np.exp(-0.5 * ((t[m] - b - off) / w) ** 2)
    x += 0.15 * np.sin(2 * np.pi * 0.25 * t)       # 호흡에 의한 기저선 흔들림
    return x + 0.03 * rng.standard_normal(n)


def _ppg(beats: np.ndarray, n: int, fs: float, rng) -> np.ndarray:
    t = np.arange(n) / fs
    x = np.zeros(n)
    for b in beats + 0.22:   # 맥파 전달 지연
        m = (t > b - 0.3) & (t < b + 0.8)
        x[m] += np.exp(-0.5 * ((t[m] - b) / 0.12) ** 2) + 0.35 * np.exp(-0.5 * ((t[m] - b - 0.3) / 0.1) ** 2)
    return 500 + 200 * x + 5 * rng.standard_normal(n)   # ADC 단위처럼 오프셋·스케일을 다르게


_REST_EFFECT = (0.0, 1.2, 0.0, 1.0)   # 기준(휴식) 구간: 평소 상태


def make_recording(subject_id: str | None, segments: list[tuple[int | None, float]],
                   base_hr: float, base_hrv: float, rng, session_id: str = "s1",
                   fs_ecg: float = 250.0, fs_ppg: float = 50.0, fs_can: float = 20.0,
                   mains_noise: float = 0.05) -> Recording:
    """segments = [(클래스 또는 None(휴식·미라벨), 길이 초), ...] 순서대로 이어 붙인 녹화."""
    ecg, ppg, labels = [], [], []
    can = {k: [] for k in ("speed", "steering_angle", "accel_pedal", "brake")}
    for k, sec in segments:
        dhr, hrv, irr, steer = _REST_EFFECT if k is None else _CLASS_EFFECT[int(k)]
        beats = _beat_times(sec, base_hr + dhr, base_hrv * hrv, irr, rng)
        n_e = int(sec * fs_ecg)
        e = _ecg(beats, n_e, fs_ecg, rng)
        e += mains_noise * np.sin(2 * np.pi * 60.0 * np.arange(n_e) / fs_ecg)   # 전원 잡음 60 Hz
        ecg.append(e)
        ppg.append(_ppg(beats, int(sec * fs_ppg), fs_ppg, rng))
        nc = int(sec * fs_can)
        tc = np.arange(nc) / fs_can
        can["speed"].append(60 + 10 * np.sin(2 * np.pi * tc / 60) + rng.standard_normal(nc))
        can["steering_angle"].append(steer * (3 * np.sin(2 * np.pi * 0.1 * tc) + rng.standard_normal(nc)))
        can["accel_pedal"].append(np.clip(20 + 5 * rng.standard_normal(nc), 0, 100))
        can["brake"].append((rng.random(nc) < 0.05).astype(float))
        labels.append(np.full(nc, -1 if k is None else int(k)))
    signals = {"ecg": np.concatenate(ecg), "ppg": np.concatenate(ppg)}
    signals.update({k: np.concatenate(v) for k, v in can.items()})
    fs = {"ecg": fs_ecg, "ppg": fs_ppg, **{k: fs_can for k in can}}
    return Recording(subject_id, session_id, signals, fs, np.concatenate(labels), fs_can)


def make_subject(subject_id: str | None, seconds_per_class: float = 180.0, rest_sec: float = 30.0,
                 n_classes: int = 2, seed: int = 0) -> Recording:
    """참가자 1명: 앞 rest_sec 휴식(미라벨) + (정상, 졸음) 을 무작위 순서로 seconds_per_class 씩 두 번."""
    rng = np.random.default_rng(seed)
    base_hr = rng.uniform(60, 80)       # 사람마다 다른 평소 심박수 → 개인화가 필요한 이유
    base_hrv = rng.uniform(0.8, 1.2)
    segs = ([(None, rest_sec)] if rest_sec > 0 else []) + \
        [(int(k), seconds_per_class) for _ in range(2) for k in rng.permutation(n_classes)]
    return make_recording(subject_id, segs, base_hr, base_hrv, rng)


def make_dataset(n_subjects: int = 6, seconds_per_class: float = 180.0, rest_sec: float = 30.0,
                 seed: int = 0) -> list[Recording]:
    return [make_subject(f"S{i:02d}", seconds_per_class, rest_sec, seed=seed + i) for i in range(n_subjects)]
