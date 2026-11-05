"""파이프라인 동작 확인용 합성 데이터.

**실제 생리 현상을 흉내 낸 데이터가 아니다.** 클래스가 바뀌면 심박수·HRV·조향 패턴이
조금씩 달라지도록 만든 장난감 신호로, 코드가 끝까지 도는지와 입력 규약이 맞는지만 확인한다.
여기서 나온 정확도는 모델 성능 근거로 쓰면 안 된다.
"""
from __future__ import annotations

import numpy as np

from .data import Recording

# 클래스별 (심박수 변화 bpm, HRV 배율, 불규칙 박동 확률, 조향 흔들림 배율). 임의값.
_CLASS_EFFECT = {
    0: (0.0, 1.3, 0.00, 1.0),
    1: (4.0, 1.0, 0.00, 1.2),
    2: (-6.0, 0.7, 0.00, 1.8),
    3: (14.0, 0.5, 0.08, 2.5),
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


def make_subject(subject_id: str, seconds_per_class: float = 180.0, fs_ecg: float = 250.0,
                 fs_ppg: float = 50.0, fs_can: float = 20.0, n_classes: int = 4,
                 seed: int = 0) -> Recording:
    rng = np.random.default_rng(seed)
    base_hr = rng.uniform(60, 80)
    base_hrv = rng.uniform(0.8, 1.2)
    order = rng.permutation(n_classes)
    ecg, ppg, can, labels = [], [], {k: [] for k in ("speed", "steering_angle", "accel_pedal", "brake")}, []
    for k in order:
        dhr, hrv, irr, steer = _CLASS_EFFECT[int(k)]
        beats = _beat_times(seconds_per_class, base_hr + dhr, base_hrv * hrv, irr, rng)
        ecg.append(_ecg(beats, int(seconds_per_class * fs_ecg), fs_ecg, rng))
        ppg.append(_ppg(beats, int(seconds_per_class * fs_ppg), fs_ppg, rng))
        nc = int(seconds_per_class * fs_can)
        tc = np.arange(nc) / fs_can
        can["speed"].append(60 + 10 * np.sin(2 * np.pi * tc / 60) + rng.standard_normal(nc))
        can["steering_angle"].append(steer * (3 * np.sin(2 * np.pi * 0.1 * tc) + rng.standard_normal(nc)))
        can["accel_pedal"].append(np.clip(20 + 5 * rng.standard_normal(nc), 0, 100))
        can["brake"].append((rng.random(nc) < 0.05).astype(float))
        labels.append(np.full(int(seconds_per_class * fs_can), int(k)))
    signals = {"ecg": np.concatenate(ecg), "ppg": np.concatenate(ppg)}
    signals.update({k: np.concatenate(v) for k, v in can.items()})
    fs = {"ecg": fs_ecg, "ppg": fs_ppg, **{k: fs_can for k in can}}
    return Recording(subject_id, "s1", signals, fs, np.concatenate(labels), fs_can)


def make_dataset(n_subjects: int = 6, seconds_per_class: float = 180.0, seed: int = 0) -> list[Recording]:
    return [make_subject(f"S{i:02d}", seconds_per_class, seed=seed + i) for i in range(n_subjects)]
