"""기준 구간 조건부 느린 갱신 시뮬레이터 (docs/state-machine-v2.md 부록 B). 모델 재학습 없음.

한 세션의 창을 시간 순서로 넣으면, 학생 S1-EP (ECG 경로) logit 과 상태 머신 v2 단계를 낸다.
adapt=True 이면 RR 특징 기준값 m 과 학생 인코더 기준 g_ref 를 같은 규칙으로 천천히 갱신한다.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .state_machine_v2 import StateMachineV2, V2Config, WindowIn


@dataclass
class AdaptCfg:
    adapt: bool = False
    tau_s: float = 1800.0        # 시간 상수 30분
    dt_s: float = 10.0
    warm_s: float = 900.0        # 기준 구간 끝 후 15분: 조건 없이 갱신
    q_max: float = 0.05
    p_max: float = 0.40          # θ_off
    hr_cap: float = 0.10         # 심박 기준이 처음 대비 ±10% 넘으면 갱신 중지


def softplus(x):
    return float(np.log1p(np.exp(x)))


def simulate(t_end: np.ndarray, X: np.ndarray, valid: np.ndarray, q: np.ndarray, m0: np.ndarray, s0: np.ndarray,
             sigma_pop: np.ndarray, G: list[np.ndarray], gref0: list[np.ndarray], params: list[dict],
             hr_idx: int, base_end: float | None, cfg: AdaptCfg, sm_cfg: V2Config | None = None):
    """t_end: 창 끝 시각 (n,), X: 원 RR 특징 (n, 36), G[k]: 시드 k 의 인코더 출력 (n, 32).
    base_end: 이 세션 안에서 기준 구간이 끝난 시각 (적응 구간 계산용, 없으면 None).
    결과: logit (n,), level (n,)"""
    n = len(t_end)
    m = m0.astype(float).copy()
    scale = np.fmax(np.nan_to_num(s0, nan=0.0), sigma_pop)
    gref = [g.astype(float).copy() for g in gref0]
    lam = cfg.dt_s / cfg.tau_s
    sm = StateMachineV2(sm_cfg or V2Config())
    stopped = False
    logit = np.full(n, np.nan)
    level = np.zeros(n, int)
    hr0 = m0[hr_idx]
    for i in range(n):
        x = X[i]
        z = np.clip(np.nan_to_num((x - m) / scale, nan=0.0), -5, 5)
        ls = []
        for k, p in enumerate(params):
            v = np.concatenate([z, G[k][i] - gref[k]]) @ p["W"].T
            ls.append(softplus(p["alpha"]) * (np.sqrt((v ** 2).sum() + 1e-8) - 1e-4) + p["b"])
        logit[i] = float(np.mean(ls))
        ok = bool(valid[i])
        pe = 1 / (1 + np.exp(-logit[i])) if ok else None
        out = sm.step(WindowIn(t=float(t_end[i]), p_ep=pe, q_ecg=float(q[i]) if np.isfinite(q[i]) else 1.0, ok_ecg=ok))
        level[i] = out.level
        if cfg.adapt and ok and not stopped and np.isfinite(q[i]) and q[i] <= cfg.q_max:
            warm = base_end is not None and t_end[i] - base_end <= cfg.warm_s
            calm = out.level == 0 and not out.candidate and pe is not None and pe < cfg.p_max
            if warm or calm:
                fin = np.isfinite(x)
                m[fin] += lam * (x[fin] - m[fin])
                for k in range(len(gref)):
                    gref[k] += lam * (G[k][i] - gref[k])
                if np.isfinite(hr0) and hr0 > 0 and abs(m[hr_idx] / hr0 - 1) > cfg.hr_cap:
                    stopped = True
    return logit, level
