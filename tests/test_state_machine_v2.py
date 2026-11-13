"""상태 머신 v2 (docs/state-machine-v2.md)."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from driver_state.state_machine_v2 import StateMachineV2, V2Config, WindowIn, choose_channel, select_baseline  # noqa: E402


def run(ps, qs=None, **kw):
    sm = StateMachineV2(V2Config())
    qs = qs if qs is not None else [0.0] * len(ps)
    return [sm.step(WindowIn(t=10.0 * (i + 1), p_ep=p, q_ecg=q, **kw)) for i, (p, q) in enumerate(zip(ps, qs))]


def first(outs, cond):
    for o in outs:
        if cond(o):
            return o.t
    return None


def test_clean_drowsy_latency():
    ps = [0.2] * 30 + [0.8] * 60
    out = run(ps)
    t_on = 300.0
    assert first(out, lambda o: o.level >= 1) - t_on <= 90          # 중앙값 10초 + 60초
    assert first(out, lambda o: o.candidate) - t_on <= 160
    assert first(out, lambda o: o.level == 2) - t_on <= 350          # + 확정 180초


def test_spike_ignored():
    ps = [0.2] * 20 + [0.95, 0.95] + [0.2] * 20                      # 2창짜리 튐
    assert max(o.level for o in run(ps)) == 0


def test_noise_burst_frozen_but_capped():
    # 잡음 낀 졸음이 계속되면 동결 상한 뒤에는 결국 올라간다
    ps = [0.2] * 30 + [0.8] * 90
    qs = [0.0] * 30 + [0.07] * 90
    out = run(ps, qs)
    assert any(o.frozen for o in out)
    assert first(out, lambda o: o.level >= 1) is not None


def test_channel_fallback():
    cfg = V2Config()
    ch, p, c = choose_channel(WindowIn(t=0, p_ep=0.9, q_ecg=0.2, p_ppg=0.3, q_ppg=0.02), cfg)
    assert ch == "PPG" and p == 0.3 and c == cfg.ppg_conf
    ch, _, _ = choose_channel(WindowIn(t=0, p_ep=0.9, q_ecg=0.2, p_ppg=0.3, q_ppg=0.2), cfg)
    assert ch == "none"
    ch, _, c = choose_channel(WindowIn(t=0, p_ep=0.9, q_ecg=0.075), cfg)
    assert ch == "EP" and abs(c - 0.5) < 1e-9


def test_unavailable_after_hold():
    out = run([0.2] * 5 + [None] * 7)
    assert out[-1].level == -1


def test_can_confirms_early():
    ps = [0.2] * 30 + [0.8] * 30
    out = run(ps, can_drowsy=True)
    assert first(out, lambda o: o.level == 2) - 300 <= 160


def test_select_baseline():
    t = np.arange(0, 900, 10.0)
    ok, idx, (a, b) = select_baseline(t, np.zeros(len(t)))
    assert ok and a == 60 and b == 360
    ok, _, _ = select_baseline(t, np.full(len(t), 0.085))
    assert not ok
