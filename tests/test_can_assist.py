"""Simulator-only CAN steering assist: no Unity, no CAN hardware. Synthetic samples only."""
from dataclasses import asdict
from pathlib import Path
import json
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from driver_state.can_assist import CanAssist, build_can_assist, read_samples, synthetic_highway, synthetic_profile
from driver_state.pi_runtime import Runtime, load_config
from driver_state.pi_sensors import Sample
from driver_state.state_machine_v2 import StateMachineV2, WindowIn
from driver_state.web_dashboard import Dashboard, validate_event


@pytest.fixture(scope='module')
def profile():
    return synthetic_profile()


def assist(profile, origin='simulated', **scenario):
    return CanAssist(lambda: synthetic_highway(400, **scenario), profile, origin=origin)


def run_until(a, t):
    payloads = []
    for step in range(1, int(t * 10) + 1):
        p = a.advance(step / 10)
        if p is not None:
            payloads.append((step / 10, p))
    return payloads


def run_until_from(a, start, t):
    for step in range(int(start * 10) + 1, int(t * 10) + 1):
        a.advance(step / 10)


def test_synthetic_profile_matches_pr13_reference(profile):
    assert profile.limits[5] == pytest.approx(0.30)


def test_window_flag_marks_deviation_windows_only(profile):
    a = assist(profile, first_deviation_s=25., deviation_every_s=30.)
    flags = []
    for end in range(10, 101, 10):   # the runtime asks right after each 10 s window closes
        run_until_from(a, end - 10, end)
        flags.append(a.window_flag(end))
    # deviation holds start at 25, 55, 85 s (1.2 s each)
    assert flags == [False, False, True, False, False, True, False, False, True, False]


def test_live_origin_never_sets_can_drowsy(profile):
    a = assist(profile, origin='live', first_deviation_s=25.)
    payloads = run_until(a, 40)
    assert a.window_flag(30) is None
    assert all(p['can_drowsy'] is None and p['rule_id'] is None for _, p in payloads)
    assert any(p['deviation_status'] == 'outside_baseline' for _, p in payloads)


def test_missing_steering_is_unavailable_not_normal(profile, tmp_path):
    path = tmp_path / 'gap.jsonl'
    rows = [r for r in synthetic_highway(30, deviation_every_s=0) if not 10e6 <= r[0] < 25e6]
    path.write_text(''.join(json.dumps(list(r)) + '\n' for r in rows))
    a = CanAssist(lambda: read_samples(path), profile)
    payloads = run_until(a, 30)
    assert a.window_flag(20) is None
    assert all(p['deviation_status'] == 'unavailable' and p['speed_kmh'] is None
               for t, p in payloads if 10.2 <= t < 25)


class SMBackend:
    """Fixed high student score through the real StateMachineV2."""
    def __init__(self):
        self.reset()

    def reset(self):
        self.sm = StateMachineV2()

    def consume(self, signals, fs, start, end, can_drowsy=None):
        wi = WindowIn(t=end, p_ep=.9, q_ecg=0., p_ppg=.9, q_ppg=0., hr_ecg=70., hr_ppg=70., can_drowsy=can_drowsy)
        out = self.sm.step(wi)
        return dict(status='ready', clean_windows=20, baseline_range=[0., 30.], input=asdict(wi), output=asdict(out))


class Sink:
    def __init__(self):
        self.events = []

    def put(self, event):
        self.events.append(event)

    def clear(self):
        self.events.clear()

    def flush(self):
        return True


def replay(can_assist, seconds=420):
    config = load_config(ROOT / 'tests/sensors.example.json')
    now = [0.]
    sink = Sink()
    runtime = Runtime(config, SMBackend(), sink, clock=lambda: now[0], origin='recorded', synchronous=True,
                      can_assist=can_assist)
    for c in ('ecg', 'ppg'):
        runtime.state(c, True, 'fixture')
    for second in range(1, seconds + 1):
        for c, v in (('ecg', 512.), ('ppg', 60000.)):
            runtime.emit(c, Sample(second - .5, v))
        now[0] = float(second)
        runtime.tick()
    return sink.events


def decisions(events):
    return [e['payload'] for e in events if e['type'] == 'ai.decision' and e['payload']['output']]


def first(events, pred):
    return next(p for p in decisions(events) if pred(p))


def test_can_confirms_drowsy_earlier_than_persistence(profile):
    base = replay(None)
    cand = first(base, lambda p: p['output']['candidate'])['output']['t']
    persist = first(base, lambda p: p['output']['level'] == 2)
    assert persist['confirmation_source'] == 'persistence' and persist['can_origin'] == 'unavailable'
    assert persist['output']['t'] >= cand + 180

    with_can = replay(assist(profile, first_deviation_s=25., deviation_every_s=30.))
    confirmed = first(with_can, lambda p: p['output']['level'] == 2)
    assert confirmed['confirmation_source'] == 'can' and confirmed['can_origin'] == 'simulated'
    assert confirmed['input']['can_drowsy'] is True
    assert cand < confirmed['output']['t'] <= cand + 30
    # Deviations before the bio candidate never raise the level by themselves.
    early = [p for p in decisions(with_can) if p['output']['t'] < cand]
    assert any(p['input']['can_drowsy'] for p in early) and all(p['output']['level'] < 2 for p in early)


def test_live_can_keeps_persistence_path(profile):
    events = replay(assist(profile, origin='live', first_deviation_s=25.))
    confirmed = first(events, lambda p: p['output']['level'] == 2)
    assert confirmed['confirmation_source'] == 'persistence' and confirmed['input']['can_drowsy'] is None


def test_runtime_can_events_pass_contract_and_dashboard(profile):
    events = replay(assist(profile, first_deviation_s=25.), seconds=200)
    validator = Dashboard().validator
    for event in events:
        validate_event(event, validator)
    can = [e for e in events if e['type'] == 'can.telemetry']
    assert can and all(e['origin'] == 'simulated' for e in can)
    assert any(e['payload']['deviation_status'] == 'outside_baseline' and e['payload']['can_drowsy'] for e in can)
    status = [e for e in events if e['type'] == 'system.status'][-1]['payload']
    assert status['can_baseline']['status'] == 'ready' and status['connections']['can'] == 'connected'
    engine = Dashboard()
    assert engine.ingest(events)['accepted'] == len(events)


def test_build_from_scenario_file_and_saved_profile(tmp_path):
    synthetic_profile('P01', data_dir=tmp_path)
    scenario = tmp_path / 'scenario.jsonl'
    scenario.write_text(json.dumps({'source': 'synthetic'}) + '\n' +
                        ''.join(json.dumps(list(r)) + '\n' for r in synthetic_highway(40, first_deviation_s=15.)))
    a = build_can_assist(scenario, tmp_path / 'profiles' / 'P01.json')
    run_until(a, 30)
    assert a.window_flag(20) is True and a.window_flag(10) is False
