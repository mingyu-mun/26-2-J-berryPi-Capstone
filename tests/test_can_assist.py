"""Simulator-only CAN assist (vehicle lane/inattention warnings): no Unity, no CAN hardware."""
from dataclasses import asdict
from pathlib import Path
import json
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from driver_state.can_assist import (CanAssist, CanSample, build_can_assist, read_samples, synthetic_highway,
                                     synthetic_profile)
from driver_state.pi_runtime import Runtime, load_config
from driver_state.pi_sensors import Sample
from driver_state.state_machine_v2 import StateMachineV2, WindowIn
from driver_state.web_dashboard import Dashboard, validate_event


def assist(origin='simulated', profile=None, seconds=400, **scenario):
    return CanAssist(lambda: synthetic_highway(seconds, **scenario), origin=origin, profile=profile)


def advance(a, start, end):
    payloads = []
    for step in range(int(start * 10) + 1, int(end * 10) + 1):
        p = a.advance(step / 10)
        if p is not None:
            payloads.append((step / 10, p))
    return payloads


def window_flags(a, ends):
    flags, last = [], 0
    for end in ends:                  # the runtime asks right after each 10 s window closes
        advance(a, last, end)
        flags.append(a.window_flag(end))
        last = end
    return flags


def test_lane_and_inattention_warnings_mark_windows():
    a = assist(first_deviation_s=20., deviation_every_s=30., lane_change_every_s=60.)
    ends = list(range(10, 121, 10))
    flags = window_flags(a, ends)
    # lane warning 20-21.5 s, inattention 50-53 s, lane 80 s, inattention 110 s;
    # signalled lane change 35-39 s (lane warning 36-37 s) and 95-99 s are excluded.
    assert flags == [False, False, True, False, False, True, False, False, True, False, False, True]


def test_turn_signal_lane_change_is_excluded_with_reason():
    a = assist(first_deviation_s=20., deviation_every_s=30., lane_change_every_s=60.)
    advance(a, 0, 40)
    assert a.window_reasons(40) == ['lane_suppressed'] and a.window_flag(40) is False
    a = assist(first_deviation_s=20., deviation_every_s=30., lane_change_every_s=0)
    advance(a, 0, 30)
    assert a.window_reasons(30) == ['lane']


def test_live_origin_never_sets_can_drowsy():
    a = assist(origin='live', first_deviation_s=25.)
    payloads = advance(a, 0, 40)
    assert a.window_flag(30) is None
    assert all(p['can_drowsy'] is None and p['rule_id'] is None for _, p in payloads)
    assert any(p['status_codes']['LaneSafe_indi'] == 5 for _, p in payloads)


def test_unknown_warning_state_or_gap_is_none(tmp_path):
    path = tmp_path / 'speed_only.jsonl'
    path.write_text(''.join(json.dumps(dict(t_us=i * 100_000, speed_kmh=100)) + '\n' for i in range(300)))
    a = CanAssist(lambda: read_samples(path))
    advance(a, 0, 30)
    assert a.window_flag(20) is None                     # warning state never received
    rows = [dict(t_us=0, LaneSafe_indi=4, Careless_Warn=0)] + \
           [dict(t_us=i * 100_000, speed_kmh=100) for i in range(1, 300) if not 100 <= i < 250]
    path.write_text(''.join(json.dumps(r) + '\n' for r in rows))
    a = CanAssist(lambda: read_samples(path))
    assert window_flags(a, [10, 20]) == [False, None]    # CAN gap 10-25 s


def test_optional_steering_rule_is_suppressed_during_turn_signal():
    profile = synthetic_profile()
    assert profile.limits[5] == pytest.approx(0.30)
    a = assist(profile=profile, first_deviation_s=20., deviation_every_s=30., lane_change_every_s=60.)
    advance(a, 0, 40)
    assert 'steering' in a.window_reasons(30) and 'steering' not in a.window_reasons(40)


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


def test_vehicle_warning_escalates_from_caution():
    base = replay(None)
    caution = first(base, lambda p: p['output']['level'] == 1)['output']['t']
    cand = first(base, lambda p: p['output']['candidate'])['output']['t']
    persist = first(base, lambda p: p['output']['level'] == 2)
    assert persist['confirmation_source'] == 'persistence' and persist['can_origin'] == 'unavailable'
    assert persist['output']['t'] >= cand + 180

    with_can = replay(assist(first_deviation_s=20., deviation_every_s=30.))
    confirmed = first(with_can, lambda p: p['output']['level'] == 2)
    assert confirmed['confirmation_source'] == 'can' and confirmed['can_origin'] == 'simulated'
    assert confirmed['input']['can_drowsy'] is True
    assert caution < confirmed['output']['t'] <= caution + 30 and confirmed['output']['t'] < cand
    # Warnings while the bio state is still normal never raise the level by themselves.
    early = [p for p in decisions(with_can) if p['output']['t'] < caution]
    assert any(p['input']['can_drowsy'] for p in early) and all(p['output']['level'] == 0 for p in early)


def test_live_can_keeps_persistence_path():
    events = replay(assist(origin='live', first_deviation_s=25.))
    confirmed = first(events, lambda p: p['output']['level'] == 2)
    assert confirmed['confirmation_source'] == 'persistence' and confirmed['input']['can_drowsy'] is None


def test_runtime_can_events_pass_contract_and_dashboard():
    events = replay(assist(first_deviation_s=20.), seconds=200)
    validator = Dashboard().validator
    for event in events:
        validate_event(event, validator)
    can = [e for e in events if e['type'] == 'can.telemetry']
    assert can and all(e['origin'] == 'simulated' and e['payload']['rule_id'] == 'vehicle_warning_sim_v1' for e in can)
    assert any(e['payload']['can_drowsy'] and e['payload']['status_codes']['LaneSafe_indi'] == 5 for e in can)
    assert any(e['payload']['can_drowsy'] and e['payload']['status_codes']['Careless_Warn'] == 2 for e in can)
    status = [e for e in events if e['type'] == 'system.status'][-1]['payload']
    assert status['can_baseline']['status'] == 'unavailable' and status['connections']['can'] == 'connected'
    assert Dashboard().ingest(events)['accepted'] == len(events)


def test_build_from_jsonl_scenario(tmp_path):
    scenario = tmp_path / 'unity.jsonl'
    rows = [dict(t_us=0, LaneSafe_indi=4, Careless_Warn=0, LTurnSig_Indi=0, RTurnSig_Indi=0)]
    rows += [dict(t_us=i * 100_000, speed_kmh=100, steering_deg=0.0) for i in range(1, 400)]
    rows += [dict(t_us=15_000_000, LaneSafe_indi=5), dict(t_us=16_000_000, LaneSafe_indi=4)]
    rows.sort(key=lambda r: r['t_us'])
    scenario.write_text(''.join(json.dumps(r) + '\n' for r in rows))
    a = build_can_assist(scenario)
    assert window_flags(a, [10, 20, 30]) == [False, True, False]


def test_blf_scenario_uses_company_mapping(tmp_path):
    can = pytest.importorskip('can')
    cantools = pytest.importorskip('cantools')
    db = cantools.database.load_file(ROOT / 'git_upload_ver1/mapping/can_list_warnings.dbc')
    indi, warn = db.get_message_by_name('INDI1'), db.get_message_by_name('WARN1')
    path = tmp_path / 'unity.blf'
    frames = [(0, indi.frame_id, indi.encode(dict(LaneSafe_indi=4, LTurnSig_Indi=0, RTurnSig_Indi=0))),
              (0.001, warn.frame_id, warn.encode(dict(Careless_Warn=0))),
              (12.05, warn.frame_id, warn.encode(dict(Careless_Warn=3))),
              (14.05, warn.frame_id, warn.encode(dict(Careless_Warn=0)))]
    frames += [(i / 10, 0x102, bytes(7) + bytes([100])) for i in range(1, 300)]
    with can.BLFWriter(str(path)) as log:
        for t, frame_id, data in sorted(frames, key=lambda f: f[0]):
            log.on_message_received(can.Message(timestamp=1000 + t, arbitration_id=frame_id, is_extended_id=False,
                                                data=data))
    a = build_can_assist(path)
    assert window_flags(a, [10, 20, 29]) == [False, True, False]
