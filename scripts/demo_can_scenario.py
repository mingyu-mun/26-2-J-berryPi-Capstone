"""시연용 CAN 시나리오 (rec_20261118 / 11-18 rec4 실측 타임라인에 맞춘 운전) → --can-sim-scenario JSONL.

Unity ``DemoDriver.cs`` 의 기본 사건표와 같은 시각을 쓴다 (기록 기준 초, 0 = 생체 기록 시작).
  0~540 s   정상: 깜빡이 켠 차로 변경 3회 (차선 경고는 깜빡이 구간이라 제외돼야 함)
  540~960 s 졸음 시작(자기보고): 핸들 놓침(부주의 1~2단계), 차선 쪽으로 흐름 2회. 생체 판정이 아직 '졸음'이
            아니므로 서버 규칙상 단계는 오르지 않는다 (CAN 은 졸음 확정에만 씀).
  960~1440 s 졸음: 30~40초마다 차선 이탈 경고, 두 번째 이탈부터 부주의 2단계(5초) → 생체 '졸음' 중이면 '졸음 확정'
  1440 s~   정상 회복: 깜빡이 차로 변경 1회

합성 시험 데이터이며 측정값이 아니다. 실제 Unity 주행 기록(drive_*_can.jsonl, SimLogs/ai/can_*.jsonl)과 형식이 같다.

    python scripts/demo_can_scenario.py -o demo_can_rec20261118.jsonl
    python scripts/can_sim_scenario.py check demo_can_rec20261118.jsonl --until 1500   # PR #17
    python scripts/run_dashboard.py --recordings-dir ... --can-sim-scenario demo_can_rec20261118.jsonl
"""
from __future__ import annotations

import argparse
import json
import math
import random

LANE_CHANGES = [150.0, 330.0, 480.0, 1560.0]
HANDS_OFF = [(600.0, 20.0), (820.0, 28.0), (1200.0, 26.0)]       # (시작, 지속)


def drifts():
    out = [(700.0, 1), (900.0, -1)]
    t, side = 960.0, 1
    while t < 1430.0:
        out.append((t, side))
        t += 32.0 if side > 0 else 38.0
        side = -side
    return out


def build_timeline(duration_s):
    """100 ms 칸마다 (LaneSafe_indi, Careless_Warn, LTurn, RTurn, 조향 편향[deg])."""
    n = int(duration_s * 10)
    lane = [4] * n
    care = [0] * n
    lt = [0] * n
    rt = [0] * n
    bias = [0.0] * n

    def idx(t):
        return max(0, min(n, int(round(t * 10))))

    for i, t0 in enumerate(LANE_CHANGES):            # 깜빡이 4초, 그 안에 차선 밟음 1초 (제외 대상)
        sig = lt if i % 2 == 0 else rt
        for k in range(idx(t0), idx(t0 + 4.0)):
            sig[k] = 1
            bias[k] = (-3.0 if sig is lt else 3.0) * (1 if (k - idx(t0)) < 15 else -1 if (k - idx(t0)) < 30 else 0)
        for k in range(idx(t0 + 1.0), idx(t0 + 2.0)):
            lane[k] = 5
    for t0, dur in HANDS_OFF:                        # 15 s → 1, 25 s → 2 (손 뗀 동안만)
        for k in range(idx(t0 + 15.0), idx(t0 + dur)):
            care[k] = max(care[k], 2 if k >= idx(t0 + 25.0) else 1)
    recent = []
    for t0, side in drifts():                        # 4 s 서서히 흐름 → 1.2 s 차선 경고 → 놀라 되돌림
        for k in range(idx(t0), idx(t0 + 4.0)):
            bias[k] = side * 1.5 * (k - idx(t0)) / 40
        for k in range(idx(t0 + 4.0), idx(t0 + 5.2)):
            lane[k] = 5
            bias[k] = -side * 8.0
        recent = [x for x in recent if t0 - x <= 60.0] + [t0]
        if len(recent) >= 2:                         # 60 s 안 두 번째 이탈 → 부주의 2(3회면 3) 5초
            lvl = 3 if len(recent) >= 3 else 2
            for k in range(idx(t0 + 4.0), idx(t0 + 9.0)):
                care[k] = max(care[k], lvl)
            if lvl == 3:
                recent = []
    return lane, care, lt, rt, bias


def write(path, duration_s=2040.0, seed=20261118):
    rng = random.Random(seed)
    lane, care, lt, rt, bias = build_timeline(duration_s)
    last = {}
    with open(path, "w", encoding="utf-8") as f:
        for ms in range(0, int(duration_s * 1000), 20):              # 50 Hz
            k = ms // 100
            t = ms / 1000
            row = {"t_us": ms * 1000,
                   "speed_kmh": round(99.5 + 0.6 * math.sin(t / 37.0) + rng.uniform(-0.2, 0.2), 2),
                   "steering_deg": round(0.8 * math.sin(t / 9.0) + bias[k] + rng.uniform(-0.4, 0.4), 2)}
            codes = {"LaneSafe_indi": lane[k], "Careless_Warn": care[k], "LTurnSig_Indi": lt[k], "RTurnSig_Indi": rt[k]}
            row.update({c: v for c, v in codes.items() if last.get(c) != v})
            last.update(codes)
            f.write(json.dumps(row) + "\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-o", "--out", default="demo_can_rec20261118.jsonl")
    ap.add_argument("--duration", type=float, default=2040.0, help="길이 [s] (기본: rec_20261118 전체)")
    a = ap.parse_args()
    write(a.out, a.duration)
    print(f"{a.out}: {a.duration:.0f}s, 차선 경고 {len(drifts())}회 + 깜빡이 차로 변경 {len(LANE_CHANGES)}회, "
          f"핸들 놓침 {len(HANDS_OFF)}회")


if __name__ == "__main__":
    main()
