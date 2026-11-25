"""
gen_synthetic_bio.py — 시뮬레이터 replay 용 합성 ECG/PPG 생성 (표준 라이브러리만 사용)

실측 데이터가 준비되기 전, 파이프라인(전처리 → 특징 → 판정 → 화면)을 끝까지 검증하기 위한 임시 데이터다.
생리학적으로 '그럴듯한' 형태일 뿐, 모델 학습용으로 쓰면 안 된다.

시나리오 (기본 20분):
  0   ~ 300 s  정상        HR ≈ 74 bpm, 보통 HRV
  300 ~ 780 s  점진적 피로  HR 74 → 60 bpm, 호흡성 동성부정맥(RSA)↑ → RMSSD↑
  780 ~ 900 s  상태 이상    HR ≈ 125 bpm 빈맥 + 불규칙 RR(심방세동 유사, P파 소실)
  900 ~ 1200 s 회복        HR ≈ 78 bpm

출력: 세션 폴더 (meta.json + ecg.csv + ppg.csv + labels.csv, 두 채널 250 Hz)

사용법 (실측 기록 없이 시뮬레이터를 돌려 볼 때):
  python Tools/gen_synthetic_bio.py --out SimLogs/bio/synthetic --minutes 20 --seed 1
  → DrivingSim 의 Bio Session Dir 에 SimLogs/bio/synthetic 지정
"""
import argparse
import csv
import json
import math
import os
import random

FS = 250


def scenario(t):
    """시각 t[s] 의 (평균 HR, RSA 진폭[s], RR 잡음[s], 불규칙 여부, 구간명)"""
    if t < 300:
        return 74.0, 0.030, 0.018, False, "정상"
    if t < 780:
        a = (t - 300) / 480.0
        return 74.0 - 14.0 * a, 0.030 + 0.035 * a, 0.018 + 0.010 * a, False, "피로(점진)"
    if t < 900:
        return 125.0, 0.010, 0.110, True, "상태이상(빈맥·불규칙)"
    return 78.0, 0.028, 0.018, False, "회복"


def gen_beats(duration, rng):
    """R 피크 시각 목록과 각 박동의 불규칙 여부"""
    beats, irr = [], []
    t = 0.4
    lf_phase = rng.random() * 6.28
    while t < duration:
        hr, rsa, noise, irregular, _ = scenario(t)
        rr = 60.0 / hr
        rr += rsa * math.sin(2 * math.pi * 0.25 * t)               # 호흡(0.25 Hz)에 따른 RSA
        rr += 0.4 * rsa * math.sin(2 * math.pi * 0.1 * t + lf_phase)  # Mayer wave (LF ≈ 0.1 Hz)
        rr += rng.gauss(0.0, noise)
        rr = max(0.33, min(1.6, rr))
        beats.append(t)
        irr.append(irregular)
        t += rr
    return beats, irr


def gauss(x, a, mu, sigma):
    return a * math.exp(-((x - mu) ** 2) / (2 * sigma * sigma))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=".")
    ap.add_argument("--minutes", type=float, default=20.0)
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    duration = args.minutes * 60.0
    n = int(duration * FS)
    ecg = [0.0] * n
    ppg = [0.0] * n

    beats, irr = gen_beats(duration, rng)
    for i, tb in enumerate(beats):
        rr_prev = tb - beats[i - 1] if i > 0 else 0.8
        s = math.sqrt(max(0.3, rr_prev))
        amp = 1.0 + rng.gauss(0, 0.04)
        # --- ECG: P, Q, R, S, T 가우시안 합 (McSharry 계열 단순화) ---
        #         (진폭 mV, R 기준 위치 s, 폭 s)
        waves = [(-0.12, -0.028, 0.009),       # Q
                 (1.15 * amp, 0.0, 0.011),     # R
                 (-0.25, 0.032, 0.010),        # S
                 (0.30, 0.27 * s, 0.055)]      # T (RR 이 길수록 늦게)
        if not irr[i]:
            waves.append((0.14, -0.19 * s, 0.025))  # P 파 (불규칙 구간에서는 소실)
        lo, hi = int((tb - 0.35) * FS), int((tb + 0.55) * FS)
        for k in range(max(0, lo), min(n, hi)):
            x = k / FS - tb
            v = 0.0
            for a, mu, sg in waves:
                v += gauss(x, a, mu, sg)
            ecg[k] += v
        # --- PPG: 맥파 전달 시간(PTT) 후 수축기 피크 + 중복맥파(dicrotic notch 이후) ---
        ptt = 0.22 + rng.gauss(0, 0.005)
        pamp = 1.0 if not irr[i] else 0.55 + 0.4 * min(1.0, rr_prev / 0.6)  # 짧은 RR 뒤 박동은 약함
        lo, hi = int((tb + ptt - 0.1) * FS), int((tb + ptt + 0.75) * FS)
        for k in range(max(0, lo), min(n, hi)):
            x = k / FS - (tb + ptt)
            ppg[k] += pamp * (gauss(x, 1.0, 0.11, 0.055) + gauss(x, 0.38, 0.33, 0.07))

    # --- 기저선 흔들림, 호흡 변조, 잡음 ---
    for k in range(n):
        t = k / FS
        ecg[k] += 0.06 * math.sin(2 * math.pi * 0.25 * t) + 0.03 * math.sin(2 * math.pi * 0.05 * t + 1.0)
        ecg[k] += rng.gauss(0, 0.015) + 0.01 * math.sin(2 * math.pi * 60 * t)  # 전원잡음(60 Hz) 약간
        resp = 1.0 + 0.08 * math.sin(2 * math.pi * 0.25 * t)
        ppg[k] = ppg[k] * resp + 0.15 * math.sin(2 * math.pi * 0.25 * t + 0.7) + rng.gauss(0, 0.01)

    # 세션 폴더 형식 (저장소 docs/data-handoff.md 와 같음): meta.json + ecg.csv + ppg.csv (value 열) + labels.csv
    os.makedirs(args.out, exist_ok=True)
    for name, x in (("ecg.csv", ecg), ("ppg.csv", ppg)):
        with open(os.path.join(args.out, name), "w", newline="", encoding="utf-8") as f:
            f.write("value\n")
            f.writelines(f"{v:.4f}\n" for v in x)
    with open(os.path.join(args.out, "meta.json"), "w", encoding="utf-8") as f:
        json.dump({"dataset_id": "synthetic", "subject_id": "synthetic", "session_id": os.path.basename(os.path.abspath(args.out)),
                   "fs": {"ecg": FS, "ppg": FS}, "is_rest": False, "prefiltered": False,
                   "note": "gen_synthetic_bio.py 합성 파형 — 파이프라인 점검용, 모델 판정의 의미 없음"}, f, ensure_ascii=False, indent=1)
    with open(os.path.join(args.out, "labels.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["start_s", "end_s", "label"])
        for s0, s1, lab in [(0, 300, "정상(합성)"), (300, 780, "피로(합성)"), (780, 900, "빈맥·불규칙(합성)"), (900, duration, "회복(합성)")]:
            if s0 < duration:
                w.writerow([s0, min(s1, duration), lab])
    print(f"저장: {args.out} ({n} 샘플 × 2채널, {len(beats)} 박동)")

if __name__ == "__main__":
    main()
