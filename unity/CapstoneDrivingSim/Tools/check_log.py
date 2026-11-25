"""
check_log.py — Unity 주행 시뮬레이터 로그 점검 + 기본 지표 + 생체신호 정렬 예시

사용법
    python check_log.py SimLogs/drive_P01_20261117_143000.csv
    python check_log.py SimLogs/drive_P01_20261117_143000.csv --plot
    python check_log.py <주행로그.csv> --sensor ppg.csv --sensor-time unix_ms   # 생체신호와 시간 정렬

필요 패키지: pandas, numpy (그래프는 matplotlib)
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def load_session(csv_path):
    """주행 CSV + 같은 이름의 _events.csv, _meta.json 을 함께 읽는다."""
    csv_path = Path(csv_path)
    df = pd.read_csv(csv_path)
    base = csv_path.with_suffix("")
    ev_path = Path(str(base) + "_events.csv")
    meta_path = Path(str(base) + "_meta.json")
    events = pd.read_csv(ev_path) if ev_path.exists() else pd.DataFrame()
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
    return df, events, meta


def timing_report(df, meta):
    """샘플링 주기와 시계 드리프트 점검"""
    dt_sim = np.diff(df["t_s"].to_numpy())
    dt_wall = np.diff(df["wall_unix_ms"].to_numpy()) / 1000.0
    # 시뮬레이션 시간과 실제 시간의 누적 차이 (프레임 드랍이 있으면 커진다)
    wall_rel = (df["wall_unix_ms"] - df["wall_unix_ms"].iloc[0]) / 1000.0
    drift = (wall_rel - (df["t_s"] - df["t_s"].iloc[0])).to_numpy()
    print("── 타이밍 ─────────────────────────────")
    print(f"  샘플 수            : {len(df)}")
    print(f"  설정 주기          : {meta.get('log_rate_hz', '?')} Hz")
    if len(dt_sim):
        print(f"  t_s 간격 (평균)    : {dt_sim.mean()*1000:.2f} ms  → {1/dt_sim.mean():.2f} Hz")
        print(f"  벽시계 간격 중앙값 : {np.median(dt_wall)*1000:.2f} ms, 95% 범위 "
              f"[{np.percentile(dt_wall,2.5)*1000:.1f}, {np.percentile(dt_wall,97.5)*1000:.1f}] ms")
        print(f"  실제-시뮬 시간 차  : 최종 {drift[-1]*1000:.0f} ms (|값|이 수백 ms 이상이면 PC 성능/프레임 드랍 점검)")


def driving_metrics(df):
    """졸음·피로 연구에서 흔히 쓰는 주행 지표 몇 가지"""
    moving = df[df["speed_kmh"] > 30]
    print("── 주행 지표 (30 km/h 이상 구간) ───────")
    if moving.empty:
        print("  주행 구간 없음")
        return
    sdlp = moving["lane_offset_m"].std()  # SDLP: 차선 내 횡위치 표준편차
    print(f"  평균 속도          : {moving['speed_kmh'].mean():.1f} km/h (표준편차 {moving['speed_kmh'].std():.1f})")
    print(f"  SDLP               : {sdlp:.3f} m   ← 졸음 시 증가하는 대표 지표")
    # 차선 이탈: 0→1 로 바뀌는 순간 수
    dep = moving["lane_departure"].to_numpy()
    n_dep = int(((dep[1:] == 1) & (dep[:-1] == 0)).sum())
    print(f"  차선 이탈 횟수     : {n_dep}")
    # 조향 반전 횟수 (핸들 각도가 방향을 2° 이상 바꾼 횟수 / 분)
    sw = moving["steering_wheel_deg"].to_numpy()
    rev, last_ext, direction = 0, sw[0], 0
    for v in sw[1:]:
        d = v - last_ext
        if direction >= 0 and d <= -2.0:
            rev += direction != 0; direction = -1; last_ext = v
        elif direction <= 0 and d >= 2.0:
            rev += direction != 0; direction = 1; last_ext = v
        elif (direction > 0 and v > last_ext) or (direction < 0 and v < last_ext):
            last_ext = v
    minutes = (moving["t_s"].iloc[-1] - moving["t_s"].iloc[0]) / 60.0
    if minutes > 0:
        print(f"  조향 반전율        : {rev / minutes:.1f} 회/분")


def align_with_sensor(drive_df, sensor_df, sensor_time_col="unix_ms", tolerance_ms=50):
    """
    생체센서 샘플마다 가장 가까운(이전) 주행 샘플을 붙인다.
    두 기록 모두 같은 PC 시계(또는 NTP 동기화된 시계)의 Unix ms 를 써야 한다.
    """
    d = drive_df.sort_values("wall_unix_ms").copy()
    s = sensor_df.sort_values(sensor_time_col).copy()
    d["wall_unix_ms"] = d["wall_unix_ms"].astype("int64")
    s[sensor_time_col] = s[sensor_time_col].astype("int64")
    return pd.merge_asof(s, d, left_on=sensor_time_col, right_on="wall_unix_ms",
                         direction="nearest", tolerance=tolerance_ms)


def plot(df, events, out_png):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    t = df["t_s"]
    fig, ax = plt.subplots(4, 1, figsize=(12, 9), sharex=True)
    ax[0].plot(t, df["speed_kmh"]); ax[0].set_ylabel("speed [km/h]")
    ax[1].plot(t, df["steering_wheel_deg"]); ax[1].set_ylabel("steering wheel [deg]")
    ax[2].plot(t, df["lane_offset_m"]); ax[2].set_ylabel("lane offset [m]")
    ax[2].axhline(0.85, ls="--", lw=0.8, c="r"); ax[2].axhline(-0.85, ls="--", lw=0.8, c="r")
    ax[3].plot(t, df["yaw_rate_dps"]); ax[3].set_ylabel("yaw rate [deg/s]")
    ax[3].set_xlabel("t [s]")
    for _, m in df[df["marker"] > 0].iterrows():
        for a in ax:
            a.axvline(m["t_s"], c="k", lw=0.8, alpha=0.6)
    fig.suptitle("Driving log (black lines = sync markers)")
    fig.tight_layout()
    fig.savefig(out_png, dpi=120)
    print(f"그래프 저장: {out_png}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv")
    ap.add_argument("--plot", action="store_true")
    ap.add_argument("--sensor", help="생체센서 CSV (Unix ms 시간 열 포함)")
    ap.add_argument("--sensor-time", default="unix_ms")
    args = ap.parse_args()

    df, events, meta = load_session(args.csv)
    print(f"참가자: {meta.get('participant_id', '?')}  시작: {meta.get('start_local', '?')}  "
          f"길이: {meta.get('duration_s', float('nan')):.1f} s")
    timing_report(df, meta)
    driving_metrics(df)
    if not events.empty:
        print("── 이벤트 ─────────────────────────────")
        print(events.to_string(index=False, max_rows=30))

    if args.sensor:
        sensor = pd.read_csv(args.sensor)
        merged = align_with_sensor(df, sensor, args.sensor_time)
        out = Path(args.sensor).with_name(Path(args.sensor).stem + "_with_drive.csv")
        merged.to_csv(out, index=False)
        miss = merged["speed_kmh"].isna().mean() * 100
        print(f"정렬 결과 저장: {out}  (매칭 실패 {miss:.1f}%)")

    if args.plot:
        plot(df, events, Path(args.csv).with_suffix(".png"))


if __name__ == "__main__":
    main()
