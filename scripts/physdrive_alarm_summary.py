"""PhysDrive 오경보율 요약 (졸음 라벨 없음 → 정상 주행으로 보고 경보를 오경보로 센다).

    python scripts/physdrive_alarm_summary.py --run old=out/physdrive/*_full --run causal=out/physdrive_causal/* --out out/pd_alarm

infer_bundle.py 의 windows.csv 를 모은다. 지표
  win_alarm_ecg : ECG 유효 창 중 학생 S1-EP logit ≥ 동작점 비율 (student_alarm)
  warn_time     : 상태 머신 주의 이상 (level ≥ 1) 시간 비율
  warn_per_h    : 상태 머신이 주의 이상으로 올라간 횟수 / 시간 (세션마다 처음부터)
"""
from __future__ import annotations

import argparse
import glob
from pathlib import Path

import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="append", required=True, help="이름=windows.csv 가 있는 폴더 glob")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    rows = []
    for spec in a.run:
        name, pat = spec.split("=", 1)
        for d in sorted(glob.glob(pat)):
            p = Path(d) / "windows.csv"
            if not p.exists():
                continue
            w = pd.read_csv(p)
            subj = Path(d).name.split("_")[0]
            for sid, g in w.groupby("session_id"):
                g = g.sort_values("t_rel")
                lv = g["level"].fillna(0).to_numpy()
                onsets = int(((lv[1:] >= 1) & (lv[:-1] < 1)).sum() + (lv[0] >= 1 if len(lv) else 0))
                v = g[g["valid"]]
                rows.append({"run": name, "subject": subj, "session": sid, "windows": len(g), "valid": len(v),
                             "alarm_valid": int(v["student_alarm"].sum()), "warn_windows": int((lv >= 1).sum()),
                             "onsets": onsets, "hours": (len(g) * 10 + 20) / 3600})
    df = pd.DataFrame(rows)
    df.to_csv(out / "per_session.csv", index=False)
    s = df.groupby("run").agg(subjects=("subject", "nunique"), sessions=("session", "size"), windows=("windows", "sum"),
                             valid=("valid", "sum"), alarm_valid=("alarm_valid", "sum"), warn_windows=("warn_windows", "sum"),
                             onsets=("onsets", "sum"), hours=("hours", "sum"))
    s["win_alarm_ecg"] = s["alarm_valid"] / s["valid"]
    s["warn_time"] = s["warn_windows"] / s["windows"]
    s["warn_per_h"] = s["onsets"] / s["hours"]
    s.to_csv(out / "summary.csv")
    print(s.round(3).to_string())


if __name__ == "__main__":
    main()
