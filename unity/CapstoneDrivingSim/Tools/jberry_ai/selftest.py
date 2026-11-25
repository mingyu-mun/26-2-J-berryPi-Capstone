"""selftest.py — 저장소 모델을 이 PC 에서 실제로 돌려 본다 (실측 기록 앞 560초, 빠른 입력)."""
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
t0 = time.time()
import torch  # noqa: E402
from engine import StreamEngine  # noqa: E402

print(f"python {sys.version.split()[0]}  torch {torch.__version__}  numpy {np.__version__}  import {time.time() - t0:.1f}s")
ROOT = HERE.parent.parent
BIO = ROOT / "SimLogs" / "bio"


def find_recording():
    """실측 기록 찾기: SimLogs/bio → (에디터) Assets/StreamingAssets/Replays → (실행 파일) *_Data/StreamingAssets/Replays.
    기준(baseline) 파일과 합성 파형은 건너뜀"""
    roots = [BIO, ROOT / "Assets" / "StreamingAssets" / "Replays"] + sorted(ROOT.glob("*_Data/StreamingAssets/Replays"))
    for r in roots:
        if not r.exists():
            continue
        for c in sorted(p.parent for p in r.glob("*/meta.json")):
            try:
                meta = json.loads((c / "meta.json").read_text(encoding="utf-8"))
            except Exception:
                continue
            if c.name == "synthetic" or "baseline" in c.name or meta.get("role") == "baseline":
                continue
            if (c / "ecg.csv").exists() and (c / "ppg.csv").exists():
                return c
    return None


if len(sys.argv) > 1:
    d = Path(sys.argv[1])
else:
    d = find_recording()
    if d is None:   # 실측 기록이 없으면 합성 파형으로 경로만 점검
        import subprocess
        d = BIO / "synthetic"
        if not (d / "meta.json").exists():
            print("실측 기록이 없어 합성 파형(10분)을 만들어 점검합니다:", d)
            subprocess.run([sys.executable, str(HERE.parent / "gen_synthetic_bio.py"), "--out", str(d), "--minutes", "10"], check=True)
m = json.loads((d / "meta.json").read_text(encoding="utf-8"))
fe, fp = m["fs"]["ecg"], m["fs"]["ppg"]
e = pd.read_csv(d / "ecg.csv")["value"].to_numpy()
p = pd.read_csv(d / "ppg.csv")["value"].to_numpy()
E = StreamEngine(fe, fp, mode="causal", session_id="selftest")
print(f"모델 로드 {time.time() - t0:.1f}s, 학생 {len(E.ep)}개, 기록 {d.name} ECG {fe:.1f}Hz PPG {fp:.1f}Hz")
res = []
t1 = time.time()
for s in range(0, 560):
    E.push("ecg", e[int(s * fe):int((s + 1) * fe)])
    E.push("ppg", p[int(s * fp):int((s + 1) * fp)])
    res += E.poll()
print(f"기준 {E.base_range} ({E.base_status}), 창 {len(res)}개, 처리 {time.time() - t1:.1f}s, 창당 중앙값 "
      f"{np.median([r.proc_ms for r in res]) if res else -1:.0f} ms")
for r in res[-3:]:
    print(f"  t_end={r.t_end:.0f}s level={r.step.level}({r.step.level_name}) p_used={r.step.p_used} ch={r.step.channel}")
sys.exit(0 if res else 2)
