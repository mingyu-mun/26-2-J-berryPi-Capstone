"""11-18 실측(rec4) 원 변환기: 연구실 PC 에서 2026-11-18 에 실제로 쓴 변환 코드를 그대로 옮긴 것.

당시에는 일회성 코드로 실행해 파일이 남지 않았고, 2026-11-23 Claude 대화 기록에서 복구했다.
계산은 그대로이고 입출력 경로만 인자로 받는다. 이 출력은 연구실 PC 의 원 변환본(data.npz)과 바이트 단위까지 같은 값이다
(docs/results/2026-11-23-repro/README.md). docs/results/2026-11-18-baseline, 2026-11-18-rec4 결과는 이 변환본으로 냈다.

    python scripts/convert_rec4_orig.py --raw <풀어 둔 raw4 폴더> --out data/converted

- ECG: ecg_1_라벨링.csv 의 0번 열(시간) · 2번 열(ECG_mV)
- PPG: ppg_1_라벨링.csv 의 1번 열(시간_FIR지연보정) · 7번 열(AC보정신호)
- 둘 다 원 시각 60.0초부터 각 신호 마지막 시각까지 균일 격자 선형 보간 (fs ECG 435.33, PPG 100.66)
- 5분 블록 라벨은 labels_orig.csv 로만 남긴다 (정,정,졸,졸,졸,정,정)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from driver_state.data_io import write_session  # noqa: E402

T0 = 60.0
FE, FP = 435.33, 100.66


def uni(t, x, fs):
    g = np.arange(T0, t[-1], 1 / fs)
    return np.interp(g, t, x)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True, help="ecg_1_라벨링.csv, ppg_1_라벨링.csv 가 있는 폴더")
    ap.add_argument("--out", required=True, help="출력 루트 (아래에 jberry_sensor/self01/2026-11-18_190640_s1)")
    a = ap.parse_args()
    R = Path(a.raw)
    e = pd.read_csv(R / "ecg_1_라벨링.csv", encoding="utf-8-sig", usecols=[0, 2])
    p = pd.read_csv(R / "ppg_1_라벨링.csv", encoding="utf-8-sig", usecols=[1, 7])
    ecg = uni(e.iloc[:, 0].to_numpy(float), e.iloc[:, 1].to_numpy(float), FE)
    ppg = uni(p.iloc[:, 0].to_numpy(float), p.iloc[:, 1].to_numpy(float), FP)
    D = Path(a.out) / "jberry_sensor" / "self01" / "2026-11-18_190640_s1"
    meta = {"dataset_id": "jberry_sensor", "subject_id": "self01", "session_id": "2026-11-18_190640_s1",
            "t0": "2026-11-18T19:07:40+09:00",
            "fs": {"ecg": FE, "ppg": FP}, "channels": ["ecg", "ppg"], "is_rest": False, "prefiltered": False,
            "sensor": {"ecg": "AD8232", "ppg": "MAX30102"},
            "notes": "11-18 측정. ecg_1 ECG_mV, ppg_1 AC보정신호(FIR 지연 보정 시각). 세션 0초 = 원본 60초. 5분 라벨: 정상,정상,졸음,졸음,졸음,정상,정상"}
    write_session(D, meta, {"ecg": ecg, "ppg": ppg}, None,
                  pd.DataFrame({"block_min": ["0-5", "5-10", "10-15", "15-20", "20-25", "25-30", "30-35"],
                                "label": ["정상", "정상", "졸음", "졸음", "졸음", "정상", "정상"]}))
    print(D, "ecg", len(ecg), "ppg", len(ppg))


if __name__ == "__main__":
    main()
