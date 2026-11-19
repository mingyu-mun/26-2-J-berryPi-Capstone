# 실측 녹음 추론·평가 (어느 PC 에서나 같은 결과)

레포만 받으면 GPU·서버·학습 데이터 없이 직접 측정한 녹음을 추론하고 평가할 수 있게 정리한 절차다.
가중치는 `models/` (2026-11-16 이후 변경 없음), 필요한 코드는 모두 `scripts/`·`src/` 에 있다.

## 1. 환경

```sh
python -m venv .venv            # Python 3.11~3.13 확인
.venv/Scripts/activate          # (Linux/macOS: source .venv/bin/activate)
pip install torch --index-url https://download.pytorch.org/whl/cu128   # GPU 없으면 기본 pip install torch
pip install -r requirements.txt
```

## 2. 변환: 측정 zip → 규약 세션 폴더

```sh
python scripts/convert_jberry_sensor.py --raw "측정 20261119_20_14.zip" --out data/converted
```

- ECG = `ecg_1_*.csv` 의 `ECG_mV` (약 435 Hz), PPG = `ppg_1_*.csv` 의 `PPG적외선원본` (약 100.7 Hz).
- 두 채널이 겹치는 구간을 균일 격자로 선형 보간한다. 0.2초 넘게 끊기면 멈춘다.
- 라벨: 표본마다 `정상/졸음` 이 있는 형식 (11-19) 과 5분마다 블록 끝에 표시하는 형식 (11-18) 을 모두 읽는다.
- 세션 시각 0 = 원 녹음 약 60초. `meta.json` 의 `raw_offset_sec` 에 남는다.
- 직접 측정한 데이터와 변환 결과(`data/`, `outputs/`, `out/`)는 커밋하지 않는다.

## 3. 추론 (실시간과 같은 방식)

```sh
python scripts/infer_bundle.py --session data/converted/jberry_sensor/self/<세션> --out out/infer [--baseline 0 540]
```

- 기준 구간을 자동으로 고른다 (60초 이후 ECG 깨끗한 창 20개). `--baseline A B` 로 직접 지정할 수 있다.
- 결과: `windows.csv` (창별 학생 logit·품질·상태 머신 단계), `summary.json`.

## 4. 평가 (라벨과 비교, 고정 설정)

```sh
python scripts/selfmeasure_eval.py --session data/converted/jberry_sensor/self/<세션> [...] --out out/selfmeasure_eval
```

- `scripts/baseline_exp_rec.py` (부록 B, 11-18 rec4 결과) 와 같은 계산이다. 경로를 고정하지 않고 라벨을 세션에서 읽는다.
- 조건 F5·F10 (처음 5·10분 고정 기준), A5·A10 (+ 조건부 느린 갱신). 학생 S1-EP 와 학습 없는 규칙(dev_core) 을 같은 기준으로 낸다.
- 평가 창은 600초 이후 (모든 조건 공통), 라벨 0/1 창만.
- `infer_bundle.py` 와 숫자가 다르다. 그쪽은 기준을 자동 선택하고, 파형 기준을 30개로 제한하고, PPG 대체를 쓴다.

## 5. 알려진 차이 (2026-11-19 확인)

11-18 rec4 결과는 이 변환기 이전에 연구실 PC 에서 따로 변환한 세션(`2026-11-18_190640_s1`)으로 냈다. 그 변환기는 레포에 없었다.
같은 가중치·같은 평가 스크립트로 두 변환을 비교한 학생 창 AUROC:

| 조건 | 연구실 PC 변환 (rec4) | `convert_jberry_sensor.py` |
| --- | --- | --- |
| F5 | 0.63 | 0.74 |
| F10 | 0.73 | 0.60 |
| A5 | 0.72 | 0.78 |
| A10 | 0.79 | 0.81 |

같은 변환·같은 스크립트로 11-19 녹음은 F5 0.24, F10 0.52, A5 0.18, A10 0.54 였다. 규칙(dev_core) 창 AUROC 는
11-18 F5 0.34·F10 0.73, 11-19 F5 0.04·F10 0.14 다 (이 변환 기준).

녹음 한 개의 AUROC 는 변환·기준 정의만으로 이 정도 흔들린다. 연구실 PC 변환기를 확인하면 이 표를 갱신하고, 둘 중 하나를 표준으로 고정한다.
