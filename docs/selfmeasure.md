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

- ECG = `ecg_1_*.csv` 의 `ECG_mV` (약 435 Hz). PPG = `ppg_1_*.csv` 의 `AC보정` 열과 `FIR지연보정` 시간 열 (약 100.7 Hz).
  11-18 rec4 를 낸 연구실 PC 변환과 같은 규칙이다 (5장). 결과는 시작점에 민감하니 바꾸지 않는다.
- 세션 시작은 원 녹음 60.0초, 균일 격자로 선형 보간한다. 0.2초 넘게 끊기면 멈춘다.
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

## 5. 변환 규칙과 결과 민감도 (2026-11-19 확인)

11-18 rec4 결과는 연구실 PC 의 변환기로 만든 세션으로 냈고, 그 변환기는 레포에 없었다. 기록에 남은 수치(PPG 이상 간격 7.4%,
조건별 AUROC)로 역추정해 `AC보정` PPG + `FIR지연보정` 시간 + 원 시각 60.0초 시작임을 확인했고, 이를 기본값으로 정했다.
ECG 는 `ECG원본`·`ECG_mV` 어느 쪽이든 결과가 같다.

학생 S1-EP 창 AUROC (`selfmeasure_eval.py`, 같은 가중치):

| 녹음 · 변환 | F5 | F10 | A5 | A10 |
| --- | --- | --- | --- | --- |
| 11-18 연구실 PC 기록 (rec4) | 0.629 | 0.731 | 0.715 | 0.788 |
| 11-18 기본값 (AC보정 + FIR 시간 + 60.0초) | 0.628 | 0.726 | 0.714 | 0.786 |
| 11-18 `--ppg-col 적외선 --ppg-time raw --start -1` | 0.743 | 0.595 | 0.776 | 0.814 |
| 11-19 기본값 | 0.17 | 0.37 | 0.17 | 0.43 |
| 11-19 `--ppg-col 적외선 --ppg-time raw --start -1` | 0.24 | 0.52 | 0.18 | 0.54 |

- 위 두 변환의 차이는 PPG 열이 아니라 세션 시작점(60.0초 vs 60.007초, 7 ms) 차이에서 온다.
  적외선 PPG 로도 시작을 60.0초로 맞추면 S1-E (ECG만) 결과가 연구실 PC 변환과 똑같고 (F5 0.616 / F10 0.722),
  PPG 를 0 으로 지우거나 ECG 전용 학생 S1-E 로 바꿔도 변환 사이 흔들림이 그대로다.
- 즉 창 격자 정렬에 민감하다. 시작점을 0.007~5초 옮기면 녹음 한 개의 창 AUROC 가 최대 ±0.15 움직인다
  (11-18 F5 0.61~0.75, F10 0.56~0.74 / 11-19 F5 0.12~0.28, F10 0.34~0.56). 기준 구간 통계와 2분·5분 특징이
  창·기준 경계에 걸친 박동에 따라 바뀌기 때문으로 보인다.
- 그래서 녹음 한 개의 AUROC 하나로 결론 내지 않는다. 여러 시작점의 평균이나 신뢰구간, 여러 녹음의 블록 단위로 본다.
- 11-19 녹음은 어느 변환에서도 라벨과 반대 또는 우연 수준이다. 변환 문제가 아니라 졸음 구간에서 심박·HRV 변화가 거의 없었다.
