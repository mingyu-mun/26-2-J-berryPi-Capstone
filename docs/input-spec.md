# 공통 입력 규약 (v0.2, 개인 Baseline 반영)

후보 모델 5종(MLP, CNN+BiLSTM, Random Forest, XGBoost, LightGBM)이 **같은 데이터를 같은 형식으로** 받도록 정한 규약입니다. 코드 기준은 `src/driver_state/spec.py`, 값은 `configs/default.yaml`입니다.

> **2026-11-05 결정**
> - 출품 시연용 모델은 **생체 단독(ECG·PPG)**으로 고정하고, CAN은 후순위로 미룹니다(`use_can: false`). 시연의 CAN 시나리오 Replay는 화면 연출용이며 모델 입력이 아닙니다.
> - 시연 흐름은 다음과 같습니다. 시연자가 주행하지 않는 상태에서 30초 calibration을 하고, 프로필을 생성한 뒤, CAN 시나리오 Replay를 실행하면서 실시간 ECG·PPG로 추론합니다.
> - 라벨 4단계 정의와 윈도우 길이는 아직 미확정이며, 아래 값은 임시 기본값입니다.

## 1. 전체 흐름과 담당 경계

```text
[데이터/전처리 담당]            [AI 모델링]
원신호 ─▶ Recording ─▶ 공통 전처리 ─▶ 윈도우 분할 ─▶ 생체 특징 8 ─┐
                       (노치·anti-aliasing·    │                    ├─ 개인화 특징 7 (Baseline 대비)
                        리샘플·대역통과·품질)  │                    │
                                              │   참가자 프로필 ────┘
                                              ▼
                              InputPipeline (학습 참가자로만 fit, JSON 저장)
                                ├─ features (N, 15)            → MLP / RF / XGBoost / LightGBM
                                └─ sequence (N, 2, 3000)
                                   + personal (N, 7)           → CNN+BiLSTM
```

## 2. Recording (전처리 → 모델 인계 형식)

| 필드 | 형식 | 설명 |
| --- | --- | --- |
| `subject_id` | 문자열 또는 `None` | 참가자 id. LOSO 분할·Baseline 키. 없으면 기준 무효(`baseline_valid=0`) |
| `session_id` | 문자열 | 측정 회차 |
| `signals` | {채널명: 1차원 배열} | `ecg`, `ppg` (CAN 채널은 `use_can`일 때만) |
| `fs` | {채널명: Hz} | 채널별 원 샘플링 주기. 채널마다 달라도 됨 |
| `labels`, `label_fs` | 정수 배열, Hz | 샘플 단위 라벨, 미라벨 구간은 `-1` |
| `t0` | 초 | 세션 시작 시각. 같은 참가자에서 가장 이른 세션이 기준 세션 |
| `is_rest` | bool | 휴식(기준 측정) 녹화. 있으면 이 녹화가 기준 세션이 되고 학습 윈도우에는 쓰지 않음 |
| `prefiltered` | bool | 데이터 담당이 필터를 이미 적용했으면 `True` (공통 전처리의 필터 단계 생략) |
| `dataset_id` | 문자열 | 여러 공개 데이터셋을 합칠 때 출처 구분 |

## 3. 공통 전처리 (윈도우 분할 전, 두 경로 공통)

`src/driver_state/preprocessing.py`. 채널마다 아래 순서로 처리합니다.

| 순서 | 처리 | 기본값 | 이유 |
| --- | --- | --- | --- |
| 1 | 노치 필터 (원 주기에서) | 60 Hz | 한국 전원 잡음 제거 |
| 2 | 저역통과 필터 (anti-aliasing, 다운샘플일 때만) | 0.4 × 100 = 40 Hz | 이 단계 없이 250 Hz → 100 Hz로 줄이면 60 Hz 잡음이 40 Hz로 접혀 들어오고(aliasing), 이후에는 걸러낼 수 없음 |
| 3 | 공통 주기 리샘플 | 100 Hz (생체 cubic spline) | 채널·데이터셋 간 주기 통일 |
| 4 | 대역통과 | ECG 0.5–40 Hz, PPG 0.5–8 Hz | 기저선 흔들림·고주파 잡음 제거 |
| 5 | 품질 검사 (윈도우 단위, 필터 전 신호 기준) | 평탄: 표준편차 < 1e-6, 포화: 최솟값·최댓값에 붙은 샘플 > 5% | 접촉 불량·포화 윈도우 표시. 학습·평가에서는 제외(`drop_bad_quality`), 추론에서는 결과에 표시 |

`prefiltered=True`이면 1·2·4단계는 건너뛰고 리샘플과 품질 검사만 합니다.

## 4. 윈도우

| 항목 | 기본값 | 근거 / 상태 |
| --- | --- | --- |
| 윈도우 길이 `window_sec` | 30 s | HRV 계산에 최소 30초 권장. 미확정 |
| 이동 간격 `stride_sec` | 10 s | 상태 갱신 주기 10초 |
| 윈도우 라벨 | 최빈값 | 최빈값 비율 < 0.8인 경계 윈도우는 제외 |
| 기준 구간 제외 | 기준 세션의 앞 30초와 겹치는 윈도우 | 기준 = 자기 자신이라 Δ≈0이 되어 성능이 부풀려지는 것을 방지 |

기호: 윈도우 샘플 수 T = window_sec × fs = 3000, 채널 수 C = 2.

## 5. features 입력: 생체 8 + 개인화 7 = 15 (순서 고정, 뒤에만 추가)

| # | 이름 | 정의 |
| --- | --- | --- |
| 0 | `ecg_hr_mean` | 평균 심박수 60000 / mean(RR) [bpm] |
| 1 | `ecg_rr_mean` | RR 간격 평균 [ms] |
| 2 | `ecg_sdnn` | RR 간격 표준편차 [ms] |
| 3 | `ecg_rmssd` | 연속 RR 차이의 제곱평균제곱근 [ms] |
| 4 | `ecg_pnn50` | 연속 RR 차이가 50 ms를 넘는 비율 |
| 5 | `ppg_pr_mean` | PPG 맥파 간격(PPI) 기반 평균 맥박수 [bpm] |
| 6 | `ppg_ppi_sdnn` | PPI 표준편차 [ms] |
| 7 | `ppg_ppi_rmssd` | PPI 연속 차이 RMS [ms] |
| 8 | `pers_hr_delta` | HR − m_HR [bpm] |
| 9 | `pers_hr_z` | (HR − m_HR) / σ_HR |
| 10 | `pers_rmssd_logratio` | ln(RMSSD / m_RMSSD) |
| 11 | `pers_rmssd_z` | (RMSSD − m_RMSSD) / σ_RMSSD |
| 12 | `pers_pr_delta` | PR − m_PR [bpm] |
| 13 | `pers_pr_z` | (PR − m_PR) / σ_PR |
| 14 | `baseline_valid` | 기준 유효 1 / 무효 0 (정규화하지 않음) |

박동 간격은 0.3~2.0초(30~200 bpm)만 쓰고, 유효 간격이 3개 미만이면 해당 특징은 `NaN`입니다. R-peak·맥파 검출은 임시 기준 구현이며, 정식 전처리(Pan-Tompkins 등)가 나오면 `features.py`의 같은 함수를 교체합니다.

## 6. 개인 Baseline

`src/driver_state/baseline.py`.

**프로필**: 사용자 s의 기준 구간에서 기준 특징 j ∈ {HR, RMSSD, PR}의 중심값 m_{s,j}를 저장합니다.

```latex
\Delta_j = x_j - m_{s,j}, \qquad z_j = \frac{x_j - m_{s,j}}{\max(\sigma_{s,j},\ \sigma_{\mathrm{pop},j})}
```

- x_j: 현재 윈도우의 특징값
- m_{s,j}: 기준 윈도우들의 중앙값. 30초 기준 구간이면 윈도우 1개라 그 값 자체
- σ_{s,j}: 개인 변동 범위 = IQR / 1.349. 기준 윈도우가 3개 이상일 때만 계산하며, 30초 시연에서는 없음
- σ_pop,j: 학습 참가자마다 그 사람 윈도우들의 표준편차를 구해 중앙값을 취한 값. fold마다 학습 참가자로만 계산하고 `pipeline.json`에 저장. 30초 시연에서는 z = Δ / σ_pop

**기준 구간 규칙**

| 상황 | 기준 구간 |
| --- | --- |
| 학습: 휴식 녹화(`is_rest`) 있음 | 휴식 녹화 앞 30초 |
| 학습: 연속 녹화만 있음 | 그 참가자의 가장 이른 녹화(`t0` 최소) 앞 30초 (라벨과 무관) |
| 학습: 참가자 id 없음 | 무효 → 개인화 특징 0, `baseline_valid=0` |
| 시연 | 시연 직전 시연자별 비주행 30초 calibration → `profile.json` (그 시연에서만 사용, 갱신 없음) |

**프로필 유효성 (calibration 품질 기준)**: 기준 구간 길이 ≥ 30초, 품질 검사 통과, 유효 RR 간격 ≥ 20개, ECG HR과 PPG 맥박수 차이 ≤ 10 bpm, 특징 계산 가능. 하나라도 실패하면 무효 프로필(사유 포함)이 되며, 시연에서는 재측정을 안내합니다.

**미등록 사용자 대비**: 학습 때 학습셋 윈도우의 20%(`train_invalid_rate`)를 일부러 기준 무효로 만들어, 모델 하나로 프로필이 없는 경우도 처리하게 합니다.

## 7. 정규화 규약

모든 통계는 **학습 fold의 학습 참가자로만** 계산합니다.

| 대상 | 방법 | fit 필요 |
| --- | --- | --- |
| sequence 생체 채널 | 윈도우·채널별 z-score | 없음 |
| features 결측 | 학습셋 특징별 중앙값으로 대체 | 있음 |
| features 값 (0~13) | 학습셋 특징별 평균·표준편차로 z-score | 있음 |
| `baseline_valid` (14) | 그대로 (0/1) | 없음 |
| personal 입력 | features의 8~14열 (정규화 후) | — |

z-score 식은 x' = (x − μ) / σ입니다. μ는 평균, σ는 표준편차이며, σ가 0에 가까우면 1로 나눕니다. fit 결과(정규화 통계 + σ_pop + 규약 전체)는 `InputPipeline.save()`로 `pipeline.json`에 저장되고, Pi 추론에서 그대로 불러옵니다.

## 8. 라벨 규약

- 모델은 정수 id `0..K-1`만 씁니다. 기본 K=4이고, 이름(매우 양호 / 양호 / 피로 / 상태 이상)은 임시 표시용입니다.
- `predict_proba`는 (N, K) 확률을, `predict`는 (N,) 정수를 반환합니다. 학습셋에 없던 클래스의 열은 0입니다.
- `binary_map`(0·1 → 정상, 2·3 → 주의)으로 이진 백업 평가를 할 수 있습니다(`--binary`).

## 9. 평가 규약

- 운전자 단위 분리: 기본은 LOSO(한 명씩 테스트)입니다.
- 각 fold에서 학습 참가자 중 1명을 검증용으로 떼어 early stopping에 씁니다.
- 테스트 참가자는 자기 프로필만 쓰고, σ_pop과 정규화 통계는 학습 참가자 값을 씁니다.
- 지표: Accuracy, Macro-F1, 클래스별 F1, 혼동행렬, batch 1 추론 지연(PC), 모델 파일 크기.
- `--compare-personal`로 개인화 없음(8개)과 있음(15개)을 같은 fold에서 비교합니다.
