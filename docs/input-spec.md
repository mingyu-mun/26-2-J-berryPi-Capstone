# 공통 입력 규약 (v0.1, 초안)

후보 모델 5종(MLP, CNN+BiLSTM, Random Forest, XGBoost, LightGBM)이 **같은 데이터를 같은 형식으로** 받도록 정한 규약입니다. 코드 기준은 `src/driver_state/spec.py`, 값은 `configs/default.yaml`입니다.

> 라벨 4단계 정의와 윈도우 길이는 아직 미확정입니다. 아래 값은 임시 기본값이며, 확정되면 설정 파일만 바꾸면 됩니다. 개인 Baseline·개인화 calibration은 이번 범위 밖입니다.

## 1. 전체 흐름과 담당 경계

```text
[데이터/전처리 담당]                     [AI 모델링]
원신호 → (필터·동기화) → Recording ──▶ 리샘플 → 윈도우 분할 → InputPipeline(fit: 학습셋만)
                                                                 ├─ sequence (N,C,T) → CNN+BiLSTM
                                                                 └─ features (N,F)   → MLP / RF / XGBoost / LightGBM
```

데이터 담당은 `Recording` 형식만 맞춰 넘기면 됩니다. 공개 데이터셋 로더, 실센서 입력, 파일 Replay 모두 같은 형식입니다.

## 2. Recording (전처리 → 모델 인계 형식)

| 필드 | 형식 | 설명 |
| --- | --- | --- |
| `subject_id` | 문자열 | 운전자 id. LOSO 분할 기준 (예: `S01`, `User1`) |
| `session_id` | 문자열 | 측정 회차 |
| `signals` | {채널명: 1차원 배열} | 채널명은 아래 표의 이름으로 고정 |
| `fs` | {채널명: Hz} | 채널별 원 샘플링 주기. 채널마다 달라도 됨 |
| `labels` | 정수 배열 또는 없음 | 샘플 단위 라벨, 미라벨 구간은 `-1` |
| `label_fs` | Hz | `labels`의 샘플링 주기 |
| `t0` | 초 | 세션 시작 시각. 채널 간 시간 정렬은 전처리 단계 책임 |

| 채널명 | 종류 | 리샘플 보간 | 비고 |
| --- | --- | --- | --- |
| `ecg` | 생체 | cubic spline | 단위 무관 (윈도우별 정규화) |
| `ppg` | 생체 | cubic spline | 단위 무관 (ADC 값 그대로 가능) |
| `speed` | CAN (선택) | 선형 | km/h |
| `steering_angle` | CAN (선택) | 선형 | deg |
| `accel_pedal` | CAN (선택) | 선형 | % |
| `brake` | CAN (선택) | 선형 | 0/1 |

CAN 채널은 `use_can: true`일 때만 필요합니다. **2026-11-05 결정: 출품 시연용 모델은 생체 단독(`use_can: false`)으로 고정하고 CAN은 후순위로 미룹니다.** 시연의 CAN 시나리오 Replay는 화면·상황 연출용이며 모델 입력이 아닙니다. CAN 관련 코드는 이후 후기 융합(late fusion) 실험을 위해 선택 기능으로 남겨 둡니다.

## 3. 리샘플과 윈도우

| 항목 | 기본값 | 근거 / 상태 |
| --- | --- | --- |
| 공통 주기 `fs` | 100 Hz | 참고논문 2와 동일. 미확정 |
| 윈도우 길이 `window_sec` | 30 s | HRV(SDNN·RMSSD) 계산에 최소 30초 권장. 미확정 (2초 / 60초 대안 검토 중) |
| 이동 간격 `stride_sec` | 10 s | 상태 갱신 주기 10초 기본안 |
| 윈도우 라벨 | 최빈값 | 최빈값 비율 < `min_label_purity`(0.8)인 경계 윈도우는 제외 |

기호: 윈도우 하나의 샘플 수 $T = \text{window\_sec} \times f_s$ (기본 3000), 채널 수 $C$ (생체 단독 2, CAN 포함 6).

## 4. 모델 입력 두 가지

**sequence 입력** (CNN+BiLSTM): $X_{seq} \in \mathbb{R}^{N \times C \times T}$. $N$은 윈도우 개수.

**features 입력** (MLP·RF·XGBoost·LightGBM): $X_{feat} \in \mathbb{R}^{N \times F}$. 특징 순서는 고정이며 새 특징은 **뒤에만 추가**합니다(순서가 바뀌면 학습된 모델과 호환되지 않음).

| # | 이름 | 정의 |
| --- | --- | --- |
| 0 | `ecg_hr_mean` | 평균 심박수 $60000 / \overline{RR}$ [bpm] |
| 1 | `ecg_rr_mean` | $\overline{RR} = \frac{1}{n}\sum_i RR_i$ [ms], $RR_i$는 i번째 R-peak 간격 |
| 2 | `ecg_sdnn` | $\sqrt{\frac{1}{n-1}\sum_i (RR_i - \overline{RR})^2}$ [ms], RR 전체 흔들림 |
| 3 | `ecg_rmssd` | $\sqrt{\frac{1}{n-1}\sum_i (RR_{i+1} - RR_i)^2}$ [ms], 박동 간 단기 변화 |
| 4 | `ecg_pnn50` | $\lvert RR_{i+1}-RR_i\rvert > 50\text{ms}$인 비율 |
| 5 | `ppg_pr_mean` | PPG 맥파 간격(PPI)으로 계산한 평균 맥박수 [bpm] |
| 6 | `ppg_ppi_sdnn` | PPI 표준편차 [ms] |
| 7 | `ppg_ppi_rmssd` | PPI 연속 차이 RMS [ms] |
| 8~13 | `can_*` (선택) | 차속 평균·표준편차, 조향각 표준편차·변화율, 가속페달 평균, 브레이크 비율 |

박동 간격은 0.3~2.0초(30~200 bpm) 범위만 사용하고, 유효 간격이 3개 미만이면 해당 특징은 `NaN`입니다. R-peak·맥파 검출은 **임시 기준 구현**(대역통과 + peak 검출)이며, 데이터 담당의 정식 전처리(Pan-Tompkins, PPG 보정 로직)가 나오면 `features.py`의 같은 함수를 교체합니다.

## 5. 정규화 규약

모든 통계는 **학습 fold의 학습 subject로만** 계산합니다. 테스트 운전자의 정보가 섞이면 LOSO 성능이 부풀려지기 때문입니다.

| 대상 | 방법 | fit 필요 |
| --- | --- | --- |
| sequence 생체 채널 | 윈도우·채널별 z-score | 없음 |
| sequence CAN 채널 | 학습셋 채널별 평균·표준편차로 z-score | 있음 |
| features 결측 | 학습셋 특징별 중앙값으로 대체 | 있음 |
| features 값 | 학습셋 특징별 평균·표준편차로 z-score | 있음 |

z-score 식: $x' = \dfrac{x - \mu}{\sigma + \epsilon}$. 여기서 $\mu$는 평균, $\sigma$는 표준편차, $\epsilon = 10^{-6}$은 0으로 나누기를 막는 작은 수입니다.

- 생체 파형을 **윈도우별**로 정규화하는 이유: ECG·PPG 진폭은 센서·증폭 회로마다 단위가 달라(공개 데이터는 mV, 우리 센서는 ADC 값) 진폭 자체에는 상태 정보가 적고 센서 차이만 큽니다. 윈도우 안에서 $\mu, \sigma$를 계산하면 이 차이가 사라집니다.
- 트리 모델은 스케일에 영향을 받지 않지만 같은 입력을 써서 모델 간 비교 조건을 맞추고, 보드 쪽 전처리 코드를 하나로 유지합니다.
- fit 결과는 `InputPipeline.save()`로 JSON에 저장되며, 보드 추론에서 같은 값을 그대로 불러 씁니다.

운전자별 평소값을 기준으로 하는 정규화(개인 Baseline 대비 편차)는 개인화 단계에서 이 위에 추가합니다.

## 6. 라벨 규약

- 모델은 정수 id `0..K-1`만 사용합니다. 기본 $K=4$, 이름(매우 양호 / 양호 / 피로 / 상태 이상)은 임시 표시용입니다.
- 출력: `predict_proba` → $(N, K)$ 확률(행 합 1), `predict` → $(N,)$ 정수.
- 학습셋에 없는 클래스가 있어도 출력은 항상 $(N, K)$이며 해당 열은 0입니다.
- 4단계가 어렵다면 `binary_map`(0·1 → 정상, 2·3 → 주의)으로 이진 백업 평가를 할 수 있습니다 (`--binary`).

## 7. 평가 규약

- 운전자 단위 분리: 기본 LOSO (한 명씩 테스트), `evaluation.n_folds`로 GroupKFold 전환.
- 각 fold에서 학습 운전자 중 1명을 검증용으로 떼어 신경망·부스팅의 early stopping에 사용.
- 지표: Accuracy, Macro-F1(클래스 불균형 대응), 클래스별 F1, 혼동행렬, batch 1 추론 지연(PC), 모델 파일 크기.
