# 후보 모델 5종 (v0.2, 개인 Baseline 반영)

코드: `src/driver_state/models/`. 모든 모델은 같은 인터페이스(`fit`, `predict_proba`, `predict`, `save`, `load`, `size_bytes`)를 가지며, `build_model(이름, 클래스 수)`로 만듭니다. 입력 형식은 [공통 입력 규약](input-spec.md)을 따릅니다.

| 이름 (`build_model`) | 입력 | 학습 파라미터 수 (기본 설정) | 보드 이식 경로 |
| --- | --- | --- | --- |
| `mlp` | features $(N, 15)$ | 약 3.2천 개 | ONNX (`export_onnx`, 입력 `x`) |
| `cnn_bilstm` | sequence $(N, 2, 3000)$ + personal $(N, 7)$ | 약 2.7만 개 | ONNX (`export_onnx`, 입력 `x`, `p`) |
| `random_forest` | features | 트리 200개 | joblib (Pi에서 scikit-learn) |
| `xgboost` | features | 부스팅 트리 최대 300개 | joblib / XGBoost 자체 포맷 |
| `lightgbm` | features | 부스팅 트리 최대 300개 | joblib / LightGBM 자체 포맷 |

공통 기호: $x$는 입력 하나, $K$는 클래스 수, $\hat{p}_k$는 클래스 $k$일 확률 예측값, $y$는 정답 클래스입니다.

## 1. MLP (다층 퍼셉트론)

$$h_1 = \mathrm{ReLU}(W_1 x + b_1),\quad h_2 = \mathrm{ReLU}(W_2 h_1 + b_2),\quad \hat{p} = \mathrm{softmax}(W_3 h_2 + b_3)$$

- $x \in \mathbb{R}^{15}$: 정규화된 특징 벡터 (생체 8 + 개인화 7).
- $W_l, b_l$: $l$번째 층의 가중치 행렬과 편향. 은닉 크기 64 → 32.
- $\mathrm{ReLU}(z) = \max(0, z)$: 음수를 0으로 만드는 활성화 함수.
- $\mathrm{softmax}(z)_k = e^{z_k} / \sum_j e^{z_j}$: 출력을 합이 1인 확률로 변환.
- 각 은닉층 뒤 Dropout 0.2 (학습 중 뉴런 20%를 무작위로 끄는 과적합 방지 기법).

## 2. CNN + BiLSTM (파형 + 개인화 보조 입력)

```text
파형 (B, C=2, T=3000)
 → Conv1d(k=15, 16ch) → BatchNorm → ReLU → MaxPool(4)   # (B,16,750)
 → Conv1d(k=7, 32ch)  → BatchNorm → ReLU → MaxPool(4)   # (B,32,187)
 → Conv1d(k=5, 32ch)  → BatchNorm → ReLU → MaxPool(2)   # (B,32,93)
 → BiLSTM(hidden 32, 양방향) → 시간축 평균              # (B,64)  ─┐
개인화 (B, P=7) → Linear(7→16) → ReLU                   # (B,16)  ─┤ concat (B,80)
                                                                    └→ Dropout 0.3 → Linear → softmax (B,K)
```

파형에서 배운 64차원 특징 h와 개인화 특징을 작은 층으로 바꾼 16차원 벡터 g를 분류층 바로 앞에서 이어 붙입니다: $[h \,\|\, g] \in \mathbb{R}^{80}$ ($\|$는 이어 붙이기). 파형 자체에는 개인화를 적용하지 않고, "평소 대비 변화" 정보는 g로만 들어갑니다. `use_personal: false`이면 개인화 경로 없이 파형만 받습니다.

- **1D 합성곱**: $y_t = \sum_{c}\sum_{j=0}^{k-1} w_{c,j}\, x_{c,\,t+j} + b$. $k$는 커널 길이(한 번에 보는 샘플 수), $c$는 입력 채널.
  첫 커널 $k=15$는 100 Hz에서 150 ms로, QRS파 폭에 맞춘 값입니다 (참고논문 2의 $k = \lfloor f_s \times 0.12 \rfloor$ 근거와 같은 발상).
- **BatchNorm**: 배치 평균 $\mu_B$와 분산 $\sigma_B^2$로 $\hat{x} = (x-\mu_B)/\sqrt{\sigma_B^2+\epsilon}$ 정규화 후 학습 가능한 $\gamma\hat{x}+\beta$. 학습을 안정시킵니다.
- **MaxPool($p$)**: $p$개 샘플마다 최댓값 하나만 남겨 시간축을 $1/p$로 줄입니다. 전체 32배 축소로 BiLSTM 계산량을 낮춥니다.
- **LSTM**: 시점 $t$마다 은닉 상태 $h_t$와 셀 상태 $c_t$를 갱신합니다.
  $$f_t = \sigma(W_f[h_{t-1}, x_t] + b_f),\ i_t = \sigma(W_i[\cdot] + b_i),\ o_t = \sigma(W_o[\cdot] + b_o),\ \tilde{c}_t = \tanh(W_c[\cdot] + b_c)$$
  $$c_t = f_t \odot c_{t-1} + i_t \odot \tilde{c}_t,\qquad h_t = o_t \odot \tanh(c_t)$$
  $f_t, i_t, o_t$는 망각·입력·출력 게이트(0~1 사이 값, $\sigma$는 시그모이드), $\odot$는 원소별 곱, $[h_{t-1}, x_t]$는 두 벡터를 이어 붙인 것입니다.
  **BiLSTM**은 정방향·역방향 LSTM의 $h_t$를 이어 붙여(32+32=64) 앞뒤 맥락을 모두 봅니다.
- 원파형을 직접 받으므로 R-peak 검출 품질에 덜 의존하지만, 데이터가 적으면 특징 기반 모델보다 불리합니다.

## 3. 신경망 공통 학습 방법

- 손실: 가중 교차엔트로피 $L = -\frac{1}{N}\sum_n w_{y_n} \log \hat{p}_{y_n}$.
  클래스 가중치 $w_k = N / (K \cdot N_k)$, $N_k$는 클래스 $k$의 학습 샘플 수 (적은 클래스일수록 가중치 큼).
- 최적화: Adam (학습률 $10^{-3}$, weight decay $10^{-4}$), 검증 손실이 `patience` 에폭 동안 줄지 않으면 조기 종료하고 가장 좋았던 가중치로 복원.

## 4. Random Forest

여러 결정 트리를 학습 데이터의 무작위 복원추출(bootstrap) 표본과 무작위 특징 부분집합(`max_features="sqrt"`, 즉 $\sqrt{F}$개)으로 따로 만든 뒤 평균을 냅니다.

$$\hat{p}_k(x) = \frac{1}{M}\sum_{m=1}^{M} \hat{p}^{(m)}_k(x)$$

$M$은 트리 개수(200), $\hat{p}^{(m)}_k(x)$는 $m$번째 트리에서 $x$가 도착한 잎(leaf)의 클래스 $k$ 비율입니다. 참고논문 1(ECG+CAN 졸음 감지)이 사용한 모델이며, 특징 중요도를 바로 볼 수 있습니다. `class_weight="balanced"`로 불균형을 보정합니다.

## 5. XGBoost / 6. LightGBM (그래디언트 부스팅)

트리를 하나씩 순서대로 추가하며, 새 트리가 이전까지의 오차를 줄이도록 학습합니다.

$$F^{(m)}(x) = F^{(m-1)}(x) + \eta\, f_m(x),\qquad \hat{p} = \mathrm{softmax}(F^{(M)}(x))$$

- $F^{(m)}$: 트리 $m$개까지 더한 클래스별 점수, $f_m$: $m$번째 트리, $\eta$: 학습률(0.05, 한 트리의 기여를 줄여 과적합 방지).
- XGBoost 목적함수: $\sum_n \ell(y_n, F(x_n)) + \sum_m \Omega(f_m)$, $\Omega(f) = \gamma T_f + \frac{\lambda}{2}\lVert w \rVert^2$.
  $\ell$은 교차엔트로피, $T_f$는 트리의 잎 개수, $w$는 잎 값, $\gamma, \lambda$는 복잡도 벌점 계수입니다.
- XGBoost는 깊이 기준(`max_depth=4`)으로 트리를 키우고, LightGBM은 잎 기준(`num_leaves=15`)으로 손실을 가장 많이 줄이는 잎부터 나눕니다. LightGBM이 보통 더 빠르고 작습니다.
- 둘 다 행·열 부분 표본(`subsample`, `colsample_bytree` 0.8)을 쓰고, 검증 운전자의 손실로 30회 개선이 없으면 조기 종료합니다.

## 7. 합성 데이터 동작 확인 결과 (성능 근거 아님)

`python scripts/run_benchmark.py --synthetic --compare-personal` · 합성 참가자 6명 (앞 30초 휴식 + 클래스 4개 × 3분), 윈도우 30초 / 10초 이동, 기준 구간 제외 후 384개 윈도우, LOSO 6 fold. 4단계 Macro-F1:

| 모델 | 개인화 없음 (특징 8) | 개인화 있음 (특징 15) | 지연 ms (batch 1, PC) |
| --- | --- | --- | --- |
| mlp | 0.667 | 0.932 | 0.04 |
| cnn_bilstm | 0.263 | 0.406 | 1.4 |
| random_forest | 0.544 | 0.895 | 7.4 |
| xgboost | 0.551 | 0.879 | 0.2 |
| lightgbm | 0.562 | 0.893 | 0.5 |

> 합성 데이터는 "사람마다 평소 심박수가 다르고, 상태에 따라 평소 대비 심박·HRV가 변한다"는 가정으로 만들었습니다. 그래서 개인화 효과가 실제보다 크게 나오는 구조입니다. 이 표는 **기준 구간 제외, fold 내 σ_pop 계산, 2-입력 CNN이 끝까지 동작하는지** 확인한 결과이며, 실제 효과는 공개 데이터셋으로 다시 측정해야 합니다. CNN+BiLSTM은 384개 윈도우로는 데이터가 부족합니다.

## 8. 다음 단계

1. 공개 데이터셋(CL-Drive 1순위) 로더 → `Recording` 변환.
2. 라벨 정의 확정 후 `configs/default.yaml`의 `label` 갱신.
3. 윈도우 길이 비교 실험(2 / 30 / 60초).
4. 공개 데이터셋으로 개인화 전/후 효과 재측정.
5. Pi 4에서 지연·메모리 측정 (양은철 파트와 연계).
