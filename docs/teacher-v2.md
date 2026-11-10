# 교사 v2 설계: NormWear + 라벨 종류별 헤드 + 맥락 게이트

> 확정: 2026-11-10 · 실험 장소: 집 PC (RTX 4070 SUPER) · 이 문서의 구조·평가·결정 기준은 **결과를 보기 전에 고정**한다.
> 바꿔야 하면 바꾼 이유와 날짜를 9장에 적고, 바꾼 뒤 평가는 다시 한 번만 한다.

## 1. 왜 바꾸나 (v1 에서 확인한 것)

| 확인한 것 | 근거 (docs/teacher.md, 2026-11-09~10 실험) |
| --- | --- |
| 데이터셋마다 "졸음"이 다른 상태 | KSS(AdVitam·DROZY)는 졸음과 싸우는 상태 = 심박↑, EEG(MPD-DF)는 잠에 빠져드는 상태 = 심박↓ |
| 헤드 하나는 다수 방향을 배움 | v1 교사가 WESAD TSST(스트레스)를 졸음으로 판정 (졸음 77%, P 중앙값 0.87) |
| 특징 실험으로는 한계 | 부호 없는 특징, MSPC, 정규화 변경 모두 LODO AUROC ±0.02 안 (최고 rrnet 0.622) |
| 시간 효과가 섞임 | 시간만으로 AdVitam 0.71, MPD-DF 0.79. 시간 추세를 빼도 DD-DB rrnet 0.72, MPD-DF 규칙 0.70 신호는 남음 |

## 2. 구조

```
창 = 30초, 10초 이동 (학생과 같은 격자)

[1] 창 인코더 (고정)      NormWear(ECG[, PPG]) → 채널별 768 → 패치 평균 → e_t (768 또는 1536)
[2] 투영                  e_t → Linear(→128) → x_t
[3] 맥락 인코더           토큰 열 [x_{t-19}, …, x_t] (지난 10분, 30초 간격 20개, 인과 마스크·빈 칸 마스크)
                          → Transformer 2층, d=128, 머리 4, 위치 임베딩 → c_t (128)
[4] 맥락 헤드             c_t → Linear → softmax → q_t (상황 4종 확률)
[5] 라벨 종류별 헤드      h_t = [x_t ‖ c_t] (256) → Linear → 로짓
                          KSS 헤드 · EEG 헤드 · 버튼 헤드
[6] 게이트                g_t = softmax(M q_t)  (M: 4×3 학습 행렬)
                          P(졸음)_t = Σ_k g_{t,k} · σ(헤드_k)
```

- NormWear: `mosaic-laboratory/normwear` (Apache 2.0). `model.safetensors` 만 쓰고 `pytorch_model.bin`(pickle)은 받지 않는다.
- 1단계에서 NormWear 는 **고정**. 특징을 한 번 뽑아 `cache/normwear/<dataset>.npy` (+ `row_id`) 로 저장하고 이후 학습은 저장된 특징만 읽는다.
- 입력 주기: NormWear 문서의 권장 주기를 집 PC 세팅 때 확인한다. 권장값이 없으면 공통 전처리(100 Hz, 대역통과)를 거친 신호를 `sampling_rate=100` 으로 넣는다. 정한 값은 9장에 기록한다.
- PPG 가 없는 창(교사 데이터 4종)은 PPG 자리를 0 으로 두고 마스크 채널 1개를 붙인다. 교사 학습은 ECG 만으로도 되게 한다.
- 맥락 길이 10분: 지난 기록이 짧으면(세션 시작 직후, DROZY·MAUS 짧은 세션) 있는 만큼만 쓰고 나머지는 마스크.
- M 초기값: 수면 박탈 → KSS, 장시간 주행 → EEG·버튼 반반, 스트레스·과제 → 균등(이 경우 정답은 정상), 평상시 → 균등.

## 3. 라벨

### 3.1 졸음 라벨 (헤드별, v1 정의 그대로)

| 헤드 | 데이터 | 졸음 1 | 정상 0 |
| --- | --- | --- | --- |
| KSS | AdVitam, DROZY | KSS ≥ 7 | KSS ≤ 5 (6 은 제외) |
| EEG | MPD-DF | 의사 판정 단계 2~4 | 단계 0 (1 은 제외) |
| 버튼 | DD-DB | 버튼 ±60초 | 버튼에서 600초 이상 |

### 3.2 확실한 각성 구간 (결합 출력 P 의 정상 0, 헤드별 손실에는 안 씀)

| 데이터 | 구간 |
| --- | --- |
| WESAD | TSST, Fun, sRead, fRead (명상·Base 는 제외) |
| CLAS | 암산·스트룹 등 과제 블록 |
| MAUS | n-back trial 1~6 |

### 3.3 맥락 라벨 (맥락 헤드)

| 상황 | 창 |
| --- | --- |
| 수면 박탈 | DROZY test2·3, AdVitam D 그룹 |
| 장시간 주행 | MPD-DF, DD-DB, AdVitam A 그룹의 주행 블록 |
| 스트레스·과제 | WESAD TSST·읽기, CLAS 과제, MAUS n-back |
| 평상시·휴식 | DROZY test1, WESAD Base·명상·Fun, 각 데이터 기준 구간 |

맥락 라벨이 데이터셋과 겹치는 문제(맥락 = 데이터셋 식별) 대응: 같은 데이터셋 안에서 상황이 바뀌는 데이터(DROZY test1 vs 2·3, AdVitam A vs D, WESAD 조건)를 모두 포함하고, 4장의 적대 손실을 쓴다.

## 4. 손실

```
L = Σ_k BCE(헤드_k, 그 헤드 라벨)             (라벨 있는 창만)
  + 1.0 · BCE(P(졸음), 결합 라벨)               (헤드 라벨 + 3.2 의 각성 0)
  + 0.5 · CE(맥락 헤드, 맥락 라벨)
  + 0.1 · CE(데이터셋 판별기 ∘ GRL, 데이터셋 id)  (c_t 에 기울기 반전, 도메인 적대)
```
- 가중치: 데이터셋 × 참가자 × 클래스 균형 (v1 과 같음).
- 최적화: AdamW lr 1e-3, weight decay 1e-4, 배치 256 시퀀스, 최대 50 epoch, 검증 손실 patience 5. 시드 0, 1, 2 세 번 돌려 평균·범위를 보고.

## 5. 평가 (고정)

- **LODO**: 교사 데이터 4종(advitam, dddb, drozy, mpddf)을 하나씩 시험. 3.2·3.3 의 WESAD·CLAS·MAUS 는 **참가자 단위 4 fold** 로 나눠 같은 fold 번호끼리 묶는다 (시험 참가자는 학습에 안 씀).
- EEG 라벨은 MPD-DF 뿐이라 MPD-DF fold 에서는 EEG 헤드가 학습되지 않는다. 그 fold 는 결합 출력 P 로 평가하고 표에 표시.
- 지표
  1. 데이터셋별·평균 AUROC (결합 P, 그리고 해당 헤드)
  2. 시간 추세 제거 AUROC (세션 안 선형 추세 제거, 2026-11-10 방식)
  3. WESAD TSST 오경보율: 학습 fold 에서 정한 τ_hi 로 졸음 판정되는 TSST 창 비율
  4. 맥락 헤드 LODO 정확도 (처음 보는 데이터셋의 상황을 맞히는가)
  5. ECE

## 6. 비교 (ablation, 같은 특징·같은 fold)

| ID | 구성 |
| --- | --- |
| B0 | v1 rrnet (기준, 0.622) |
| B1 | dev_core 규칙 (참고, 0.665) |
| A1 | NormWear + 헤드 1개 (선형) |
| A2 | NormWear + 라벨별 헤드 (맥락 없음, 결합은 평균) |
| A3 | NormWear + 라벨별 헤드 + 맥락 게이트 (본안) |
| A4 | A3 에서 도메인 적대 손실 제거 |

## 7. 결정 기준 (미리 정함)

- A3 채택: 평균 LODO AUROC 가 A2 보다 0.02 이상 높고, **TSST 오경보율이 B0·A2 보다 낮을 때**.
- A3 가 A2 와 0.02 안이면 단순한 A2 를 쓴다. A2 가 A1 보다 0.02 안이면 A1.
- 어떤 구성도 B0 를 넘지 못하면 NormWear 특징 대신 RR 인코더로 같은 구조를 시험한다 (v1 rrnet 인코더 재사용).
- 문헌 비교: DROZY 참가자 5-fold AUROC (Kralikova 2026 의 0.753 과 같은 방식)도 함께 보고.

## 8. 보드 탑재 (다음 단계)

NormWear(약 2억 파라미터)는 라즈베리파이에 올리지 않는다. 학생 = 작은 창 인코더(수만 파라미터)가 NormWear 특징을 흉내 내도록(MSE) + 결합 P 를 따르도록(KL) 증류하고, 그 위에 작은 맥락 모듈(토큰 20개, d=32)을 얹어 ONNX 로 내보낸다.

## 9. 변경 기록

| 날짜 | 바꾼 것 | 이유 |
| --- | --- | --- |
| 2026-11-10 | 최초 확정 | — |

## 10. 집 PC 준비 순서

1. 이 PC(DESKTOP-5QLNVTK)에서 `git push` (미푸시 커밋 반영)
2. 집 PC: 레포 clone, `claude/project-thread-tbz3az` 브랜치, Python 3.11 + CUDA PyTorch, `pip install -r requirements.txt`
3. `Downloads\datasets\converted` (약 1.8 GB) 와 `code\work\teacher\cache` (약 0.4 GB) 복사
4. NormWear 코드·`model.safetensors` 내려받기, 창 몇백 개로 속도·출력 형태 확인
5. 특징 추출 → A1~A4 학습·평가 → 5장 지표로 보고
