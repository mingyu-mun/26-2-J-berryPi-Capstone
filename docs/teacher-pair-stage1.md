# 쌍 비교 교사 1단계 구현 규격 (teacher-pair, RR 전용)

> 작성: 2026-11-11 · 실험 장소: 학교 PC (GPU 없음, 기존 RR 캐시) · 코드 기준: `claude/project-thread-tbz3az` 의 `d2f02e4`
> 이 문서의 쌍 규칙·조건 정의·개인 기준·지표·판정은 **결과를 보기 전에 고정**한다. 바꾸면 11장에 날짜와 이유를 적고, 바꾼 뒤 평가는 한 번만 다시 한다.

범위는 1단계 네 조건 **BT, P0, P1, P2** 다. NormWear 를 붙이는 P3~P6 은 이 문서 범위 밖이고, 1단계 관문(8.3)을 넘은 뒤 별도 규격으로 쓴다.

## 0. 설계 문서에서 바뀐 것

| 항목 | 설계 문서 | 이 규격 | 이유 |
| --- | --- | --- | --- |
| 창 격자 | 30초, 겹침 없음 | 30초, 10초 이동 | `TeacherSpec.stride_sec = 10`. 기존 `windows.parquet` 을 그대로 쓴다 |
| 개인 기준 | 첫 녹화 시작 3분 | 기존 `baseline_plan` 의 기준 구간 (기본 300초) | B1(dev_core)과 같은 기준이어야 공정하다 |
| 주 지표 | 시간 추세 제거 AUROC 4종 평균 | 시간 층화 AUROC (8.1 M1) | 선형 추세 제거는 DROZY 에서 계산이 안 되고 AdVitam 에서는 판단이 어렵다 (`results/2026-11-10/README.md` 4장) |
| λ_T, λ_S | 내부 LODO 로 선택 | 고정값 | λ_S 를 고를 지표가 WESAD 밖에 없는데 WESAD 는 평가 전용이다 |
| 평가 점수 | 2분 이동 평균 후 | 평활 없음 | 특징이 이미 120·300초 문맥을 담는다. B0·B1 도 평활하지 않는다 |
| RR 특징 차원 | d_r 미정 | 36 | `TeacherSpec.baseline_feature_names` |

## 1. 쓰는 데이터와 열

새로 뽑는 특징은 없다.

| 파일 | 열 |
| --- | --- |
| `cache/windows.parquet` | 키 `dataset_id, subject_id, session_id, t_start`, 그리고 `group, t_rel, valid, y, profile_key, row_id, inducer`, 특징 36열 |
| `cache/profiles.parquet` | `profile_key, baseline_valid, baseline_sec`, 기준 중앙값 `m_<특징>` 36열 |
| `cache/rr/**.npz` | `RRSeries.meta` (`baseline_range`, `baseline_per_session`, `nback`), `is_rest`, `t0` |
| `<converted>/<dataset>/…/labels_orig.csv` | CLAS 자극 시각, WESAD 조건 구간 |

학생용 3종(MAUS, CLAS, WESAD)은 같은 형식의 캐시를 따로 만든다 (9.2).

### 1.1 창마다 덧붙이는 열 (`pairs.annotate`)

| 열 | 정의 |
| --- | --- |
| `tbin` | AdVitam: `t_rel < 300` 이면 0, 아니면 1. 그 외: `floor(t_rel / 300)` |
| `ses_rank` | 같은 `group` 안에서 휴식 세션을 뺀 세션을 `t0` 순으로 세운 순번 (0부터) |
| `same_as_base` | `session_id` 가 `profile_key` 의 `#` 뒤 세션 이름과 같으면 1 |
| `base_kind` | `rest_session` / `range` / `default` (3.1) |
| `cond` | 학생용 데이터만: `task` / `stress` / `other` / `mixed` (4.3) |

AdVitam 의 `tbin` 이 따로인 이유: 라벨이 주행 앞 300초와 끝 300초에만 있고 주행 길이가 사람마다 달라서, 끝 구간을 위치로 맞춘다.

## 2. 입력 특징 (36열)

`spec.baseline_feature_names` 순서 그대로. 추세 2열(`trend_*`)과 `NON_INPUT` 열은 넣지 않는다.

| 스케일 | 특징 | 수 |
| --- | --- | --- |
| 30초 | `hr, rr_mean, sdnn, rmssd, pnn50, sd1, sd2, sd1_sd2` + `_30` | 8 |
| 120초 | 위 8개 + `lf_log, hf_log, lf_hf_log, lf_nu, dfa_a1, sampen` + `_120` | 14 |
| 300초 | 120초와 같은 14개 + `_300` | 14 |

변환 순서:
1. 로그: 이름 앞부분이 `hr, rr_mean, sdnn, rmssd, sd1, sd2` 인 18열에 자연로그. 0 이하는 결측.
2. 표준화: 학습 fold 의 유효 창으로 평균·표준편차. 가중치는 데이터셋 균등 × 참가자 균등. 교사 3종과 CLAS·MAUS 를 함께 쓴다.
3. 자르기: ±5.
4. 결측: 표준화 뒤 0. 결측 표시 열은 만들지 않는다.

쌍 결측 규칙: 쌍의 한쪽에서라도 결측인 특징은 양쪽 모두 0 으로 둔다.

## 3. 개인 기준

개인 기준은 기준 구간에서 구한 특징 중앙값 벡터 x_ref 이고, 기준 점수는 그 벡터를 같은 모델에 넣은 값이다. 구간 정의는 기존 `features.baseline_plan` 을 바꾸지 않고 쓴다.

### 3.1 기준 구간

1. 세션 메타에 `baseline_per_session` 이 있으면 세션마다 자기 구간.
2. 아니면 휴식 세션(`is_rest`)이 있을 때 그 세션, 없으면 `t0` 가 가장 이른 세션.
3. 구간은 메타의 `baseline_range`, 없으면 `[0, 300)` 초. 세션 길이를 넘지 않게 자른다.

| 데이터 | 기준 구간 | `base_kind` | 비고 |
| --- | --- | --- | --- |
| AdVitam | `baseline` 세션 앞 300초 | `rest_session` | `baseline` 표시가 없는 참가자는 block1 앞 300초 → `default` |
| DROZY | 가장 이른 test 앞 300초 | `default` | 그 세션의 300초 이전 창은 없다. D07·D09 는 test2 가 기준 |
| MPD-DF | `drive` 앞 300초 | `default` | |
| DD-DB | 회차마다 앞 300초 | `default` | `baseline_per_session` |
| MAUS | `rest` 세션 앞 300초 | `rest_session` | 휴식이 비면 가장 이른 trial → `default` (M003) |
| CLAS | `begin` 화면 60초 | `range` | 120·300초 묶음은 기준이 없다 |
| WESAD | `Base` 조건 전체 (약 20분) | `range` | |

유효 조건: 구간 길이 60초 이상이고 30초 묶음 첫 특징의 중앙값이 유한하면 `baseline_valid = 1`.

### 3.2 기준 벡터와 기준 점수

```
x_ref[j] = profiles.m_<특징 j>        (2장 변환 적용)
M_t[j]   = 1 (x_t[j] 와 x_ref[j] 가 모두 유한) / 0
d_t      = f(x_t ⊙ M_t) − f(x_ref ⊙ M_t)
```

- `m_<특징>` 은 스케일이 기준 구간보다 길면 결측이다. 그 특징은 M_t 로 양쪽에서 빠진다.
- 평가와 라벨링 점수는 d_t. `baseline_valid = 0` 인 창은 d_t 를 내지 않는다.
- P2 는 헤드별로 d_U, d_D 를 같은 식으로 구하고, 결합은 d_t = s(x_t) − s(x_ref) (s 는 5.2 결합 점수).

### 3.3 기준이 정상 상태라는 보장은 없다

기준 구간 라벨로 참가자를 거르지 않는다. 대신 보고:
- AdVitam `KSS_0 ≥ 7` 참가자 수
- DROZY 기준 세션 KSS ≥ 7 참가자 수, 기준이 test1 이 아닌 참가자 (D07, D09)

## 4. 쌍 규칙

모든 쌍은 같은 `profile_key` 안에서만. 공통: 두 창 모두 `valid = True`. 같은 세션의 두 창은 시작 시각 차이 300초 이상.

### 4.1 R 쌍 (순위)

졸음 창 a (y=1), 정상 창 b (y=0). 같은 `profile_key`. 세션이 다르거나, 같은 세션이면 |t_rel_a − t_rel_b| ≥ 300. 세션이 다르고 tbin_a = tbin_b 이면 tm = 1. 요구: 점수 a > 점수 b.

| 데이터 | 쌍 | tm |
| --- | --- | --- |
| DROZY | 졸음 세션 창 vs 정상 세션 창 (항상 세션 다름) | tm=1: 같은 tbin 끼리 |
| AdVitam | 같은 블록의 끝(졸음) vs 앞(정상), 다른 블록 사이 모든 졸음·정상 조합 | tm=1: 다른 블록의 앞 vs 앞, 끝 vs 끝 |
| DD-DB | 같은 회차의 버튼 ±60초 창 vs 600초 이상 떨어진 창 | 없음 |
| MPD-DF | 같은 주행의 단계 2~4 창 vs 단계 0 창, 300초 이상 간격 | 없음 |

### 4.2 T 쌍 (시간 불변)

같은 `session_id`, y_a = y_b, |t_rel_a − t_rel_b| ≥ 300. 요구: |점수 a − 점수 b| ≤ m.

- AdVitam 앞·끝 KSS 모두 5 이하인 블록의 앞 vs 끝 포함.
- DROZY 는 기준 세션이 아닌 세션에서만.
- 라벨 -1 창 안 씀. 세션을 넘는 T 쌍 없음.

### 4.3 S 쌍 (스트레스)

기준 벡터 x_ref vs 과제 창 b. 조건: 데이터셋 clas 또는 maus, cond_b = task, base_kind ∈ {rest_session, range}, baseline_valid = 1. 요구: 점수 b ≤ 점수 x_ref + m (한쪽 방향만).

`cond`:
- MAUS = trial1~trial6 모든 창 (M003 은 default 라 제외)
- CLAS = 블록 종류 Math Test, Logic(IQ) Test, Stroop Test 구간에 완전히 들어가는 창은 `task`; 영상·그림·Neutral 은 `other`, 경계 걸친 창은 `mixed`
- WESAD = 학습에 안 씀; TSST 구간에 완전히 들어가는 창은 `stress` (M5 평가용)

CLAS 는 `labels_orig.csv` 의 stimulus 이름 → Block_details 블록 종류 대응표 필요. 첫 단계에서 고유 이름 목록을 뽑아 10장에 적고 점수 보기 전에 확정.

### 4.4 표집

배치 쌍 256개: R 128, T 77, S 51.

1. 데이터셋: 그 종류 쌍이 있는 학습 데이터셋 중 균등
2. 참가자: 그 종류 쌍이 하나라도 있는 group 중 균등
3. 창
   - R: 졸음 창 a 균등; 후보 C = 4.1 조건 만족 정상 창; C 안에 tm=1 창이 있으면 확률 0.5 로 그중에서, 아니면 C 전체에서 b 균등; C 가 비면 a 재추출 (최대 20회)
   - T: T 쌍 있는 세션 균등 → 쌍 있는 라벨 균등 → a 균등 → 조건 만족 b 균등
   - S: 과제 창 b 균등, 상대는 그 창의 x_ref

- 난수: `np.random.default_rng(1000 * seed + fold_index)`
- LODO 에서 빼는 데이터셋은 1번에서 빠진다. CLAS·MAUS 는 모든 fold 학습에 포함. WESAD 는 어느 fold 에도 안 들어감.
- 쌍 목록을 미리 다 만들지 않는다. 참가자별 (session, t_rel, y, tbin) 배열만 들고 그때그때 뽑는다.

### 4.5 학습 전에 채우는 표 (pair-stats)

열: R 쌍 가능 참가자 / 그중 tm=1 가능 / T 쌍 가능 참가자 / S 쌍 가능 참가자 / 유효·라벨 창.
행: AdVitam (5,365 창), DD-DB (7,594), DROZY (1,072), MPD-DF (26,110), CLAS, MAUS (S 만).
R 쌍 가능 참가자 5명 미만 데이터셋은 결과 표에 표시. 규칙은 안 바꿈.

## 5. 조건 정의

| | BT | P0 | P1 | P2 |
| --- | --- | --- | --- | --- |
| 입력 | 시간·세션 위치 4열 | RR 36열 | RR 36열 | RR 36열 |
| 모델 | 로지스틱 회귀 | MLP 헤드1 | MLP 헤드1 | MLP 헤드2 |
| 학습 단위 | 창 | 창 | 쌍 | 쌍 |
| 손실 | BCE | BCE | L_R + L_T + L_S | L_R + 헤드 보조 + L_T + L_S |
| CLAS·MAUS | 아니오 | 아니오 | 예(S) | 예(S) |

### 5.1 BT

- BT-t: 점수 = t_rel. 학습 없음.
- BT-fit: 입력 [t_rel/600, (t_rel/600)², ses_rank, same_as_base]. `LogisticRegression(C=1.0)`, sample_weights 가중, LODO.

BT-t 원 AUROC 는 `results/2026-11-10/README.md` 4장 "시간만으로" 줄(AdVitam 0.707, DD-DB 0.346, MPD-DF 0.785)과 ±0.01 안에서 맞아야 한다. 안 맞으면 원인 찾은 뒤 진행.

### 5.2 P1, P2

```
몸통: x(36) → Linear(36,32) → GELU → Dropout(0.3) → Linear(32,32) → GELU → Dropout(0.3) → h(32)
P1: s = Linear(32,1)(h)                                           (파라미터 2,273)
P2: s_U = Linear(32,1)(h), s_D = Linear(32,1)(h),
    s = logaddexp(s_U, s_D) − ln 2                                (2,306)
```

쌍의 두 입력은 같은 가중치를 각각 통과.

손실:
```
L_R(a,b) = softplus(−(s_a − s_b))
L_T(a,b) = max(0, |s_a − s_b| − m)²
L_S(b)   = softplus(s_b − s_ref − m)
P1: L = mean L_R(s) + λ_T·mean L_T(s) + λ_S·mean L_S(s)
P2: L = mean L_R(s) + λ_h·[mean_{KSS 쌍} L_R(s_U) + mean_{EEG 쌍} L_R(s_D)]
        + λ_T·mean[L_T(s_U)+L_T(s_D)] + λ_S·mean[L_S(s_U)+L_S(s_D)]
```
- KSS 쌍 = AdVitam·DROZY R 쌍, EEG 쌍 = MPD-DF R 쌍. DD-DB R 쌍은 결합 점수 s 만 감독.
- MPD-DF 를 빼는 fold 에서는 EEG 쌍 항이 0. 표에 표시.

### 5.3 P0

P1 과 같은 몸통·헤드·정규화·학습 길이. 학습 창: 교사 3종의 valid 이고 y ≥ 0 인 창. 손실: sample_weights(dataset, group, y) 가중 BCE; 배치는 창 256개를 가중치에 비례해 뽑음. 결측 창별 0. T·S 없음. CLAS·MAUS 안 씀. 점수 두 가지: d_t (3.2 기준 차감)와 원 로짓 s_t. P1 과 비교는 d_t.

### 5.4 고정 하이퍼파라미터

- m = 0.5; λ_h, λ_T, λ_S = 0.5, 1.0, 0.5
- AdamW lr 1e-3 고정, weight decay 1e-2; 2,000 스텝, 조기 종료·검증 분할 없음
- 입력 잡음: 표준화 뒤 N(0, 0.1²), 쌍 두 쪽에 서로 다른 잡음
- 스케일 묶음 드롭: 쌍마다 확률 0.1 로 120초 묶음, 0.1 로 300초 묶음을 양쪽 같이 0 (P0 에도 창 단위로 같은 확률)
- 시드 0, 1, 2, 3, 4

## 6. LODO 절차

fold 는 교사 4종을 하나씩 뺀 4개.

1. 시험 데이터셋을 뗀다. 학습 = 나머지 교사 3종 + CLAS + MAUS.
2. 학습 창으로 2장 표준화 통계.
3. 조건·시드마다 2,000 스텝 학습.
4. 학습 창 중 valid, y ≥ 0, baseline_valid = 1 인 창의 d 로 보정: P = σ(a·d + b), sample_weights 가중. a ≤ 0 이면 a = 1e-3 (rule.py 와 같은 처리).
5. 시험 데이터셋 d, P. 시드 5개는 보정 로짓 a·d + b 를 평균.
6. 같은 fold 모델로 WESAD 전체 창의 d, P (M5 용).

`oof_lodo.parquet` 열: `row_id, dataset_id, group, session_id, t_rel, tbin, y, fold`, 조건별 `d_<조건>`, `p_<조건>`, 시드별 `d_<조건>_s<k>`.

## 7. 평가 행

모든 조건과 B0·B1 을 같은 행에서 비교: valid, y ≥ 0, baseline_valid = 1, 비교하는 모든 조건 점수가 유한한 창.

- B0 (rrnet): 기존 `oof_lodo.parquet` 의 `p_rrnet` 을 `row_id` 로 잇는다.
- B1 (dev_core): fold 마다 `PersonalNormalizer(spec).fit(학습 창, prof).transform(시험 창, prof)["dev_core"]`.

재현 확인: 행 선택 전 전체 행에서 B0·B1 원 AUROC 가 README 1장 값(AdVitam/DD-DB/DROZY/MPD-DF: rrnet 0.529/0.558/0.744/0.656, dev_core 0.571/0.536/0.898/0.655)과 ±0.005 안인지 먼저 본다.

## 8. 지표와 판정

### 8.1 지표

| 지표 | 정의 |
| --- | --- |
| M1 | 시간 층화 AUROC (주 지표, 4종 평균) |
| M2 | 원 AUROC (`evaluate.metrics`) |
| M3 | 선형 추세 제거 AUROC (세션마다 점수를 t_rel 에 선형 회귀한 잔차, DD-DB·MPD-DF 만) |
| M4 | 참가자 내 AUROC (두 라벨 다 있는 참가자 평균) |
| M5 | WESAD TSST 오경보율 |
| M6 | ECE (`calibrate.ece`, P 기준) |
| M7 | DROZY 참가자 5-fold AUROC (DROZY 만으로 학습, S 쌍 포함, 원 AUROC, 문헌 0.753 과 비교) |

M1: 데이터셋 D, 층 b = tbin
```
w_i  = 1 / (그 참가자가 층 b 에서 가진 같은 라벨 창 수)
A_b  = 가중 AUROC (층 b 졸음 vs 정상, 가중치 w)
W_b  = (층 b 졸음 창 있는 참가자 수) × (정상 창 있는 참가자 수)
M1(D) = Σ W_b·A_b / Σ W_b   (두 라벨 다 있는 층만)
M1   = 4종 평균
```
- 비교에 쓰인 창 비율을 함께 보고. BT-t 는 M1 0.5 근처여야 함. BT-fit 의 M1 이 DROZY 기준 세션 효과 크기를 보여 줌.

M5 (fold 모델 4개 각각 → 평균, 대상 cond = stress 창):
1. P ≥ 0.5 비율 (v1 77%)
2. P 중앙값 (v1 0.87)
3. 민감도 50% 고정 오경보율: 그 fold 시험 데이터셋 졸음 창 d 의 가중 중앙값을 θ, d ≥ θ 인 TSST 창 비율

조건 간 비교는 3번. B0 의 3번은 fold 모델 저장하도록 고쳐 재실행해야 나옴; 못 하면 B0 은 1·2번 기존 값만.

### 8.2 불확실성

참가자 단위 쌍체 부트스트랩: 데이터셋마다 group 복원 추출 2,000회 (시드 0), 같은 표본에서 두 조건 지표 차이, 2.5~97.5 백분위 = 95% 구간. 시드별 M1 최소·최대도 기록.

### 8.3 판정 (결과 보기 전 고정)

- 지표 유효: BT-t 의 M1 이 네 데이터셋 모두 0.45~0.55
- 1단계 관문 통과: 지표 유효하고 P1 또는 P2 의 M1 평균이 B0 보다 높다
- 헤드 수: M1 에서 P2 − P1 95% 구간 하한 > 0 이면 2개, 아니면 1개
- 쌍 학습 효과 인정: M1 에서 P1 − P0 95% 구간 하한 > 0
- 중단: 관문 못 넘으면 공개 데이터 한계로 보고 자체 측정을 앞당긴다

지표 유효 조건이 깨지면 tbin 폭을 150초로 한 번만 줄이고 11장에 기록. 그 뒤엔 안 바꿈.

## 9. 구현

### 9.1 파일

- `src/driver_state/teacher/pairs.py`: annotate, baseline_kinds, tag_conditions, PairIndex, PairSampler, pair_stats
- `src/driver_state/teacher/models/pairnet.py`: 몸통·헤드, 손실 세 항, PairNet.fit(sampler), score(x, mask)
- `src/driver_state/teacher/pair_eval.py`: LODO, BT·P0, 기준 차감, 보정, oof_lodo.parquet
- `src/driver_state/teacher/metrics_time.py`: M1, M3, M4, M5, 부트스트랩
- `scripts/teacher.py`: pair-stats, pair-eval 하위 명령
- `tests/test_pairs.py`

기존 TeacherModel 인터페이스는 쓰지 않는다 (TeacherInput 이 NON_INPUT 열을 빼는데 표집에 group, session_id, t_rel 이 필요). 이 열들은 표집기만 보고 모델 입력에는 안 들어감.

### 9.2 실행 순서

```
python scripts/teacher.py rr --data <converted>\maus <converted>\clas <converted>\wesad --out <work>\cache_student\rr --jobs 4
python scripts/teacher.py features --rr <work>\cache_student\rr --out <work>\cache_student --no-tacho
python scripts/teacher.py pair-stats --cache <work>\cache --student <work>\cache_student --data <converted> --out <work>\outputs\pair
(재현 확인 5.1, 7장 뒤)
python scripts/teacher.py pair-eval --cache <work>\cache --student <work>\cache_student --data <converted> --b0-oof <work>\outputs\cfg_...\oof_lodo.parquet --conds BT P0 P1 P2 --seeds 0 1 2 3 4 --out <work>\outputs\pair
```
산출물: `pair_stats.json, oof_lodo.parquet, wesad_scores.parquet, report_pair.md, report_pair.json`.

### 9.3 테스트 (`teacher.synthetic` 합성 데이터)

test_pairs_same_profile, test_pairs_gap (같은 세션 쌍 t_rel 차 ≥ 300), test_r_labels (R=(1,0), T 같은 라벨, -1 없음), test_tm_flag, test_lodo_exclusion (빼는 데이터셋 row_id 가 표집·표준화 통계에 없음), test_wesad_never_trained, test_pair_mask, test_ref_score (x_t = x_ref 이면 d_t = 0), test_m1_time_only (점수=t_rel 이면 M1 0.45~0.55, 점수=y 이면 1.0), test_determinism.

## 10. 구현하면서 채울 것

> 아래는 2026-11-11 본 실행 **전에** 채웠다 (pair-stats, BT 전용 재현 실행).

### 10.1 쌍 통계 (4.5, tbin 150초 기준)

| 데이터 | R 쌍 가능 참가자 | 그중 tm=1 | T 쌍 가능 참가자 | S 쌍 가능 참가자 | 유효·라벨 창 |
| --- | --- | --- | --- | --- | --- |
| AdVitam | 15 | 11 | 49 | 0 | 5,365 |
| DD-DB | 10 | 0 | 10 | 0 | 7,594 |
| DROZY | 9 | 9 | 12 | 0 | 1,072 |
| MPD-DF | 38 | 0 | 50 | 0 | 26,110 |
| CLAS | 0 | 0 | 0 | 36 | 0 |
| MAUS | 0 | 0 | 0 | 21 | 0 |

R 쌍 가능 참가자 5명 미만 데이터셋은 없다. 유효·라벨 창 수는 4.5 표의 기대값과 같다.
CLAS 는 60명 중 36명만 S 쌍이 가능하다 (기준 60초 구간의 30초 특징이 유효하고 과제 창이 유효한 참가자).

### 10.2 CLAS 자극 → 블록 종류 대응표

`labels_orig.csv` 의 고유 자극 이름(숫자를 # 로 바꾼 패턴): `#.png`, `#a.png`, `#r.png`, `#.#.png`, `#.#.#.png`, `#.PNG`(소문자화 후 `#.png`), `#.mp4`, `#.jpg`, `#_#.png`, `#_#_#.png`, `n#.jpg`, `neutral#.mp4`, `neutral.mp4`, `begin.png`, `mathtestanswer.png`, `strooptestanswer.png`, `iqtestanswer.png`, `end.jpg`, `all.png`.
같은 패턴(`#.png` 등)이 여러 과제에 섞여 있어 이름만으로는 블록을 정할 수 없다. 그래서 `_stimuli.txt` 의 구조를 따라 **휴식 화면 → 과제 → 정답 화면** 순서로 구간을 정한다.

| 구간 | 시작 | 끝 | 블록 종류 | cond |
| --- | --- | --- | --- | --- |
| 수학 | `begin.png` 또는 `n#.jpg` 다음 자극 | `mathtestanswer.png` 직전 | Math Test | task |
| 스트룹 | `n#.jpg` 다음 자극 | `strooptestanswer.png` 직전 | Stroop Test | task |
| IQ | `n#.jpg` 다음 자극 | `iqtestanswer.png` 직전 | Logic(IQ) Test | task |
| 정답 화면, 영상, 그림, `neutral*`, `n#.jpg`, `begin`, `all`, `end` | | | Response / Video clip / Pictures / Neutral / Baseline | other |

결과: 과제 구간 Math 59, Stroop 60, IQ 60 (Part 하나는 `mathtestanswer.png` 가 없어 수학 구간 없음). 창 cond: task 3,305 / other 9,137 / mixed 897. MAUS task 3,668. WESAD stress 987 / other 5,973 / mixed 79.

### 10.3 기준이 졸음이던 참가자 (3.3)

- AdVitam `KSS_0 ≥ 7`: 10명
- DROZY 기준 세션 KSS ≥ 7: 0명. 기준이 test1 이 아닌 참가자: D07(test2), D09(test2)
- `base_kind`: AdVitam rest_session 63, CLAS range 60, WESAD range 15, MAUS rest_session 21 + default 1 (M003), DD-DB·DROZY·MPD-DF default

### 10.4 재현 확인 (5.1, 7장)

| 확인 | AdVitam | DD-DB | DROZY | MPD-DF | 기준값과 차이 |
| --- | --- | --- | --- | --- | --- |
| B0 rrnet 원 AUROC | 0.529 | 0.558 | 0.744 | 0.656 | 0.000 |
| B1 dev_core 원 AUROC | 0.571 | 0.536 | 0.898 | 0.655 | 0.000 |
| BT-t "시간만으로" (README 4장 방식: 같은 세션 안 AUROC) | 0.707 | 0.346 | — | 0.785 | 0.000 |
| M3 B0 (선형 추세 제거) | 0.494 | 0.721 | — | 0.592 | 0.000 |

README 4장의 "시간만으로" 값은 풀링한 원 AUROC 가 아니라 같은 세션 안 AUROC 를 졸음×정상 수로 가중 평균한 값이라, 그 방식으로 재현했다. 원 AUROC(M2)로는 AdVitam 0.565, DD-DB 0.521, DROZY 0.345, MPD-DF 0.708.

### 10.5 지표 유효 조건 (8.3) 확인

tbin 300초에서 BT-t 의 M1 = AdVitam 0.532, DD-DB 0.502, DROZY 0.521, **MPD-DF 0.574** → 0.45~0.55 를 벗어남.
8.3 규칙대로 tbin 폭을 **150초로 한 번** 줄였다 (11장). 150초에서 BT-t M1 = 0.532 / 0.508 / 0.494 / 0.537 로 네 데이터셋 모두 범위 안 → 지표 유효. 이후 바꾸지 않는다.
참고 (150초, BT 전용 실행): BT-fit M1 = 0.583 / 0.639 / **0.906** / 0.471, B0 = 0.541 / 0.587 / 0.736 / 0.581, B1 = 0.581 / 0.582 / 0.837 / 0.548.
BT-fit 의 DROZY 0.906 은 "기준 세션인가 + 세션 순번" 만으로 맞힌 값이라, DROZY 의 M1 은 기준 세션 효과가 크다는 뜻이다.

### 10.6 구현 해석 (규격 문구가 두 가지로 읽히는 곳)

- P0 손실: "sample_weights 가중 BCE; 배치는 가중치에 비례해 뽑음" → 가중치에 비례해 뽑고 배치 안에서는 가중 없이 BCE 평균 (가중 BCE 와 기댓값이 같다. 둘 다 적용하면 가중이 두 번 들어간다).
- 평가 점수는 드롭아웃을 끈 상태 (`model.eval()`), 잡음·스케일 드롭 없음.
- d_<조건> 열 = 시드 5개 d 의 평균, p_<조건> = 시드 5개 보정 로짓 평균의 시그모이드. P0 의 원 로짓은 `d_P0raw`.
- M7 은 DROZY 참가자를 시드 0 순열로 5묶음. 학습 = DROZY 학습 참가자 (R·T) + CLAS·MAUS (S), 점수 = 시드 평균 보정 로짓.
- B1 (dev_core) 의 M5 θ 는 같은 fold 시험 데이터셋 졸음 창의 dev_core 가중 중앙값.

## 11. 변경 기록

| 날짜 | 바꾼 것 | 이유 |
| --- | --- | --- |
| 2026-11-11 | 최초 작성 | — |
| 2026-11-11 | tbin 폭 300초 → **150초** (AdVitam 은 앞·끝 2층 그대로) | 8.3 지표 유효 조건: 300초에서 BT-t M1 이 MPD-DF 0.574 로 범위를 벗어남. 규칙대로 한 번만 줄임 (10.5) |
