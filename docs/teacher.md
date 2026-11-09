# 교사 모델 (ECG 정상/졸음 라벨러) v0.1

교사는 졸음 라벨이 있는 공개 ECG 데이터로 학습해, 라벨이 없는 ECG 데이터의 30초 창(10초 이동)마다 보정된 P(졸음)을 붙이는 오프라인 라벨러입니다. 학생 모델의 학습 라벨만 만들며, 상태 머신과는 무관합니다. 코드는 `src/driver_state/teacher/`, 실행은 `scripts/teacher.py`, 설정은 `configs/teacher.yaml`입니다.

## 1. 흐름

```text
원 데이터 ─(어댑터)─▶ 규약 폴더 (data.npz, ECG만) ─rr─▶ RR 캐시 ─features─▶ windows.parquet + profiles.parquet (+ tacho.npy)
   ─evaluate─▶ LODO 예측 oof_lodo.parquet + report_lodo.md ─tau─▶ tau.json ─fit─▶ teacher_bundle/ ─label─▶ 대상 세션마다 teacher_p.csv, labels_window.csv
```

| 명령 | 하는 일 | 결과 |
| --- | --- | --- |
| `inspect --src` | 원 데이터의 파일 구성·헤더 출력 (어댑터 작성용) | 화면 출력 |
| `rr --data --out --jobs` | 공통 전처리 → R-peak(포물선 보간) → RR, 이상 RR 표시, 10초 블록 품질 | `cache/rr/**.npz` |
| `features --rr --out --jobs` | 창 그리드, 3개 스케일 특징, 참가자 Baseline 프로필, RR 타코그램 | `windows.parquet`, `profiles.parquet`, `tacho.npy` |
| `evaluate --cache --out [--mode lodo\|subject\|both]` | 후보 비교, 온도 보정, 지표 | `oof_*.parquet`, `report_*.md/json` |
| `tau --oof [--exclude]` | LODO 예측에서 τ_hi/τ_lo 선택 | `tau.json` |
| `fit --cache --oof --out` | 선택된 구조를 전체 데이터로 재학습, T 는 LODO 예측으로 | `teacher_bundle/` |
| `label --bundle --data [--out]` | 대상 데이터에 P(졸음)·라벨 | 세션마다 `teacher_p.csv`, `labels_window.csv`, `teacher_label.json` |
| `demo` | 합성 데이터로 전 과정 | `outputs/teacher_demo/` |

## 2. 결정 사항

| 항목 | 값 | 이유 |
| --- | --- | --- |
| 창 | 30초 / 10초 이동, 학생과 같은 격자 | 교사 라벨과 학생 창이 `t_start` 로 1:1 |
| 전처리 | 학생과 동일 (`prepare_channel`), `mains_hz` 있으면 노치만 변경 | 교사가 보는 신호 = 학생이 보는 신호 |
| 특징 | 30초·2분·5분 스케일의 시간영역·푸앵카레, 2분·5분의 LF/HF·DFA α1·표본 엔트로피, 앞뒤 5분 HR·RMSSD 기울기 (원 특징 38개 + 개인 정규화 55개) | 기기 불변인 RR 기반, 거친 라벨에 맞는 수 분 문맥 |
| 입력에서 뺀 것 | 경과 시간·시각·창 순번, 데이터셋·유발 요인·센서, 유효 RR 비율 같은 품질 지표 | 지름길 방지 |
| 개인 Baseline | 기준 구간 5분 (60초 이상이면 있는 만큼), z = (x−m)/max(σ_s, σ_pop), 양수 특징은 ln(x/m) 도 | 긴 스케일 특징의 기준값 확보, σ_s 계산 가능 |
| 기준 없음 대비 | 학습 창 20% 를 일부러 기준 무효로 | 기준 구간이 없는 대상 데이터 |
| 가중치 | 데이터셋 합 동일 → 참가자 동일 → 정상·졸음 합 동일 | 라벨이 촘촘한 데이터셋이 지배하지 않게 |
| 확률 보정 | 온도 스케일링 (T 는 0.25–4 로 제한) | 순위(AUROC)는 그대로, τ 가 의미를 갖게 |

## 3. 후보와 선택 규칙

| 후보 | 내용 |
| --- | --- |
| `rule` | 규칙 기준선: −z(HR 2분) + z(RMSSD 2분). 학습은 확률 변환 2개 값뿐. 교사가 넘어야 할 최소 기준 |
| `lgbm` | A. LightGBM (num_leaves 15, depth 4, 강한 정규화, early stopping) — 원 특징 + 개인 정규화 |
| `lgbm_nopers` | A 에서 개인 정규화를 뺀 비교안 |
| `rrnet` | B. 5분 RR 타코그램(4 Hz, 1200점 × 3채널) 1D CNN + BiGRU, 약 2.5만 파라미터 |
| `ensemble` | C. 보정된 A·B 로짓 평균 |

선택: LODO AUROC(데이터셋 평균)가 가장 높은 후보. 차이 0.01 이하면 ECE 가 낮은 쪽, 그래도 비슷하면 `lgbm → lgbm_nopers → rrnet → ensemble` 순으로 단순한 쪽. 규칙 기준선은 선택 대상이 아니라 비교 기준입니다.

평가: LODO 4 fold. 각 fold 에서 데이터셋마다 참가자 20% 를 검증용으로 떼어 early stopping 과 T 에 씁니다. 지표는 데이터셋별 AUROC·균형 정확도(P ≥ 0.5)·ECE·Brier, 유발 요인별 AUROC, LightGBM 상위 특징입니다. `--mode subject` 는 데이터셋을 섞은 참가자 단위 5-fold 보조 평가입니다.

## 4. τ

`tau` 는 LODO 예측(처음 보는 데이터셋에 대한 예측, 실제 라벨)으로 고릅니다. τ_hi 는 P ≥ τ_hi 인 창의 졸음 비율이 **모든 데이터셋에서** 0.85 이상인 가장 낮은 값, τ_lo 는 τ_hi 아래에서 정상 비율이 0.90 이상인 가장 높은 값입니다. 목표를 못 채우면 `achieved: false` 와 이유를 남기며, 이때는 표를 보고 사람이 정합니다. 라벨이 거친 데이터셋(예: 이벤트 기반 DD-Database)은 `--exclude` 로 τ 계산에서 뺄 수 있습니다.

## 5. 출력 (대상 세션 폴더 또는 `--out`)

| 파일 | 열 |
| --- | --- |
| `teacher_p.csv` | `t_start, t_end, p_drowsy, valid, teacher_version` |
| `labels_window.csv` | `t_start, label` (1 졸음, 0 정상, -1 미라벨·무효 창) |
| `teacher_label.json` | `label_source, teacher_version, tau_hi, tau_lo, window_sec, stride_sec` |

교사는 기준 구간(5분)과 겹치는 창에는 라벨을 내지 않습니다. 학생은 기준 구간이 30초라 30초–5분 구간의 창은 라벨이 없습니다(-1).

## 6. 합성 데이터 동작 확인 (성능 근거 아님)

`python scripts/teacher.py demo` (4개 합성 데이터셋 × 6명 × 30분, 4코어 약 4분). 합성 신호는 "졸음이면 HR↓·HF↑"를 그대로 심은 장난감이라 규칙 기준선이 거의 완벽하게 맞습니다. 아래 수치는 코드가 끝까지 돌고 지표가 계산된다는 것만 보여 줍니다.

| 후보 | AUROC (데이터셋 평균) | 균형 정확도 | ECE |
| --- | --- | --- | --- |
| rule | 0.999 | 0.932 | 0.156 |
| lgbm | 0.985 | 0.946 | 0.105 |
| lgbm_nopers | 0.832 | 0.694 | 0.261 |
| rrnet | 0.996 | 0.948 | 0.064 |
| ensemble | 0.990 | 0.948 | 0.066 |

합성에서도 보이는 점: 개인 정규화를 빼면 AUROC 가 0.15 떨어집니다(사람마다 평소 심박을 크게 다르게 만든 탓). 이벤트 기반 라벨 데이터셋은 "정상" 창에 실제로는 졸린 창이 섞여 졸음 정밀도 0.85 를 만족하는 τ 가 없었고, `--exclude` 로 빼야 τ 가 정해졌습니다. 실데이터의 DD-Database 에서도 같은 일이 생길 수 있습니다.

## 7. 어댑터 (`src/driver_state/adapters/`, 실행 `scripts/convert.py`)

```text
python scripts/convert.py advitam --src <Exp4 압축 해제 폴더> --out <규약 폴더 루트> [--kss-label-sec 300]
python scripts/convert.py mpddf   --src <MPD-DF 폴더> --out <규약 폴더 루트>
python scripts/convert.py dddb    --src <DD-Database 폴더> --out <규약 폴더 루트>
python scripts/convert.py drozy   --src <DROZY 폴더> --out <규약 폴더 루트>
python scripts/convert.py maus    --src <MAUS 폴더>  --out <규약 폴더 루트>
python scripts/convert.py mand    --src <manD extracted> --out <규약 폴더 루트> [--participants P13 …] [--dehold]
```

AdVitam 은 `Exp4.zip` 에서 `Raw/Physio/Txt` 와 `Preprocessed/Questionnaire` 만 풀면 됩니다 (11 GB).

| 데이터 | 용도 | 세션 | 라벨 (창 수: 졸음 / 정상) | 실데이터 확인 (2026-11-09) |
| --- | --- | --- | --- | --- |
| AdVitam Exp4 | 교사 학습 | 63명, baseline(is_rest) + block1·2 (189개, ECG 1000 Hz) | 주행 끝 5분에 KSS_B_k (1160 / 1526) | 유효 창 96%. 53_AC16 은 시간 열이 hrs 단위 |
| MPD-DF | 교사 학습 | 50명, drive (2시간, ECG 1024 Hz) | 의사가 EEG 로 30초마다 매긴 단계의 변화점을 다음 변화점까지. 0 정상, 2–4 졸음, 1 제외 (1978 / 25 905) | 유효 창 94%. 졸음 창이 있는 참가자 40명. 3–4(수면)만 졸음으로 보면 4명뿐이라 논문이 drowsy 로 부르는 2 를 포함 |
| DD-Database | 교사 학습 | 10명 × 2회 (20개, ECG 128 Hz) | 버튼 ±60초 졸음, 10분 이상 떨어지면 정상 (4651 / 3078) | 유효 창 99%. 시작·끝 30초 안의 Event 는 표시로 보고 뺌 |
| DROZY | 교사 학습 | 14명, test1–3 (36개, 각 10분, ECG 512 Hz) | 테스트별 KSS 1개를 세션 전체에 (≥7 졸음 15세션 분량 870창, ≤5 정상 568창) | 유효 창 70%. D01·D04·D12 는 T파가 커서 R-peak 검출이 흔들림 (유효 0–18%) |
| MAUS | 라벨링 대상 | 22명, rest(기준) + trial1–6 (153개, ECG·PPG 256 Hz) | 없음 (n-back, NASA-TLX 는 labels_orig) | 유효 창 98%. 003 은 휴식 녹화가 비어 있음 |
| manD | 라벨링 대상 | 31명, 시나리오 6종 (183개) | 없음 (감정 시나리오는 labels_orig) | **유효 창 0.1%**. 256 Hz 표본 유지 신호로 실효 주기가 ECG 28–41 Hz, BVP 약 6 Hz. `--dehold` 보간 후에도 0.5% |

교사 4종 합계: 창 71 844개, 유효 68 235개, 라벨 37 951개 (`features` 344초, i5-10400).

## 8. 남은 일

- 실데이터 LODO, τ, 대상 데이터(MAUS 등) 라벨링 — 노트북에서 실행.
- 학생 로더가 `labels_window.csv` 를 우선 읽도록 한 줄 수정 (학생 쪽 작업).
