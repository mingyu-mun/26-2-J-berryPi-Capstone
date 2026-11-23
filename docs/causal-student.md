# 인과(실시간) 학생 재학습 (2026-11-23)

## 1. 문제

기기에서 실시간으로 추론할 때는 창 [t, t+30] 의 판정을 t+30 시점에 내야 하므로 그 뒤 표본을 볼 수 없다.
기존 학생(S1-EP, S1-E, `models/`)의 입력은 다음 곳에서 미래를 본다.

| 위치 | 기존 | 미래를 보는 양 |
| --- | --- | --- |
| RR 특징 120초 스케일 (`teacher/features.py` `session_windows`) | 창 중심 c=t+15 앞뒤 60초 | 45초 |
| RR 특징 300초 스케일 | 창 중심 앞뒤 150초 | 135초 |
| 추세 특징 (`trend_values`, 학생 입력 아님) | 중심 앞뒤 300초 | 285초 |
| 타코그램 (`tachogram`, rrnet 교사 전용) | 중심 앞뒤 150초 (+3초) | 135초 |
| 필터 (노치, anti-aliasing, 대역통과, R-peak 검출·정밀화, 학생 파형) | `filtfilt` / `sosfiltfilt` (영위상) | 필터 임펄스 응답 길이 (0.5 Hz 고역에서 수 초) |
| 학생 파형 64 Hz 리샘플 (`student_cache.prep`) | `resample_poly` (대칭 FIR) | FIR 반 길이 (약 0.15초) |

30초 스케일은 원래 [t, t+30] 이라 문제없다. 11-20 점검(재학습 없이 입력만 인과로)에서 rec4 학생 창 AUROC 가
F10 0.726 → 0.516, A10 0.785 → 0.629 로 떨어져, 기존 학생이 미래 문맥에 기대고 있음을 확인했다.

## 2. 코드 변경 (브랜치 `claude/causal-student`)

- `TeacherSpec.causal` (기본 False). True 면
  - 스케일 L 구간 = [e − L, e], e = t + 30 (창 끝). 추세 = [e − 600, e], 타코그램 = [e − 300, e] (창 끝 뒤 박동 안 씀).
  - `extract_rr` 안의 모든 필터를 인과로 (`driver_state/filtering.py` 스위치: `sosfilt`·`lfilter`).
- `driver_state/filtering.py`: `sos_filter`, `ba_filter`, `causal_filters(on)` 컨텍스트. `preprocessing.py`, `features.py`
  (R-peak·맥파 검출), `teacher/rr.py` 가 이 스위치를 쓴다. 기본은 기존과 같은 영위상.
- `student_cache.prep(..., causal)`: 인과 대역통과 + 인과 다상 FIR 리샘플 (`causal_resample`). `--causal` 옵션.
- `infer_bundle.py`, `selfmeasure_eval.py`: 번들 `spec.causal` 을 따라 자동으로 인과 입력을 만든다.
- `teacher.py --causal rr|features`: 인과 RR·창 특징 캐시.
- `causal_relabel.py`: 기존 F2C 라벨 표에 인과 RR z (`zc_*`), `dev_core_c`, `logit_c` 를 붙인다.
- `student_train.py --causal-input [--causal-target]`, `distill.Data`: 학생 RR 입력 = `zc_*`.
- `causal_compare.py`: 공개 4종 LODO 비교표. `make_causal_bundle.py`: `models_causal/` 번들 (기존 `models/` 유지).
- `run_causal_server.sh`: 서버 전체 실행.

기본값(causal=False)에서 rec4 `selfmeasure_eval` 결과가 기준 출력(F5 0.629 / F10 0.731 / A5 0.715 / A10 0.788)과 같음을 확인했다.

### 남는 비인과 요소 (작음, 이번 범위 밖)

- R-peak 검출 높이 임계 (`detect_ecg_peaks`: 세션 전체 99 백분위), PPG 돌출도 (세션 전체 표준편차), PPG 맥박 주기 (세션 전체 중앙값).
  세션 전체 통계라 진폭 수준 정도만 새어 들어온다. 기기에서는 이동 백분위로 바꿔야 한다.
- 100 Hz 공통 리샘플 `CubicSpline` 은 전역 보간이지만 영향이 이웃 몇 표본 안에서 지수적으로 줄어든다.

## 3. 재학습 계획

교사는 다시 학습하지 않는다. 교사는 오프라인 라벨러라 미래 문맥을 써도 되고, 학생이 "지금까지의 신호로 오프라인 교사의 판정을
맞추도록" 배우는 것이 실시간 목표와 맞다. 바뀌는 것은 학생 입력(파형 필터, RR z)뿐이다.

| 조건 | 학생 RR 입력 | 파형 | 목표 | 용도 |
| --- | --- | --- | --- | --- |
| 기존 (old) | 중심 정렬 zr | filtfilt | 오프라인 교사 logit | 기준 (이미 있음, `out/student`) |
| 본안 causal | 인과 zc | 인과 | 오프라인 교사 logit | 채택 후보 |
| 비교 causal_ct | 인과 zc | 인과 | 인과 z 로 다시 매긴 교사 logit_c | 목표 선택 확인 |
| 규칙 rule_old / rule_causal | dev_core (중심 / 인과) | - | - | 학습 없는 기준 |

학습 설정은 기존과 같다 (S1-EP·S1-E, 4 fold LODO, 시드 3, 3000 step). 데이터: 교사 4종 + 학생 3종 (CLAS·MAUS·WESAD).

실행 장소: 학습 1회가 이 PC CPU 에서 약 66분 (1.33 초/step, 실측), 서버 GPU 에서 약 2분이다. LODO 2조건 + 비교안 + 최종이
약 30회라 CPU 로는 하루 넘게 걸리고, 교사 라벨 표(`f2c_labels.parquet`, NormWear 임베딩 포함)도 서버에만 있다. 그래서 서버에서 돌린다
(`run_causal_server.sh`, 예상 RR 캐시 약 20분 + 파형 캐시 약 30분 + 학습 약 1.5시간).

## 4. 평가

- 공개 4종 LODO: M1 (시간 층 안 참가자 가중 AUROC, 데이터셋 평균), 참가자 단위 부트스트랩 95% 구간 (`causal_compare.py`).
  같은 창 집합에서 old / causal / causal_ct / rule_old / rule_causal 비교. WESAD 스트레스 오경보 S1 은 `report_student.md`.
- rec4 (11-18 실측): `selfmeasure_eval.py --models models_causal` 로 F5 / F10 / A5 / A10 창 AUROC, 규칙 dev_core 는 같은 실행의 `win_auc_rule`.
- 판정 기준 (제안): 인과 학생의 공개 M1 이 기존 학생보다 0.03 넘게 낮지 않고, 인과 규칙보다 높으면 기기용으로 채택.
  rec4 는 한 사람 한 녹음이라 참고 지표로만 쓴다.

## 5. 결과

(실행 후 기록)
