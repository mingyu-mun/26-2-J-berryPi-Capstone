# 중간 산출물 (2026-11-13)

교사 → 학생 → 상태 머신까지 한 번 이어 붙인 상태를 중간 산출물로 확정한다.

## 구성

| 단계 | 결정 | 근거 |
| --- | --- | --- |
| 교사 | **F2C**: RR z 36 + NormWear z 32 → ‖W z‖ (W 8×68), 스트레스 음성 포함 | `docs/teacher-normwear-stage1.md`, `docs/results/2026-11-12-normwear` (M1 0.641, S1 0.410) |
| 라벨링 | 교사 4종 LODO oof, CLAS·MAUS 최종 교사, WESAD 묶음 교차 | `docs/student-f2c-stage1.md` 1장, `docs/results/2026-11-12-f2c-label` |
| 학생 | **S1-EP** (PPG+ECG): 1D CNN 인코더(약 5만 파라미터) + ECG RR z, 교사 W 물려받음, 특징 증류 | `docs/results/2026-11-12-student` (M1 0.668, S1 0.149, 교사 일치 ρ 0.68~0.83) |
| 상태 머신 | **v2**: 기준 수집 품질 조건, 창 신뢰도, 3창 중앙값, 잡음 동결(30초, 5분당 60초), PPG 대체 판정, 주의 진입 40초 | `docs/state-machine-v2.md`, `docs/results/2026-11-13-sm-v2` |

## 미채택 실험 (기록)

- 쌍 학습 (teacher-pair), RR 인코더 거리 (R1~R3), NormWear 단독 (F2N), 점수만 증류 (S2-E)
- RR 품질 가중 (부록 A): 잡음 창이 졸음과 함께 나타나 판정력 손실 (`docs/results/2026-11-12-rrgate`)

## 실측 확인

- 1차 (정상, ECG 이상 간격 15%): 기준 수집 실패 → 재측정 안내.
- 2차 (졸린 상태, 이상 간격 4%): 학생 경보 24%, 상태 머신 v2 주의 18% (2회), 졸음 확정 없음.

## 남은 과제

- 상태 머신 v2: 공개 정상 오경보 1.38/h (목표 1), WESAD 3.5/h. 대화·흥분 상황 직접 측정 필요.
- CAN 교차 확인 규칙 규격
- PPG 대체 경로의 공개 데이터 검증 (PRV 기반 LODO)
- ONNX/int8 내보내기, 라즈베리파이 추론 시간 측정

## 실행 위치

- 학습·추론: 연구실 GPU 서버 `~/jberry` (repo, data, out)
- 스크립트: `scripts/normwear_embed.py`, `teacher.py anomaly-eval / f2c-label`, `student_cache.py`, `student_train.py`, `selfmeasure_infer.py`, `sm_v2_eval.py`
