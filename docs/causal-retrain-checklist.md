# 실시간(인과) 재학습 점검표

결정 (2026-11-26): 실시간 판정은 GPU 서버를 쓸 수 있을 때 재학습한다. 발표 시연의 실시간 부분은 "동작 시연"으로 두고, 성능은 오프라인 결과로 말한다.
이 표의 모든 항목을 고치고 **4절 관문을 통과하기 전에는 재학습을 시작하지 않는다.**

## 1. 왜 다시 하나 (11-18 rec4 근거)

- 이전 "인과" 학생(A)은 필터·RR 창만 과거 전용으로 바꾸고, R-peak 임계(세션 전체 통계)와 이상 박동 판정(뒤 박동 5개)은 미래를 본 채 학습·평가했다.
- 실시간 경로(웹보드 재생 = Pi와 같은 경로)로 rec4를 돌리면 기존 임계로는 기준 계산부터 실패했다 (유효 창 5개).
- rolling 임계로 박동을 되찾아도 창별 점수는 오프라인과 상관 0.07, 창 AUROC 0.67 → 0.54, 경보 일치 48%, 단계 일치 56%.

## 2. 미래를 보는 지점 전부 (코드 위치)

| # | 지점 | 위치 | 고칠 방향 |
|---|---|---|---|
| 1 | 노치·anti-aliasing·대역통과 `filtfilt`/`sosfiltfilt` | `preprocessing.prepare_channel`, `_band` | 단방향 `sosfilt`(지연 보정) 또는 창 끝 여유 구간을 버림 |
| 2 | R-peak 높이 임계 = 신호 전체 99 백분위 | `features.detect_ecg_peaks` | 과거 블록만 (`threshold="rolling"`, PR #17) |
| 3 | R-peak 정밀 위치용 QRS 대역 `sosfiltfilt`, ±40 ms 탐색 | `teacher/rr.extract_rr`, `refine_peaks` | 단방향 필터, 창 끝 40 ms 이내 박동은 다음 창으로 미룸 |
| 4 | 이상 박동 판정 = 앞뒤 5박동 중앙값 | `teacher/rr.flag_abnormal` (`median_filter` 중심) | 과거 11박동 중앙값 |
| 5 | PPG 맥박 주기 = 녹음 전체 스펙트럼 중앙값, prominence = 전체 std | `features.detect_ppg_peaks_adaptive` | 과거 구간만으로 주기·prominence 갱신 |
| 6 | PPG 정밀 위치 ±80 ms, 0.5–8 Hz `sosfiltfilt` | `teacher/rr.extract_rr` (ppg) | 3번과 같음 |
| 7 | 30·120·300초 RR 특징을 창 중심에 맞춤 | `teacher/features` (오프라인), `streaming_model.trailing_features` (실시간) | 학습도 창 끝까지만 (trailing) |
| 8 | 파형 입력 대역통과 `sosfiltfilt` + `resample_poly` | `streaming_model.wave_windows`, 학생 캐시 생성 | 단방향 필터, 학습 캐시도 같은 함수로 다시 생성 |
| 9 | 기준 구간 선택이 녹음 전체 품질을 봄 | `infer_bundle` 자동 기준, `select_baseline` 호출부 | 기준은 이미 지난 구간(실시간처럼 360초 시점)으로만 선택 |
| 10 | cubic 리샘플이 이웃 표본 사용 | `preprocessing.resample` | 창 끝 1~2표본 영향, 관문 시험에서 허용 오차로 확인 |

학습 데이터 캐시(`data/teacher_cache`, `student_wave`)는 위 함수로 **모두 다시 만든다.** 예전 캐시를 재사용하지 않는다.

## 3. 학습 전 확인

- 학습·평가·실시간이 **같은 특징 함수**를 부르는지 확인한다 (오프라인 전용 경로를 따로 두지 않는다).
- 서버 사용은 매번 먼저 허락받고, GPU 사용 여부를 확인하고, 결과는 새 폴더에만 쓴다.

## 4. 관문 (통과해야 재학습)

1. **자르기 시험:** 실측(11-18 rec4, 11-19)과 공개 데이터 각 1개 이상, ECG·PPG 모두.
   모든 창 끝 t에 대해 `신호[:t]`로 계산한 RR 특징·z 입력·유효 판정·파형 입력이 전체 녹음으로 계산한 값과 같아야 한다 (cubic 경계 등 허용 오차를 미리 정함).
2. **실시간 경로 재생:** `scripts/rpeak_stream_check.py`로 박동 재현율 확인, 웹보드 재생(LocalReplay)으로 기준 계산이 성공하는지 확인.
3. **학습 후:** 평가를 `infer_bundle`이 아니라 실시간 경로(웹보드 재생 또는 같은 함수)로도 내서, 오프라인 평가와 창별 점수 상관·경보 일치를 함께 보고한다.
