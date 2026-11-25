# Tools/jberry_ai — 시뮬레이터용 J_Berry PI 실시간 추론 서버

저장소 [`mingyu-mun/26-2-J-berryPi-Capstone`](https://github.com/mingyu-mun/26-2-J-berryPi-Capstone)의 **S1-EP 학생 모델(seed 3개 평균)과 상태 머신 v2**를 이 PC의 CPU에서 돌린다.
Unity는 센서 역할을 한다. 실측 ECG·PPG 원신호를 기록된 원래 주기로 흘려보내고, 차량 신호를 함께 보낸다. 서버는 판정 결과를 화면으로 돌려준다.

```
Unity BioReplay (실측 기록, ECG 435 Hz · PPG 101 Hz 원신호)
   │  AiLink  UDP 5005: hello / bio / can
   ▼
server.py ── engine.StreamEngine ── repo/ (저장소 코드·가중치, 수정 없음)
   │  UDP 5006: ui (10 Hz: 상태 머신 출력, 기준 수집 상태, 전처리 파형)
   ├─ SimLogs/ai/events_<세션>.jsonl   웹보드 규약 v1.2 이벤트 4종 (bio.wave · ai.decision · can.telemetry · system.status)
   ├─ SimLogs/ai/windows_<세션>.csv    창마다 WindowIn · StepOut · logit
   └─ (선택) --webboard http://127.0.0.1:8000 → /api/events (WebboardPublisher)
```

## 설치와 실행

1. Unity 메뉴 **Capstone > AI 추론 환경 설치 (Python venv)** 를 실행한다.
   - Python 3.10~3.13을 찾아 `.venv`를 만든다. 이 PC에서는 3.14가 기본이라 `py -3.13`을 사용했다.
   - `requirements.txt`와 torch(CPU)를 설치한다.
   - 마지막으로 `selftest.py`를 돌린다.
   - 결과는 `SimLogs/ai/setup_status.txt`, 전체 출력은 `setup_log.txt`에 남는다.
2. Play를 누르면 `AiLink`가 `.venv\Scripts\python.exe server.py`를 자동으로 띄우고, Play를 끝내면 서버도 끈다.
   - 서버만 따로 띄우려면 `run_server.bat`을 실행한다.

## DrivingSim(SimBootstrap) Inspector 설정

| 필드 | 기본값 | 의미 |
|---|---|---|
| `bioSessionDir` | (비움 → `SimLogs/bio/rec_20261118`) | `docs/data-handoff.md` 형식 세션 폴더: `meta.json` + `ecg.csv` + `ppg.csv` (`value` 열), 선택으로 `labels.csv` |
| `bioStartOffsetSec` | 0 | 기록의 몇 초부터 재생할지 |
| `aiPrerollBeforeStart` | true | 시작 위치 앞의 기록을 서버에 미리 빠르게 넣는다. 기준 수집(약 7~8분)을 건너뛰고 바로 판정 화면을 볼 수 있다. 실제 측정값만 넣으며, 화면에 "앞 N s 빠른 입력"으로 표시된다. |
| `aiMode` | `causal` | 아래 "두 가지 모드" 참고 |
| `aiWebboardUrl` | (비움) | 팀 웹보드 대시보드로도 이벤트를 보낼 때 입력 |

## 두 가지 모드 (중요)

저장소의 교사 RR 특징 36개는 창의 **중심** $c = t + 15$ s를 기준으로 계산한다. 스케일 $L \in \{30, 120, 300\}$ s마다 구간 $[c - L/2,\ c + L/2)$을 쓴다 (`teacher/features.py: session_windows`).

- $t$: 30초 창의 시작 시각
- $c$: 창의 중심 시각
- $L$: 특징을 계산하는 구간 길이(스케일)

따라서 300초 스케일은 창이 끝난 뒤 $c + 150 - (t + 30) = 135$초의 **미래** 심박까지 쓴다. 오프라인 batch(`scripts/infer_bundle.py`)에서는 문제가 없지만, 실시간에는 이 데이터가 아직 존재하지 않는다.

| 모드 | 구간 | 판정 출력 시점 | batch와의 일치 (rec_20261118, 같은 기준 구간) |
|---|---|---|---|
| `exact` | 학습과 같은 중심 구간 | 창 끝 + 140 s (미래 135 s + 필터 가장자리 5 s) | 상태 일치 94~95 %, logit 차이 중앙값 0.14~0.16 |
| `causal` (기본) | $[t_{end} - L,\ t_{end})$, 과거만 사용 | 창 끝 + 5 s | 상태 일치 38 %, logit 상관 0.19 → **학습 때와 입력 분포가 다름** |

rec_20261118 한 세션에서 자기보고 라벨(10~25분 졸음)과 비교한 결과는 아래와 같다. 기준 구간 이후 창만 비교했다.

| 경로 | 창 AUROC | 5분 블록 AUROC | 주의 이상 비율 | 졸음 확정 구간 (원본 시각) | 졸음 블록에서 '졸음' 비율 / 정상 블록에서 '졸음' 비율 |
|---|---|---|---|---|---|
| batch (`infer_bundle.py`) | 0.59 | 0.78 | 0.91 | 1005 ~ 2100 s | 0.23 / 0.64 |
| 스트리밍 exact | 0.64 | 0.89 | 0.87 | 1035 ~ 1950 s | 0.20 / 0.53 |
| 스트리밍 causal | 0.58 | 0.67 | 0.77 | 1065 ~ 1500 s | 0.48 / 0.00 |

**한 명 · 한 세션 결과라 일반화할 수 없다.**
- causal 모드는 학습 분포와 달라 창 단위 분리는 약해졌다.
- 대신 정상 블록에서 졸음이 남아 있는 현상(batch 0.64)이 사라졌다.
- 실시간 시연용 모델이라면 **causal 특징으로 다시 증류(distill)하는 것**을 팀에서 검토할 만하다.

## 저장소 코드와 다른 점 (실시간이라 불가피한 것만)

1. **R-peak / 맥파 검출 임계값.**
   - 저장소의 `detect_ecg_peaks`는 높이 임계값을 *세션 전체* 에너지 $e = (\text{5–15 Hz 대역통과 ECG})^2$의 99 백분위 × 0.3으로 잡는다.
   - 실시간에는 앞부분만 있으므로, 움직임 잡음이 몇 초만 있어도 임계값이 튀어 박동을 대량으로 놓쳤다 (rec 첫 460초: 618 → 247 박동).
   - 그래서 **조각마다 같은 값을 구해 그 중앙값**으로 바꿨다. 조각 길이는 버퍼 길이 ÷ 6을 10~60초로 제한한다 (실측 시작 직후 짧은 버퍼 대응, 2026-11-21). PPG의 `0.3·std` 기준도 같은 방식으로 바꿨다.
   - 전체 기록에서 확인한 결과, batch 박동 2726개가 모두 ±20 ms 안에서 다시 검출되고 12개만 추가된다. PPG는 2652개가 그대로 같다.
2. **기준 수집.** `select_baseline`(60초 대기, 깨끗한 창 20개, 최대 600초)을 받은 데이터에 5초마다 적용한다. 600초 안에 실패하면 `retry_required`를 표시한다. 기록 재생은 재측정을 할 수 없으므로 `infer_bundle --force`와 같이 기본 구간으로 진행하고, 화면에 "(강제)"로 표시한다.
3. **판정 대상 창.** 기준 구간과 겹치지 않는 첫 창부터 10초마다 판정한다. 버퍼 끝 5초는 필터 가장자리라 쓰지 않는다.

그 밖의 계산(전처리, RR 추출, 개인 정규화 z, 학생 파형 입력, logit, 시그모이드, `StateMachineV2.step`)은 저장소 함수를 그대로 호출한다.

## 규약(WEBBOARD_DATA_CONTRACT v1.2) 준수

- 생체 출처는 `recorded`(기록 재생), CAN 출처는 `simulated`(시뮬레이터 합성)이다. 화면에도 배지로 계속 표시한다.
- `can_drowsy`는 저장소 규칙 `vehicle_warning_sim_v1`(`src/driver_state/can_assist.py`, PR #16)로 만든다 (2026-11-26). 판정 창 10초 안에 Unity 차선 이탈 경고(`LaneSafe_indi=5`, 방향지시등이 켜져 있거나 꺼진 뒤 3초 이내는 제외) 또는 부주의 경고(`Careless_Warn≥2`)가 있으면 `true`, 없으면 `false`. Unity가 경고 상태 코드를 보내지 않으면(구버전) 예전처럼 `null`이다.
  - 상태 머신 v2(main)는 생체 판정이 **주의** 이상일 때 `can_drowsy=true`면 졸음으로 격상한다. 정상 단계에서는 차량 경고만으로 경고하지 않는다.
  - 졸음에 처음 오른 창의 `can_drowsy`가 `true`면 `confirmation_source="can"`, 아니면 `persistence`.
  - 받은 CAN은 `SimLogs/ai/can_<세션>.jsonl`에 저장소 `--can-sim-scenario` 형식으로도 남긴다.
  - **필요한 저장소 파일:** `repo/`에 main의 `src/driver_state/state_machine_v2.py`(주의 단계 CAN 격상), `src/driver_state/can_assist.py`, `git_upload_ver1/steering_profile.py`가 있어야 한다. 없으면 서버가 시작할 때 경고를 찍고 `can_drowsy=null`로 동작한다.
- 화면의 상태는 `StateMachineV2.step` 출력(level, candidate, frozen …)만 쓰며, UI가 따로 판정하지 않는다.
- CAN 값은 `Can list.xlsx`의 표현 범위로 양자화한다.
  - speed: 0~255 정수
  - steering: 0.1° 단위
  - accel_pedal_raw: 0~254
  - brake_pedal_raw: 0~127
  - GearPosition: 0=P, 1=R, 2=N, 3=D
- 환산식이 확정되지 않은 `*_pct`와 `road_wheel_angle_deg`는 `null`이다.
- 원신호와 운전자 데이터는 저장소에 올리지 않는다. `SimLogs/`는 이 PC에만 둔다.

## 성능 (이 PC)

- 모델 로드 약 6 s
- 창 하나 판정 약 70~100 ms
- 10초마다 RR 전체를 다시 추출해도 0.3 s 미만

## 파일

- `engine.py`: 스트리밍 엔진. 표본을 쌓다가 10초마다 창을 판정한다.
- `server.py`: UDP 서버. 규약 이벤트를 기록하고 화면 패킷을 보낸다.
- `setup_env.py`, `selftest.py`, `requirements.txt`, `run_server.bat`
- `repo/`: 저장소에서 추론에 필요한 파일만 복사한 것 (`SOURCE.txt`에 커밋 기록)
  - 가중치 사용 조건은 `repo/models/README.md` 참고 (WESAD 학습분은 비상업 용도)
