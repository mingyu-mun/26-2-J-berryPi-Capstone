# J_Berry PI · 웹보드 통합 데이터 규약 v1.2 (팀 제안안)

- 작성 기준: 2026-11-15, GitHub `mingyu-mun/26-2-J-berryPi-Capstone` `main`의 확인된 코드 및 팀 카카오톡 논의.
- 상태: **UI 제작을 위한 제안 규약**. 현재 GitHub에 구현된 공식 WebSocket API가 아님. 실제 구현 완료나 성능 인증을 의미하지 않음.
- 목적: ECG·PPG 파형 / 생체 특징·AI 출력 / CAN 속력·조향각·주행 판단을 하나의 시간축에서 표시.
- 금지: 원본에 없는 생체값, AI 확률, 차량 속력·조향각, 졸음 라벨을 그럴듯하게 생성하지 않는다. 누락·미수신은 `null`, 사용 불가능한 파형 채널은 `null`. 합성 CAN은 출처 `simulated` 표시.

## 1. 기존 구현 확인과 신규 제안의 경계

| 항목 | 사실 또는 설계 상태 | 출처 |
|---|---|---|
| 학생 모델 | `S1-EP` (교사 F2C를 증류). 30초 ECG·PPG 파형, 학생 입력 64 Hz×1920 표본/채널, 파형은 창별 정규화 | `scripts/student_cache.py`, `docs/MILESTONE-2026-11-13.md` |
| 기존 공통 전처리 | `prepare_channel`은 공통 `spec.fs=100 Hz`, ECG 0.5–40 Hz / PPG 0.5–8 Hz. `prefiltered` 입력 예외 | `src/driver_state/preprocessing.py`, `src/driver_state/spec.py` |
| 학생 전처리 | `student_cache.prep`은 64 Hz로 리샘플, ECG 0.5–25 Hz / PPG 0.5–8 Hz; `cut`에서 30초 창별 z-score | `scripts/student_cache.py` |
| 학생 결과 파일 | `scripts/selfmeasure_infer.py`의 `windows_scores.csv` 열 `student_S1-EP_p`, `student_S1-EP_logit`, `student_S1-EP_prv_p`(가중치 존재 시). `t_rel`은 창 시작 | `scripts/selfmeasure_infer.py` |
| 상태 머신 원본 | `WindowIn` 10개 입력, `StepOut` 9개 출력; 정상/주의/졸음/측정 불가 및 후보 | `src/driver_state/state_machine_v2.py` |
| CAN | 기존 분류 모델 입력에서는 제외. 독립 개인 주행 기준선·주행 변화 표시를 시뮬레이션으로 준비; 최신 상태 머신은 별도 `can_drowsy` 교차 확인 슬롯 있음 | 카톡 11/8~11/12, `docs/state-machine-v2.md` |
| CAN 신호 정의 | `Can list.xlsx`의 `CAN DB!B2:P27` 25개 신호를 확인. 표의 정의를 규약에 전사했지만 실차 프레임으로 검증된 것은 아님 | 사용자 제공 `Can list.xlsx`, `CAN_MAPPING_FROM_XLSX.md` |
| 아래 JSON 전송 방식·웹보드 필드 | **이번 문서에서 새로 정한 연결 계층**. 원본 소스 코드에 그대로 존재한다고 주장하지 않음 | 본 규약 |

### 꼭 지킬 구분

- **AI의 `p_ep`**는 모델의 졸음 점수. **최종 상태**는 `StepOut.level`/`candidate`로만 표시한다. `p_ep>0.5`만으로 '졸음'을 표시하지 않는다.
- **StateMachineV2의 `confidence`**는 신호 품질에 따른 상태 머신 가중치이지, AI의 예측 신뢰도나 의학적 정확도가 아니다.
- **CAN 패턴 이탈**은 졸음의 독립적인 증거가 아님. 주행 패턴 알림(`deviation_status`)과 졸음 교차 확인(`can_drowsy`)을 별개 필드로 구분한다.
- 이전 `InputSpec`의 30초 시연 Baseline은 구버전; 현재 `StateMachineV2`는 시작 안정화 60초 + 기준 구간 300초, 최대 600초 및 품질 조건을 정의한다.

## 2. 전체 데이터 연결과 책임

```text
[센서/데이터 서동균]  →  실제 전처리 파형, fs, 단위, HR/PR, 품질
              ├─ bio.wave ──────────────────────────────┐
              └─ S1-EP 모델 입력 ──[AI 문민규]── p_ep ─┐│
                                                     ││
[차량/CAN 양은철] → CAN 디코딩·단위 변환·출처 표기 ─┐││
                      ├─ can.telemetry ──────────────┤││
                      └─ can_drowsy (규칙 검증 후) ──┘││
                                       [Pi/통합 서버]││
                       WindowIn → StateMachineV2.step ││
                               └── ai.decision ───────┤│
                        system.status ────────────────┤│
                                  [웹보드] ◀──────────┴┘
```

- `bio.wave`: 센서/전처리 측이 공급, UI에서는 받은 파형을 그리기만 한다.
- `ai.decision`: AI + 품질 + CAN을 결합하는 백엔드가 공급; 최종 출력은 **실제 원본 Python `StepOut` 직렬화**.
- `can.telemetry`: CAN 측이 공급. 제공된 `Can list.xlsx`의 25개 신호 정의를 참고하되 실제 수신 프레임과 대조해 디코딩하며, 합성이라면 출처를 명시.
- `system.status`: 백엔드가 공급. 두 종류의 Baseline 상태와 연결상태를 분리 관리.

## 3. 공통 메시지 봉투 (모든 이벤트)

```text
{
  schema_version: "1.1",           # 규약 버전
  type: "bio.wave" | "ai.decision" | "can.telemetry" | "system.status",
  session_id: string,              # 같은 세션에서 불변
  origin: "live" | "recorded" | "simulated" | "unavailable",
  t_rel_s: number,                 # 공통 세션 시작 후 경과 초. 측정 시각 기준
  payload: { ... }                 # 메시지별 필드
}
```

- 시간이 다른 스트림을 **같은 세션 기준 `t_rel_s`** 로 정렬한다. 컴퓨터 수신 시각이 아닌 **측정/시뮬레이터 발생 시각**을 사용한다.
- 실제 데이터의 장치 시계 기준이 서로 다르면 서버에서 명시적으로 동기화 후 게시한다. 시간축 동기화가 되지 않으면 CAN 교차 확인 결과를 `null` 처리한다.
- 동일 세션 재시작은 새 `session_id`를 쓰고 상태 머신 `reset()`을 호출한다.
- 메시지 순서가 뒤바뀔 수 있으므로 프런트는 `t_rel_s` 기준으로 정렬하고 구간 중복은 중복 표시하지 않는다.
- `origin=simulated`는 **CAN 시나리오**에만 사용한다. 실제 모델 출력과 혼합할 때 실측/합성을 둘 다 화면에 명시한다.

## 4. `bio.wave`: 전처리 ECG·PPG 파형

**웹보드에 표시:** ECG 파형 선 그래프, PPG 파형 선 그래프, 각 샘플의 시간축, 신호 품질/전처리 경로.

`payload`:

| 필드 | 형식 | 규칙 |
|---|---|---|
| `ecg`, `ppg` | `WaveChannel` 또는 `null` | 수신하지 못한 채널은 `null`; 0으로 채우지 않음 |
| `WaveChannel.t_start_rel_s` | number | 해당 묶음의 첫 샘플 세션 기준 시각 |
| `WaveChannel.fs_hz` | number > 0 | **실제로 전달한 파형의** 샘플링 속도 |
| `WaveChannel.samples` | number 배열 | 시간 순서의 **전처리된 실제 표본**, 변형 없이 전달 |
| `WaveChannel.unit` | `mV`/`adc_count`/`a.u.`/`z_score` | 장치 보정으로 mV가 보장되지 않으면 임의로 mV 표기하지 않음 |
| `WaveChannel.stage` | `common_preprocessed`/`student_model_input` | 공통 전처리 100 Hz와 학생 입력 64 Hz를 명확히 구분 |
| `WaveChannel.quality_ok` | boolean/null | 해당 채널의 실제 품질검사 결과. 미검증 `null` |

- `samples[i]`의 시각은 `t_start_rel_s + i / fs_hz` (초). 두 채널의 시작 시각·샘플링 주파수가 다를 수 있다.
- **공통 전처리 100 Hz와 S1-EP 모델용 64 Hz는 다른 경로**다. 그래프용 파형에 `student_model_input`을 쓰는 경우 창별 z-score와 30초 경계를 알아야 하고, 창 간 진폭이 서로 다를 수 있다.
- UI용 파형은 권장상 `common_preprocessed` 사용. 백엔드에 그 결과가 없다면 실제 `student_model_input`을 사용하되 단계와 단위를 표시하고 다른 처리 경로를 만들어낸 척하지 않는다.
- 파형 업데이트 권장: 0.5~1초 단위 묶음 전송(신규 UI 제안). 모델 판정 10초 주기와 무관.
- R-peak 점·PPG peak 점은 제공되지 않은 경우 추측해서 그리지 않으며, 후속 버전에서 **실제 검출 인덱스가 있을 때만** 별도 확장.

## 5. `ai.decision`: AI 입력·최종 상태

**웹보드에 표시:** 최종 상태, 졸음 후보, S1-EP 점수 추이, 상태 머신 누적 점수, HR/PR 및 신호 품질, CAN 교차 확인 가능 여부.

`payload`에는 `model_id="S1-EP"`, `state_machine_id="v2"`, `window_start_s`, `window_end_s`, `baseline_status`, `bio_origin`, `can_origin`, `input`, `output`, `confirmation_source`가 있다. `bio_origin`/`can_origin`은 혼합 출처(실측 생체 + 합성 CAN)를 명확하게 구분하기 위한 신규 필수 필드다.

### 5.1 `input`: Python `WindowIn`에 연결

| 이름 | 형식 | 원본과의 관계 |
|---|---|---|
| `t` | number/null | 30초 창 **종료** 시각(초). `t_rel+30` 변환은 시작 시각이 확인된 CSV에만 적용 |
| `p_ep` | number 0~1/null | `student_S1-EP_p` |
| `q_ecg` | number 0~1/null | `q_abn` (ECG 이상 RR 비율) |
| `p_ppg` | number 0~1/null | `student_S1-EP_prv_p` (출력 있으면) |
| `q_ppg` | number 0~1/null | `q_ppg` |
| `hr_ecg` | number/null | `hr_ecg` (bpm) |
| `hr_ppg` | number/null | `hr_ppg` (bpm) |
| `ok_ecg` | boolean/null | **별도 품질검사**. CSV의 `valid`와 무조건 동일시 금지 |
| `ok_ppg` | boolean/null | 별도 품질검사. CSV `valid_ppg`와 무조건 동일시 금지 |
| `can_drowsy` | boolean/null | 동기화된 CAN 규칙 판정; 미수신/미검증은 null |

원본 `WindowIn`에는 일부 기본값이 있지만 **어댑터는 관측값 누락을 0/True로 임의 보충하지 않는다**. 유효성 검증 불가 시 `output=null` 처리.

### 5.2 `output`: 실제 Python `StepOut`의 9개 필드 그대로

| 이름 | 형식 | UI 규칙 |
|---|---|---|
| `t` | number | 창 종료 시각(s) |
| `level` | -1/0/1/2 | `-1 측정 불가`, `0 정상`, `1 주의`, `2 졸음` |
| `level_name` | string | 원본 상태명 그대로 |
| `candidate` | boolean | `level=1 && candidate=true`면 **졸음 후보** |
| `score` | number 0~1/null | 3창 중앙값과 신뢰도 반영, **원시 p와 다름** |
| `p_used` | number 0~1/null | 실제 사용한 채널 예측값 |
| `confidence` | number 0~1 | 품질 가중치이지 AI의 모델 정확도 아님 |
| `channel` | `EP`/`PPG`/`none` | 실제 사용한 채널 경로 |
| `frozen` | boolean | ECG 잡음 관련 상승 누적 동결 여부 |

- `baseline_status != ready`면 `output=null` 및 UI `기준 측정 중/재측정`으로 표시한다.
- 상태 머신이 이전 상태를 보존해도 `channel=none`이면 `최근 창 판정 불가 · 직전 상태 유지`를 추가 표기한다.
- `bio_origin`과 `can_origin`은 각각 생체 데이터와 CAN 데이터의 출처이며, `can_origin=simulated`일 때 웹보드에 CAN 기반 판단이 합성 데이터에서 나온 결과임을 지속 표시한다.
- `confirmation_source`(`"can" | "persistence" | null`)는 **이번 규약의 신규 보조 필드**다. 실제 확정 전이의 근거를 백엔드가 증명할 수 있을 때만 채우며, 단지 `level=2`라고 해서 임의로 추론하지 않는다.
- AI의 logit을 웹 UI에 노출하는 건 선택사항이다. 모델 출력 점수와 교사 F2C의 `teacher_p`를 혼동하지 않는다.

### 5.3 동작 판정 기준 (코드의 현재 기본값)

- 30초 분석 창 / 10초 간격 갱신.
- `score>0.55` 누적 40초: 정상 → 주의.
- 이후 누적 60초: 주의 → 졸음 후보(`level=1`, `candidate=true`).
- CAN의 **검증된** `can_drowsy=true` 또는 졸음 후보 추가 180초: 졸음 확정.
- `score<0.40` 상태로 60초 유지: 한 단계 하강.
- 유효 채널 없음 60초 지속: 측정 불가.
- 원본 문서 일부에는 첫 상승 지속 시간이 60초로 적혔으나, 현 `V2Config.t_up1=40.0`; **코드 기준으로 운용하며 팀 확인 필요**.

## 6. `can.telemetry`: 속력·조향각·페달·차량 상태 (**엑셀 대조 후 수정**)

이번 버전은 **업로드된 `Can list.xlsx`의 `CAN DB!B2:P27`을 직접 대조**했다. 이 문서에 있는 CAN ID·비트 위치·스케일은 **원본 표의 기재값**이지 실차 검증 완료 사실이 아니다. 25개 전체 신호는 [`CAN_MAPPING_FROM_XLSX.md`](CAN_MAPPING_FROM_XLSX.md) 및 `can_signal_catalog.json`에서 확인한다.

### 6.1 기본 화면 및 모델 교차확인 필드

| UI 필드 | 제공된 엑셀의 실제 원천 | 적용/표시 규칙 |
|---|---|---|
| `speed_kmh` | `Speed`, `0x102` CARINFO, 100ms, start 56, len 8, Unsigned, factor 1, offset 0, `km/h` | 원본 디코드값 0~255. 수신·검증 전 null |
| `steering_angle_deg` | `SAS_angleVal`, `0x125`, 10ms, start 24, len 16, Signed, factor 0.1 | **핸들 조향각°** = signed raw×0.1. **raw=32767은 오류**, 화면은 null. 좌/우 부호 정의 미기재 |
| `road_wheel_angle_deg` **신규** | 동일 SAS, 원본 설명의 **조향비 15:1** | 유도값 = 핸들각/15 (실제 바퀴각 측정 아님). 적용 타당성 불확실하면 null |
| `steering_rate_deg_s` | 실제 조향각·연속 타임스탬프 | **계산값** Δ핸들각/Δ초. 필드 갱신주기 10ms를 시간차로 단정 금지 |
| `accel_pedal_raw` **신규** | `Accel_Pedal_Pos`, `0x371`, 이벤트, start 31, len 8 | 원본 정수 0~254. Unit 공란/Factor 1 |
| `brake_pedal_raw` **신규** | `Brake_Pedal_Pos`, `0x371`, 이벤트, start 0, len 8 | 원본 정수 0~127. Unit 공란/Factor 1 |
| `accelerator_pct` | `Accel_Pedal_Pos`, 원본 설명에 `0~254 (0~100%)` | 공식 환산계수 미기재. **정규화식 검증 전 null**; raw 값은 표시 가능 |
| `brake_pct` | `Brake_Pedal_Pos`, 원본 설명에 `0~127 (0~100%)` | 공식 환산계수 미기재. **정규화식 검증 전 null** |
| `baseline_coverage_pct` | 별도 CAN 개인 Baseline 로직(엑셀 없음) | 실제 알고리즘 확보율만 0~100. 자료 없으면 null |
| `deviation_status` | 별도 주행 패턴 비교 알고리즘(엑셀 없음) | `within_baseline`/`outside_baseline`/`unavailable`. 졸음 판단 아님 |
| `can_drowsy` | 별도 검증된 규칙 필요(엑셀에 **없음**) | `true`/`false`는 실제 규칙으로 판정 시만. 그 전에는 **null** |
| `rule_id` | CAN 판정 구현 | 실제 판정 규칙의 버전. 없으면 null |
| `signal_map_id` | 수신 측이 적용한 사양 식별자 | `Can list.xlsx` 사양을 사용했다는 사실과 실차 검증 여부를 구분. 매핑 미확인 시 null |

### 6.2 원시 상태 코드(선택)

`status_codes`는 Excel의 **원래 `Signal name`을 키로 사용하는 선택 객체**이며, 해당 원본 값이 실제 수신·디코딩된 경우에만 전송한다. 기어 위치(`GearPosition`: 0=P,1=R,2=N,3=D), 차선 유지 보조 ON/OFF(`LaneKeep`), 방향지시등(`LTurnSig_Indi`/`RTurnSig_Indi`), 핸들 파지(`HOD_Dir_Status`), 부주의 경고 표시(`Careless_Warn`), ESC(`ESC_Indi`), 차선안전 표시(`LaneSafe_indi`), 차량/타이어 상태 등 **21개**를 지원한다.

- 저장·전송 값은 원본 **코드/정수**이며, UI가 원본 값 설명표를 참고해 문자열로 표시한다. 알 수 없는 코드는 정상/부주의/졸음으로 임의 매핑하지 않는다.
- `Careless_Warn`은 차량의 표시신호일 뿐 **졸음 진단 정답이 아님**. `LaneKeep`은 보조 기능 ON/OFF이지 차선 이탈량이 아님.
- `HOD_Dir_Status`: `0x208`, start bit **80**, length **3** → 정의대로라면 페이로드가 최소 **11바이트**라 **CAN FD 여부** 및 하드웨어 호환 검증 필수. raw `7`은 INVALID, `5`와 `6`은 예약값.
- 미수신인 상태 신호는 JSON에서 생략하거나 null. 이벤트 기반인 경우 마지막 값은 **측정 시각을 붙여 이전 관측값이라고 표시**하고 영구적으로 '현재 상태'처럼 표시하지 않는다.

### 6.3 시간과 품질

- `signal_times_s` **필수 신규 객체**: `speed`, `steering`, `accelerator`, `brake` 각각 세션 기준 실제 관측 시각(number/null). `can.telemetry` 봉투의 `t_rel_s` 하나만으로 10ms/100ms/이벤트 주기를 모두 표현할 수 없다.
- `status_signal_times_s` **선택 객체**: 각 상태 신호의 실제 관측 시각(number/null).
- 메시지 원본 주기: `CARINFO` **100ms**, `SAS_FD_01_10ms` **10ms**, `HOD_FD_01_100ms` **100ms**, `E_EMS11` **비주기(이벤트 기반)**. 통신 주기와 측정 주기 혼동 금지.
- **오류 코드**: SAS raw `32767` → 조향값 null(정상 3276.7°로 표시 금지). 범위 이탈·해석 미확정 상태도 값으로 만들어 채우지 않는다.
- **페달 % 변환 주의**: 엑셀에는 `Factor=1`, `Offset=0`, `Unit` 공란이고, 설명에 raw 구간을 0~100%라고 표현했다. `raw/254×100`, `raw/127×100`은 *선형이라고 가정했을 때의 제안식*일 뿐, 엑셀의 공식 DBC 정의는 아니다. 우선 raw를 표시하고 정규화는 양은철님과 확인한다.
- 실제 CAN 프레임/수신 라이브러리와 대조 전에는 `bit numbering`, 버스 CAN FD, 페이로드 길이 등을 **검증 필요**로 둔다.
- CAN 주행 패턴 이상과 `can_drowsy`는 독립 항목이다. 자동 졸음 확정은 별도 검증이 끝나기 전까지 켜지지 않는다.

## 7. `system.status`: 베이스라인·연결상태

`payload`:

- `bio_baseline`: `status` = `waiting | collecting | ready | retry_required`, `clean_windows`=정수/null, `required_clean_windows`=20(현 `V2Config`), `elapsed_s`=경과초/null. 시작 대기 60초, 5분 기준, 최대 10분.
- `can_baseline`: `status` = `unavailable | collecting | ready`, `coverage_pct`=0~100/null, `elapsed_s`=경과초/null. **생체 Baseline과 별개**.
- `connections`: `{ecg, ppg, can, model}` 각각 `connected | disconnected | unknown`. 실제 확인 결과만 사용.
- 임의의 Baseline 평균 심박수, 조향각 편차를 만들지 않음. 실제 계산값이 있으면 향후 버전에서 별도 필드 추가.
- **UI에서 Baseline 갱신 버튼**은 요청/승인을 서버로 보내는 별도의 제어 API가 있어야 하며, 버튼 클릭만으로 기준을 업데이트했다고 표시하지 않는다.

## 8. 통신/시간축/결측 규칙 (팀 제안)

- 기본 전송 제안: Raspberry Pi 또는 중계 서버의 **WebSocket** 단일 채널. 메시지는 JSON. 임시 단계에는 파일 리플레이도 **동일한 네 가지 메시지**로 보내도록 어댑터 작성.
- 화면에서 신호를 받은 시각과 측정된 시각은 구분. `t_rel_s`는 측정 시각; 수신 지연을 측정한 경우 UI의 통신 상태에 별도 표시.
- 제안 stale 판단: 생체 파형 3초 이상 미수신 → `신호 지연`, CAN 2초 이상 미수신 → `CAN 지연`, AI 판정 30초 이상 갱신 없음 → `추론 지연`. **이 값은 팀이 조정 가능한 UI 통신 정책이며 모델 임계값 아님**.
- CAN-AI 동기화: 같은 세션 시간축에서 판정창 종료 시점과 CAN 이벤트의 시간차가 팀이 정한 `max_age_s`보다 크면 `can_drowsy=null`. 초기 제안 `max_age_s=1.0`, 실제 CAN 생성 주기를 확인한 후 최종 채택. **이전 CAN 값을 무기한 마지막값 유지(Forward Fill)하여 판정하지 않는다**.
- 원본 데이터는 버퍼/파일에 보관하더라도, 웹에는 필요한 파형 구간과 계산 완료된 상태값만 스트리밍. 가공 중 데이터는 빈칸으로 두고 결측을 0으로 표기하지 않는다.
- Python 모델·상태 머신은 서버에서 실행. **브라우저에서 판정 로직을 재구현하지 않는다**.
- AI 판정에 필요한 품질 정보가 미수신이라면 파이썬의 `WindowIn` 기본값(`q_ecg=0.0`, `ok_ecg=True`)으로 안전한 신호인 척 처리하지 않는다. 결측은 모델 호출 전에 검증하고, 조건이 불충분하면 `output=null`로 유지한다.
- `NaN`/`Inf`는 JSON에서 불허. 비유한 숫자는 `null`로 변환, 원인이 품질 오류라면 오류 상태도 유지.
- 동일 사용자의 개인정보 식별자는 서버에만 보관하고 UI에는 익명 `session_id` 사용.

## 9. 웹보드 화면 요소와 데이터 매핑

| 화면 위치 | 실제 표시항목 | 메시지의 필드 |
|---|---|---|
| 최상단 상태 카드 | 정상 / 주의 / 졸음 후보 / 졸음 / 측정 불가 / 대기 | `ai.decision.payload.output.level`, `candidate`, `baseline_status` |
| AI 영역 | 학생 졸음 점수, 누적 점수, 판정 경로, 잡음 동결 | `input.p_ep`, `output.score`, `output.channel`, `output.frozen` |
| ECG 파형 | 시각별 실제 ECG 파형 | `bio.wave.payload.ecg` |
| PPG 파형 | 시각별 실제 PPG 파형 | `bio.wave.payload.ppg` |
| 바이탈 카드 | ECG HR, PPG PR, 센서 이상 간격, 품질 | `ai.decision.payload.input.hr_ecg`, `hr_ppg`, `q_ecg`, `q_ppg`, `ok_ecg`, `ok_ppg` |
| 차량 카드 | 속력·핸들 조향각·변화율·바퀴각 유도값 | `can.telemetry.payload.speed_kmh`, `steering_angle_deg`, `steering_rate_deg_s`, `road_wheel_angle_deg` |
| 주행 패턴 | 가속/제동 raw 우선, 패턴 이탈, CAN 징후 | `accel_pedal_raw`, `brake_pedal_raw`, 검증 후 `accelerator_pct`, `brake_pct`, `deviation_status`, `can_drowsy` |
| 차량 보조 상태 | 기어, 방향지시등, 핸들 파지, 부주의 경고·차선안전 표시 | `can.telemetry.payload.status_codes` (원본 Signal name별 정수 + 매핑표) |
| CAN 측정 시각 | 속력·조향·페달 각 값의 실제 마지막 관측 시각 | `can.telemetry.payload.signal_times_s`, 필요시 `status_signal_times_s` |
| 개인화/연결 | 생체·CAN 기준 진행률, 채널 연결 | `system.status.payload.bio_baseline`, `can_baseline`, `connections` |
| 시연 출처 표기 | 실측/재생/합성 구분 | 각 이벤트 `origin` 및 `ai.decision.payload.bio_origin`, `can_origin` |

*ECG/PPG의 평균 심박수와 차량 속도는 서로 다른 출처이며 같은 “현재 값”이라고 동기 표시하지 않는다. 카드에는 마지막 **유효 측정 시각**을 표시한다.*

## 10. 팀별 인계 / 미결정 항목

| 담당 | 지금 제공해야 할 데이터·코드 | 아직 확인할 부분 |
|---|---|---|
| 서동균(센서/전처리) | 전처리 ECG/PPG 배열과 측정시각·실제 fs, 단위, 채널별 품질 | 100 Hz 공통 경로 파형 실시간 제공 여부, 파형 보정 단계 |
| 문민규(AI) | S1-EP 확률, 모델 로드·추론 함수, `WindowIn` 구성/StepOut 반환 | 실시간 Pi 구동, PPG 대체 경로 검증, CSV와 `ok_ecg`/`ok_ppg` 매핑 |
| 양은철(CAN) | 실제 시뮬레이터 값, 속력·조향각 디코딩 스펙, CAN Baseline 진행률, 패턴 결과 | `Can list.xlsx` 제공 ID·스케일은 전사 완료. 실 CAN 프레임 대조/비트 numbering, 페달 % 환산식, 조향 좌우 부호, HOD CAN FD, `can_drowsy` 규칙 검증은 남음 |
| 균도(UI/통합) | 이벤트 수신 서버·웹 차트·상태 표시·출처 표시·재생 화면 | 통합 시각 동기화, 교차 확인 가용성, 시연 실측/합성 구분 |

### 합의해야 할 4가지

1. UI에 사용할 ECG/PPG 파형 경로: 공통 전처리 **100 Hz**인지 학생 파형 **64 Hz**인지. 둘은 별개.
2. CAN ID·비트·스케일은 `Can list.xlsx`에서 확인했음. **실제 프레임 디코드 검증, 조향 부호 좌/우 기준, 페달 % 환산, HOD CAN FD**를 팀에서 확인.
3. CAN 교차확인: `can_drowsy` 규칙이 구현·검증되었는지. 아니면 현재 **주행 패턴 표시만** 구현.
4. `state_machine_v2.py`의 첫 주의 진입 40초와 문서 일부의 60초 표기 불일치 확인.

## 11. 검수 기준 (원본 데이터가 확보되면)

- 실제 파형 버퍼를 연결해 시간축·샘플링 속도·표시 단위가 맞는지 확인.
- 실제 학생 모델 추론 확률 + 품질값을 원본 Python `StateMachineV2.step`에 전달하여 `StepOut`과 웹 상태가 같은지 검증.
- `Can list.xlsx`의 0x102 속력·0x125 조향각·0x371 페달을 실제 또는 출처가 표시된 리플레이 프레임으로 검증. SAS raw=32767 오류 처리, HOD CAN FD 지원 여부, 상태 코드 원문 매핑을 확인.
- CAN 미수신일 때 빈칸과 `can_drowsy=null`, 합성 CAN 때 출처 라벨, 실측 CAN 때 실측 라벨이 정확한지 확인.
- 모든 숫자는 측정/추론/검증된 계산의 출처로 추적 가능해야 함.
- v1.2 JSON Schema로 입력 구조를 검증. 구조 검증만으로 센서 데이터의 진위가 검증되는 것은 아님.

## GitHub 원문

- [StateMachineV2](https://github.com/mingyu-mun/26-2-J-berryPi-Capstone/blob/main/src/driver_state/state_machine_v2.py)
- [추론 CSV 생성](https://github.com/mingyu-mun/26-2-J-berryPi-Capstone/blob/main/scripts/selfmeasure_infer.py)
- [공통 전처리](https://github.com/mingyu-mun/26-2-J-berryPi-Capstone/blob/main/src/driver_state/preprocessing.py)
- [학생 입력 전처리](https://github.com/mingyu-mun/26-2-J-berryPi-Capstone/blob/main/scripts/student_cache.py)
- [상태 머신 설계](https://github.com/mingyu-mun/26-2-J-berryPi-Capstone/blob/main/docs/state-machine-v2.md)
