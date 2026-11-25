# CAN 신호 디코딩 매핑 · Can list.xlsx 원본 근거

## A. 화면 우선 신호 4종: 원본 엑셀 B3:P8

| 웹보드 필드 | 엑셀 Signal name | CAN ID | 주기 | start bit / length | 비트순서·자료형 | Factor + Offset | 단위·변환 |
|---|---|---|---|---|---|---|---|
| `speed_kmh` | `Speed` | `0x102` (`CARINFO`) | 100ms | 56 / 8 | Intel / Unsigned | 1 + 0 | km/h = raw (0~255) |
| `steering_angle_deg` | `SAS_angleVal` | `0x125` (`SAS_FD_01_10ms`) | 10ms | 24 / 16 | Intel / Signed | 0.1 + 0 | **핸들 각도(°)** = raw × 0.1; raw=32767 → 오류, 표시 제외 |
| `accel_pedal_raw` | `Accel_Pedal_Pos` | `0x371` (`E_EMS11`) | 비주기(이벤트) | 31 / 8 | Intel / Unsigned | 1 + 0 | raw 0~254; 설명에는 0~100% 대응이라고 쓰였으나 Unit·Factor에 % 변환식은 없음 |
| `brake_pedal_raw` | `Brake_Pedal_Pos` | `0x371` (`E_EMS11`) | 비주기(이벤트) | 0 / 8 | Intel / Unsigned | 1 + 0 | raw 0~127; 설명에는 0~100% 대응이라고 쓰였으나 Unit·Factor에 % 변환식은 없음 |

**부가 계산(원본 문구 또는 개발 제안과 구분):**

- `road_wheel_angle_deg`는 원본 설명의 조향비 15:1을 적용한 **유도값**인 `steering_angle_deg / 15`다. 차량에서 실제 바퀴 각도를 측정한 값이 아니다. 실제 조향비/센서 의미 확인 전에는 null 권장.
- 조향각의 **양수/음수가 좌·우 어느 방향인지 원본 엑셀에 없음**. 화면에서 부호 그대로 표기하고 좌/우 설명은 확정하지 않는다.
- `steering_rate_deg_s`는 실제 수신한 연속 핸들 각도와 타임스탬프에서 Δθ/Δt로 계산하는 유도값이다. 10ms는 메시지 주기이지, 실제 유효 표본 간격을 반드시 10ms라고 가정할 근거는 아니다.
- `accelerator_pct`·`brake_pct`는 **일단 null**. 원본 `Factor=1, Offset=0, Unit` 공란이며, 설명의 0~100% 문구만으로 선형 환산을 **공식 DBC 변환식으로 확정할 수 없다**. 팀이 매핑을 검증해 채택하면 사용할 수 있는 *제안 환산식*은 `raw×100/254`, `raw×100/127`이지만 실측/제조사 검증 전에는 적용하지 않는다.

## B. 나머지 21개 신호 (화면 선택 항목)

표의 코드는 원본 엑셀에 기재된 원시 코드이며 **운전자 졸음 정답이 아니다**. 프런트엔드는 `status_codes`에서 같은 `Signal name` 키로 받아 표의 해석을 보여준다. 상세 코드별 의미는 `can_signal_catalog.json`에 전사했다.

| 기능 | Signal name | CAN ID | 주기 | 시작/길이 | 원본 값 설명 (요약) |
|---|---|---|---|---|---|
| 기어위치 | `GearPosition` | `0x102` | 100ms | 0/4 | 0=P / 1=R / 2=N / 3=D |
| 차선유지기능 | `LaneKeep` | `0x102` | 100ms | 7/1 | 0=OFF / 1=ON |
| 좌측 방향지시등 | `LTurnSig_Indi` | `0x101` | 비주기(이벤트 기반) | 32/4 | 0=Off / 1=Blinking / 2=FastBlink / 3=SlowBlink |
| 우측 방향지시등 | `RTurnSig_Indi` | `0x101` | 비주기(이벤트 기반) | 40/4 | 0=Off / 1=Blinking / 2=FastBlink / 3=SlowBlink |
| 운전석 도어 | `FlDoor` | `0x102` | 100ms | 4/1 | 0=Locked / 1=Open |
| 동승석 도어 | `FrDoor` | `0x102` | 100ms | 5/1 | 0=Locked / 1=Open |
| 트렁크 상태 | `TrunkState` | `0x102` | 100ms | 6/1 | 0=Closed / 1=Open |
| 운전석 안전벨트 경고 | `DrvSeatBelt_Warn` | `0x100` | 비주기(이벤트 기반) | 4/4 | 0=Off / 1=On |
| ESC 작동 상태 | `ESC_Indi` | `0x101` | 비주기(이벤트 기반) | 16/4 | 0=Off / 1=Fault / 2=Operating / 3=ESC_OFF |
| 운전자 부주의 경고 | `Careless_Warn` | `0x100` | 비주기(이벤트 기반) | 24/4 | 0=Off / 1=Yellow / 2=YellowBlink / 3=YellowWarn |
| 차선안전 표시 | `LaneSafe_indi` | `0x101` | 비주기(이벤트 기반) | 28/4 | 0=Off / 1=Yellow / 2=GreenBlink / 3=Gray / 4=Green / 5=YellowWarn |
| 차선유지 표시 | `LaneKeep_indi` | `0x101` | 비주기(이벤트 기반) | 24/4 | 0=Off / 1=Yellow / 2=WhiteBlink / 3=Gray / 4=Green |
| 주행모드 | `DriveMode_Indi` | `0x101` | 비주기(이벤트 기반) | 12/4 | 0=Off / 1=ECO / 2=SPORT / 3=SNOW / 4=MY_DRIVE |
| 회생제동 단계 | `ReGenBrakeLevel` | `0x102` | 100ms | 52/4 | 0=Level0 / 1=Level1 / 2=Level2 / 3=Level3 |
| 핸들 파지 상태 | `HOD_Dir_Status` | `0x208` | 100ms | 80/3 | 0=HANDS OFF / 1=HAND TOUCH (SOFT) / 2=HAND TOUCH (STRONG) / 3=HAND GRIP (SOFT) / 4=HAND GRIP (STRONG) / 5=RESERVED / 6=RESERVED / 7=INVALID |
| 좌측 타이어 공기압 | `tirepressureFL` | `0x103` | 비주기(이벤트 기반) | 36/6 | 0~63: 물리값 범위; Physical = Raw × 1 + (0) |
| 우측 타이어 공기압 | `tirepressureFR` | `0x103` | 비주기(이벤트 기반) | 42/6 | 0~63: 물리값 범위; Physical = Raw × 1 + (0) |
| 후방 좌측 타이어 공기압 | `tirepressureRL` | `0x103` | 비주기(이벤트 기반) | 48/6 | 0~63: 물리값 범위; Physical = Raw × 1 + (0) |
| 후방 우측 타이어 공기압 | `tirepressureRR` | `0x103` | 비주기(이벤트 기반) | 54/6 | 0~63: 물리값 범위; Physical = Raw × 1 + (0) |
| 타이어 공기압 경고 | `TirePressure_Warn` | `0x100` | 비주기(이벤트 기반) | 16/4 | 0=Off / 1=On / 2=Blinking |
| 주행거리 | `Odometer` | `0x102` | 100ms | 32/20 | 0~999999: 물리값 범위; Physical = Raw × 1 + (0) |

## C. 디코더 검증 필요 항목

1. **신호 설명의 출처:** `Can list.xlsx` `CAN DB!B2:P27` (25개). ID와 비트 필드는 **제공된 문서상 정의**이지, 캡처한 실차 프레임과 대조 완료한 값이 아니다.
2. **비트 오더:** 25개 모두 원본에 `Intel (Little)`로 기재되어 있다. 원본에 명시되지 않은 CAN DBC의 start-bit 인덱스 기준(LSB0 등)은 수신기/디코더에서 실제 프레임으로 검증해야 한다.
3. **HOD CAN-FD 주의:** `HOD_Dir_Status`는 `0x208`, Start Bit=80, Length=3. 이 정의대로라면 **최소 11바이트 프레임 페이로드**가 필요하여 클래식 CAN의 8바이트 프레임만 처리하는 장치에서는 수신할 수 없다. 해당 장치/프레임이 CAN FD인지 확인하기 전에는 '핸들 미파지'로 판정하지 않는다.
4. **부주의 경고:** `Careless_Warn`은 원본에서도 **표시 신호**이며, 졸음의 정답 라벨로 사용해서는 안 된다. `LaneKeep`도 보조 기능 ON/OFF이지 차선 이탈 측정값은 아니다.
5. **장애값:** `SAS_angleVal` raw=32767은 Error Indicator. `HOD_Dir_Status` raw=7 INVALID, 5/6 RESERVED. 알 수 없는 코드·범위 이탈 값은 화면의 정상 수치로 사용하지 않는다.
6. **주기:** `Speed`/`CARINFO`=100ms, `SAS_angleVal`=10ms, `HOD_Dir_Status`=100ms, 방향지시등·페달·경고 표시 등은 **비주기(이벤트 기반)**. `can.telemetry` 하나의 수신 시각을 모든 값의 시각처럼 사용하지 않으며, 주요 네 값은 `signal_times_s`에 각각 측정시각을 기록한다.
7. **물리값:** `ReGenBrakeLevel`은 회생제동 단계이지 브레이크 페달 입력 %가 아니다. `Odometer`는 누적 주행거리 km. 네 바퀴 공기압은 `psi`. RPM·차선 카메라의 차선 편차량·요 레이트는 **이 엑셀에 정의되지 않았다**.
8. **CAN 교차 확인:** 이 엑셀은 CAN 신호 **디코딩 정의**만 제공하며 `can_drowsy=true`를 산출할 검증된 임계값이나 분류기는 제공하지 않는다. 따라서 별도 규칙이 없으면 `can_drowsy=null`; 차량 조향 신호가 있다는 이유로 자동 졸음 확정 금지.
   - **2026-11-25 팀 결정 (시뮬레이터 한정):** `rule_id=vehicle_warning_sim_v1`로 시뮬레이터(`origin="simulated"`)의 차선 경고 `LaneSafe_indi=5`(방향지시등 구간 제외)와 부주의 경고 `Careless_Warn≥2`를 `can_drowsy=true` 보조 입력으로 쓴다. 4번 원칙대로 **정답 라벨로 쓰지 않으며**, 상태 머신 v2에서 생체 판정이 주의 이상일 때만 졸음 격상 근거가 된다. 검증된 실차 규칙이 아니므로 실차·기록 출처는 계속 `can_drowsy=null`. 근거: `src/driver_state/can_assist.py`, `docs/state-machine-v2.md` 8장.
