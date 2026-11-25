# Capstone Driving Sim — Unity 운전 시뮬레이터 환경 구축 가이드

J_Berry PI 캡스톤 (On-Device AI 기반 운전자 상태 분석)용 **운전 데이터 수집 시뮬레이터**.
사람이 키보드/게임패드로 고속도로를 운전하는 동안 **CAN과 유사한 차량 신호**와 **차선 기준 주행 지표**를
50 Hz CSV로 기록한다. 같은 시간대의 ECG/PPG 기록과 **Unix 시간(ms)** 으로 맞춰 쓰는 것을 전제로 한다.

- 외부 3D 에셋 없이 코드만으로 도로·차량·카메라·HUD가 생성된다. 에셋 스토어 차량 모델·노면 머티리얼을 끼워 넣을 수도 있다 (3-2절).
- 대상 환경: Windows 10/11 x64, **Unity 6.3 LTS (6000.3.x)**, Universal 3D(URP) 템플릿

---

## 1. 설치

### 1-1. Unity Hub
1. https://unity.com/download 에서 Unity Hub 설치 (Unity 계정 로그인 필요)
2. Hub 첫 실행 시 무료 **Personal** 라이선스 활성화

### 1-2. Unity Editor
1. Hub → **Installs** → **Install Editor** → **Unity 6.3 (LTS)** 선택
   - 6.3 LTS는 2027년 12월까지 지원된다. (6.0 LTS는 2026년 10월에 지원 종료)
2. 모듈 선택
   - ✅ **Microsoft Visual Studio Community 2022** (이미 VS Code를 쓰면 생략 가능 — 아래 참고)
   - ✅ **Windows Build Support (IL2CPP)** — 실험용 .exe로 빌드할 때 필요 (Mono 빌드만 할 거면 생략 가능)
   - ✅ Documentation (선택)
3. 설치 용량은 대략 수~십수 GB. SSD 권장.

> VS Code를 쓰는 경우: VS Code에 **C# Dev Kit** + **Unity** 확장 설치 → Unity에서
> `Edit > Preferences > External Tools > External Script Editor` 를 VS Code로 지정.

### 1-3. 프로젝트 생성
1. Hub → **Projects** → **New project** → 템플릿 **Universal 3D** (URP)
2. 프로젝트 이름 `CapstoneDrivingSim`
3. **저장 경로는 영문/숫자만** (예: `C:\Unity\CapstoneDrivingSim`). 한글·공백이 섞인 경로는 패키지 빌드 오류가 나는 경우가 있다.
4. Unity Cloud 연결 / Version Control 옵션은 꺼도 된다.

---

## 2. 스크립트 넣기

1. 이 압축의 `Assets/CapstoneSim` 폴더를 프로젝트의 `Assets/` 아래로 그대로 복사
   ```
   CapstoneDrivingSim/
   ├─ Assets/
   │  └─ CapstoneSim/
   │     ├─ Scripts/   (런타임 스크립트 9개)
   │     └─ Editor/    (메뉴 스크립트 1개)
   ├─ Tools/check_log.py   ← 프로젝트 밖 아무 데나 둬도 됨
   └─ .gitignore           ← 팀 GitHub 저장소 쓸 때 프로젝트 루트에
   ```
2. Unity 창으로 돌아가면 자동 컴파일된다. **Console 창(Ctrl+Shift+C)에 빨간 에러가 없어야 한다.**
3. 입력 설정 확인
   - `Window > Package Manager` → **Input System** 패키지가 설치되어 있는지 확인 (Unity 6 템플릿에는 기본 포함)
   - `Edit > Project Settings > Player > Other Settings > Active Input Handling` 이
     **Input System Package (New)** 또는 **Both** 인지 확인
   - 스크립트는 구 Input Manager만 켜진 경우에도 동작하지만(키보드만), 게임패드를 쓰려면 새 Input System이 필요하다.

---

## 3. 실행

1. `Assets/Scenes/SampleScene` 을 연 상태에서 상단 메뉴 **Capstone > 주행 시뮬레이터 씬 구성**
   → Hierarchy에 `DrivingSim` 오브젝트가 생긴다. (메뉴가 안 보이면 컴파일 에러부터 확인)
2. `DrivingSim` 선택 → Inspector에서 설정
   | 항목 | 설명 | 기본값 |
   |---|---|---|
   | Participant Id | 참가자 ID (파일명에 들어감) | P01 |
   | Session Note | 메모 (meta.json에 기록) | |
   | Log Rate Hz | CSV 기록 주기 | 50 |
   | Output Folder | 비우면 `<프로젝트>/SimLogs` | |
   | Physics Hz | 물리 갱신 주기 | 100 |
   | Straight Length / Curve Radius | 트랙 직선 길이 / 곡선 반경 [m] | 1500 / 300 (한 바퀴 ≈ 4.9 km) |
   | Start Lane | 시작 차로 (1=1차로, 2=2차로) | 2 |
   | Night | 야간 시나리오 (어둡게 + 전조등) | off |
3. **Ctrl+S** 로 씬 저장 → **Play** (Ctrl+P)
4. Game 창을 한 번 클릭해 포커스를 준 뒤 조작. 큰 화면으로 하려면 Game 창 탭 우클릭 → Maximize.

### 조작키
| 키보드 | 게임패드 | 기능 |
|---|---|---|
| W / ↑ | RT | 가속 |
| S / ↓ / Space | LT | 브레이크 (정지 상태에서 계속 누르면 후진) |
| A·D / ←→ | 왼쪽 스틱 | 조향 |
| **L** | Start | **기록 시작 / 정지** |
| **M** | A | **동기화 마커** (화면 우상단 흰 사각형이 0.3초 깜빡임) |
| Q / E | LB / RB | 좌 / 우 깜빡이 (토글, 차로 변경을 마치면 자동으로 꺼짐) |
| **G** | X | **조향 보조 세기** (끔 → 약 → 강, 기본 강) |
| K | | 부주의 경고 시연 (4초간 강제로 켬) |
| **F8** | | **시연 운전 시나리오** 시작/정지 (AI 세션 중, 운전 키를 누르면 바로 멈춤) |
| R | Select | 차선 위로 복귀 (도로 이탈·전복 시) |
| C | Y | 운전석 시점 ↔ 추적 시점 |
| **T** | | **조향 방식 전환** (자동 → 키보드 → 마우스 → 게임패드) |
| H | | HUD 표시 토글 |

### 3-0. 운전 보조와 차량 경고 (2026-11-26)
코너에서 차선 이탈이 불가피할 만큼 어려워서 **조향 보조**를 넣었고, 저장소 CAN 보조 확정 규칙(PR #16, `vehicle_warning_sim_v1`)이 쓰는 **차선 이탈 경고·부주의 경고·방향지시등**을 Unity 가 직접 만든다. 화면 위 가운데(AI 배너 아래)에 경고 배너, 계기판 위에 보조 상태가 보인다.

| 기능 | 동작 | 회사 CAN 코드 |
|---|---|---|
| 조향 보조 (`DriverAssist.cs`) | 운전자 핸들각에 차로 중심 추종 조향(곡률 피드포워드 + 편차·헤딩 되먹임, `CarAutopilot` 과 같은 식)을 상한을 두고 더한다. **강**: 손을 떼도 코너를 돈다(보정 상한 핸들각 60°, 운전자 입력은 절반만 반영, 커브 앞에서 횡가속도 2.7 m/s² 이하가 되도록 자동 감속, 최고 120 km/h). **약**: 곡선 유지 정도만(상한 12°). 깜빡이를 켜면 차로 변경을 위해 잠시 꺼지고 30 km/h 미만·도로 밖에서는 동작 안 함 | — |
| 차선 이탈 경고 (`VehicleWarnings.cs`) | 60 km/h 이상에서 차체가 차선을 밟으면(\|차로 편차\| > (3.5 − 차폭)/2) 경고음 + 배너. 깜빡이를 켰으면 경고 안 함. 최소 1초 유지 | `LaneSafe_indi` 3 비활성 / 4 활성 / **5 경고** |
| 부주의 경고 | 핸들 조작 없이 15/25/40초 → 1/2/3단계, 최근 60초 안에 차선 이탈 경고가 2번째로 나면 2(3번째면 3)를 **5초 동안만** 표시하고 셈을 비운다, K 키 → 2 (4초). 핸들을 조작하면 '핸들 조작 없음'은 바로 꺼진다 | `Careless_Warn` 0~3 |
| 방향지시등 | Q/E 토글. 옆 차로로 옮겨 중심 ±0.6 m 에 1초 머물면 자동으로 끄고, 20초가 지나도 끈다. 딸깍 소리 | `LTurnSig_Indi`, `RTurnSig_Indi` 0/1 |

- 조향 보조가 **강**이면 손을 떼고 달려도 차로를 지킨다. 이때는 '핸들 조작 없음' 부주의 경고가 15초부터 켜진다 (실차의 '핸들을 잡으세요'와 같은 성격). 졸음 시연에서 차선 이탈을 보여 주려면 **G 로 약/끔**으로 바꾸거나 보정 상한보다 크게 꺾는다.
- 수치(세기별 이득·상한, 경고 기준)는 Play 중 `DrivingSim` 의 `DriverAssist`·`VehicleWarnings` Inspector 에서 바로 바꿀 수 있다.
- 차량 경고는 **졸음 정답 라벨이 아니라 보조 입력**이다. 정상/주의/졸음 단계는 생체 신호만으로 정하고, 생체 판정이 **졸음**인 동안 차량 경고(차선 이탈·부주의 2단계 이상)가 있으면 **졸음 확정**으로 표시한다. 차량 경고만으로 단계가 오르지는 않는다.

### 3-0-1. 시연 운전 시나리오 (`DemoDriver.cs`)
시연 영상용으로 rec_20261118(11-18 rec4) 자기보고 라벨에 맞춘 운전을 자동으로 한다. 시각은 생체 기록 기준 초(0 = 기록 시작)이다.

| 기록 시각 | 운전 |
|---|---|
| 0~540 s 정상 | 차로 중앙 유지, 150·330·480 s 에 깜빡이 켜고 차로 변경 (차선 경고 제외 확인) |
| 540~960 s 졸음 시작 | 600·820 s 핸들 놓침(부주의 1~2단계), 700·900 s 차선 쪽으로 흐름. 생체가 아직 졸음이 아니므로 단계는 오르지 않는다 |
| 960~1440 s 졸음 | 30~40초마다 좌우 번갈아 차선 이탈 경고, 1200 s 핸들 놓침 → 생체가 졸음이면 **졸음 확정** |
| 1440 s~ 회복 | 1560 s 깜빡이 차로 변경 |

- 메뉴에서 rec_20261118 재생으로 주행 시작 → `F8` (또는 `DrivingSim` 의 `SimBootstrap > Demo Scenario` 를 켜 두면 세션 시작과 함께 자동). 시작 오프셋을 주면 그 시각부터 이어서 운전한다.
- 시나리오 동안 조향 보조는 끔으로 바뀌고, 끝나면 원래 세기로 돌아온다.
- 서버 없이 같은 사건표의 CAN 파일이 필요하면 저장소 `scripts/demo_can_scenario.py` 로 만든다 (`--can-sim-scenario` 형식).

### 3-1. 조향 방식 (v2)
핸들 각도를 상태값으로 두고, 입력 장치마다 핸들을 돌리는 방식을 다르게 했다. 바퀴각 = 핸들각 ÷ 15.

| 방식 | 동작 | 권장 용도 |
|---|---|---|
| **마우스** | 화면 가로 위치 = 핸들 위치. 화면 하단 가이드 막대 끝이 최대 조향 | **게임패드가 없을 때 데이터 수집용 1순위** (아날로그라 부드러움) |
| 게임패드 | 왼쪽 스틱 = 핸들 위치. 중앙 부근은 둔하게(expo) → 고속 미세 조향 쉬움 | 가장 자연스러움 |
| 키보드 | 누르는 시간에 따라 점점 빨리 돎(톡 치면 미세 조향), 떼면 천천히 복원 | 테스트용 |

- **속도 감응형**: 정지 시 핸들 ±450°까지, 100 km/h 이상에서는 ±60°까지만 → 고속에서 확 꺾여 스핀하는 일 방지
- **애커먼 조향**(안쪽 바퀴를 더 꺾음) + **안티롤바** + 타이어 마찰 재설정으로 코너에서 덜 미끄러지고 덜 기울어짐
- 수치는 `EgoCar > CarController` Inspector에서 Play 중에도 바로 조정 가능 (Play 종료 시 원복되므로 맞는 값을 메모해 둘 것)
  - 고속에서 더 예민하게: `High Speed Lock Deg` ↑ / 둔하게: ↓
  - 마우스 감도: `Mouse Range` (작을수록 예민)
  - 키보드 미세 조향: `Kb Ramp Time` ↑ 하면 짧게 누를 때 더 조금 돎
- 조향 장치에 따라 핸들각 신호의 모양이 달라지므로, 사용한 방식이 `meta.json` 의 `steer_mode` 에 기록된다. **한 실험 안에서는 같은 장치로 통일**할 것.

### 3-2. 외형 바꾸기 (에셋 스토어 차량·도로)
에셋을 넣지 않아도 아스팔트·잔디 텍스처, 가드레일, 나무가 코드로 생성된다. 더 사실적으로 하려면:

1. 에셋 스토어에서 에셋을 받아 `Window > Package Manager > My Assets` 에서 Import
2. 머티리얼이 분홍색이면 URP로 변환: `Window > Rendering > Render Pipeline Converter` → *Material Upgrade* 체크 → Initialize And Convert
3. `DrivingSim` Inspector에 끌어다 놓기
   | 칸 | 넣을 것 |
   |---|---|
   | **Car Model Prefab** | 차량 프리팹 (Project 창에서 드래그). 스크립트·물리는 자동 제거되고 외형만 씀 |
   | Car Model Yaw Offset | 차가 옆/뒤를 보고 있으면 90, -90, 180 중 하나 |
   | Model Wheel Names / Model Steering Wheel Name | 바퀴·핸들 자동 인식이 실패했을 때만 오브젝트 이름 입력 |
   | Invert Steering Wheel Visual | 실내 핸들이 반대로 돌면 체크 |
   | Driver Eye Adjust | 운전석 시점이 어색하면 위치 미세 조정 (예: y +0.05) |
   | **Road Material** | 아스팔트 머티리얼 (**차선 표시가 없는 것** — 차선은 코드가 그림) |
   | Ground Material | 잔디/흙 머티리얼 |
4. Play → Console에 `[CarModel] 바퀴 자동 인식: FL=..., FR=...` 로그가 뜨면 성공

**추천 에셋** (가격·호환성은 받기 전에 페이지에서 재확인)
| 용도 | 에셋 | 비고 |
|---|---|---|
| 차량 (무료) | Sport Car Free | 무료, Unity 6 기반 (2026-02 갱신) |
| 차량 (무료) | Realistic Car Controller (Free) | 무료, Unity 6 기반. 차량 모델만 가져다 씀 (조작 스크립트는 자동 제거) |
| 차량 (유료, **실내 포함**) | Modern Vehicles Pack – Interior & Exterior | 세단 2 + SUV 1, 실내·외 모두 모델링 → 운전석 시점이 가장 사실적 |
| 차량 (유료) | Realistic Sport Car 03 | 바퀴·핸들이 분리된 모델 |
| 노면 (무료) | Urban Road Pack Lite – Clean & Weathered | 무료, URP/HDRP 지원 — 아스팔트 머티리얼만 사용 |

- 운전석 시점 현실감은 **실내(대시보드·핸들)가 모델링된 차량**이 좌우한다. 외관만 있는 모델은 운전석 시점에서 차체가 거의 안 보인다.
- 과제 예산의 소프트웨어활용비로 유료 에셋을 사는 것도 방법이다.
- 도로 자체를 EasyRoads3D 같은 도로 제작 툴로 바꾸면, 차선 편차·차로 번호 계산(`TrackBuilder`)을 새 도로에 맞게 다시 짜야 한다. 지금 구조에서는 **노면 머티리얼만 교체하는 방식**을 권장.

---

## 4. 출력 파일

`L` 로 기록을 시작/정지하면 `SimLogs/` 에 세 파일이 생긴다.

```
drive_P01_20261117_143000.csv          주행 신호 (50 Hz)
drive_P01_20261117_143000_events.csv   이벤트 (기록 시작/정지, 마커, 깜빡이, 리셋, 랩)
drive_P01_20261117_143000_meta.json    세션 정보
drive_P01_20261117_143000_can.jsonl    저장소 --can-sim-scenario 형식 CAN (속도·핸들각 50 Hz + 차량 경고 상태 코드)
```

`_can.jsonl` 은 저장소에서 바로 쓸 수 있다: `python scripts/can_sim_scenario.py check drive_..._can.jsonl --verbose`, `python scripts/run_dashboard.py --can-sim-scenario drive_..._can.jsonl`. AI 세션(메뉴의 주행 시작)에서는 서버도 받은 CAN 을 `SimLogs/ai/can_<세션>.jsonl` 로 같은 형식으로 남긴다.

### 4-1. 주행 신호 컬럼
| 컬럼 | 단위 | 설명 | 실차 CAN 대응 |
|---|---|---|---|
| `wall_unix_ms` | ms | PC 벽시계 UTC Unix 시간. **생체신호와 맞출 때 기준** | 타임스탬프 |
| `t_s` | s | 기록 시작부터의 시뮬레이션 시간 (간격이 정확히 균일) | |
| `sample` | | 샘플 번호 | |
| `speed_kmh` | km/h | 차속 (후진은 음수) | 차속 |
| `accel_pedal` | 0~1 | 가속 페달 위치 | APS |
| `brake_pedal` | 0~1 | 브레이크 페달 위치 | 브레이크 |
| `steer_input` | -1~1 | 조향 입력 (−왼쪽, +오른쪽) | |
| `steering_wheel_deg` | deg | 핸들 각도 = 바퀴각 × 기어비(15) | SAS (조향각 센서) |
| `wheel_angle_deg` | deg | 앞바퀴 조향각 | |
| `yaw_rate_dps` | deg/s | 요레이트 (+오른쪽 회전) | ESC 요레이트 |
| `long_accel_mps2` | m/s² | 종가속도 (+가속) | ESC 가속도 |
| `lat_accel_mps2` | m/s² | 횡가속도 (+오른쪽) | ESC 가속도 |
| `engine_rpm` | rpm | 엔진 회전수 (자동변속 **모사값**) | EMS RPM |
| `gear` | | −1=R, 0=정지, 1~6 (**모사값**) | TCU 기어 |
| `turn_signal` | | −1=좌, 0=없음, +1=우 | 방향지시등 |
| `track_s_m` | m | 트랙 진행 거리 | (없음 — 시뮬 전용) |
| `lap` | | 랩 수 | |
| `lane` | | 1=1차로, 2=2차로, 0=도로 밖 | LKAS 차로 인식 |
| `lateral_pos_m` | m | 1·2차로 경계선 기준 횡위치 (+오른쪽) | |
| `lane_offset_m` | m | **현재 차로 중심 대비 편차** (+오른쪽) | LKAS 횡편차 |
| `heading_error_deg` | deg | 차량 방향 − 도로 방향 | LKAS 헤딩 |
| `road_curvature` | 1/m | 도로 곡률 (직선 0, 곡선 1/R) | LKAS 곡률 |
| `lane_departure` | 0/1 | 차체가 차선을 밟음 (\|lane_offset\| > (3.5−1.8)/2 = 0.85 m) | LDW |
| `marker` | | 그 순간 눌린 마커 번호 (평소 0) | |
| `lane_safe_indi` | | 차선 이탈 경고 상태 (3 비활성 / 4 활성 / 5 경고) | LaneSafe_indi |
| `careless_warn` | | 부주의 경고 단계 0~3 | Careless_Warn |
| `assist_wheel_deg` | deg | 조향 보조가 더한 핸들각 (`steering_wheel_deg` 에 포함) | |

- `engine_rpm`, `gear` 는 차량 동역학이 아니라 속도 구간으로 만든 **그럴듯한 모사값**이다. 모델 특징으로 쓰기 전에 팀에서 판단할 것.
- 실제 차량 CAN에는 `lane_offset` 류 신호가 LKAS 장착 차량에만 있다. 시뮬에서는 정확히 계산되므로 **"실차에서 얻을 수 있는 신호"와 "시뮬 전용 신호"를 구분**해서 쓰는 것이 좋다.

### 4-2. 왜 시간 컬럼이 두 개인가
- Unity의 물리 갱신(FixedUpdate)은 화면 프레임마다 몰아서 실행된다. 그래서 `wall_unix_ms` 간격은 20 ms 근처에서 ±1프레임(≈16 ms) 흔들리고, `t_s` 간격은 정확히 20 ms다.
- PC가 느려 프레임이 크게 끊기면 `t_s` 가 실제 시간보다 뒤처진다. `check_log.py` 가 이 차이를 보고한다 (수백 ms 이상이면 그래픽 품질을 낮추거나 다른 프로그램을 끌 것).

---

## 5. 생체센서(ECG/PPG)와 시간 맞추기

권장 순서 (위가 더 정확):

1. **센서를 같은 PC에 연결** (Arduino/ESP32 등으로 USB 시리얼 수신) 하고, PC 쪽 수신 프로그램이 샘플마다 Unix ms를 붙여 저장
   → 시계가 하나라 정렬 오차가 사실상 시리얼 지연(수 ms)뿐.
2. **라즈베리파이에서 따로 기록**한다면 두 기기의 시계를 NTP로 맞춘다
   - Windows: 관리자 cmd에서 `w32tm /resync`
   - Raspberry Pi: `chrony` 또는 `systemd-timesyncd` 활성화 후 `timedatectl` 로 동기화 확인
   - 실험 시작 전에 두 기기 시간 차를 한 번 기록해 둘 것
3. **마커로 사후 보정**: 기록 시작 직후 `M` 을 누르는 순간 센서 쪽에도 식별 가능한 이벤트(예: 센서 쪽 버튼 입력)를 함께 남기면, 두 시계가 어긋나 있어도 오프셋을 계산할 수 있다.

정렬은 `check_log.py --sensor` 가 `pandas.merge_asof` 로 해 준다 (아래 6절).

---

## 6. 로그 점검 (Python)

```bash
pip install pandas numpy matplotlib
python Tools/check_log.py SimLogs/drive_P01_20261117_143000.csv --plot
python Tools/check_log.py SimLogs/drive_P01_20261117_143000.csv --sensor ppg.csv --sensor-time unix_ms
```
- 실제 기록 주기·시계 드리프트, 평균 속도, **SDLP**(차선 내 횡위치 표준편차 — 졸음 시 커지는 대표 지표), 차선 이탈 횟수, 조향 반전율, 이벤트 목록을 출력
- `--plot` : 속도·핸들각·차선 편차·요레이트 그래프 PNG 저장 (마커 위치 표시)
- `--sensor` : 생체센서 CSV의 각 샘플에 가장 가까운 주행 샘플을 붙여 `<센서파일>_with_drive.csv` 로 저장

---

## 7. 실험용 실행 파일(.exe) 만들기

에디터 없이 다른 PC에서 돌리려면:
1. `File > Build Profiles` → Windows 선택 → `SampleScene` 이 Scene List에 있는지 확인 → **Build**
2. 실행 시 참가자 ID 지정
   ```
   CapstoneDrivingSim.exe -participant P03 -note "야간_2회차"
   ```
3. 로그는 `<exe가 있는 폴더>\SimLogs` 에 저장된다.

---

## 8. 문제 해결

| 증상 | 원인 / 해결 |
|---|---|
| `InvalidOperationException: You are trying to read Input using the UnityEngine.Input class...` | 이 프로젝트 스크립트에서는 나지 않는다. 다른 스크립트가 구 Input을 쓰는 경우 → Active Input Handling 을 **Both** 로 |
| 도로·차량이 **분홍색** | 렌더 파이프라인과 셰이더 불일치. Universal 3D 템플릿으로 만들었는지 확인 |
| `Capstone` 메뉴가 안 보임 | Console의 컴파일 에러 해결 / `Editor` 폴더가 `Assets/CapstoneSim/Editor` 에 있는지 확인 |
| 차가 튀거나 미끄러짐 | `DrivingSim > Physics Hz` 를 100 이상으로. `EgoCar > CarController` 의 조향·토크 값 조정 |
| 외부 차량의 바퀴가 안 돎 / 엉뚱한 부품이 돎 | Console의 `[CarModel]` 로그 확인 → `Model Wheel Names` 에 FL, FR, RL, RR 순서로 바퀴 오브젝트 이름 직접 입력 (Play 중 Hierarchy의 `EgoCar > Model_...` 아래에서 이름 확인) |
| 외부 차량이 땅에 묻히거나 뜸 | 모델 피벗 문제. `Driver Eye Adjust` 와 별개로, 바퀴 이름을 직접 지정하면 대부분 해결 |
| 실내 핸들이 엉뚱한 축으로 돎 | `EgoCar > CarController > Steering Wheel Axis` 에 (0,0,1) / (0,1,0) / (1,0,0) 중 하나 입력 |
| 키 입력이 안 먹힘 | Game 창을 클릭해 포커스. 게임패드는 Play 전에 연결 |
| 기록 중 Play를 멈춤 | 자동으로 기록이 정지·저장된다 (`OnDisable`) |

---

## 9. 파일 구성

| 파일 | 역할 |
|---|---|
| `SimBootstrap.cs` | Play 시 도로·지면·차량·카메라·조명·로거·HUD 생성, 외부 모델/머티리얼 연결, 깜빡이/리셋/시점 키 처리 |
| `TrackBuilder.cs` | 스타디움형 편도 2차로 순환로 + 가드레일·나무 생성, 차량 위치 → 차로·편차·헤딩 오차 계산 |
| `CarController.cs` | WheelCollider 차량 (핸들각 기반 조향·애커먼·안티롤바, 페달 램프, 자동변속·RPM 모사, 가속도·요레이트 측정) |
| `CarModelFitter.cs` | 외부 차량 프리팹 정리(스크립트·물리 제거), 크기·위치 보정, 바퀴·핸들 자동 인식 |
| `ProceduralTextures.cs` | 아스팔트·잔디·HUD 핸들 아이콘 텍스처 생성 |
| `SimInput.cs` | 키보드/마우스/게임패드 입력 (새 Input System + 구 Input Manager 겸용) |
| `DrivingDataLogger.cs` | 50 Hz CSV + 이벤트 CSV + meta.json + `_can.jsonl` 기록 |
| `DriverAssist.cs` | 조향 보조 (차로 유지, G 키로 세기 전환) |
| `VehicleWarnings.cs` | 차선 이탈 경고·부주의 경고·방향지시등 자동 해제 + 경고음, 회사 CAN 상태 코드 |
| `DemoDriver.cs` | 시연 운전 시나리오 (생체 기록 시각에 맞춘 자동 운전, F8) |
| `FollowCamera.cs` | 운전석 1인칭 / 추적 3인칭 카메라 |
| `SimHUD.cs` | 속도·기어·차로 편차·기록 상태·마커 플래시 표시 |
| `Editor/CapstoneSimMenu.cs` | `Capstone` 메뉴 (씬 구성, 로그 폴더 열기) |
| `Tools/check_log.py` | 로그 점검, 주행 지표, 생체신호 정렬 |

## 10. 다음 확장 후보
- 실험 중 일정 간격으로 **KSS(졸음 척도) 자가평가 팝업**을 띄워 라벨을 같은 로그에 기록
- 주변 차량(앞차) 추가 → 차간거리/TTC 신호
- 차선 이탈 시 경고음 → HMI 피드백 시나리오 검증
