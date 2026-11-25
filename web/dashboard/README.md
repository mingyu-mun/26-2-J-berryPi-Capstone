# 사용자별 모델 기준과 로컬 생체 리플레이

사용자를 선택하고 모델용 baseline을 측정·저장·불러온 뒤, 서버의 원본 ECG·PPG 기록으로 새 졸음 점수와 상태 판정을 계산한다. CAN 표시 영역은 없으며 CAN 입력 규약과 수집 코드는 유지한다. 실제 입력 전에는 계기판을 비우고 **판정 대기**를 표시한다.

## Pi에서 실행

팀의 수집·모델 송신 프로그램을 유지하며 화면만 교체하려면 **외부 이벤트 수신 모드**를 사용한다. 기존 웹서버를 종료한 뒤 새 브랜치의 Python 서버를 실행한다. 팀 프로그램의 규약 v1.2 `POST /api/events` 송신은 그대로 사용한다.

```bash
python3 -m venv .venv-dashboard
.venv-dashboard/bin/python -m pip install -r web/dashboard/requirements.txt
.venv-dashboard/bin/python scripts/run_dashboard.py --external-events --host 0.0.0.0 --port 8000
```

기존 프로그램이 `create_server()`를 직접 실행하는 구성도 외부 수신 모드로 동작한다. 모델·센서 프로그램을 별도로 바꿀 필요가 없으며 이 모드의 웹서버는 PyTorch 모델을 실행하지 않는다. 사용자 기준·수집 제어는 송신 프로그램이 담당하고, 로컬 기록 설정 버튼은 표시하지 않는다. CAN 이벤트는 수신하지만 차량 화면은 표시하지 않는다. 직접 WebSocket 연결과 화면에서 선택한 사용자 기준의 실시간 송신기 자동 적용은 지원하지 않는다.

정적 HTML만 복사하는 대신 이 브랜치의 Python 서버·검증기·화면을 함께 사용한다. 같은 규약의 `bio.wave`, `ai.decision`, `can.telemetry`, `system.status`를 검증한다. 다른 입력 규약이나 팀원의 별도 미공유 수정은 현장에서 확인해야 한다.

**로컬 원본 파일로 기준을 만들고 재생할 때**는 아래 실행을 사용한다. 원본 재생 중에는 선택한 기준으로 새 점수와 상태를 계산한다.

Pi 4 / Raspberry Pi OS 64비트 / Python 3.11을 권장한다. Python 3.10~3.12 환경에서 Pi 프로젝트 루트의 다음 명령을 실행한다.

```bash
python3 -m venv .venv-local
.venv-local/bin/python -m pip install -r tests/requirements.txt
.venv-local/bin/python scripts/run_dashboard.py --recordings-dir /home/pi/data/normal2 --data-dir /home/pi/jberry-users
```

`--recordings-dir`는 PC의 디스크나 Pi SD카드의 실제 원본 폴더, `--data-dir`는 사용자·기준 저장 폴더다. `--models`로 모델 폴더, `--port`로 포트를 바꿀 수 있다. 원본 폴더는 기본적으로 `outputs/local-recordings`, 저장 폴더는 `outputs/webboard-users`다. 파일이 없으면 빈 목록을 표시한다.

사용자·저장 기준·기존 결과 기록은 `--data-dir`의 SQLite·NPZ 파일로 Pi SD카드에 보관하며 서버 재시작 후에도 유지한다. 원본 측정 파일과 사용자 기준은 Git에 포함하지 않는다.

같은 Wi-Fi의 노트북에서는 `http://<Pi IP>:8000`으로 접속한다. Python 서버는 기본 `0.0.0.0`으로 수신한다. 핫스팟의 기기 간 통신 허용 여부도 확인한다.

이 실행기는 **로컬 기록 시험용**이며 USB·I²C를 열지 않는다. `tests/run.py` 센서 수집 코드는 유지한다. 센서 실행기의 외부 이벤트 수신 화면과 로컬 기록 제어를 동시에 운영하지 않는다. 새 모델 프로필 제어는 위의 `run_dashboard.py` 실행에서 제공한다.

### 포함된 실물 Pi 센서 실행기

이 브랜치의 수집기까지 사용할 때는 `tests`에서 설치·장치 설정·실행을 진행한다. 위의 외부 수신 모드는 팀의 기존 수집기를 유지하는 경로다.

```bash
cd tests
bash pi_test.sh setup
nano sensors.local.json
bash pi_test.sh check
bash pi_test.sh run
```

이전 `pi-integration-test/` 설치가 남아 있으면 `check`·`run`은 새 위치에 없는 설정·가상환경을 이전 위치에서 사용한다. `setup`은 `tests/.venv`를 준비하고 기존 로컬 설정을 `tests/sensors.local.json`으로 복사하되, 새 위치의 설정이 있으면 유지한다. 이전 설정·가상환경·로그는 삭제하지 않는다.

- Pi 4 / Raspberry Pi OS 64비트, ARM64 Python 3.10~3.12를 지원하며 3.11을 권장한다. OS에 git·python3-venv·i2c-tools·libgomp1·libstdc++6가 필요하다.
- AD8232 → Nano → USB, MAX30102 → Pi I²C 연결이다. 모듈의 전원·논리 전압 사양, I²C 활성화와 `dialout`·`i2c` 장치 권한을 확인한다. USB 경로는 `/dev/serial/by-id/`, MAX30102는 bus 1·주소 0x57·100Hz가 예시다.
- `setup`은 로컬 설정이 없을 때 기존 설정 또는 예시를 복사한다. 기본 USB 설정은 115200bps·CSV 두 필드·첫 필드 ADC다. 두 번째 필드 의미를 추측하지 않는다. 실제 출력이 다르면 형식과 열을 맞춘다.
- 예시 ECG 435.276Hz·ADC 1023·5000mV는 실제 장치 설정과 확인한다. 표본 수 시계는 누락·지연을 모두 검출하지 못하므로 장치 시각·순번이 있으면 설정에 반영한다.
- `check`는 설정·패키지·가중치 확인이며 USB·I²C를 열지 않는다. `run`은 수집부터 웹서버까지 실행하므로 같은 포트의 별도 웹서버·시리얼 모니터·기존 수집기를 먼저 종료한다. 다른 포트는 `run --port 8002`로 지정한다.
- 새 로그는 `tests/results/<실행별 폴더>/run.log`다. 센서 수신 Hz, 기준 완료, 모델 처리 시간, 연결 끊김·재연결을 확인한다. 실물 Pi·센서와 실제 처리 속도는 현장 시험 대상이다.
- 이 실행기의 모델은 과거 구간을 사용하는 실험용 연결이다. 화면에서 저장한 사용자 기준을 실물 수집기에 자동 적용하는 연결은 아직 포함하지 않는다.

## 화면에서 시험하기

**사용자 설정·시작** 버튼 하나로 아래 과정을 같은 창에서 진행한다.

1. **사용자:** 닉네임을 입력해 **등록·다음**을 누르거나 기존 사용자를 선택하고 다음으로 이동한다.
2. **기준 선택:** 최초에는 **새 기준 만들기**에서 깨어 있는 안정 상태의 ECG·PPG 원본을 선택한다. **기준 계산으로**를 누른다.
3. **계산:** 공급 속도를 정하고 **기준 계산 시작**을 누른다. 처음 60초를 제외한 이후 5분 구간과 품질을 확인하며 준비되면 자동으로 멈춘다. 부족하면 최대 약 10분까지 확인하고 실패 이유를 표시한다. 일시정지·재개·취소를 제공한다. 창을 닫아도 계산은 계속되며 다시 열거나 페이지를 새로고침하면 서버의 진행 상태를 복원한다.
4. **저장:** 계산 완료 시 자동으로 저장 단계로 이동한다. 이름을 입력해 **측정한 모델 기준 저장·다음**을 누른다. SQLite에는 사용자·버전·품질·원본 출처를, NPZ에는 통계와 파형 참조값을 보관한다.
5. **재생:** 기준을 만든 기록과 별도로 **새로 판정할 원본 기록**을 선택한다. 공급 속도를 정하고 **선택 기준으로 재생 시작**을 누른다. 원본 준비에 성공하면 운전자 화면으로 이동하고 새 점수와 상태를 계산한다. 원본 준비 실패 시 창을 유지하며 이유와 재시도를 안내한다. 진행 중 재시작은 선택한 기록의 처음부터 처리한다.

기존 사용자는 2단계에서 **저장된 기준 불러오기**를 선택하고 **기준 불러오기·재생 준비**를 누르면 계산·저장을 건너뛴다. 모델·처리 설정이 맞지 않으면 같은 화면에 이유를 표시한다. 이전 단계로 돌아가도 입력한 닉네임·기준 이름·기록 선택을 유지하고 사용자별 초안을 분리한다. 버튼이 비활성화되면 바로 아래에 필요한 다음 행동을 표시한다.

주 화면의 **설정·재생 제어**는 접고 펼칠 수 있다. 접혀 있어도 사용자·실제 기준·입력 기록·재생 상태 요약을 유지하며 계기판·운전자 상태·파형은 계속 표시한다. 접기와 창 닫기는 모델 처리를 중단하지 않는다. 설정 접기 상태는 같은 브라우저에 기억한다.

재생 도구의 **일시정지·재개·처음으로·종료**로 제어한다. 처음으로는 새 세션을 만들고 처음에서 일시정지한다. 재개하면 원본부터 다시 처리한다. 끝에서 자동 반복하지 않는다. **설정하고 시작**은 같은 안내 창을 열어 사용자 기준과 판정할 기록을 확인한다.

상단에는 서버가 확정한 사용자·기준·원본·재생 상태를 표시한다. 사용자나 기준을 바꾸면 이전 파형·점수·상태 누적을 비우고, 이전 작업의 늦은 결과를 버린다. 한 서버에서 활성 측정 세션은 하나다. 여러 브라우저는 같은 세션의 제어와 결과를 공유한다.

기준은 처음부터 불러올 수 있지만 **현재 입력의 30·120·300초 특징**은 새로 준비한다. 원본 재생의 모델 점수는 보수적으로 현재 세션의 약 300초 기록이 관측된 뒤 계산한다. 슬롯·필터 지연으로 시작 시점에는 차이가 있을 수 있다. 이전 기준 파형을 현재 입력 앞에 붙이지 않는다. 빠른 재생은 기록 시간을 빠르게 진행하는 기능 시험이며 Pi의 실시간 처리 속도를 증명하지 않는다.

운전자 상태는 상태 머신의 실제 출력을 표시한다. 준비 전에는 **판정 대기**, 졸음 후보는 메인에서 **주의**로 묶고 상세에 후보 여부를 표시한다. 신호 불량·처리 오류를 정상으로 표시하지 않는다. ECG 계기판의 외곽 색상은 기준 심박 편차이고 졸음 점수의 경계가 아니다. 파형 아래 품질·시간 문구는 표시하지 않고, 해당 정보는 접힌 **세부 데이터와 상태 기록 → 신호·판정 정보**에서 확인한다.

## 시뮬레이터 CAN 보조 확정 (`--can-sim`)

생체 졸음 판정을 **시뮬레이터 차량 경고(차선 경고·부주의 경고)**로 보조 확정한다. 실차 CAN 데이터셋과 실차 주행 기준이 없으므로 **시뮬레이터 출처(`origin="simulated"`)에서만** 동작한다. 차량 경고는 졸음 정답 라벨이 아니라 보조 입력이다.

```bash
python scripts/run_dashboard.py --can-sim                          # 내장 합성 주행 (Unity 없이 시연·시험)
python scripts/run_dashboard.py --can-sim --can-sim-every 20       # 이상 상황 간격(초) 조정
python scripts/run_dashboard.py --can-sim-scenario unity_drive.jsonl   # Unity 기록 (JSONL 또는 BLF)
python scripts/run_dashboard.py --can-sim --can-sim-steering       # 선택: PR #13 개인 조향 기준 이탈도 조건에 추가
```

**`can_drowsy` 규칙 (`rule_id=vehicle_warning_sim_v1`, `src/driver_state/can_assist.py`)** — 10초 판정 창마다:

| 결과 | 조건 |
|---|---|
| `true` | 차선 경고 `LaneSafe_indi=5`(YellowWarn) — 단 방향지시등이 켜져 있거나 꺼진 뒤 3초 이내면 정상 차로 변경으로 제외 / 부주의 경고 `Careless_Warn≥2`(YellowBlink 이상) / (선택) 개인 조향 기준 이탈 |
| `false` | 경고 없음, 경고 상태를 한 번 이상 받았고 창의 절반 이상 CAN 수신 |
| `null` | 그 밖 (경고 상태 미수신, CAN 끊김), 또는 실차·기록 출처 |

- 상태 머신 v2: 생체 판정이 **주의** 이상(점수 > 0.55, 잡음 동결 아님)일 때 `can_drowsy=true`면 바로 졸음으로 격상하고, 졸음 후보에서는 3분 추가 대기 없이 확정한다. 정상 단계에서는 CAN을 쓰지 않으므로 차량 경고만으로 경고를 시작하지 않는다. 이때 `confirmation_source="can"`. 주의 단계 격상은 `V2Config.can_from_caution=False`로 끌 수 있다 (`docs/state-machine-v2.md` 8장).
- 판정 재생(기준 계산 제외)마다 CAN 시나리오가 재생 0초부터 같은 시계로 함께 흘러간다.
- 화면에는 `can_origin=simulated`일 때만 운전자 상태 아래에 `차량 경고: …` 한 줄을 표시한다(화면에는 시뮬레이터 표기를 하지 않고 발표에서 설명). `can.telemetry`의 `status_codes`에 네 신호 원시값을 싣는다. 실차·기록 CAN은 표시하지 않는다.
- 내장 합성 주행: 20초부터 30초마다 차선 경고(1.5초)와 부주의 경고(3초)를 번갈아 내고, 35초부터 60초마다 방향지시등을 켠 차로 변경(그 안에서 차선 경고 1초, 제외 대상)을 넣는다. 시연용이며 성능 근거가 아니다.

### 통합 테스트 도구

`scripts/can_sim_scenario.py`: `make`(합성 CAN JSONL 생성), `check`(10초 창별 `can_drowsy`·원인 출력, Unity 기록 점검), `e2e`(가짜 고점수 모델 + 실제 상태 머신으로 CAN 유무 비교·규약 검증, 모델 가중치 불필요), `bio`(합성 원본 ECG·PPG 기록 폴더 생성, 재생 흐름 점검용). CAN 시나리오는 판정 재생 기록보다 길어야 한다.

### Unity 출력 형식

Unity의 차선 이탈 경고·부주의 경고 기능 출력을 회사 CAN 매핑(`Can list.xlsx`) 형식으로 내보낸다. 둘 중 하나:

1. **JSONL** (권장): 한 줄에 객체 하나. `t_us`(주행 시작 기준 마이크로초, 오름차순)만 필수이고 나머지는 바뀐 값만 넣어도 된다. 경고·방향지시등은 이벤트 신호라 마지막 값을 유지한다. 시작 시 네 상태 코드를 한 번 모두 보낸다.
   ```json
   {"t_us": 0, "LaneSafe_indi": 4, "Careless_Warn": 0, "LTurnSig_Indi": 0, "RTurnSig_Indi": 0}
   {"t_us": 100000, "speed_kmh": 99.8, "steering_deg": -0.4}
   {"t_us": 15000000, "LaneSafe_indi": 5}
   ```
2. **BLF**: 회사 매핑 프레임. 속도 `0x102`(bit 56/8), 조향 `0x125`(bit 24/16, ×0.1°), 부주의 경고 `0x100` `Careless_Warn`(bit 24/4), 차선안전·방향지시등 `0x101` `LaneSafe_indi`(28/4)·`LTurnSig_Indi`(32/4)·`RTurnSig_Indi`(40/4). DBC: `git_upload_ver1/mapping/can_list.dbc`, `can_list_warnings.dbc`.

값: `LaneSafe_indi` 0 Off / 1 Yellow / 2 GreenBlink / 3 Gray / 4 Green / 5 YellowWarn, `Careless_Warn` 0 Off / 1 Yellow / 2 YellowBlink / 3 YellowWarn, 방향지시등 0 Off / 1~3 점멸.

## 원본 기록 폴더 형식

서버는 지정 폴더와 바로 아래 기록 폴더의 원본 쌍을 찾는다. 한국어 원본 이름과 `meta.json`이 한 쌍으로 명확하면 자동 조회한다. 여러 metadata가 있으면 `notes`에서 해당 `원본_*_YYYYMMDD_HHMMSS.csv`를 명시한 파일이 하나일 때만 연결한다. 시간상 가까운 파일이나 Hz를 추측하지 않는다.

여러 파일이 있어 모호하거나 파일 이름이 다르면 [recording.example.json](recording.example.json)을 **Download raw file**로 받고 원본 폴더의 `recording.json`으로 저장해 상대 경로를 지정한다. 예시는 `정상상태측정2`의 실제 파일 이름을 가리키며 측정값을 포함하지 않는다. 여러 기록은 하위 폴더마다 `recording.json`을 두면 된다.

- ECG: 시간·원본 ADC의 2열 CSV. ADC 열 이름에 `adc`, `raw`, `원본` 중 하나가 필요하다.
- PPG: 시간·RED·IR의 3열 CSV. IR 열 이름에 `ir` 또는 `적외선`이 필요하다.
- metadata: `fs.ecg`, `fs.ppg`에 실제 표본 주기 정보를 제공한다. 가공된 한 열 `value` CSV나 판정 JSON은 이 입력이 아니다.
- 원본 ADC→mV·PPG 보정 설정은 `tests/sensors.example.json`을 사용한다. 현재 기본 ADC 범위는 0~1023, 기준 전압은 5000mV다. 다른 원본에서는 설정을 먼저 확인한다.
- 각 CSV는 128MB 이하, 공통 구간은 31초 이상이어야 한다. 기준 생성에는 더 긴 양호한 기록이 필요하다.
- 수신 시간의 지터·FIFO 순서를 그대로 모델 시간으로 사용하지 않고 원본 표본 순서와 metadata의 fs로 기록 시계를 복원한다. 실제 하드웨어 동기화·누락 검증과 구분한다.
- 지정 폴더 밖의 파일·심볼릭 링크 대상은 읽지 않는다. 파일이 바뀌면 목록을 새로고침해 다시 선택한다.

## 모델 기준의 보관과 호환성

`<data-dir>/library.sqlite3`의 기존 사용자·결과 보관함을 유지하고 `model_profiles`를 추가한다. 실제 모델 기준은 `<data-dir>/profiles/<profile_id>.npz`에 원자적으로 저장한다. 사용자별 RR 특징의 중앙값·변동폭과 모델 3개 × 경로 2개 × 32차원 파형 참조값, 특징 순서·측정 품질·원본 출처·모델 및 처리 코드 해시를 함께 보관한다. pickle을 사용하지 않는다. 추정할 수 없는 일부 통계의 누락도 보존하고 기존 정규화의 집단 변동폭 보완을 유지한다.

불러올 때 소유 사용자, 파일 무결성, 배열 크기, 필수 특징·기준 품질, 가중치·전처리 버전, ADC 전압과 PPG 보정 설정을 검사한다. 부적합하면 적용하지 않고 이유를 반환한다. 원본별 fs는 출처로 보관하고 각 기록의 metadata에 맞춰 같은 공통·모델 주파수로 처리한다.

프로필은 사용자가 명시적으로 선택한다. 자동 사용자 식별 기능은 아니다. 타인의 기준을 적용하면 사람 간 차이를 졸음 변화로 해석할 수 있으므로 올바른 사용자·기준·기록 연결을 확인한다.

## 기존 결과 기록 보기

**보조 도구 → 저장 결과 JSON 보기**는 브라우저의 JSON/JSONL 파일 또는 조회 사용자의 기존 기준 수집 결과 기록을 읽어, 저장된 파형과 판정을 그대로 표시한다. 조회할 사용자는 별도로 선택하며 실제 추론 사용자를 바꾸지 않는다. 4MB·10,000개 이벤트 이하의 기존 v1.2 이벤트를 검증한다. 재생·일시정지·다음 데이터·처음으로를 제공한다. **모델을 재실행하거나 선택한 모델 프로필을 적용하는 기능과 분리**하며 JSON을 모델용 기준으로 등록하지 않는다.

저장 기준에서 온 평균 심박수에는 선택 필드 `source_session_id`와 `profile_id`를 함께 붙인다. 기준 측정 시각은 출처 세션의 시각으로 보존하고 현재 입력 시각으로 바꾸지 않는다. 두 필드가 없으면 기존 같은 세션 기준 검증을 유지한다. CAN schema 분기는 변경하지 않았다.

## 검증

```bash
python -m pytest -q tests/test_model_profiles.py tests/test_local_replay.py tests/test_pi_sensors.py tests/test_web_dashboard.py tests/test_webboard_library.py tests/test_state_machine_v2.py tests/test_webboard_connector.py
node --test tests/webboard_state.test.cjs
node --check web/dashboard/app.js
python -m pytest -q tests/test_guided_setup_ui.py
```

브라우저 검사는 선택 개발 의존성인 Playwright와 시스템 Chromium이 있을 때 실행한다. 명시적 시험용 원본·모델 출력으로 HTTP·저장 API와 화면의 단계 전환·복원·오류 재시도·사용자 분리·결과 재생을 확인하며, 앱에 가상 입력을 추가하지 않는다. 별도로 실제 원본과 S1-EP 가중치를 사용한 전체 흐름도 확인한다.

가중치와 모델 판정 규칙은 그대로 사용한다. `causal_trailing_v1`은 미래 구간을 사용하지 않는 실험용 연결 방식이며, 판정 정확도·실물 Pi 처리 속도는 별도 검증 대상이다. CPU Python과 실제 원본 기록으로 기능을 검증하고 브라우저 배치를 확인한다. 실물 센서 수집과 Pi의 실제 처리 속도는 현장에서 별도로 확인한다.
