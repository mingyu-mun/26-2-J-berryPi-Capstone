# J_BERRY PI 웹 계기판

검정에 가까운 배경, 얇은 눈금과 절제된 시안 포인트의 로컬 웹 화면입니다. 주 화면은 심박·운전자 상태·속력·실제 파형·기준 수집 현황에 집중하고, 점수와 품질 정보는 접힌 상세 영역에 둡니다. 데이터 연결 설정은 오른쪽 위 버튼으로 엽니다.

**실제 입력 없이는 측정값·파형·상태를 생성하지 않습니다.** 이전 모의 점수, 합성 파형, 시나리오 버튼과 데모 생성 API는 제거했습니다. 초기 상태는 빈 계기와 데이터 수신 대기입니다.

## 실행

Windows PowerShell, 기존 저장소 폴더에서:

```powershell
cd C:\Users\JU\berryPi-Capstone
.\.venv-web\Scripts\python.exe -m pip install -r web\dashboard\requirements.txt
.\.venv-web\Scripts\python.exe scripts\run_dashboard.py
```

가상 환경이 없다면 먼저 `py -m venv .venv-web`을 실행합니다. PC 브라우저에서 `http://localhost:8000`을 엽니다.

Raspberry Pi OS, 실제 저장소 경로에서:

```bash
python3 -m venv .venv-web
.venv-web/bin/python -m pip install -r web/dashboard/requirements.txt
.venv-web/bin/python scripts/run_dashboard.py
```

같은 네트워크의 휴대폰·PC에서 `http://<Pi IP>:8000`으로 접속합니다. Pi의 주소는 `hostname -I`로 확인합니다. 포트 변경은 `--port 8080`, 종료는 Ctrl+C입니다. 모든 정적 파일은 로컬에서 제공되므로 최초 패키지 설치 후 외부 CDN이 필요하지 않습니다. 모델 가중치·센서 드라이버를 실행하는 서버는 아닙니다.

## 입력 방법

첨부받은 `contract/`의 규약과 JSON Schema를 그대로 보관했습니다. 문서 일부 예시의 `schema_version: 1.1` 표기는 동봉 Schema·types.ts의 `1.2`와 다릅니다. 이번 구현은 **동봉 JSON Schema의 1.2를 기준**으로 검증합니다. 내부 식별자는 화면 제목·운전자 상태 영역에 표시하지 않습니다.

지원 이벤트는 `bio.wave`, `ai.decision`, `can.telemetry`, `system.status`입니다. 누락 필드를 0 또는 true로 채우지 않습니다.

### 이 Pi의 수신 데이터

기본 입력입니다. 외부 수집·추론 프로그램이 실제 이벤트를 HTTP로 보냅니다.

```python
import json
from urllib.request import Request, urlopen

# event: 수집/전처리/추론 프로그램이 만든 규약 이벤트 dict
# 임의 측정값이나 파형을 만들지 않고, 실제 프로그램의 출력을 사용합니다.
def publish(event):
    request = Request('http://127.0.0.1:8000/api/events',
                      data=json.dumps(event, allow_nan=False).encode('utf-8'),
                      headers={'Content-Type': 'application/json'}, method='POST')
    with urlopen(request, timeout=3) as response:
        return json.load(response)
```

`POST /api/events`는 단일 이벤트 또는 최대 1,000개 이벤트 배열, 2MB 이하를 받습니다. 봉투·payload를 Schema로 검증하고 유한한 숫자·품질·판정 시각·상태 필드의 일치 여부를 추가 확인합니다. 실패하면 JSON 오류를 반환하며 해당 요청은 반영하지 않습니다. 임의 상태를 계산하거나 모델 점수만으로 정상·졸음을 결정하지 않습니다.

브라우저는 `GET /api/webboard?after=<이벤트 번호>`를 0.5초마다 읽습니다. 같은 Pi를 보는 여러 기기는 이 입력을 함께 봅니다. 서버는 최근 이벤트를 제한된 메모리에만 보관합니다. 재시작하면 수신 대기로 돌아갑니다. 이 서버는 신뢰된 로컬 네트워크 안에서 사용하며 로그인 기능은 없습니다.

### 외부 실시간 연결

설정에서 외부 실시간 연결을 선택하고 실제 송신 프로그램의 `ws://` 또는 `wss://` 주소를 입력합니다. 외부 WebSocket 서버는 같은 이벤트를 JSON 텍스트로 보내야 합니다. 현재 Pi 서버 자체는 HTTP 수신·조회 서버이며, 별도 WebSocket 서버를 자동으로 실행하지 않습니다.

연결 성공만으로 센서가 정상이라고 표시하지 않습니다. 실제 파형·판정·연결 상태가 각각 도착해야 해당 영역이 동작합니다. WebSocket 입력은 현재 브라우저에서만 보며 다른 기기로 전달되지 않습니다.

### 기록 파일

JSON 이벤트 배열 또는 한 줄에 이벤트 하나인 JSONL 파일을 선택합니다. 4MB 이하, 10,000개 이하 이벤트를 검증한 뒤 읽습니다. 파일은 브라우저 안에서 처리하며 Pi나 외부 서버에 업로드하지 않습니다. 일반 ECG CSV 또는 예전 점수만 있는 `windows_scores.csv`를 직접 해석하지 않습니다. 최종 상태와 출처를 포함하는 이벤트로 변환한 기록이 필요합니다.

재생 버튼은 파일을 선택한 후에만 활성화됩니다. 실제 기록의 측정 시각 간격으로 재생하며 일시정지·다음 데이터·처음으로 기능이 있습니다. 세션별 순서를 유지하고 세션 안에서 측정 시각을 정렬합니다. 끝에서 자동 반복하지 않습니다. 같은 세션에 기록된 공백을 임의 표본으로 채우지 않습니다.

## 화면의 데이터 원칙

- 운전자 상태는 `ai.decision.payload.output`에 담긴 Python 최종 결과만 표시합니다. `candidate`는 level=1일 때 후보로 표시합니다.
- 기준 준비 전에는 output이 null이어야 하며 기준 측정 중 또는 재측정 필요를 표시합니다. `system.status`를 받으면 생체 기준과 CAN 기준을 따로 표시합니다. 기준 평균·심박 변화율은 규약에 없어 만들어내지 않습니다.
- 모델 점수·누적 점수·품질 가중치는 상세 영역에서만 표시합니다. 품질 가중치는 모델 정확도나 의학적 신뢰도가 아닙니다.
- 파형은 받은 표본과 `t_start_rel_s + i/fs_hz`를 그대로 사용합니다. 실제 단위·주파수를 표시하며 중복 구간을 합치고 공백은 끊어서 그립니다. 두 파형의 가로축은 공통 세션 시간입니다. 가짜 R파·peak·센서 신호는 없습니다.
- 실시간 파형 3초, CAN 2초, 판정 30초 지연 시 해당 현재 값을 숨기거나 지연으로 표시합니다. 이는 UI의 통신 정책입니다. CAN은 봉투 시각 외에 개별 신호의 실제 관측 시각도 검사합니다. 이벤트 기반 페달이 오래 갱신되지 않으면 현재 값으로 계속 유지하지 않습니다.
- 미수신과 0은 다릅니다. 미수신은 `··`, 수신한 속력 0은 `0`입니다.
- 페달은 받은 raw를 표시합니다. 검증된 percent가 실제 payload에 있을 때만 %를 표시하며 raw를 임의 환산하지 않습니다. 조향각의 좌/우 부호를 추측하지 않고 받은 부호와 °를 표시합니다.
- 생체·차량 출처를 따로 표시하고 CAN 시뮬레이션 표시는 지속 유지합니다. CAN 패턴 이탈을 졸음으로 바꾸지 않습니다.
- 새 session_id에서는 이전 파형·상태 기록·시간축을 지웁니다. 이미 끝난 세션의 늦은 패킷으로 되돌아가지 않습니다. 동일 이벤트 반복 수신으로 지연 시간을 새로 갱신하지 않습니다.

## 원래 상태 머신과의 관계

모델과 원본 `StateMachineV2` 코드는 변경하지 않았습니다. 통합 송신 프로그램이 품질·개인 기준을 확인하고 상태 머신을 실행한 뒤, 실제 `StepOut`을 직렬화해 `ai.decision`으로 보내야 합니다. 현재 화면용 서버는 그 결과를 표시하는 수신 계층입니다. 이전 `/api/window`, `/api/control`, `/api/replay`의 데모·점수 변환 기능은 제거했습니다.

CAN 매핑·단위·금지 오류값은 동봉 `contract/CAN_MAPPING_FROM_XLSX.md`를 참고하세요. UI는 CAN 프레임을 디코딩하거나 미검증 졸음 규칙을 생성하지 않습니다.

## 부팅 시 실행

수동 실행 확인 후 `deploy/jberry-dashboard.service.example`의 사용자명과 모든 경로를 실제 Pi에 맞게 고칩니다.

```bash
sudo cp web/dashboard/deploy/jberry-dashboard.service.example /etc/systemd/system/jberry-dashboard.service
sudo systemctl daemon-reload
sudo systemctl enable --now jberry-dashboard
journalctl -u jberry-dashboard -f
```

중지·해제는 `sudo systemctl disable --now jberry-dashboard`입니다. 수동 서버와 같은 포트에서 동시에 실행하지 않습니다.

## 검사

```bash
python -m pytest -q tests/test_web_dashboard.py tests/test_state_machine_v2.py
node --test tests/webboard_state.test.cjs
node --check web/dashboard/app.js
```

Node는 개발 검사용이며 Pi에서 화면을 실행하는 데 필요하지 않습니다. 테스트의 수치는 검증용 fixture이며 앱의 시연 데이터로 공급하지 않습니다. 브라우저에서 별도로 실제 수신 파일·지연·모바일 배치를 확인할 수 있습니다.
