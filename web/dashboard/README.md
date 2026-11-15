# Raspberry Pi 운전자 상태 웹 대시보드

사용자가 제공한 `index.html`의 어두운 패널·청록색 강조 디자인을 바탕으로 한 로컬 웹 앱입니다. Raspberry Pi가 Python 서버를 실행하고 휴대폰·PC는 브라우저로 접속합니다. 인터넷·Node.js·외부 CDN이 필요하지 않습니다(최초 Python 패키지 설치 제외).

- 모델의 내부 구현·가중치는 비워 두었습니다. 센서 드라이버도 실행하지 않습니다.
- 서버는 원본 `driver_state.state_machine_v2.StateMachineV2`를 실행합니다. 브라우저의 별도 JS 상태 머신은 없습니다.
- 하나의 Pi는 하나의 입력 소스·타임라인을 공유합니다. 다른 기기에서 재생·초기화하면 모든 화면에 반영됩니다. 메모리 상태는 서버 재시작 때 초기화됩니다.
- CAN PR의 강한 감속을 자동으로 졸음 확인으로 변환하지 않습니다. 데모의 CAN true는 상태 전이 시험만을 위한 입력입니다.

## 1. Raspberry Pi에서 실행

Raspberry Pi OS 64비트, Python 3.10 이상을 권장합니다. 아래 명령은 이 저장소의 루트에서 실행합니다. 웹 화면만 사용할 때는 루트 `requirements.txt`의 대형 ML 패키지를 설치할 필요가 없습니다.

```bash
cd ~/berryPi-Capstone   # 실제로 저장소를 복제한 경로로 변경
python3 -m venv .venv-web
.venv-web/bin/python -m pip install -r web/dashboard/requirements.txt
.venv-web/bin/python scripts/run_dashboard.py
```

`venv`를 만들 수 없다면 Pi에서 `sudo apt install python3-venv`를 먼저 실행합니다. 서버 종료는 `Ctrl+C`입니다.

Pi 자체 브라우저에서는 `http://localhost:8000`을 엽니다. 휴대폰·PC는 같은 Wi-Fi/LAN에 연결한 다음 `hostname -I`로 확인한 Pi의 주소를 사용합니다. 예: `http://192.168.0.20:8000`. 다른 기기에서 `localhost`를 입력하면 그 기기 자신을 가리킵니다. 게스트 Wi-Fi의 기기 간 격리가 켜져 있으면 접속되지 않을 수 있습니다.

8000번 포트를 이미 쓰는 경우 `--port 8080`으로 실행하고 브라우저 주소도 8080으로 바꿉니다. 이 앱은 같은 신뢰된 LAN 안의 공동 조작용으로, 로그인 기능은 없으며 인터넷 포트 개방 없이 사용합니다.

Windows PC에서도 먼저 실행할 수 있습니다.

```powershell
py -m venv .venv-web
.\.venv-web\Scripts\python.exe -m pip install -r web/dashboard/requirements.txt
.\.venv-web\Scripts\python.exe scripts/run_dashboard.py
```

## 2. 화면 확인

### 데모

기본 모드입니다. 처음에는 수신 대기이며, `+ 10초 판정` 또는 `자동 재생`으로 모의 점수를 넣습니다. 정상·졸음 추정·회복·ECG 잡음(PPG 대체)·모두 불량·직접 지정 점수를 선택할 수 있습니다. 기본 자동 재생은 **실제 0.55초마다 논리 시간 10초를 진행하는 가속 시연**입니다. `--interval 10`은 10초마다 한 번 진행합니다. 실시간 모델 출력 수신과 구별됩니다.

이 모드에서만 합성 ECG·PPG 파형을 표시합니다. 별도의 신경망이 이 파형을 분석하는 것은 아닙니다. Baseline도 준비됐다고 가정한 시연이며 실제 기준을 수집·학습하지 않습니다.

### 추론 CSV 재생

화면에서 `CSV 선택`으로 `selfmeasure_infer.py`의 `windows_scores.csv`를 불러옵니다. 선택 파일은 **브라우저에서 해당 Pi 서버로 전송**되며 서버는 메모리에서 처리합니다. 파일 이름으로 디스크에 저장하지 않습니다. 1MB 이하 파일을 지원합니다.

연결 시험용 합성 CSV는 `web/dashboard/samples/demo_scores.csv`입니다. 이것은 정확도 평가 자료가 아닙니다.

| 열 | 처리 |
|---|---|
| `student_S1-EP_p` 또는 `student_S1-EP_logit` | 모델 점수, logit은 sigmoid 변환 |
| `t_rel` 또는 `t` | `t_rel`은 30초 창 시작, `t`는 창 끝 시각 [초] |
| `q_abn`, `valid` | ECG 품질과 유효 여부. 없으면 ECG를 유효하다고 가정하지 않음 |
| `student_S1-EP_prv_p`, `q_ppg`, `valid_ppg` | PPG 대체 경로. 값·품질·유효 상태가 있어야 사용 |
| `hr_ecg` 또는 `hr_30`, `hr_ppg` | 표시 및 센서 불일치 검사에 사용하는 심박수 |
| `can_drowsy` | 별도 검증된 외부 확인 값. 누락 시 null, 데모 CAN 선택값을 주입하지 않음 |
| `session_id`, 선택적으로 `dataset_id`, `group` | 세션 경계를 구분 |

박동 원본 CSV를 업로드하는 기능은 아닙니다. 원본 수집·전처리·모델 추론은 별도 파이프라인에서 수행해야 합니다. 품질 필드가 없는 점수만의 파일은 정상으로 꾸미지 않고 유효 입력이 없는 것으로 처리합니다.

세션 변경, 10초 간격 단절, 파일 반복 시 상태 누적을 초기화합니다. 각 세션의 t_rel은 해당 세션 시작 기준으로 기록해야 합니다. 파일 끝에서는 자동 재생을 멈추며, 반복은 체크했을 때만 수행합니다. 기록 재생 중에는 실제 시간의 통신 끊김 검사를 적용하지 않습니다.

### 실시간 연결 대기

실제 연결은 다음처럼 시작합니다.

```bash
.venv-web/bin/python scripts/run_dashboard.py --mode live
```

브라우저의 입력 소스에서도 실시간으로 변경할 수 있습니다. 모델 출력이 오기 전에는 **수신 대기**이며, 마지막 유효 요청 후 20초가 지나면 **수신 지연**으로 표시하고 이전 판정을 숨깁니다. 새 점수가 들어와도 수신이 장시간 끊겼거나 창 시간에 공백이 있으면 누적 판정을 초기화합니다.

## 3. 추론 프로그램 연결 API

별도 추론 프로그램이 **30초 창의 결과를 10초마다 한 번** 보냅니다. 센서의 100Hz 원시 샘플을 이 API로 보내면 안 됩니다. 초기 개인 기준 생성·정규화·품질 계산은 추론 프로그램이 담당합니다. 기존 30초 Baseline 논의와 현행 v2의 1분 대기+5분 기준은 이 화면에서 임의로 바꾸지 않습니다.

```python
import json
from urllib.request import Request, urlopen

window = {
    "t": 30.0,            # 창 끝 시각. 다음 요청은 40.0, 50.0, ...
    "p_ep": 0.24,         # 학생 모델 점수. 없으면 None
    "q_ecg": 0.02,        # ECG 이상 간격 비율
    "ok_ecg": True,       # 전처리 품질 검사
    "p_ppg": None,        # 검증된 PPG 대체 경로 점수, 있을 때만 제공
    "q_ppg": None,
    "ok_ppg": False,
    "hr_ecg": 73.0,
    "hr_ppg": 72.0,
    "can_drowsy": None,   # 확인되지 않은 CAN 이벤트를 True로 바꾸지 않음
}
req = Request("http://127.0.0.1:8000/api/window",
              data=json.dumps(window).encode(),
              headers={"Content-Type": "application/json"}, method="POST")
with urlopen(req, timeout=3) as response:
    result = json.load(response)
```

위 코드는 데이터 구조 설명용입니다. 실제 실행에서는 모의 숫자를 실제 추론·품질 출력으로 교체합니다. 점수·품질은 0~1의 유한한 숫자, 유효 여부는 JSON true/false입니다. `t`는 순서대로 증가해야 합니다. 같은 시각을 중복 전송하지 않습니다.

두 센서 모두 신호가 없을 때도 10초마다 `p_ep=null`, `p_ppg=null`, `ok_ecg=false`, `ok_ppg=false`와 새 `t`를 보내면 원본 상태 머신의 60초 유지 후 측정 불가 전이를 확인할 수 있습니다. 요청 자체가 끊긴 통신 문제는 별도로 20초 수신 지연에 표시됩니다.

| 경로 | 역할 |
|---|---|
| `GET /api/status` | 모든 화면이 공유하는 상태·입력·설정·추이·이벤트 |
| `POST /api/window` | 실시간 모드에서 모델 출력 창 수신 |
| `POST /api/control` | `action`: start, pause, step, reset, settings, source |
| `POST /api/replay` | `{ "name": "windows_scores.csv", "csv": "..." }` |

제어 예시: `{ "action":"source", "mode":"live" }`. 설정 예시: `{ "action":"settings", "scenario":"drowsy" }`. 잘못된 값은 JSON 오류 응답으로 반환합니다.

## 4. 부팅 후 자동 실행 (선택)

먼저 수동 실행을 확인한 뒤 `deploy/jberry-dashboard.service.example`의 사용자명과 모든 경로를 실제 Pi에 맞게 수정합니다. 예시의 `pi`가 현재 사용자명과 같다고 가정하면 안 됩니다.

```bash
sudo cp web/dashboard/deploy/jberry-dashboard.service.example /etc/systemd/system/jberry-dashboard.service
sudo systemctl daemon-reload
sudo systemctl enable --now jberry-dashboard
systemctl status jberry-dashboard
journalctl -u jberry-dashboard -f
```

서비스 기본 모드는 live라 실제 점수가 올 때까지 대기합니다. 수동 서버와 같은 포트에서 동시에 실행하지 않습니다. 중지·자동 실행 해제는 `sudo systemctl disable --now jberry-dashboard`입니다.

## 5. 확인 방법

```bash
python -m pytest -q tests/test_web_dashboard.py tests/test_state_machine_v2.py
node --check web/dashboard/app.js  # 개발 PC의 JS 문법 검사. Pi 실행에는 Node 불필요
```

HTTP/API, 입력 검증, CSV 경계, 상태 머신 동작을 테스트합니다. 화면 테스트는 별도 브라우저에서 데모→졸음→신호 불량→회복, CSV, live 대기·수신·지연 순서로 확인합니다.
