# CAN Edge 웹 데모와 라즈베리 파이 실시간 CAN 변환기

## 브라우저에서 개인 조향 기준 확인

[`index.html`](index.html)을 브라우저로 열면 파이, Python, 서버 없이 실행됩니다. 회사 매핑으로 가상 CAN 바이트를 해석하고, 가상의 운전자 A·B 정상 주행을 각각 두 번 생성해 개인 기준을 만든 뒤 같은 평가 주행에 적용합니다.

- **0.65초**를 선택하면 100 km/h에서 +5.2° 조향을 유지했을 때 A만 이탈로 표시됩니다.
- **1.20초**를 선택하면 A와 B 모두 이탈로 표시됩니다. 슬라이더에서 다른 유지 시간을 정한 뒤 **가상 주행 실행**을 눌러도 됩니다.
- 조향 그래프, 각 프로필의 생성 기준과 첫 이탈 시각, 원본 CAN 바이트와 디코딩 값을 확인할 수 있습니다. 웹 데모의 운전자 A·B는 합성 프로필입니다. 별도 문서의 Unity P01·P02는 해당 실험 기록입니다.

GitHub에서 웹으로 공유하려면 이 폴더의 내용이 저장소 루트에 오도록 올리고, **Settings → Pages**에서 게시할 브랜치와 `/(root)`를 선택합니다. 이 폴더를 다른 저장소의 하위 폴더로 올렸다면 Pages가 그 하위 폴더를 제공하는지 확인한 뒤 페이지 주소에 해당 폴더 경로를 붙입니다. [GitHub Pages 게시 설정](https://docs.github.com/en/pages/getting-started-with-github-pages/configuring-a-publishing-source-for-your-github-pages-site)

브라우저 데모는 회사 CAN 매핑과 파이 코드의 조향 판정을 JavaScript로 구현합니다. 아래 파이 실행 절차와 Python 테스트도 이 폴더에 있습니다.

개발 환경에 Python 의존성과 Node.js가 있다면 `python tests/check_web_parity.py`로 웹과 Python의 기준 및 이탈 시각을 비교할 수 있습니다.

## 라즈베리 파이 CAN 변환기

`can_live.py`를 켜 두면 SocketCAN으로 들어오는 프레임에서 속도와 핸들 조향각을 디코딩하고, 실시간으로 웹보드 `can.telemetry` v1.2 이벤트를 출력합니다. 이 폴더만 복사해서 사용합니다. `--can-list`는 `Can list.xlsx`의 `CAN DB` 7~8행에서 추출한 두 신호의 매핑을 사용합니다. [SocketCAN 인터페이스 문서](https://python-can.readthedocs.io/en/stable/interfaces/socketcan.html)

## 1. 노트북에서 먼저 확인하기

프로젝트 폴더가 없는 노트북에서는 Python 환경을 아래처럼 만듭니다. Windows PowerShell에서 이 README가 있는 폴더를 연 뒤 실행합니다.

```powershell
py -3.12 -m venv .venv
& .\.venv\Scripts\python.exe -m pip install -r requirements.txt
& .\.venv\Scripts\python.exe .\highway_demo.py
```

고속도로 99~100 km/h 주행을 가정해 회사 표의 ID와 주기로 가상 CAN 프레임을 보냅니다. 작은 좌우 보정, +5° 조향을 약 1.2초 동안 유지, 조향 오류 코드, 조향 입력 약 0.5초 중단, 전체 신호 중단을 확인합니다. 데모 이벤트 출처는 `simulated`입니다.

- [실시간 출력 JSONL](examples/highway_can_events.jsonl)은 웹보드의 **기록 파일**에서 바로 엽니다.
- [가상 CAN 원본 BLF](examples/highway_demo.blf)에는 CAN ID와 데이터 바이트가 들어 있습니다. [속도·조향 매핑](mapping/can_list.dbc)은 원본 표에서 추출한 두 신호의 DBC입니다. 새 BLF를 만들려면 `highway_demo.py --blf 새이름.blf`로 실행합니다.

## 2. 라즈베리 파이에서 실제 CAN 받기

Raspberry Pi OS에서 차량용 CAN 어댑터가 SocketCAN 장치로 준비되어 있어야 합니다. `ip -details link show can0`으로 실제 장치 이름과 상태를 확인합니다. 차량의 비트레이트와 CAN FD 여부는 어댑터 설정에 따릅니다. 이 스크립트는 들어오는 프레임을 읽으며 인터페이스 설정을 변경하지 않습니다.

```bash
cd can_edge
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt

.venv/bin/python can_live.py \
  --channel can0 --can-list \
  --session-id drive-01 | tee can_events.jsonl
```

차량의 어댑터가 CAN FD로 설정되어 있으면 `--fd`를 붙입니다. 표에는 전체 DLC, 프레임 형식, 실제 버스 채널이 명시되어 있지 않습니다. 이 매핑은 표준 ID를 사용하고 속도는 8바이트부터, 조향은 5바이트부터 읽습니다. 조향 프레임에 뒤따르는 바이트는 허용합니다. 다른 차량 매핑을 쓰려면 `--can-list` 대신 `--dbc /path/to/vehicle.dbc --speed 메시지.신호 --steering 메시지.신호`를 지정합니다.

- 이벤트는 한 줄에 한 개씩 표준 출력으로 나옵니다. `tee`를 쓰면 동시에 파일에 남습니다. 켜 둔 상태에서 CAN이 오면 바로 처리하고, `Ctrl+C`로 종료합니다.
- 실행 명령에서 파이프 앞에 `--webboard http://127.0.0.1:8000`을 추가하면 같은 이벤트를 웹보드의 `/api/events`로 보냅니다. 실제 웹보드 주소를 사용합니다. HTTP 전송은 수신과 별개로 실행하고, 실패·대기열 초과 건수는 종료할 때 표준 오류에 표시합니다. 표준 출력에는 JSONL만 나옵니다.
- CAN 단독 실행의 0초는 수신기 시작 시간입니다. ECG·PPG와 묶을 때는 `--session-id`를 동일하게 맞추고, `--session-start-unix-s`에 생체 세션 시작의 UNIX 초를 지정합니다. CAN 어댑터의 수신 시각과 생체 기록 시각은 공통 시계 기준이어야 합니다.

## 3. 운전자별 정상 주행 수집 → 기준 생성 → 실시간 판정

회사 매핑 모드에서 같은 운전자의 정상 직선 주행을 다른 세션 ID로 두 번 기록합니다. 각 기록에 95~100 km/h 조건을 만족하는 구간이 20초 이상, 합계 60초 이상 필요합니다. 아래 예시는 여유를 두고 45초씩 수집합니다. 차로 위치 입력 없이 동작하므로 수집을 시작하는 사람이 정상 직선 구간을 지정합니다.

```bash
.venv/bin/python can_live.py --can-list --channel can0 --collect-normal P01 --session-id p01-normal-01 --duration-s 45
.venv/bin/python can_live.py --can-list --channel can0 --collect-normal P01 --session-id p01-normal-02 --duration-s 45
.venv/bin/python steering_profile.py --participant P01 p01-normal-01 p01-normal-02
.venv/bin/python can_live.py --can-list --channel can0 --assess P01 --session-id p01-assess-01
```

다른 운전자는 운전자 ID를 바꿔 같은 절차를 진행합니다. CAN FD 인터페이스에서는 `can_live.py` 명령에 `--fd`를 추가합니다. 수집은 지정 시간 경과 또는 `Ctrl+C`로 정상 종료합니다. 수집·기준 생성·판정 모두 `--data-dir`로 같은 데이터 위치를 지정할 수 있습니다.

| 단계 | 저장 위치 |
|---|---|
| 정상 기록 | `data/normal/P01/p01-normal-01.jsonl` 등 |
| 개인 기준 | `data/profiles/P01.json` |
| 실시간 조향 판정 | `data/decisions/P01/p01-assess-01.jsonl` |

기준은 각도 구간 1·2·5·10°별로 완료된 조향 구간을 최소 5개 확보했을 때, 유지시간의 95백분위에 0.10초를 더해 생성합니다. 판정은 같은 방향의 조향이 개인 기준을 넘고 Unity 차량 모델의 중앙 출발 경계 도달 추정 시간의 절반 이상 지속됐을 때 `deviation`을 기록합니다. 기록에는 사용한 각도 구간, 실제 핸들각과 속도, 유지시간과 기준 시간이 들어갑니다. 그 외에는 `normal`, `out_of_scope` 또는 `unavailable` 상태를 기록합니다. 조향 이탈은 표준 오류에도 표시합니다.

장치 없이 전체 경로를 재현하려면 프로젝트 폴더에서 `python -m unittest discover -s tests -v`를 실행합니다. 가상 CAN으로 두 운전자의 정상 기록을 각각 수집하고, 생성한 두 프로필에 같은 평가 프레임을 넣습니다. 5° 기준은 A 0.30초, B 0.90초였습니다. +5.2°를 0.65초 유지하면 A에만 이탈이 기록되고, 1.20초 유지하면 둘 다 이탈이 기록됩니다. 속도 또는 조향이 끊기면 판정 상태가 `unavailable`로 바뀝니다.

## 출력과 판정 규약

선택한 CAN 프레임마다 디코딩합니다. 개인 판정은 조향 프레임이 들어올 때 평가하고, 웹보드에는 바뀐 결과를 초당 최대 10회 전달합니다. 회사 매핑의 유효 시간은 속도 120ms, 조향 80ms입니다. 속도는 표의 100ms 주기에 수신 여유 20ms를 더했고, 조향 프레임 간격이 80ms를 넘으면 지속시간을 다시 셉니다. `signal_times_s`에는 원래 관측 시각을 유지하고, 기한이 지난 값과 시각은 `null`로 표시합니다. 실차 수신의 `origin`은 `live`입니다.

| 회사 신호 | CAN ID | 표의 주기 | 디코딩 |
|---|---|---|---|
| `CARINFO.Speed` | `0x102` | 100ms | 시작 비트 56, 8비트 unsigned little-endian, 원시값 그대로 km/h |
| `SAS_FD_01_10ms.SAS_angleVal` | `0x125` | 10ms | 시작 비트 24, 16비트 signed little-endian, 원시값 × 0.1°, 원시값 32767은 `null` |

개인 조향 결과는 위의 판정 파일에 저장합니다. `--assess` 중에는 웹보드용 `can.telemetry`에도 직전 출력 이후의 판정을 `deviation_status`(`within_baseline`/`outside_baseline`, 판정 불가는 `unavailable`)와 `rule_id=steer_hold_personal_v1`로 싣습니다. 짧은 이탈도 다음 출력에 남깁니다. `can_drowsy`는 `--origin simulated`(Unity 등 시뮬레이터 CAN)일 때만 채우고, 실차(`live`)는 `null`을 유지합니다. 생체 졸음 판정과의 결합은 웹보드 서버의 `--can-sim`을 참고하세요(`web/dashboard/README.md`). 현재 결합 규칙은 시뮬레이터 차량 경고(차선·부주의, DBC `mapping/can_list_warnings.dbc`)가 기본이고 이 조향 규칙은 선택 조건입니다.

## Unity 개인 조향 기준 검증

Unity 직선 주행 12회의 기록을 P01·P02 프로필에 각각 적용했습니다. 같은 조향 기록에서 판정이 달라진 조건과, 시험용 졸음 상태가 조향 이탈과 겹칠 때 추가 경보가 발생한 결과를 정리했습니다. [실험 조건과 결과](docs/unity_personal_steering_validation.md)
