# CAN 속도 기반 개인 감속 탐지

BLF 또는 ASC에서 프로젝트의 속도 신호를 복호화한다. 알려진 감속 이벤트로 개인 기준을 만들고 별도의 로그에서 경고 시각을 찾는 Python 3.10 이상 CLI다. 이 폴더에서 아래 순서로 실행한다.

```powershell
python -m pip install -r requirements.txt
python deceleration.py fit samples/baseline.blf samples/baseline_windows.json baseline.json
python deceleration.py replay samples/strong_decel.blf baseline.json
python deceleration.py replay samples/usual_decel.blf baseline.json
```

## 비교 규칙

- 기준 로그의 경계 파일에서 감속 구간을 읽고, 각 구간의 최대 1초 평균 감속을 계산한다. 감속 구간 검출은 기준용 경계 파일을 제공하는 방식이며 실시간 탐지에는 경계 파일을 사용하지 않는다.
- 기준 속도 35·50·65 km/h에서 ±5 km/h 범위를 사용한다. 각 속도대에 유효 감속이 4회 이상 있으면 최대 관측값에 0.45 m/s²를 더한다. 부족한 속도대는 경계가 없다고 저장한다.
- 주행 중에는 1초 전과 현재의 속도 차이를 3.6으로 나눈다. 표본이 정확히 1초 전에 없으면 구간 안에서 보간한다. CAN 간격이 허용 범위를 벗어나면 판정 상태를 초기화하고 다시 1초가 확보될 때까지 기다린다.
- 경계 초과가 시작한 시각, 비교 속도대, 감속값과 경계값을 JSON으로 출력한다. 기준이 매칭되지 않은 감속 표본 수도 제공한다.

동봉한 기준 및 평가 로그는 기존 합성 감속 실험의 표본이다. 강한 감속은 20.9초에 경고 1회, 평소 감속은 0회다. 속도 정의인 CAN ID 0x102와 바이트 배치는 이 프로젝트 로그 기준이며, 다른 차종에 적용할 때는 차량의 신호 정의를 확인해야 한다.
