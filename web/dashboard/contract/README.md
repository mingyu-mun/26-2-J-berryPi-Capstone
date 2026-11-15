# J_Berry PI 웹보드 통합 데이터 규약 v1.2

업로드된 **`Can list.xlsx`의 `CAN DB` 시트(B2:P27, 25개 신호)**를 직접 검토하여 CAN 신호 필드와 화면 표시 규칙을 보강한 버전입니다. 기존 v1.1의 ECG/PPG, S1-EP, `StateMachineV2` 데이터 구조는 유지합니다.

## 파일

- `WEBBOARD_DATA_CONTRACT.md`: 전체 데이터 구조 및 화면 데이터 필드/통신/검증 규칙.
- `CAN_MAPPING_FROM_XLSX.md`: Excel에서 확인한 CAN ID, 주기, bit 위치, 단위/배율, 오류값 및 미확정 항목.
- `can_signal_catalog.json`: 25개 신호의 엑셀 원본 열값 전사 데이터 (신호 설명도 원문 보존). **측정 CSV가 아님**.
- `webboard_event_v1_2.schema.json`: JSON Schema Draft 2020-12, `can.telemetry` 확장.
- `types.ts`: JSON 메시지 TypeScript 타입.
- `empty_status_template.json`: 수치 미포함 대기 상태 메시지.

## 중요한 수정

1. 차량 속력: `0x102` `Speed` start56 len8 Unsigned factor1 offset0, km/h (100ms).
2. 조향각: `0x125` `SAS_angleVal` start24 len16 Signed factor0.1. **핸들 각도**. raw 32767 오류값 제외 (10ms). 바퀴각 핸들각÷15는 원본 설명의 **유도값**.
3. 가속/브레이크: `0x371` `Accel_Pedal_Pos`, `Brake_Pedal_Pos`, 이벤트 신호. raw 범위(0..254 / 0..127)를 우선 제공. Factor 1과 Unit 공란이므로 **% 환산은 검증 전 미표시**.
4. 방향지시등, 차선안전, 핸들 파지, 부주의 경고 표시 등 나머지 21개 상태/보조신호를 원본 Signal name 기준 `status_codes` 선택 필드로 추가.
5. 속력(100ms)·조향(10ms)·페달(이벤트) 수신 주기가 다르므로 `signal_times_s` 필수화.
6. `HOD_Dir_Status` StartBit 80, Length 3은 **최소 11바이트 프레임** 필요. CAN FD 장비 호환 확인 필수.
7. `Careless_Warn`은 경고 **표시 신호**이며 졸음 정답 데이터가 아니다. CAN-ID 매핑만으로 `can_drowsy` 규칙이 완성되지 않는다.

## 근거와 한계

- CAN: 사용자 업로드 `Can list.xlsx` (표의 내용은 전사했으나 **실제 차종/펌웨어/CAN 프레임 디코드로 검증하지 않음**).
- 모델: GitHub `mingyu-mun/26-2-J-berryPi-Capstone`의 `state_machine_v2.py`, `selfmeasure_infer.py`, 전처리 코드.
- 서버·웹소켓·합성 주행 데이터·실측 생체신호 자체는 이 파일 묶음에 포함되지 않음.
- 엑셀에 존재하지 않는 CAN RPM·차선 편차값을 임의로 만들지 않음.

## 검증

```bash
python -m pip install jsonschema
python -c "import json; from pathlib import Path; from jsonschema import Draft202012Validator; p=Path('.'); schema=json.loads((p/'webboard_event_v1_2.schema.json').read_text()); Draft202012Validator.check_schema(schema); data=json.loads((p/'empty_status_template.json').read_text()); Draft202012Validator(schema).validate(data); print('PASS')"
```
