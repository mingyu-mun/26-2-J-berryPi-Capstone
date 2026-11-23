# 재현 기준 출력 (2026-11-23, 연구실 PC DESKTOP-5QLNVTK, CPU)

다른 PC 에서 같은 추론이 나오는지 비교하기 위한 기준 결과다. 모두 `main` (12d5ce0) 의 코드와 `models/` 가중치로,
GPU·서버 없이 이 PC CPU 에서 다시 돌렸다. 환경은 `pip-freeze-cpu.txt` (Python 3.11.9, torch 2.15.0+cpu).

## rec4 (2026-11-18 실측, 라벨 정,정,졸,졸,졸,정,정)

입력: `drive-download-20261118T124939Z-1-001.zip` (sha256 앞 16자 `fe8a00e12771b4c0`). 직접 측정한 원본이라 레포에 넣지 않았다.

```sh
python scripts/convert_jberry_sensor.py --raw drive-download-20261118T124939Z-1-001.zip --out data/converted
python scripts/infer_bundle.py   --session data/converted/jberry_sensor/self/2026-11-18_190640 --out out/infer
python scripts/selfmeasure_eval.py --session data/converted/jberry_sensor/self/2026-11-18_190640 --out out/selfmeasure_eval
```

- 변환: 2040초, fs ECG 435.33 / PPG 100.66 Hz, 세션 0초 = 원 녹음 60.000초.
- `rec4/infer/`: `windows.csv` (창 160개, 유효 147, 학생 logit 4종, 규칙 점수 `dev_core`, 상태 머신 단계), `summary.json`
  (기준 60~450초 자동 선택, 학생 경보율 61.9%, 주의 이상 84.4%).
- `rec4/eval/`: `results.csv` + 조건별 창 점수. 학생 S1-EP 창 AUROC F5 0.628 / F10 0.726 / A5 0.714 / A10 0.785,
  규칙 dev_core F5 0.340 / F10 0.744. `docs/selfmeasure.md` 5장 표의 "11-18 기본값" 행과 같다.

다른 PC 에서는 위 명령을 돌린 뒤 `windows.csv`·`results.csv` 를 이 폴더와 비교하면 된다 (부동소수 끝자리 차이는 허용).

## 레포에 넣지 않은 것

| 항목 | 위치 (연구실 PC) | 이유 |
| --- | --- | --- |
| 실측 원본·변환본 (rec1~rec4) | `Downloads/datasets/selfmeasure`, 각 zip | 본인 생체 신호, 공개 레포 (docs/selfmeasure.md 2장 규칙) |
| 공개 데이터 원본·변환본·캐시 (DROZY, MAUS, AdVitam, CLAS, WESAD, UL-DD 등) | `Downloads/datasets`, `code/work/teacher/cache*` (약 0.8 GB) | 데이터셋 라이선스상 재배포 금지 |
| 교사 v1 번들 (rrnet) | `code/work/teacher/teacher_bundle` | 최종 추론에 쓰지 않음 (현재 교사·학생은 `models/`) |
| NormWear 가중치 | HF `mosaic-laboratory/normwear` | 775 MB, 원 배포처에서 받음 (`models/README.md`) |
| PhysDrive 추론 결과 (`windows.csv`) | 연구실 서버 `~/jberry/out/physdrive` | 이 PC 에 없음. PhysDrive 원본은 재배포 금지라 결과 표만 `2026-11-17-physdrive` 에 있음 |
