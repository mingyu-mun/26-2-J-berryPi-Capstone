# 재현 기준 출력 (2026-11-23, 연구실 PC DESKTOP-5QLNVTK, CPU)

다른 PC 에서 같은 추론·평가가 나오는지 비교하기 위한 기준 결과다. 표준 변환기 (`scripts/convert_jberry_sensor.py`,
11-18 연구실 PC 원 변환 코드를 복구해 확정) 와 `models/` 가중치로, GPU·서버 없이 이 PC CPU 에서 돌렸다.
환경은 `pip-freeze-cpu.txt` (Python 3.11.9, torch 2.15.0+cpu).

## rec4 (2026-11-18 실측, 5분 라벨 정,정,졸,졸,졸,정,정)

입력: `drive-download-20261118T124939Z-1-001.zip` (sha256 앞 16자 `fe8a00e12771b4c0`). 직접 측정한 원본이라 레포에 넣지 않았다.

```sh
python scripts/convert_jberry_sensor.py --raw drive-download-20261118T124939Z-1-001.zip --out data/converted --labels-array
S=data/converted/jberry_sensor/self01/2026-11-18_190640_s1
python scripts/infer_bundle.py     --session $S --out out/infer
python scripts/selfmeasure_eval.py --session $S --out out/selfmeasure_eval
python scripts/baseline_exp_rec.py --self-dir data/converted/jberry_sensor/self01 --out out/baseline_exp_rec   # rec1·rec2 는 없으면 건너뜀
```

- 변환: `--labels-array` 없이 돌리면 연구실 PC 원 변환본과 `data.npz`·`meta.json`·`labels_orig.csv` 가 모두 같다 (확인함).
  `--labels-array` 는 1 Hz 라벨 배열만 더한다 (신호 동일).
- `rec4/infer/`: `windows.csv` (창 161, 유효 148, 학생 logit 4종, 규칙 점수 `dev_core`, 상태 머신 단계), `summary.json`
  (기준 자동 선택, 학생 경보율 61.5%). 원 변환본에 돌린 결과와 파일이 같다.
- `rec4/selfmeasure_eval/`: 학생 S1-EP 창 AUROC F5 0.629 / F10 0.731 / A5 0.715 / A10 0.788.
- `rec4/baseline_exp_rec/`: 11-18 원 평가 스크립트 결과. `docs/results/2026-11-18-baseline/rec_results.csv` 와 값이 같다.

다른 PC 에서는 위 명령을 돌린 뒤 이 폴더의 파일과 비교하면 된다 (부동소수 끝자리 차이는 허용).

## 원본 확인 기록

- PR #12 의 변환기는 다른 PC 가 결과 수치로 역추정한 것이었다. 2026-11-23 에 11-18 대화 기록에서 실제 변환 코드를 찾아 표준으로 바꾸고
  역추정본은 폐기했다. 역추정본은 끝 표본이 ECG 64·PPG 1 개 짧아 평가 창이 142 → 132 개, AUROC 가 0.001~0.005 낮았다.
- `infer_bundle.py`, `baseline_exp_rec.py`, `src/`, `models/` 는 이 PC 에서 쓴 원본과 같다. `baseline_exp_rec.py` 에는
  경로 인자 `--self-dir` 와 없는 세션 건너뛰기만 더했다. `selfmeasure_eval.py` (PR #12) 는 표준 변환기 출력에서 위 원 결과와 같은 값을 낸다.
- rec1~rec3 (11-12·11-15) 의 원 변환 코드는 복구하지 않았다 (측정 앱 출력 형식이 달라 표준 변환기 대상이 아님).

## 레포에 넣지 않은 것

| 항목 | 위치 (연구실 PC) | 이유 |
| --- | --- | --- |
| 실측 원 녹음·변환본 (rec1~rec4) | `Downloads/datasets/selfmeasure`, 각 zip | 본인 생체 신호, 공개 레포 (사용자 결정: 넣지 않음) |
| 공개 데이터 원본·변환본·캐시 (DROZY, MAUS, AdVitam, CLAS, WESAD, UL-DD 등) | `Downloads/datasets`, `code/work/teacher/cache*` (약 0.8 GB) | 데이터셋 라이선스상 재배포 금지 |
| 교사 v1 번들 (rrnet) | `code/work/teacher/teacher_bundle` | 최종 추론에 쓰지 않음 (현재 교사·학생은 `models/`) |
| NormWear 가중치 | HF `mosaic-laboratory/normwear` | 775 MB, 원 배포처에서 받음 (`models/README.md`) |
| PhysDrive 추론 결과 (`windows.csv`) | 연구실 서버 `~/jberry/out/physdrive` | 이 PC 에 없음. PhysDrive 원본은 재배포 금지라 결과 표만 `2026-11-17-physdrive` 에 있음 |
