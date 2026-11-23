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

## 연구실 PC 원본 vs main 재구성 (2026-11-23 확인)

main 의 `scripts/convert_jberry_sensor.py`·`scripts/selfmeasure_eval.py` 는 PR #12 에서 다른 PC 가 결과 수치로 역추정해 만든 재구성이다.
나머지 코드 (`infer_bundle.py`, `baseline_exp_rec.py`, `src/`, `models/`) 는 이 PC 에서 쓴 원본과 같다. 위 rec4 절은 재구성 코드로 돌린 결과다.

**원 변환기 (rec4)**: `scripts/convert_rec4_orig.py`. 11-18 에 일회성 코드로 실행해 파일이 없었고, 그날 Claude 대화 기록에서 복구했다.
계산은 그대로, 입출력 경로만 인자로 받는다. 이 PC 의 원 변환본과 `data.npz`(ecg·ppg 배열)·`meta.json`·`labels_orig.csv` 가 모두 똑같다.

```sh
python scripts/convert_rec4_orig.py --raw <풀어 둔 drive-download-20261118...zip 폴더> --out data/orig
python scripts/baseline_exp_rec.py  --self-dir data/orig/jberry_sensor/self01 --out out/rec4_orig   # 원 평가 (rec1·rec2 는 없으면 건너뜀)
python scripts/infer_bundle.py      --session data/orig/jberry_sensor/self01/2026-11-18_190640_s1 --out out/infer_orig
```

- `orig/rec_results.csv`: 원 평가를 원 변환본에 돌린 결과. 11-18 커밋 결과 (`docs/results/2026-11-18-baseline/rec_results.csv`) 와 **값이 똑같다**
  (rec4 F5 0.629 / F10 0.731 / A5 0.715 / A10 0.788). `baseline_exp_rec.py` 는 경로 인자 `--self-dir` 와 없는 세션 건너뛰기만 더했다.
- `orig/infer_rec4/`: 원 변환본에 `infer_bundle.py` (창 161, 유효 148, `dev_core` 포함).
- 재구성 변환기와 차이: 신호 값은 float 반올림 수준 (ECG 최대 0.0004 mV), 끝이 ECG 64·PPG 1 표본 짧다. 그래도 평가 창이 142 → 132 개로 달라
  AUROC 가 0.001~0.005 낮다. 원 결과를 재현하려면 `convert_rec4_orig.py` 를 쓴다.
- rec1~rec3 (11-12·11-15) 의 원 변환 코드는 아직 복구하지 않았다. 실측 신호 파일은 레포에 넣지 않는다 (사용자 결정 2026-11-23).

## 레포에 넣지 않은 것

| 항목 | 위치 (연구실 PC) | 이유 |
| --- | --- | --- |
| 실측 원 녹음·변환본 (rec1~rec4) | `Downloads/datasets/selfmeasure`, 각 zip | 본인 생체 신호, 공개 레포 (사용자 결정: 넣지 않음) |
| 공개 데이터 원본·변환본·캐시 (DROZY, MAUS, AdVitam, CLAS, WESAD, UL-DD 등) | `Downloads/datasets`, `code/work/teacher/cache*` (약 0.8 GB) | 데이터셋 라이선스상 재배포 금지 |
| 교사 v1 번들 (rrnet) | `code/work/teacher/teacher_bundle` | 최종 추론에 쓰지 않음 (현재 교사·학생은 `models/`) |
| NormWear 가중치 | HF `mosaic-laboratory/normwear` | 775 MB, 원 배포처에서 받음 (`models/README.md`) |
| PhysDrive 추론 결과 (`windows.csv`) | 연구실 서버 `~/jberry/out/physdrive` | 이 PC 에 없음. PhysDrive 원본은 재배포 금지라 결과 표만 `2026-11-17-physdrive` 에 있음 |
