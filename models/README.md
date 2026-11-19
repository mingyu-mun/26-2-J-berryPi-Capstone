# 모델 번들 (중간 산출물 2026-11-13)

다른 기기에서 학습 데이터·GPU 없이 추론하기 위한 파일이다. 전부 합쳐 약 1.4 MB 이고 pickle 파일은 없다.

| 파일 | 내용 |
| --- | --- |
| `student/S1-EP_s{0,1,2}.safetensors` | **주 학생** S1-EP (ECG+PPG 30초 파형 + ECG RR 특징), 시드 3개 (각 약 5만 파라미터) |
| `student/S1-E_s{0,1,2}.safetensors` | 비교용 학생 S1-E (ECG만) |
| `teacher/f2c_final.npz` | 교사 F2C 최종 W(8×68)·α·b, 시드 5개 |
| `teacher/nw_pca.npz` | NormWear 768차원 → 32차원 PCA 축 (CLAS·MAUS 로 맞춤) |
| `bundle.json` | RR 특징 규약(spec), 개인 정규화 통계(σ_pop), NormWear σ_pop, 동작점(경보 임계 logit), 파형 규약, NormWear 출처·sha256 |

가중치를 만든 명령은 `scripts/export_bundle.py` (GPU 서버), 근거 결과는 `docs/MILESTONE-2026-11-13.md` 에 있다.

## 학생만 추론 (CPU, 권장)

```sh
pip install -r requirements.txt          # torch, scipy, pandas, safetensors ...
python scripts/infer_bundle.py --session <세션 폴더> --out out/infer
```

- 세션 폴더 형식은 `docs/data-handoff.md` 를 따른다: `meta.json` + `ecg.csv` + `ppg.csv` (또는 `data.npz`).
- 직접 측정한 zip 은 `scripts/convert_jberry_sensor.py` 로 바꾼다. 변환부터 평가까지 전체 절차는 [`docs/selfmeasure.md`](../docs/selfmeasure.md) 에 있다.
- 같은 사람의 세션 여러 개를 넣으면 첫 세션의 기준 구간을 모두에 쓴다.
- 기준 수집 (상태 머신 v2 2장):
  - 첫 세션의 60초 이후에서 ECG 이상 간격 5% 이하인 창 20개를 모아 5분 기준을 만든다.
  - 10분 안에 못 모으면 재측정을 안내하고 끝난다.
  - `--force` 를 주면 0~300초를 기준으로 계속 진행하고, `--baseline A B` 를 주면 기준 구간을 직접 지정한다.
- 결과:
  - `windows.csv`: 창별 학생 logit (EP / PPG 대체 / ECG만), 품질(q_ecg, q_ppg), 상태 머신 단계(level 0 정상, 1 주의, 2 졸음; candidate; channel; frozen)
  - `summary.json`: 세션별 경보율, 주의 이상 시간 비율 등
- 30분 녹음 기준 노트북 CPU 로 약 20초 걸린다.

## 교사 F2C 도 함께 (선택, NormWear 필요)

NormWear 체크포인트는 775 MB 라 저장소에 넣지 않았다 (GitHub 파일 한도 100 MB). 아래처럼 받는다.

```sh
git clone https://github.com/Mobile-Sensing-and-UbiComp-Laboratory/NormWear.git <dir>/NormWear
cd <dir>/NormWear && git checkout 07517fc && cd -
# 체크포인트: https://huggingface.co/mosaic-laboratory/normwear 의 model.safetensors
#   (huggingface-cli download mosaic-laboratory/normwear model.safetensors --local-dir <ckpt_dir>)
python scripts/infer_bundle.py --session <세션 폴더> --out out/infer --teacher \
    --normwear-dir <dir> --ckpt <ckpt_dir>/model.safetensors
```

- 받은 파일의 sha256 이 `bundle.json` 의 `teacher.normwear.sha256` 과 같은지 확인한다.
- 원저자의 `.pth`·`.bin` (pickle) 체크포인트는 쓰지 않는다. safetensors 만 쓴다.
- NormWear 는 Apache 2.0 이다.
- NormWear 는 CPU 에서도 돌지만 30분 녹음에 수 분 걸린다.

## 검증

- 연구실 서버 결과와 같은지 확인했다. 실측 2차 녹음, 기준 0~300초 기준으로 창 112개의 logit 최대 차이:
  - 학생 EP 0.0004, PPG 대체 0.0005, ECG만 0.0004 (CPU 와 GPU 차이)
  - 교사 0.00001

## 라이선스·사용 조건

- 학생·교사 가중치는 WESAD (비상업 연구 목적 공개 데이터) 의 스트레스 구간을 음성으로 써서 학습했다. **비상업 용도로만 쓴다.** 데이터 자체는 저장소에 없다.
- 직접 측정한 생체 데이터는 저장소에 올리지 않는다.
