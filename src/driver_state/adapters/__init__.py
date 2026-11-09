"""공개 데이터 원본 → 전달 규약 폴더 (docs/data-handoff.md) 어댑터.

각 모듈의 convert(src, out, ...) 가 data/<dataset_id>/<subject>/<session>/ 를 쓰고
쓴 세션 수와 메모를 담은 dict 를 돌려준다. 원본 폴더는 읽기만 한다.

| 모듈 | 용도 | 채널 | 라벨 |
| --- | --- | --- | --- |
| drozy | 교사 학습 | ecg | KSS → 이진 (LabelRule.kss) |
| mand | 교사 라벨링 대상 (학생) | ecg, ppg | 없음 (감정 시나리오는 labels_orig) |
| maus | 교사 라벨링 대상 (학생) | ecg, ppg | 없음 (n-back 수준은 labels_orig) |
"""
from . import drozy, mand, maus  # noqa: F401

ADAPTERS = {"drozy": drozy, "mand": mand, "maus": maus}
