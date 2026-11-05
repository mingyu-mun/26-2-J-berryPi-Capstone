"""ECG 기반 정상/졸음 교사 모델 (오프라인 라벨러).

교사는 공개 데이터(AdVitam Exp4, MPD-DF, DD-Database, DROZY)의 KSS·피로 라벨로 학습하고,
라벨이 없는 ECG 데이터의 30초 창(10초 이동)마다 P(졸음)을 붙인다. 상태 머신과는 무관하다.
설계: docs/teacher.md
"""
