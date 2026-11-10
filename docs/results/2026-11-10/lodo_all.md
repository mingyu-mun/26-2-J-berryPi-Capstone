# 교사 후보 비교 (LODO)

선택: **rrnet** (AUROC 데이터셋 평균 기준, 차이 0.01 이하면 ECE·단순성)

| 후보 | AUROC (데이터셋 평균) | 균형 정확도 | ECE (평균) | AUROC (전체) | Brier |
| --- | --- | --- | --- | --- | --- |
| rule | 0.484 | 0.464 | 0.227 | 0.352 | 0.253 |
| lgbm | 0.586 | 0.567 | 0.204 | 0.580 | 0.240 |
| lgbm_nopers | 0.601 | 0.574 | 0.205 | 0.624 | 0.227 |
| rrnet | 0.622 | 0.533 | 0.214 | 0.672 | 0.211 |
| ensemble | 0.617 | 0.569 | 0.215 | 0.639 | 0.220 |

규칙 기준선 대비: rrnet 가 AUROC +0.137.

## 데이터셋별 AUROC

| 후보 | advitam | dddb | drozy | mpddf |
| --- | --- | --- | --- | --- |
| rule | 0.452 | 0.449 | 0.321 | 0.715 |
| lgbm | 0.605 | 0.560 | 0.617 | 0.562 |
| lgbm_nopers | 0.612 | 0.633 | 0.552 | 0.607 |
| rrnet | 0.529 | 0.558 | 0.744 | 0.656 |
| ensemble | 0.567 | 0.574 | 0.718 | 0.611 |

## 유발 요인별 AUROC

| 후보 | none | prolonged_driving | sleep_deprivation |
| --- | --- | --- | --- |
| rule | 0.495 | 0.313 | 0.379 |
| lgbm | 0.642 | 0.581 | 0.564 |
| lgbm_nopers | 0.634 | 0.644 | 0.564 |
| rrnet | 0.496 | 0.696 | 0.521 |
| ensemble | 0.549 | 0.653 | 0.550 |

## fold 별 LightGBM 상위 특징 (gain)

- advitam: sdnn_120, sd2_300, sdnn_300, lf_hf_log_300, lf_nu_300, sd2_120, sd1_300, z_rr_mean_300
- dddb: sd1_300, z_hr_300, sampen_300, rmssd_300, trend_hr_slope, pnn50_300, sd2_300, hr_300
- drozy: sdnn_300, sd2_300, z_sd2_300, hf_log_300, lf_log_300, hr_300, z_sd1_sd2_300, sdnn_120
- mpddf: sdnn_300, z_sd1_sd2_300, sd2_300, hr_300, z_pnn50_300, trend_hr_slope, hr_120, z_rr_mean_300
