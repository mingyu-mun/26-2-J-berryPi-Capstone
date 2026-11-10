# 교사 후보 비교 (LODO)

선택: **lgbm_abs** (AUROC 데이터셋 평균 기준, 차이 0.01 이하면 ECE·단순성)

| 후보 | AUROC (데이터셋 평균) | 균형 정확도 | ECE (평균) | AUROC (전체) | Brier |
| --- | --- | --- | --- | --- | --- |
| rule | 0.529 | 0.512 | 0.198 | 0.525 | 0.249 |
| lgbm | 0.583 | 0.577 | 0.192 | 0.616 | 0.238 |
| lgbm_abs | 0.598 | 0.578 | 0.197 | 0.580 | 0.239 |
| lgbm_nopers | 0.601 | 0.574 | 0.205 | 0.624 | 0.227 |
| rrnet | 0.571 | 0.544 | 0.253 | 0.462 | 0.321 |

규칙 기준선 대비: lgbm_abs 가 AUROC +0.069.

## 데이터셋별 AUROC

| 후보 | advitam | dddb | drozy | mpddf |
| --- | --- | --- | --- | --- |
| rule | 0.535 | 0.536 | 0.398 | 0.647 |
| lgbm | 0.607 | 0.586 | 0.544 | 0.596 |
| lgbm_abs | 0.627 | 0.639 | 0.558 | 0.567 |
| lgbm_nopers | 0.612 | 0.633 | 0.552 | 0.607 |
| rrnet | 0.502 | 0.582 | 0.600 | 0.602 |

## 유발 요인별 AUROC

| 후보 | none | prolonged_driving | sleep_deprivation |
| --- | --- | --- | --- |
| rule | 0.555 | 0.541 | 0.483 |
| lgbm | 0.632 | 0.635 | 0.561 |
| lgbm_abs | 0.605 | 0.595 | 0.596 |
| lgbm_nopers | 0.634 | 0.644 | 0.564 |
| rrnet | 0.489 | 0.471 | 0.506 |

## fold 별 LightGBM 상위 특징 (gain)

- advitam: sdnn_120, lf_nu_300, sd2_300, sd1_300, r_rmssd_300, hr_120, r_hr_120, r_sd2_120
- dddb: sd1_300, sampen_120, hr_120, sd1_sd2_120, z_rmssd_300, z_sd1_sd2_300, z_hr_300, sd2_300
- drozy: hr_300, sdnn_300, hf_log_300, sd2_300, lf_log_300, pnn50_300, rr_mean_300, sdnn_120
- mpddf: sdnn_300, sd2_300, hr_300, sampen_300, lf_log_300, rmssd_300, sampen_120, trend_hr_slope
