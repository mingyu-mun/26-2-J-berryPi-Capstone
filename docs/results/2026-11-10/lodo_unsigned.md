# 교사 후보 비교 (LODO)

선택: **lgbm_nopers** (AUROC 데이터셋 평균 기준, 차이 0.01 이하면 ECE·단순성)

| 후보 | AUROC (데이터셋 평균) | 균형 정확도 | ECE (평균) | AUROC (전체) | Brier |
| --- | --- | --- | --- | --- | --- |
| rule | 0.484 | 0.464 | 0.227 | 0.352 | 0.253 |
| lgbm | 0.586 | 0.567 | 0.204 | 0.580 | 0.240 |
| lgbm_signabs | 0.592 | 0.594 | 0.200 | 0.587 | 0.236 |
| lgbm_abs | 0.606 | 0.574 | 0.211 | 0.595 | 0.232 |
| lgbm_dev | 0.598 | 0.591 | 0.207 | 0.656 | 0.228 |
| lgbm_nopers | 0.601 | 0.574 | 0.205 | 0.624 | 0.227 |

규칙 기준선 대비: lgbm_nopers 가 AUROC +0.117.

## 데이터셋별 AUROC

| 후보 | advitam | dddb | drozy | mpddf |
| --- | --- | --- | --- | --- |
| rule | 0.452 | 0.449 | 0.321 | 0.715 |
| lgbm | 0.605 | 0.560 | 0.617 | 0.562 |
| lgbm_signabs | 0.612 | 0.522 | 0.663 | 0.570 |
| lgbm_abs | 0.632 | 0.518 | 0.637 | 0.637 |
| lgbm_dev | 0.628 | 0.535 | 0.602 | 0.628 |
| lgbm_nopers | 0.612 | 0.633 | 0.552 | 0.607 |

## 유발 요인별 AUROC

| 후보 | none | prolonged_driving | sleep_deprivation |
| --- | --- | --- | --- |
| rule | 0.495 | 0.313 | 0.379 |
| lgbm | 0.642 | 0.581 | 0.564 |
| lgbm_signabs | 0.622 | 0.584 | 0.595 |
| lgbm_abs | 0.644 | 0.591 | 0.572 |
| lgbm_dev | 0.620 | 0.664 | 0.607 |
| lgbm_nopers | 0.634 | 0.644 | 0.564 |

## fold 별 LightGBM 상위 특징 (gain)

- advitam: sdnn_120, sd2_300, sdnn_300, lf_hf_log_300, lf_nu_300, sd2_120, sd1_300, z_rr_mean_300
- dddb: sd1_300, z_hr_300, sampen_300, rmssd_300, trend_hr_slope, pnn50_300, sd2_300, hr_300
- drozy: sdnn_300, sd2_300, z_sd2_300, hf_log_300, lf_log_300, hr_300, z_sd1_sd2_300, sdnn_120
- mpddf: sdnn_300, z_sd1_sd2_300, sd2_300, hr_300, z_pnn50_300, trend_hr_slope, hr_120, z_rr_mean_300
