# 교사 후보 비교 (LODO)

선택: **rrnet** (AUROC 데이터셋 평균 기준, 차이 0.01 이하면 ECE·단순성)

| 후보 | AUROC (데이터셋 평균) | 균형 정확도 | ECE (평균) | AUROC (전체) | Brier |
| --- | --- | --- | --- | --- | --- |
| lgbm | 0.586 | 0.567 | 0.204 | 0.580 | 0.240 |
| lgbm_abs | 0.606 | 0.574 | 0.211 | 0.595 | 0.232 |
| lgbm_mspc | 0.582 | 0.568 | 0.195 | 0.603 | 0.224 |
| lgbm_abs_mspc | 0.597 | 0.578 | 0.198 | 0.629 | 0.221 |
| rrnet | 0.622 | 0.533 | 0.214 | 0.672 | 0.211 |
| rrnet_mspc | 0.612 | 0.538 | 0.224 | 0.625 | 0.223 |
| rrnet_dev | 0.610 | 0.525 | 0.224 | 0.592 | 0.223 |

## 데이터셋별 AUROC

| 후보 | advitam | dddb | drozy | mpddf |
| --- | --- | --- | --- | --- |
| lgbm | 0.605 | 0.560 | 0.617 | 0.562 |
| lgbm_abs | 0.632 | 0.518 | 0.637 | 0.637 |
| lgbm_mspc | 0.609 | 0.583 | 0.585 | 0.552 |
| lgbm_abs_mspc | 0.635 | 0.538 | 0.633 | 0.584 |
| rrnet | 0.529 | 0.558 | 0.744 | 0.656 |
| rrnet_mspc | 0.517 | 0.567 | 0.708 | 0.658 |
| rrnet_dev | 0.531 | 0.527 | 0.727 | 0.656 |

## 유발 요인별 AUROC

| 후보 | none | prolonged_driving | sleep_deprivation |
| --- | --- | --- | --- |
| lgbm | 0.642 | 0.581 | 0.564 |
| lgbm_abs | 0.644 | 0.591 | 0.572 |
| lgbm_mspc | 0.643 | 0.612 | 0.571 |
| lgbm_abs_mspc | 0.642 | 0.626 | 0.586 |
| rrnet | 0.496 | 0.696 | 0.521 |
| rrnet_mspc | 0.477 | 0.622 | 0.510 |
| rrnet_dev | 0.492 | 0.576 | 0.522 |

## fold 별 LightGBM 상위 특징 (gain)

- advitam: sdnn_120, sd2_300, sdnn_300, lf_hf_log_300, lf_nu_300, sd2_120, sd1_300, z_rr_mean_300
- dddb: sd1_300, z_hr_300, sampen_300, rmssd_300, trend_hr_slope, pnn50_300, sd2_300, hr_300
- drozy: sdnn_300, sd2_300, z_sd2_300, hf_log_300, lf_log_300, hr_300, z_sd1_sd2_300, sdnn_120
- mpddf: sdnn_300, z_sd1_sd2_300, sd2_300, hr_300, z_pnn50_300, trend_hr_slope, hr_120, z_rr_mean_300
