# 교사 후보 비교 (LODO)

선택: **lgbm** (AUROC 데이터셋 평균 기준, 차이 0.01 이하면 ECE·단순성)

| 후보 | AUROC (데이터셋 평균) | 균형 정확도 | ECE (평균) | AUROC (전체) | Brier |
| --- | --- | --- | --- | --- | --- |
| rule | 0.404 | 0.443 | 0.202 | 0.412 | 0.254 |
| lgbm | 0.597 | 0.549 | 0.201 | 0.451 | 0.245 |
| lgbm_nopers | 0.546 | 0.538 | 0.213 | 0.359 | 0.262 |
| rrnet | 0.593 | 0.546 | 0.231 | 0.564 | 0.260 |
| ensemble | 0.591 | 0.525 | 0.211 | 0.516 | 0.253 |

규칙 기준선 대비: lgbm 가 AUROC +0.193.

## 데이터셋별 AUROC

| 후보 | advitam | dddb | drozy | mpddf |
| --- | --- | --- | --- | --- |
| rule | 0.458 | 0.550 | 0.325 | 0.284 |
| lgbm | 0.601 | 0.502 | 0.776 | 0.509 |
| lgbm_nopers | 0.622 | 0.479 | 0.706 | 0.376 |
| rrnet | 0.497 | 0.572 | 0.747 | 0.556 |
| ensemble | 0.546 | 0.545 | 0.753 | 0.519 |

## 유발 요인별 AUROC

| 후보 | none | prolonged_driving | sleep_deprivation |
| --- | --- | --- | --- |
| rule | 0.500 | 0.419 | 0.376 |
| lgbm | 0.594 | 0.400 | 0.584 |
| lgbm_nopers | 0.637 | 0.284 | 0.616 |
| rrnet | 0.481 | 0.576 | 0.532 |
| ensemble | 0.514 | 0.487 | 0.559 |

## fold 별 LightGBM 상위 특징 (gain)

- advitam: sdnn_120, sd2_300, sdnn_300, lf_hf_log_300, lf_nu_300, sd2_120, sd1_300, z_rr_mean_300
- dddb: hr_300, rr_mean_300, pnn50_300, pnn50_120, sampen_300, trend_hr_slope, z_rr_mean_300, z_pnn50_300
- drozy: sd2_300, hr_120, lf_nu_300, sd1_300, r_hr_120, z_rr_mean_300, z_sd1_sd2_300, z_lf_nu_120
- mpddf: z_sd1_sd2_300, hr_300, sd1_sd2_300, rmssd_300, r_rr_mean_120, rr_mean_300, sd1_300, z_pnn50_300
