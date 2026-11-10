# 교사 후보 비교 (LODO)

선택: **lgbm_nopers** (AUROC 데이터셋 평균 기준, 차이 0.01 이하면 ECE·단순성)

| 후보 | AUROC (데이터셋 평균) | 균형 정확도 | ECE (평균) | AUROC (전체) | Brier |
| --- | --- | --- | --- | --- | --- |
| rule | 0.583 | 0.493 | 0.309 | 0.549 | 0.261 |
| lgbm | 0.505 | 0.513 | 0.270 | 0.490 | 0.250 |
| lgbm_nopers | 0.553 | 0.531 | 0.267 | 0.459 | 0.250 |
| rrnet | 0.529 | 0.510 | 0.318 | 0.528 | 0.258 |
| ensemble | 0.531 | 0.518 | 0.306 | 0.531 | 0.255 |

규칙 기준선 대비: lgbm_nopers 가 AUROC -0.031.

## 데이터셋별 AUROC

| 후보 | advitam | mpddf |
| --- | --- | --- |
| rule | 0.452 | 0.714 |
| lgbm | 0.517 | 0.493 |
| lgbm_nopers | 0.541 | 0.564 |
| rrnet | 0.508 | 0.550 |
| ensemble | 0.511 | 0.551 |

## 유발 요인별 AUROC

| 후보 | none | prolonged_driving | sleep_deprivation |
| --- | --- | --- | --- |
| rule | 0.472 | 0.714 | 0.433 |
| lgbm | 0.526 | 0.493 | 0.512 |
| lgbm_nopers | 0.553 | 0.564 | 0.501 |
| rrnet | 0.528 | 0.550 | 0.494 |
| ensemble | 0.532 | 0.551 | 0.495 |

## fold 별 LightGBM 상위 특징 (gain)

- advitam: trend_hr_slope, trend_rmssd_slope, z_sd2_300, r_sd2_300, r_hr_30, sd1_sd2_300, r_sdnn_300, r_rr_mean_30
- mpddf: sd2_300, z_sd2_300, z_rr_mean_300, z_sd1_120, lf_nu_300, z_pnn50_120, z_lf_nu_300, sampen_120
