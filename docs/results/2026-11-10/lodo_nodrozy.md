# 교사 후보 비교 (LODO)

선택: **lgbm** (AUROC 데이터셋 평균 기준, 차이 0.01 이하면 ECE·단순성)

| 후보 | AUROC (데이터셋 평균) | 균형 정확도 | ECE (평균) | AUROC (전체) | Brier |
| --- | --- | --- | --- | --- | --- |
| rule | 0.538 | 0.545 | 0.236 | 0.483 | 0.252 |
| lgbm | 0.515 | 0.557 | 0.235 | 0.505 | 0.244 |
| lgbm_nopers | 0.523 | 0.566 | 0.243 | 0.518 | 0.243 |
| rrnet | 0.479 | 0.470 | 0.288 | 0.342 | 0.280 |
| ensemble | 0.485 | 0.509 | 0.250 | 0.422 | 0.261 |

규칙 기준선 대비: lgbm 가 AUROC -0.023.

## 데이터셋별 AUROC

| 후보 | advitam | dddb | mpddf |
| --- | --- | --- | --- |
| rule | 0.452 | 0.448 | 0.715 |
| lgbm | 0.527 | 0.571 | 0.448 |
| lgbm_nopers | 0.605 | 0.562 | 0.401 |
| rrnet | 0.446 | 0.550 | 0.440 |
| ensemble | 0.446 | 0.559 | 0.451 |

## 유발 요인별 AUROC

| 후보 | none | prolonged_driving | sleep_deprivation |
| --- | --- | --- | --- |
| rule | 0.472 | 0.493 | 0.433 |
| lgbm | 0.564 | 0.444 | 0.540 |
| lgbm_nopers | 0.639 | 0.453 | 0.555 |
| rrnet | 0.489 | 0.280 | 0.420 |
| ensemble | 0.490 | 0.373 | 0.420 |

## fold 별 LightGBM 상위 특징 (gain)

- advitam: sd2_300, lf_nu_300, z_rmssd_300, z_lf_nu_300, z_sd2_300, hr_120, sd2_120, r_hr_120
- dddb: sd2_300, sdnn_300, trend_hr_slope, z_sd1_sd2_300, sampen_300, hf_log_300, pnn50_300, z_rr_mean_300
- mpddf: sd2_300, sdnn_300, pnn50_300, hf_log_300, z_lf_log_300, rr_mean_300, hr_300, sdnn_120
