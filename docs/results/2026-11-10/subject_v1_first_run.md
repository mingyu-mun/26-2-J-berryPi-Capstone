# 교사 후보 비교 (SUBJECT)

선택: **rrnet** (AUROC 데이터셋 평균 기준, 차이 0.01 이하면 ECE·단순성)

| 후보 | AUROC (데이터셋 평균) | 균형 정확도 | ECE (평균) | AUROC (전체) | Brier |
| --- | --- | --- | --- | --- | --- |
| rule | 0.452 | 0.465 | 0.241 | 0.407 | 0.263 |
| lgbm | 0.483 | 0.534 | 0.201 | 0.501 | 0.240 |
| lgbm_nopers | 0.491 | 0.532 | 0.221 | 0.475 | 0.248 |
| rrnet | 0.604 | 0.604 | 0.245 | 0.560 | 0.239 |
| ensemble | 0.533 | 0.561 | 0.195 | 0.528 | 0.241 |

규칙 기준선 대비: rrnet 가 AUROC +0.153.

## 데이터셋별 AUROC

| 후보 | advitam | dddb | drozy | mpddf |
| --- | --- | --- | --- | --- |
| rule | 0.583 | 0.256 | 0.500 | 0.468 |
| lgbm | 0.538 | 0.437 | 0.488 | 0.469 |
| lgbm_nopers | 0.547 | 0.498 | 0.516 | 0.401 |
| rrnet | 0.588 | 0.317 | 0.799 | 0.713 |
| ensemble | 0.586 | 0.356 | 0.634 | 0.555 |

## 유발 요인별 AUROC

| 후보 | none | prolonged_driving | sleep_deprivation |
| --- | --- | --- | --- |
| rule | 0.637 | 0.388 | 0.560 |
| lgbm | 0.538 | 0.476 | 0.511 |
| lgbm_nopers | 0.678 | 0.433 | 0.489 |
| rrnet | 0.604 | 0.522 | 0.693 |
| ensemble | 0.608 | 0.491 | 0.617 |

## fold 별 LightGBM 상위 특징 (gain)

- k0: hr_120, sd2_300, z_sd1_sd2_300, z_hf_log_300, sd1_300, lf_nu_300, pnn50_300, r_rr_mean_300
- k1: sampen_300, sampen_120, z_pnn50_300, hf_log_300, lf_nu_300, sd2_300, r_hr_120, sd1_300
- k2: lf_hf_log_300, sd2_300, sdnn_300, sd1_300, sampen_300, trend_hr_slope, rmssd_300, sd1_sd2_300
- k3: z_rr_mean_300, sd2_300, sd1_300, pnn50_300, sdnn_120, hr_300, z_sd1_sd2_300, r_rr_mean_300
- k4: sdnn_300, rr_mean_300, lf_log_300, hr_300, sd2_300, sd1_sd2_300, hf_log_300, z_hr_300
