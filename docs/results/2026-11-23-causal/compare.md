# 공개 4종 LODO: 인과 학생 vs 기존 학생 vs 규칙

평가 창 39257 (모든 비교 열이 유한한 창)

| 점수 | M1 | advitam | dddb | drozy | mpddf |
| --- | --- | --- | --- | --- | --- |
| rule_old | 0.644 | 0.579 | 0.581 | 0.867 | 0.550 |
| rule_causal | 0.651 | 0.566 | 0.582 | 0.933 | 0.523 |
| old:S1-EP | 0.673 | 0.607 | 0.672 | 0.923 | 0.489 |
| old:S1-E | 0.675 | 0.606 | 0.671 | 0.930 | 0.492 |
| old:S1-EP_ecg | 0.673 | 0.607 | 0.672 | 0.923 | 0.489 |
| old:S1-EP_ppg | nan | nan | nan | nan | nan |
| causal:S1-EP | 0.654 | 0.590 | 0.657 | 0.899 | 0.473 |
| causal:S1-E | 0.652 | 0.581 | 0.656 | 0.892 | 0.479 |
| causal:S1-EP_ecg | 0.654 | 0.590 | 0.657 | 0.899 | 0.473 |
| causal:S1-EP_ppg | nan | nan | nan | nan | nan |
| causal_ct:S1-EP | 0.655 | 0.597 | 0.670 | 0.866 | 0.488 |
| causal_ct:S1-EP_ecg | 0.655 | 0.597 | 0.670 | 0.866 | 0.488 |
| causal_ct:S1-EP_ppg | nan | nan | nan | nan | nan |

| 차이 | 값 | 95% 구간 |
| --- | --- | --- |
| causal:S1-EP − old:S1-EP | -0.018 | [-0.044, +0.006] |
| causal:S1-EP − rule_causal | +0.004 | [-0.033, +0.043] |
| old:S1-EP − rule_old | +0.028 | [-0.011, +0.069] |
| causal:S1-E − old:S1-E | -0.022 | [-0.049, -0.000] |
| causal_ct:S1-EP − causal:S1-EP | +0.001 | [-0.014, +0.015] |
