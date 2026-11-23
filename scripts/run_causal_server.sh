#!/usr/bin/env bash
# 인과(실시간) 학생 재학습 전체 (docs/causal-student.md 3장). GPU 서버에서:
#   cd ~/jberry/repo && git fetch origin claude/causal-student && git checkout claude/causal-student
#   nohup setsid bash scripts/run_causal_server.sh > ~/jberry/out/causal.log 2>&1 &
# 단계마다 결과 파일이 있으면 건너뛴다 (중단 후 다시 돌려도 됨).
set -euo pipefail
J=${J:-$HOME/jberry}
D=$J/data; O=$J/out; C=$D/converted
PY=${PY:-$J/repo/.venv/bin/python}
JOBS=${JOBS:-20}
cd "$J/repo"
T="$C/advitam $C/dddb $C/drozy $C/mpddf"; S="$C/clas $C/maus $C/wesad"
step() { echo "=== $(date '+%F %T') $*"; }

# 1. 인과 RR·창 특징 (CPU)
[ -f $D/teacher_cache_c/windows.parquet ] || { step "RR 교사 4종";
  $PY -u scripts/teacher.py --causal rr --data $T --out $D/teacher_cache_c/rr --jobs $JOBS
  $PY -u scripts/teacher.py --causal features --rr $D/teacher_cache_c/rr --out $D/teacher_cache_c --jobs $JOBS --no-tacho; }
[ -f $D/cache_student_c/windows.parquet ] || { step "RR 학생 3종";
  $PY -u scripts/teacher.py --causal rr --data $S --out $D/cache_student_c/rr --jobs $JOBS
  $PY -u scripts/teacher.py --causal features --rr $D/cache_student_c/rr --out $D/cache_student_c --jobs $JOBS --no-tacho; }
[ -f $D/cache_prv_c/windows.parquet ] || { step "PRV 학생 3종 (기존 cache_prv 와 같은 범위)";
  $PY -u scripts/teacher.py --causal rr --channel ppg --data $S --out $D/cache_prv_c/rr --jobs $JOBS
  $PY -u scripts/teacher.py --causal features --rr $D/cache_prv_c/rr --out $D/cache_prv_c --jobs $JOBS --no-tacho; }

# 2. 라벨 표에 인과 RR z 붙이기 (교사 목표는 그대로)
[ -f $O/f2c_label_c/f2c_labels.parquet ] || { step "relabel";
  $PY -u scripts/causal_relabel.py --label-dir $O/f2c_label --cache $D/teacher_cache_c --student $D/cache_student_c --out $O/f2c_label_c; }

# 3. 인과 파형 캐시 (row_id·profile_key 는 기존 캐시 기준이라 --cache/--student 는 기존 것)
[ -f $D/student_wave_c/index.parquet ] || { step "파형 캐시 (인과)";
  $PY -u scripts/student_cache.py --causal --labels $O/f2c_label_c/f2c_labels.parquet --data $C \
      --cache $D/teacher_cache --student $D/cache_student --out $D/student_wave_c; }

COMMON="--label-dir $O/f2c_label_c --wave $D/student_wave_c --cache $D/teacher_cache_c --prv $D/cache_prv_c \
  --s0-oof $O/anomaly/oof_lodo.parquet --f2c-oof $O/anomaly_c/oof_lodo.parquet --causal-input"

# 4. LODO (S1-EP, S1-E × 4 fold × 시드 3) — 본안: 인과 입력 + 오프라인 교사 목표
[ -f $O/student_causal/report_student.json ] || { step "LODO 인과 입력";
  $PY -u scripts/student_train.py $COMMON --conds S1-EP S1-E --out $O/student_causal; }
# 5. 비교안: 목표도 인과 (S1-EP 만)
[ -f $O/student_causal_ct/report_student.json ] || { step "LODO 인과 입력 + 인과 목표";
  $PY -u scripts/student_train.py $COMMON --causal-target --conds S1-EP --out $O/student_causal_ct; }

# 6. 비교표
step "비교"
$PY -u scripts/causal_compare.py --old $O/student/student_scores.parquet --new $O/student_causal/student_scores.parquet \
  --new-ct $O/student_causal_ct/student_scores.parquet --labels $O/f2c_label_c/f2c_labels.parquet \
  --s0-oof $O/anomaly/oof_lodo.parquet --out $O/causal_compare

# 7. 최종 학생 (전체 데이터, 기기용) + 동작점
[ -f $O/student_causal_final/thresholds.json ] || { step "최종 학생";
  $PY -u scripts/student_train.py $COMMON --final --conds S1-EP S1-E --scores $O/student_causal/student_scores.parquet \
      --out $O/student_causal_final; }
step "끝"
