#!/bin/sh
# The studies this revision adds, in the order they must run.
# The cost study is last and alone: it is the only one whose result is a wall
# clock time, and it must not share the machine with a six-worker pool.
set -e
PY="C:/Users/24481084/AppData/Local/miniconda3/python.exe"
cd "$(dirname "$0")"

echo "=== [1/3] pointwise coverage of the two intervals ==="
"$PY" coverage_study.py 400 6

echo "=== [2/3] size and power of the constancy test ==="
"$PY" power_study.py 200 99 6

echo "=== [3/3] cost scaling at both sample sizes, single worker ==="
"$PY" clsir_study.py cost --reps 20 --n 500 2000 \
  --designs bin2 bin10 poi10 gau2 poi2 binskew \
  --cost-sizes 25 50 100 200 400 \
  --methods CLSIR-QS CLSIR-OS OSL LMLE-warm LMLE FZ FIRTH CGA \
  --workers 1 --out ../results/v5

echo "V6 DONE"
