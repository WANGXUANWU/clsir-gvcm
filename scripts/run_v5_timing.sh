#!/bin/sh
# Re-measure everything whose result is a wall clock time, on a quiet machine.
#
# The first pass of the cross-validated analysis was timed while figures were
# being drawn and the draft compiled in parallel, which showed up as the warm
# started fully iterated estimator coming out slower than the cold started one
# in Example 2 -- an ordering that cannot happen and is pure contention.  This
# script repeats both timing studies with nothing else running, and widens the
# cost study to all six designs so that the Gaussian qualification in Section S3
# has data behind it.
set -e
PY="C:/Users/24481084/AppData/Local/miniconda3/python.exe"
cd "$(dirname "$0")"
OUT=../results/v5

echo "=== [1/2] cost scaling, all six designs, single worker ==="
"$PY" clsir_study.py cost --reps 20 --n 2000 \
  --designs bin2 bin10 poi10 gau2 poi2 binskew \
  --cost-sizes 25 50 100 200 400 \
  --methods CLSIR-QS CLSIR-OS OSL LMLE-warm LMLE FZ FIRTH CGA \
  --workers 1 --out $OUT

echo "=== [2/2] complete analysis with a cross-validated bandwidth ==="
"$PY" clsir_study.py cv --reps 60 --n 500 2000 --designs bin2 bin10 poi2 \
  --c-grid 0.5 0.65 0.8 1.0 1.3 1.7 --grid-size 101 --folds 5 \
  --methods CLSIR-QS CLSIR-OS LMLE LMLE-warm OSL FZ CGA \
  --workers 6 --out $OUT

echo "V5 TIMING DONE"
