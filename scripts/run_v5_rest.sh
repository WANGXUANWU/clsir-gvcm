#!/bin/sh
# The rest of the v5 study, after the bandwidth grid.
#
# Only the CLSIR estimators changed, and only for a binary response, but each
# study below is re-run over the whole set of designs it covered in v4 so that
# results/v5 is self-contained and nothing has to be merged across versions.
set -e
PY="C:/Users/24481084/AppData/Local/miniconda3/python.exe"
cd "$(dirname "$0")"
OUT=../results/v5

echo "=== [1/4] mean fitted coefficient curves ==="
"$PY" make_curves.py --reps 200

echo "=== [2/4] complete analysis with a cross-validated bandwidth ==="
"$PY" clsir_study.py cv --reps 60 --n 500 2000 --designs bin2 bin10 poi2 \
  --c-grid 0.5 0.65 0.8 1.0 1.3 1.7 --grid-size 101 --folds 5 \
  --methods CLSIR-QS CLSIR-OS LMLE LMLE-warm OSL FZ CGA \
  --workers 6 --out $OUT

echo "=== [3/4] cost scaling (single worker, clean wall clock) ==="
"$PY" clsir_study.py cost --reps 20 --n 2000 --designs bin10 poi10 \
  --cost-sizes 25 50 100 200 400 \
  --methods CLSIR-QS CLSIR-OS OSL LMLE-warm LMLE FZ FIRTH CGA \
  --workers 1 --out $OUT

echo "=== [3b/5] real data application (binary, so the pilot ratio changed) ==="
"$PY" real_data.py > ../results/v5_real.log 2>&1 && tail -40 ../results/v5_real.log

echo "=== [4/5] parallel scaling of the update ==="
"$PY" parallel_bench.py --designs bin10 poi10 --n 2000 --grid-size 1600 \
  --workers 1 2 3 4 5 6 --repeats 5 --out $OUT

echo "=== [5/5] robustness of the quadratic update ==="
"$PY" curv_ridge.py --reps 200 || echo "  (curv_ridge failed, continuing)"

echo "V5 REST DONE"
