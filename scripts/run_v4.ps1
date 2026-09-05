# The whole numerical study, in the order the paper needs it.
# Run from the simulations directory.  Everything lands in ../results/v4.
$ErrorActionPreference = "Continue"
$PY = "C:\Users\24481084\AppData\Local\miniconda3\python.exe"
Set-Location $PSScriptRoot
$OUT = "../results/v4"

Write-Output "=== [1/8] bandwidth grid, six examples ==="
& $PY clsir_study.py grid --reps 500 --n 500 2000 `
  --designs bin2 bin10 poi2 poi10 gau2 binskew `
  --c-grid 0.5 0.65 0.8 1.0 1.3 1.7 --grid-size 101 `
  --methods CLSIR-OS CLSIR-QS LMLE OSL FZ CGA GA LC-OS `
  --workers 6 --out $OUT

Write-Output "=== [2/8] the penalized competitor, Bernoulli designs ==="
& $PY clsir_study.py grid --reps 500 --n 500 2000 `
  --designs bin2 bin10 binskew `
  --c-grid 0.5 0.65 0.8 1.0 1.3 1.7 --grid-size 101 `
  --methods FIRTH --workers 6 --out ../results/v4_firth

Write-Output "=== [3/8] mean fitted coefficient curves ==="
& $PY make_curves.py

Write-Output "=== [4/8] cost scaling (single worker, clean wall clock) ==="
& $PY clsir_study.py cost --reps 20 --n 2000 --designs bin10 poi10 `
  --cost-sizes 25 50 100 200 400 `
  --methods CLSIR-QS CLSIR-OS OSL LMLE-warm LMLE FZ FIRTH CGA `
  --workers 1 --out $OUT

Write-Output "=== [5/8] parallel scaling of the update ==="
& $PY parallel_bench.py --designs bin10 poi10 --n 2000 --grid-size 1600 `
  --workers 1 2 3 4 5 6 --repeats 5 --out $OUT

Write-Output "=== [6/8] real data application ==="
& $PY real_data.py

Write-Output "=== [7/8] robustness of the quadratic update, and theory checks ==="
& $PY curv_ridge.py --reps 200
& $PY verify_theory.py C9 C10

Write-Output "=== [8/8] complete analysis with a cross-validated bandwidth ==="
& $PY clsir_study.py cv --reps 60 --n 500 2000 --designs bin2 bin10 poi2 `
  --c-grid 0.5 0.65 0.8 1.0 1.3 1.7 --grid-size 101 --folds 5 `
  --methods CLSIR-QS CLSIR-OS LMLE LMLE-warm OSL FZ CGA `
  --workers 6 --out $OUT

Write-Output "V4 DONE"
