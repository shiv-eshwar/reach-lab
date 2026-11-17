#!/bin/sh
set -eu
cd "$(dirname "$0")"
venv_dir=${REACH_VENV:-.venv}
uv venv --allow-existing --python 3.11 "$venv_dir"
uv pip sync --python "$venv_dir/bin/python" requirements.txt
for step in 01 02 03 04 07 08 09; do
    "$venv_dir/bin/python" src/"$step"_*.py
done
if [ "${1:-}" = "--benchmark" ]; then
    REACH_BENCHMARK_DIR=results/corrected_benchmark OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 "$venv_dir/bin/python" src/23_model_benchmark.py
fi
"$venv_dir/bin/python" demo/build.py
