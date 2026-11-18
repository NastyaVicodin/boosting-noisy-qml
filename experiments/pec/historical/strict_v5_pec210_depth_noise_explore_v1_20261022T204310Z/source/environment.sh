#!/bin/bash
# Source with the absolute experiment directory as the first argument.
set -euo pipefail
export LD_LIBRARY_PATH=/easybuild/software/bzip2/1.0.8-GCCcore-13.2.0/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
export PYTHONDONTWRITEBYTECODE=1
export JAX_PLATFORMS=cpu
export JAX_ENABLE_COMPILATION_CACHE=false
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export XDG_CACHE_HOME="$1/cache"
export MPLCONFIGDIR="$1/cache/matplotlib"
