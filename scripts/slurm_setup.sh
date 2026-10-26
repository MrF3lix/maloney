# Shared setup for the *.submit scripts (sourced, not run): pick the GPU flavour, point uv at a
# per-flavour venv on the node's scratch disk, sync it, and define `run`, which executes a
# command or, with DRY_RUN=1, only prints it (to check the command lines without SLURM).

run() {
    if [ "${DRY_RUN:-0}" = "1" ]; then
        printf '+'; printf ' %q' "$@"; printf '\n'
    else
        "$@"
    fi
}

# Hydra otherwise hides the stack trace (and e.g. the name of a missing file) in the job log.
export HYDRA_FULL_ERROR=1

if [ ! -f .env ] && [ -z "${WANDB_API_KEY:-}" ]; then
    echo "WARNING: no .env with WANDB_API_KEY in $(pwd); wandb logging will fail (copy .env from your Mac)" >&2
fi

# Batch jobs run a non-interactive shell that skips ~/.bashrc, so uv's install dir isn't on PATH.
export PATH="${HOME}/.local/bin:${HOME}/.cargo/bin:${PATH}"
if ! command -v uv >/dev/null 2>&1; then
    echo "ERROR: uv not found on $(hostname) (PATH=${PATH}). Run 'which uv' on the login node and add its dir above." >&2
    exit 1
fi

if [ "${DRY_RUN:-0}" = "1" ]; then
    ACCEL_EXTRA=cu128
    echo "[dry run] assuming ${ACCEL_EXTRA}, skipping GPU check and uv sync"
    return 0
fi

if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L 2>/dev/null | grep -q '^GPU '; then
    # torch's cu128 builds dropped Pascal (P40/P100, compute capability 6.x); those need cu126.
    MIN_CC=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | sort -n | head -1)
    if [ "${MIN_CC%%.*}" -lt 7 ]; then
        ACCEL_EXTRA=cu126
    else
        ACCEL_EXTRA=cu128
    fi
elif [ -e /dev/kfd ] || (command -v rocm-smi >/dev/null 2>&1 && rocm-smi --showid >/dev/null 2>&1); then
    ACCEL_EXTRA=rocm
else
    echo "ERROR: no NVIDIA or AMD GPU found on $(hostname). Refusing to fall back to CPU." >&2
    exit 1
fi
echo "[$(date +%T)] Node $(hostname): detected ${ACCEL_EXTRA} GPUs"

# Hugging Face model weights (Whisper) are cached once on the scratch disk, not in the home quota.
export HF_HOME="${HF_HOME:-/raid/persistent_scratch/saaf/huggingface}"
export UV_PROJECT_ENVIRONMENT="/raid/persistent_scratch/saaf/venvs/maloney-${ACCEL_EXTRA}"
export UV_LINK_MODE=copy
export UV_HTTP_TIMEOUT=900

if [ "${ACCEL_EXTRA}" = "rocm" ]; then
    export MIOPEN_USER_DB_PATH="/raid/persistent_scratch/saaf/miopen/${SLURM_JOB_ID:-$$}"
    export MIOPEN_CUSTOM_CACHE_DIR="${MIOPEN_USER_DB_PATH}"
    mkdir -p "${MIOPEN_USER_DB_PATH}"
    trap 'rm -rf "${MIOPEN_USER_DB_PATH}"' EXIT
fi

uv sync --extra "${ACCEL_EXTRA}" --extra wandb
