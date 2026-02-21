#!/usr/bin/env bash
#SBATCH -J color_bypass_followup_big_20260221_123927
#SBATCH -p cuda
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=12
#SBATCH --mem=80G
#SBATCH --time=12:00:00
#SBATCH --array=1-36%1
#SBATCH -o /mnt/scratchc/fmlab/zuberi01/phd/color-bypass-bench/results/color_bypass_followup_big_20260221_123927/slurm_logs/%A_%a.out
#SBATCH -e /mnt/scratchc/fmlab/zuberi01/phd/color-bypass-bench/results/color_bypass_followup_big_20260221_123927/slurm_logs/%A_%a.err

set -euo pipefail

log() {
  printf '[%s] %s\n' "$(date -u +'%Y-%m-%dT%H:%M:%SZ')" "$*" >&2
}

RUN_ROOT=/mnt/scratchc/fmlab/zuberi01/phd/color-bypass-bench/results/color_bypass_followup_big_20260221_123927
PROJECT_ROOT=/mnt/scratchc/fmlab/zuberi01/phd/color-bypass-bench
CONFIG_SNAPSHOT=/mnt/scratchc/fmlab/zuberi01/phd/color-bypass-bench/results/color_bypass_followup_big_20260221_123927/config_snapshot.yaml
SHARD_DIR="$RUN_ROOT/shards/shard_$(printf '%04d' "$SLURM_ARRAY_TASK_ID")"
SHARD_FILE="$SHARD_DIR/shard_input.jsonl"

if [ ! -f "$SHARD_FILE" ]; then
  log "ERROR: shard file missing: $SHARD_FILE"
  exit 1
fi

CONDA_BASE_HINT=${LLM_CONDA_BASE:-${CONDA_PREFIX:-}}
CONDA_EXE_HINT=${LLM_CONDA_EXE:-$(command -v conda || true)}

if [ -n "$CONDA_BASE_HINT" ] && [ -f "$CONDA_BASE_HINT/etc/profile.d/conda.sh" ]; then
  # shellcheck source=/dev/null
  source "$CONDA_BASE_HINT/etc/profile.d/conda.sh"
elif [ -n "$CONDA_EXE_HINT" ] && [ -x "$CONDA_EXE_HINT" ]; then
  CONDA_BASE_FROM_EXE="$($CONDA_EXE_HINT info --base 2>/dev/null || true)"
  if [ -n "$CONDA_BASE_FROM_EXE" ] && [ -f "$CONDA_BASE_FROM_EXE/etc/profile.d/conda.sh" ]; then
    # shellcheck source=/dev/null
    source "$CONDA_BASE_FROM_EXE/etc/profile.d/conda.sh"
  fi
fi

if ! command -v conda >/dev/null 2>&1; then
  for candidate in "$HOME/miniforge3" "$HOME/mambaforge" "$HOME/anaconda3" "$HOME/miniconda3"; do
    if [ -f "$candidate/etc/profile.d/conda.sh" ]; then
      # shellcheck source=/dev/null
      source "$candidate/etc/profile.d/conda.sh"
      break
    fi
  done
fi

if ! command -v conda >/dev/null 2>&1; then
  log "ERROR: conda not found on compute node."
  exit 1
fi

conda activate llm_ollama

export PATH="$HOME/.local/bin:$PATH"
export LD_LIBRARY_PATH="$HOME/.local/lib/ollama:${LD_LIBRARY_PATH:-}"
OLLAMA_BIN="$(command -v ollama || true)"
if [ -z "$OLLAMA_BIN" ] && [ -x "$HOME/.local/bin/ollama" ]; then
  OLLAMA_BIN="$HOME/.local/bin/ollama"
fi
if [ -z "$OLLAMA_BIN" ]; then
  log "ERROR: ollama not found in PATH."
  exit 1
fi

export OLLAMA_HOST=127.0.0.1:11434
if [ -z "${OLLAMA_MODELS:-}" ]; then
  user_name="$(whoami)"
  group_name="$(id -gn 2>/dev/null || true)"

  pick_models_dir() {
    local base candidate
    if [ -n "${SCRATCH:-}" ]; then
      candidate="$SCRATCH/.ollama/models"
      if mkdir -p "$candidate" >/dev/null 2>&1; then
        echo "$candidate"
        return 0
      fi
    fi

    for base in /mnt/scratchc /scratchc /mnt/scratch /scratch; do
      [ -d "$base" ] || continue
      for candidate in         "$base/$user_name/.ollama/models"         "$base/$group_name/$user_name/.ollama/models"; do
        if mkdir -p "$candidate" >/dev/null 2>&1; then
          echo "$candidate"
          return 0
        fi
      done
    done

    candidate="$HOME/.ollama/models"
    mkdir -p "$candidate"
    echo "$candidate"
  }

  OLLAMA_MODELS="$(pick_models_dir)"
  export OLLAMA_MODELS
fi
mkdir -p "$OLLAMA_MODELS"

cd "$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT/src:${PYTHONPATH:-}"

"$OLLAMA_BIN" serve >"$SHARD_DIR/ollama_serve.log" 2>&1 &
OLLAMA_PID="$!"

cleanup() {
  local rc=$?
  if kill -0 "$OLLAMA_PID" >/dev/null 2>&1; then
    kill "$OLLAMA_PID" >/dev/null 2>&1 || true
    wait "$OLLAMA_PID" >/dev/null 2>&1 || true
  fi
  exit "$rc"
}
trap cleanup EXIT

ready=0
for _ in $(seq 1 150); do
  if "$OLLAMA_BIN" list >"$SHARD_DIR/ollama_models_runtime.txt" 2>/dev/null; then
    ready=1
    break
  fi
  sleep 1
done
if [ "$ready" -ne 1 ]; then
  log "ERROR: ollama server did not become ready."
  exit 1
fi

if [ ! -s "$RUN_ROOT/ollama_models.txt" ]; then
  cp "$SHARD_DIR/ollama_models_runtime.txt" "$RUN_ROOT/ollama_models.txt" || true
fi

if [ "0" -eq 1 ]; then
  python -m ollama_color_bypass_bench.hpc.runtime run-shard --shard "$SHARD_FILE" --config "$CONFIG_SNAPSHOT" --rerun
else
  python -m ollama_color_bypass_bench.hpc.runtime run-shard --shard "$SHARD_FILE" --config "$CONFIG_SNAPSHOT"
fi

log "Shard $SLURM_ARRAY_TASK_ID finished."
