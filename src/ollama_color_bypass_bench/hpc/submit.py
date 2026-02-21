"""SLURM array script rendering and submission."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
import shlex
import subprocess

from ..config import BenchmarkConfig


@dataclass(frozen=True)
class SubmissionResult:
    """Result of an sbatch submission."""

    job_id: str
    stdout: str
    stderr: str


def parse_job_id(sbatch_stdout: str) -> str:
    text = sbatch_stdout.strip()
    if not text:
        raise RuntimeError("sbatch returned empty output; unable to parse job id")

    first_line = text.splitlines()[0].strip()
    candidate = first_line.split(";", 1)[0].strip()
    if candidate.isdigit():
        return candidate

    match = re.search(r"\d+", candidate)
    if match:
        return match.group(0)

    raise RuntimeError(f"Could not parse job id from sbatch output: {first_line}")


def _sbatch_header_lines(
    *,
    run_id: str,
    config: BenchmarkConfig,
    shard_count: int,
    logs_dir: Path,
) -> list[str]:
    slurm_cfg = config.hpc.slurm

    lines = [
        "#!/usr/bin/env bash",
        f"#SBATCH -J {run_id}",
        f"#SBATCH -p {slurm_cfg.partition}",
        f"#SBATCH --gres={slurm_cfg.gres}",
        f"#SBATCH --cpus-per-gpu={slurm_cfg.cpus_per_gpu}",
        f"#SBATCH --mem={slurm_cfg.mem}",
        f"#SBATCH --time={slurm_cfg.time_limit}",
        f"#SBATCH --array=1-{shard_count}%{slurm_cfg.array_parallelism}",
        f"#SBATCH -o {logs_dir / '%A_%a.out'}",
        f"#SBATCH -e {logs_dir / '%A_%a.err'}",
    ]
    if slurm_cfg.qos:
        lines.append(f"#SBATCH --qos={slurm_cfg.qos}")
    if slurm_cfg.account:
        lines.append(f"#SBATCH --account={slurm_cfg.account}")

    return lines


def render_submit_array_script(
    *,
    run_dir: str | Path,
    run_id: str,
    config: BenchmarkConfig,
    config_snapshot_path: str | Path,
    shard_count: int,
    rerun_completed_shards: bool,
    project_root: str | Path,
) -> Path:
    """Write the SLURM array submission script for this run."""

    if shard_count <= 0:
        raise ValueError("shard_count must be > 0")

    run_root = Path(run_dir).resolve()
    project_root_path = Path(project_root).resolve()
    config_snapshot = Path(config_snapshot_path).resolve()
    logs_dir = run_root / "slurm_logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    script_path = run_root / "submit_array.sh"

    slurm_cfg = config.hpc.slurm
    conda_base_hint = (
        shlex.quote(slurm_cfg.conda_base_hint.strip())
        if slurm_cfg.conda_base_hint.strip()
        else "${LLM_CONDA_BASE:-${CONDA_PREFIX:-}}"
    )
    conda_exe_hint = (
        shlex.quote(slurm_cfg.conda_exe_hint.strip())
        if slurm_cfg.conda_exe_hint.strip()
        else "${LLM_CONDA_EXE:-$(command -v conda || true)}"
    )

    header_lines = _sbatch_header_lines(
        run_id=run_id,
        config=config,
        shard_count=shard_count,
        logs_dir=logs_dir,
    )

    body = f"""
set -euo pipefail

log() {{
  printf '[%s] %s\\n' "$(date -u +'%Y-%m-%dT%H:%M:%SZ')" "$*" >&2
}}

RUN_ROOT={shlex.quote(str(run_root))}
PROJECT_ROOT={shlex.quote(str(project_root_path))}
CONFIG_SNAPSHOT={shlex.quote(str(config_snapshot))}
SHARD_DIR="$RUN_ROOT/shards/shard_$(printf '%04d' "$SLURM_ARRAY_TASK_ID")"
SHARD_FILE="$SHARD_DIR/shard_input.jsonl"

if [ ! -f "$SHARD_FILE" ]; then
  log "ERROR: shard file missing: $SHARD_FILE"
  exit 1
fi

CONDA_BASE_HINT={conda_base_hint}
CONDA_EXE_HINT={conda_exe_hint}

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

conda activate {shlex.quote(slurm_cfg.conda_env)}

export PATH="$HOME/.local/bin:$PATH"
export LD_LIBRARY_PATH="$HOME/.local/lib/ollama:${{LD_LIBRARY_PATH:-}}"
export OLLAMA_NOPROGRESS=1
OLLAMA_BIN="$(command -v ollama || true)"
if [ -z "$OLLAMA_BIN" ] && [ -x "$HOME/.local/bin/ollama" ]; then
  OLLAMA_BIN="$HOME/.local/bin/ollama"
fi
if [ -z "$OLLAMA_BIN" ]; then
  log "ERROR: ollama not found in PATH."
  exit 1
fi

export OLLAMA_HOST=127.0.0.1:{int(slurm_cfg.ollama_port)}
if [ -z "${{OLLAMA_MODELS:-}}" ]; then
  user_name="$(whoami)"
  group_name="$(id -gn 2>/dev/null || true)"

  pick_models_dir() {{
    local base candidate
    if [ -n "${{SCRATCH:-}}" ]; then
      candidate="$SCRATCH/.ollama/models"
      if mkdir -p "$candidate" >/dev/null 2>&1; then
        echo "$candidate"
        return 0
      fi
    fi

    for base in /mnt/scratchc /scratchc /mnt/scratch /scratch; do
      [ -d "$base" ] || continue
      for candidate in \
        "$base/$user_name/.ollama/models" \
        "$base/$group_name/$user_name/.ollama/models"; do
        if mkdir -p "$candidate" >/dev/null 2>&1; then
          echo "$candidate"
          return 0
        fi
      done
    done

    candidate="$HOME/.ollama/models"
    mkdir -p "$candidate"
    echo "$candidate"
  }}

  OLLAMA_MODELS="$(pick_models_dir)"
  export OLLAMA_MODELS
fi
mkdir -p "$OLLAMA_MODELS"

cd "$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT/src:${{PYTHONPATH:-}}"

"$OLLAMA_BIN" serve >"$SHARD_DIR/ollama_serve.log" 2>&1 &
OLLAMA_PID="$!"

cleanup() {{
  local rc=$?
  if kill -0 "$OLLAMA_PID" >/dev/null 2>&1; then
    kill "$OLLAMA_PID" >/dev/null 2>&1 || true
    wait "$OLLAMA_PID" >/dev/null 2>&1 || true
  fi
  exit "$rc"
}}
trap cleanup EXIT

ready=0
for _ in $(seq 1 {int(slurm_cfg.ollama_ready_timeout_seconds)}); do
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

if [ "{1 if rerun_completed_shards else 0}" -eq 1 ]; then
  {shlex.quote(slurm_cfg.python_executable)} -m ollama_color_bypass_bench.hpc.runtime run-shard --shard "$SHARD_FILE" --config "$CONFIG_SNAPSHOT" --rerun
else
  {shlex.quote(slurm_cfg.python_executable)} -m ollama_color_bypass_bench.hpc.runtime run-shard --shard "$SHARD_FILE" --config "$CONFIG_SNAPSHOT"
fi

log "Shard $SLURM_ARRAY_TASK_ID finished."
""".strip("\n")

    content = "\n".join(header_lines) + "\n\n" + body + "\n"
    script_path.write_text(content, encoding="utf-8")
    script_path.chmod(0o755)
    return script_path


def submit_array_script(script_path: str | Path, *, dependency: str | None = None) -> SubmissionResult:
    """Submit a rendered array script with ``sbatch --parsable``."""

    script = Path(script_path)
    cmd = ["sbatch", "--parsable"]
    if dependency:
        cmd.extend(["--dependency", dependency])
    cmd.append(str(script))
    submit = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        check=False,
    )
    if submit.returncode != 0:
        stderr = submit.stderr.strip() or "sbatch failed"
        raise RuntimeError(f"sbatch failed for {script}: {stderr}")

    job_id = parse_job_id(submit.stdout)
    return SubmissionResult(job_id=job_id, stdout=submit.stdout, stderr=submit.stderr)
