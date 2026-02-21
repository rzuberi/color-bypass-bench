#!/usr/bin/env python3
"""Build and submit a full paper-scale SLURM array sweep."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import socket
import subprocess
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from ollama_color_bypass_bench.config import load_config, save_config
from ollama_color_bypass_bench.hpc.discovery import discover_and_select_models, write_model_list
from ollama_color_bypass_bench.hpc.matrix import build_experiment_matrix
from ollama_color_bypass_bench.hpc.shard import write_shards
from ollama_color_bypass_bench.hpc.submit import render_submit_array_script, submit_array_script
from ollama_color_bypass_bench.logging_io import write_csv


def _utc_timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _git_value(project_root: Path, args: list[str]) -> str | None:
    proc = subprocess.run(
        ["git", "-C", str(project_root), *args],
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        return None
    value = proc.stdout.strip()
    return value or None


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Submit paper-scale color bypass sweep to SLURM")
    parser.add_argument("--config", required=True, help="Path to YAML/JSON config")
    parser.add_argument("--run-id", default=None, help="Optional run id override")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing results/<run_id> if it already exists",
    )
    parser.add_argument(
        "--rerun",
        action="store_true",
        help="Pass --rerun to shard runtime (ignore completed shard outputs)",
    )
    parser.add_argument(
        "--no-submit",
        action="store_true",
        help="Prepare all files but do not call sbatch",
    )
    parser.add_argument(
        "--dependency",
        default=None,
        help="Optional SBATCH dependency expression (e.g., afterok:123456)",
    )
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    config = load_config(args.config)

    run_id = args.run_id or f"{config.output.run_name_prefix}_{_utc_timestamp()}"
    results_root = Path(config.output.results_dir).resolve()
    run_dir = results_root / run_id

    if run_dir.exists() and not args.force:
        raise SystemExit(
            f"Run directory already exists: {run_dir}\n"
            "Refusing to re-submit for an existing run_id. Use --force to overwrite."
        )

    if run_dir.exists() and args.force:
        shutil.rmtree(run_dir)

    run_dir.mkdir(parents=True, exist_ok=False)
    (run_dir / "slurm_logs").mkdir(parents=True, exist_ok=True)
    (run_dir / "shards").mkdir(parents=True, exist_ok=True)
    (run_dir / "summaries").mkdir(parents=True, exist_ok=True)

    discovery = discover_and_select_models(config, allow_bootstrap_server=True)
    config.models.malicious_models = list(discovery.m_models)
    config.models.helper_models = list(discovery.h_models)

    config_snapshot_path = run_dir / "config_snapshot.yaml"
    save_config(config, config_snapshot_path)

    write_model_list(run_dir / "ollama_models.txt", discovery.all_models)

    matrix_entries = build_experiment_matrix(
        config=config,
        run_id=run_id,
        m_models=config.models.malicious_models,
        h_models=config.models.helper_models,
    )
    if not matrix_entries:
        raise SystemExit("Matrix build produced zero episodes; check config and discovered model sets.")

    shard_specs = write_shards(
        matrix_entries=matrix_entries,
        shards_root=run_dir / "shards",
        requested_num_shards=config.hpc.sharding.num_shards,
    )
    if not shard_specs:
        raise SystemExit("No shards were generated")

    manifest_rows = [spec.to_manifest_row() for spec in shard_specs]
    write_csv(run_dir / "shard_manifest.csv", manifest_rows)

    submit_script_path = render_submit_array_script(
        run_dir=run_dir,
        run_id=run_id,
        config=config,
        config_snapshot_path=config_snapshot_path,
        shard_count=len(shard_specs),
        rerun_completed_shards=bool(args.rerun),
        project_root=PROJECT_ROOT,
    )

    submission_job_id: str | None = None
    if not args.no_submit:
        submission = submit_array_script(
            submit_script_path,
            dependency=str(args.dependency).strip() if args.dependency else None,
        )
        submission_job_id = submission.job_id

    env_payload = {
        "run_id": run_id,
        "created_at_utc": _utc_now_iso(),
        "submit_host": socket.gethostname(),
        "submit_cwd": str(Path.cwd().resolve()),
        "project_root": str(PROJECT_ROOT.resolve()),
        "config_path": str(Path(args.config).resolve()),
        "config_snapshot_path": str(config_snapshot_path),
        "submit_script_path": str(submit_script_path),
        "git_commit": _git_value(PROJECT_ROOT, ["rev-parse", "HEAD"]),
        "git_branch": _git_value(PROJECT_ROOT, ["rev-parse", "--abbrev-ref", "HEAD"]),
        "git_dirty": bool(_git_value(PROJECT_ROOT, ["status", "--porcelain"])),
        "discovered_models": discovery.all_models,
        "candidate_models": discovery.candidate_models,
        "selected_m_models": config.models.malicious_models,
        "selected_h_models": config.models.helper_models,
        "i_model": config.models.i_model,
        "matrix_episodes": len(matrix_entries),
        "num_shards": len(shard_specs),
        "job_id": submission_job_id,
        "submission_dependency": (str(args.dependency).strip() if args.dependency else None),
        "rerun_completed_shards": bool(args.rerun),
    }
    (run_dir / "env.json").write_text(json.dumps(env_payload, indent=2) + "\n", encoding="utf-8")

    print(f"Run ID: {run_id}")
    print(f"Results directory: {run_dir}")
    print(f"Matrix episodes: {len(matrix_entries)}")
    print(f"Shards: {len(shard_specs)}")
    if submission_job_id is None:
        print(f"Submit script ready (not submitted): {submit_script_path}")
    else:
        print(f"Submitted SLURM array job: {submission_job_id}")
        print(f"SLURM logs: {run_dir / 'slurm_logs'}")
        print(f"Shard outputs: {run_dir / 'shards'}")


if __name__ == "__main__":
    main()
