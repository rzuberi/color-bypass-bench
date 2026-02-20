"""Pairwise sweep runner for malicious/helper model combinations."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
import random
from typing import Any

from .analysis import compute_summary_rows
from .config import BenchmarkConfig, load_config, save_config
from .episode import EpisodeJob, run_episode
from .logging_io import append_jsonl, ensure_results_tree, write_csv
from .ollama_client import OllamaChatClient


def _build_jobs(config: BenchmarkConfig, run_id: str) -> list[EpisodeJob]:
    jobs: list[EpisodeJob] = []
    episode_counter = 0

    for m_model in config.models.malicious_models:
        for h_model in config.models.helper_models:
            for trial_index in range(config.experiment.n_trials):
                seed = config.experiment.base_seed + trial_index
                variant_rng = random.Random(seed)
                task_variant = variant_rng.choice(config.experiment.task_variants)
                jobs.append(
                    EpisodeJob(
                        run_id=run_id,
                        episode_id=f"{run_id}_{episode_counter:06d}",
                        m_model=m_model,
                        h_model=h_model,
                        trial_index=trial_index,
                        seed=seed,
                        task_variant=task_variant,
                    )
                )
                episode_counter += 1
    return jobs


def _run_single_job(job: EpisodeJob, config: BenchmarkConfig) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    client = OllamaChatClient(config.ollama)
    return run_episode(client=client, config=config, job=job)


def run_sweep(config: BenchmarkConfig) -> dict[str, Path]:
    """Run a full (M,H) sweep and write episodes/turns/summary outputs."""

    run_id = (
        f"{config.output.run_name_prefix}_"
        f"{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
    )

    dirs = ensure_results_tree(config.output.results_dir)
    episodes_path = dirs["episodes"] / f"{run_id}.jsonl"
    turns_path = dirs["turns"] / f"{run_id}.jsonl"
    summary_path = dirs["summaries"] / f"{run_id}.csv"
    config_snapshot_path = dirs["summaries"] / f"{run_id}_config_snapshot.yaml"

    save_config(config, config_snapshot_path)

    jobs = _build_jobs(config, run_id)
    episode_records: list[dict[str, Any]] = []

    if config.experiment.parallel_workers <= 1:
        client = OllamaChatClient(config.ollama)
        for job in jobs:
            episode_record, turn_records = run_episode(client=client, config=config, job=job)
            append_jsonl(episodes_path, episode_record)
            for turn_record in turn_records:
                append_jsonl(turns_path, turn_record)
            episode_records.append(episode_record)
    else:
        max_workers = config.experiment.parallel_workers
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(_run_single_job, job, config) for job in jobs]
            for future in as_completed(futures):
                episode_record, turn_records = future.result()
                append_jsonl(episodes_path, episode_record)
                for turn_record in turn_records:
                    append_jsonl(turns_path, turn_record)
                episode_records.append(episode_record)

    summary_rows = compute_summary_rows(episode_records)
    write_csv(summary_path, summary_rows)

    return {
        "episodes_path": episodes_path,
        "turns_path": turns_path,
        "summary_path": summary_path,
        "config_snapshot_path": config_snapshot_path,
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run color bypass benchmark sweep")
    parser.add_argument("--config", required=True, help="Path to YAML/JSON benchmark config")
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    config = load_config(args.config)
    outputs = run_sweep(config)

    print(f"Episodes: {outputs['episodes_path']}")
    print(f"Turns: {outputs['turns_path']}")
    print(f"Summary: {outputs['summary_path']}")
    print(f"Config snapshot: {outputs['config_snapshot_path']}")


if __name__ == "__main__":
    main()
