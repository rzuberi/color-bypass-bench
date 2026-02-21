"""Runtime shard execution logic for SLURM array tasks."""

from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import os
import json
from pathlib import Path
import subprocess
import time
from typing import Any

from ..config import BenchmarkConfig, load_config
from ..episode import EpisodeJob, run_episode
from ..logging_io import append_jsonl, write_csv
from ..ollama_client import OllamaChatClient
from .discovery import discover_and_select_models, write_model_list
from .matrix import MatrixEntry

try:
    import fcntl
except ImportError:  # pragma: no cover
    fcntl = None  # type: ignore[assignment]


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows

    with path.open("r", encoding="utf-8") as handle:
        for raw in handle:
            stripped = raw.strip()
            if not stripped:
                continue
            rows.append(json.loads(stripped))
    return rows


def _load_shard_entries(path: Path) -> list[MatrixEntry]:
    payload = _read_jsonl(path)
    return [MatrixEntry.from_dict(row) for row in payload]


def _ollama_show(model_name: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["ollama", "show", model_name],
        capture_output=True,
        text=True,
        check=False,
    )


def _ollama_models_root() -> Path:
    configured = os.environ.get("OLLAMA_MODELS", "").strip()
    if configured:
        return Path(configured).expanduser()
    return (Path.home() / ".ollama" / "models").expanduser()


def _acquire_pull_lock(lock_path: Path):
    if fcntl is None:
        return None
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    handle = lock_path.open("a+", encoding="utf-8")
    fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
    return handle


def _release_pull_lock(handle: Any) -> None:
    if handle is None or fcntl is None:
        return
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
    finally:
        handle.close()


def _ensure_model_available(model_name: str, *, auto_pull: bool) -> None:
    show = _ollama_show(model_name)
    if show.returncode == 0:
        return

    if not auto_pull:
        stderr = show.stderr.strip() or "model not found"
        raise RuntimeError(f"Model '{model_name}' is not available locally: {stderr}")

    lock_handle = _acquire_pull_lock(_ollama_models_root() / ".pull.lock")
    try:
        # Another shard may have completed the pull while this shard waited on the lock.
        if _ollama_show(model_name).returncode == 0:
            return

        last_error = "pull failed"
        for attempt in range(1, 4):
            pull = subprocess.run(
                ["ollama", "pull", model_name],
                capture_output=True,
                text=True,
                check=False,
            )
            if pull.returncode == 0 and _ollama_show(model_name).returncode == 0:
                return

            stderr = pull.stderr.strip()
            if stderr:
                last_error = stderr
            if attempt < 3:
                time.sleep(float(attempt))

        raise RuntimeError(
            f"Failed to pull model '{model_name}' after 3 attempts: {last_error}"
        )
    finally:
        _release_pull_lock(lock_handle)


def _compute_shard_summary_rows(
    *,
    run_id: str,
    shard_id: int,
    episodes: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str], dict[str, float]] = defaultdict(
        lambda: {
            "episodes": 0,
            "compromised": 0,
            "abstained": 0,
            "had_invalid": 0,
            "turns_total": 0,
            "compromise_turn_total": 0,
            "compromised_with_turn": 0,
            "final_distance_sum": 0.0,
            "final_distance_count": 0,
        }
    )

    for episode in episodes:
        key = (
            str(episode.get("m_model", "")),
            str(episode.get("h_model", "")),
            str(episode.get("i_model", "")),
        )
        agg = grouped[key]
        agg["episodes"] += 1
        agg["compromised"] += 1 if bool(episode.get("compromised")) else 0
        agg["abstained"] += 1 if bool(episode.get("abstained")) else 0
        agg["had_invalid"] += 1 if bool(episode.get("had_invalid")) else 0
        agg["turns_total"] += int(episode.get("turns_executed", 0) or 0)

        compromise_turn = episode.get("compromise_turn")
        if compromise_turn is not None:
            agg["compromise_turn_total"] += int(compromise_turn)
            agg["compromised_with_turn"] += 1

        final_distance = episode.get("final_distance_to_red")
        if final_distance is not None:
            agg["final_distance_sum"] += float(final_distance)
            agg["final_distance_count"] += 1

    rows: list[dict[str, Any]] = []
    for (m_model, h_model, i_model), agg in sorted(grouped.items()):
        rows.append(
            {
                "run_id": run_id,
                "shard_id": shard_id,
                "m_model": m_model,
                "h_model": h_model,
                "i_model": i_model,
                "episodes": int(agg["episodes"]),
                "compromised": int(agg["compromised"]),
                "abstained": int(agg["abstained"]),
                "had_invalid": int(agg["had_invalid"]),
                "turns_total": int(agg["turns_total"]),
                "compromise_turn_total": int(agg["compromise_turn_total"]),
                "compromised_with_turn": int(agg["compromised_with_turn"]),
                "final_distance_sum": float(agg["final_distance_sum"]),
                "final_distance_count": int(agg["final_distance_count"]),
            }
        )

    return rows


def _metadata_is_complete(metadata_path: Path, expected_episodes: int) -> bool:
    if not metadata_path.exists():
        return False
    try:
        payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return False
    if not isinstance(payload, dict):
        return False
    return bool(payload.get("completed")) and int(payload.get("completed_episodes", -1)) == expected_episodes


def run_shard(
    *,
    shard_path: str | Path,
    config_path: str | Path,
    rerun: bool = False,
) -> dict[str, Any]:
    """Execute a single shard file and write shard-local outputs."""

    shard_file = Path(shard_path).resolve()
    shard_dir = shard_file.parent
    metadata_path = shard_dir / "metadata.json"
    episodes_path = shard_dir / "episodes.jsonl"
    turns_path = shard_dir / "turns.jsonl"
    shard_summary_path = shard_dir / "shard_summary.csv"

    config: BenchmarkConfig = load_config(config_path)
    entries = _load_shard_entries(shard_file)
    expected_episodes = len(entries)
    shard_id = int(shard_dir.name.split("_")[-1]) if "_" in shard_dir.name else 0
    run_id = entries[0].run_id if entries else "unknown"

    if not rerun and _metadata_is_complete(metadata_path, expected_episodes):
        return {
            "run_id": run_id,
            "shard_id": shard_id,
            "expected_episodes": expected_episodes,
            "completed_episodes": expected_episodes,
            "skipped": True,
        }

    if rerun:
        for path in (episodes_path, turns_path, shard_summary_path, metadata_path):
            if path.exists():
                path.unlink()

    discovered_all_models: list[str] = []
    discovered_m_models: list[str] = []
    discovered_h_models: list[str] = []
    discovery_error: str | None = None
    try:
        discovered = discover_and_select_models(config, allow_bootstrap_server=False)
        discovered_all_models = list(discovered.all_models)
        discovered_m_models = list(discovered.m_models)
        discovered_h_models = list(discovered.h_models)
    except RuntimeError as exc:
        # Do not fail the shard up front; required models are enforced below
        # and can be auto-pulled per config if absent.
        discovery_error = str(exc)

    required_models: set[str] = {config.models.i_model}
    for entry in entries:
        required_models.add(entry.m_model)
        required_models.add(entry.h_model)

    for model_name in sorted(required_models):
        _ensure_model_available(
            model_name,
            auto_pull=config.hpc.runtime.auto_pull_missing_models,
        )
    runtime_models = sorted(set(discovered_all_models).union(required_models))
    write_model_list(shard_dir / "ollama_models_runtime.txt", runtime_models)

    existing_episode_ids: set[str] = set()
    if episodes_path.exists() and not rerun:
        for row in _read_jsonl(episodes_path):
            episode_id = row.get("episode_id")
            if isinstance(episode_id, str) and episode_id:
                existing_episode_ids.add(episode_id)

    started_at = _utc_now()
    client = OllamaChatClient(config.ollama)

    for entry in entries:
        if not rerun and entry.episode_id in existing_episode_ids:
            continue

        job = EpisodeJob(
            run_id=entry.run_id,
            episode_id=entry.episode_id,
            m_model=entry.m_model,
            h_model=entry.h_model,
            trial_index=entry.trial_index,
            seed=entry.seed,
            task_variant=entry.task_variant,
        )
        episode_record, turn_records = run_episode(client=client, config=config, job=job)
        append_jsonl(episodes_path, episode_record)
        for turn_record in turn_records:
            append_jsonl(turns_path, turn_record)

    episodes = _read_jsonl(episodes_path)
    shard_summary_rows = _compute_shard_summary_rows(
        run_id=run_id,
        shard_id=shard_id,
        episodes=episodes,
    )
    write_csv(shard_summary_path, shard_summary_rows)

    completed_episodes = len(episodes)
    completed = completed_episodes == expected_episodes
    ended_at = _utc_now()

    metadata = {
        "run_id": run_id,
        "shard_id": shard_id,
        "shard_file": str(shard_file),
        "expected_episodes": expected_episodes,
        "completed_episodes": completed_episodes,
        "completed": completed,
        "rerun": rerun,
        "started_at_utc": started_at,
        "ended_at_utc": ended_at,
        "models_discovered_runtime": discovered_all_models,
        "m_models_runtime": discovered_m_models,
        "h_models_runtime": discovered_h_models,
        "discovery_error": discovery_error,
        "models_required_runtime": sorted(required_models),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")

    if not completed:
        raise RuntimeError(
            f"Shard incomplete: expected {expected_episodes} episodes, found {completed_episodes}"
        )

    return {
        "run_id": run_id,
        "shard_id": shard_id,
        "expected_episodes": expected_episodes,
        "completed_episodes": completed_episodes,
        "skipped": False,
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="HPC runtime entrypoints")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_shard_parser = subparsers.add_parser("run-shard", help="Run one shard file")
    run_shard_parser.add_argument("--shard", required=True, help="Path to shard_input.jsonl")
    run_shard_parser.add_argument("--config", required=True, help="Path to run config snapshot")
    run_shard_parser.add_argument("--rerun", action="store_true", help="Ignore existing outputs and rerun")

    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    if args.command == "run-shard":
        result = run_shard(shard_path=args.shard, config_path=args.config, rerun=bool(args.rerun))
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
