#!/usr/bin/env python3
"""Generate cross-run HPC status tables and plots for README reporting."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import re
import subprocess
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd


STATE_ORDER = ("COMPLETED", "FAILED", "CANCELLED", "RUNNING", "PENDING")
STATE_COLORS = {
    "COMPLETED": "#16a34a",
    "FAILED": "#dc2626",
    "CANCELLED": "#f59e0b",
    "RUNNING": "#2563eb",
    "PENDING": "#64748b",
}


def _read_env(run_dir: Path) -> dict[str, Any]:
    env_path = run_dir / "env.json"
    if not env_path.exists():
        raise FileNotFoundError(f"Missing env.json in {run_dir}")
    return json.loads(env_path.read_text(encoding="utf-8"))


def _sacct_rows(job_id: str) -> list[dict[str, str]]:
    proc = subprocess.run(
        [
            "sacct",
            "-j",
            job_id,
            "--format=JobID,State,ExitCode,Elapsed",
            "-n",
            "-P",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"sacct failed for {job_id}: {proc.stderr.strip() or 'unknown error'}")

    rows: list[dict[str, str]] = []
    for raw in proc.stdout.splitlines():
        parts = raw.strip().split("|")
        if len(parts) < 4:
            continue
        rows.append(
            {
                "job_id": parts[0],
                "state": parts[1].split()[0],
                "exit_code": parts[2],
                "elapsed": parts[3],
            }
        )
    return rows


def _state_counts(rows: list[dict[str, str]], array_job_id: str) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for row in rows:
        jid = row["job_id"]
        if not jid.startswith(f"{array_job_id}_"):
            continue
        if "." in jid:
            continue
        counts[row["state"]] += 1
    return dict(counts)


def _canonical_reason(text: str) -> str:
    stripped = text.strip()
    if not stripped:
        return "Unknown"
    lowered = stripped.lower()
    if "no candidate models matched discovery filters" in lowered:
        return "Discovery filter mismatch"
    if "could not connect to ollama server" in lowered:
        return "Ollama server unavailable"
    if "failed to pull model" in lowered:
        return "Ollama pull race/failure"
    return stripped


def _reason_from_err(err_path: Path) -> str:
    text = err_path.read_text(encoding="utf-8", errors="replace")
    match = re.search(r"RuntimeError:\s*(.+)", text)
    if match:
        return _canonical_reason(match.group(1))
    match = re.search(r"Error:\s*(.+)", text)
    if match:
        return _canonical_reason(match.group(1))
    return "Unknown"


def _episode_stats(run_dir: Path) -> dict[str, int]:
    stats = Counter(
        {
            "episodes_logged": 0,
            "compromised_episodes": 0,
            "abstained_episodes": 0,
            "invalid_episodes": 0,
        }
    )
    for path in sorted((run_dir / "shards").glob("shard_*/episodes.jsonl")):
        with path.open("r", encoding="utf-8") as handle:
            for raw in handle:
                line = raw.strip()
                if not line:
                    continue
                payload = json.loads(line)
                stats["episodes_logged"] += 1
                if bool(payload.get("compromised")):
                    stats["compromised_episodes"] += 1
                if bool(payload.get("abstained")):
                    stats["abstained_episodes"] += 1
                if bool(payload.get("had_invalid")):
                    stats["invalid_episodes"] += 1
    return dict(stats)


def _completed_shards(run_dir: Path) -> tuple[int, int]:
    completed = 0
    total = 0
    for meta_path in sorted((run_dir / "shards").glob("shard_*/metadata.json")):
        total += 1
        try:
            payload = json.loads(meta_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        if bool(payload.get("completed")):
            completed += 1
    return completed, total


def _plot_states(df: pd.DataFrame, output_path: Path) -> None:
    runs = df["run_id"].tolist()
    x = list(range(len(runs)))
    bottom = [0] * len(runs)

    plt.figure(figsize=(10, 5))
    for state in STATE_ORDER:
        values = df[state].astype(int).tolist()
        plt.bar(
            x,
            values,
            bottom=bottom,
            label=state.title(),
            color=STATE_COLORS.get(state, "#94a3b8"),
        )
        bottom = [b + v for b, v in zip(bottom, values)]

    plt.xticks(x, runs, rotation=20, ha="right")
    plt.ylabel("Shard tasks")
    plt.title("HPC Array Outcomes by Run")
    plt.legend(loc="upper right", ncols=2)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=180)
    plt.close()


def _plot_reasons(reason_df: pd.DataFrame, output_path: Path) -> None:
    if reason_df.empty:
        return
    top = (
        reason_df.groupby("reason", as_index=False)["count"]
        .sum()
        .sort_values("count", ascending=False)
        .head(8)
    )
    plt.figure(figsize=(10, 4.5))
    plt.barh(top["reason"], top["count"], color="#ef4444")
    plt.gca().invert_yaxis()
    plt.xlabel("Count")
    plt.title("Top Failure Reasons Across HPC Runs")
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=180)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate HPC overview assets for README")
    parser.add_argument(
        "--runs",
        nargs="+",
        required=True,
        help="Run directories under results/ (full paths or relative).",
    )
    parser.add_argument(
        "--output_csv",
        default="results/summaries/hpc_runs_overview.csv",
        help="Output CSV for run-level state/episode summary.",
    )
    parser.add_argument(
        "--failure_csv",
        default="results/summaries/hpc_failure_reasons.csv",
        help="Output CSV for run-level failure reason counts.",
    )
    parser.add_argument(
        "--figures_dir",
        default="docs/figures",
        help="Directory for generated PNG figures.",
    )
    args = parser.parse_args()

    overview_rows: list[dict[str, Any]] = []
    reason_rows: list[dict[str, Any]] = []

    for run_arg in args.runs:
        run_dir = Path(run_arg).resolve()
        env = _read_env(run_dir)
        run_id = str(env.get("run_id", run_dir.name))
        job_id = str(env.get("job_id") or "").strip()
        if not job_id:
            continue

        rows = _sacct_rows(job_id)
        counts = defaultdict(int, _state_counts(rows, job_id))
        completed_meta, metadata_total = _completed_shards(run_dir)
        episode_stats = _episode_stats(run_dir)

        overview_rows.append(
            {
                "run_id": run_id,
                "job_id": job_id,
                "configured_shards": int(env.get("num_shards", 0) or 0),
                "configured_episodes": int(env.get("matrix_episodes", 0) or 0),
                "COMPLETED": int(counts["COMPLETED"]),
                "FAILED": int(counts["FAILED"]),
                "CANCELLED": int(counts["CANCELLED"]),
                "RUNNING": int(counts["RUNNING"]),
                "PENDING": int(counts["PENDING"]),
                "metadata_files": metadata_total,
                "completed_shards": completed_meta,
                **episode_stats,
            }
        )

        run_reason_counts: Counter[str] = Counter()
        for err_path in sorted((run_dir / "slurm_logs").glob("*.err")):
            run_reason_counts[_reason_from_err(err_path)] += 1
        for reason, count in sorted(run_reason_counts.items(), key=lambda item: (-item[1], item[0])):
            reason_rows.append(
                {
                    "run_id": run_id,
                    "job_id": job_id,
                    "reason": reason,
                    "count": int(count),
                }
            )

    overview_df = pd.DataFrame(overview_rows)
    reason_df = pd.DataFrame(reason_rows)

    output_csv = Path(args.output_csv).resolve()
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    overview_df.sort_values("run_id").to_csv(output_csv, index=False)

    failure_csv = Path(args.failure_csv).resolve()
    failure_csv.parent.mkdir(parents=True, exist_ok=True)
    reason_df.sort_values(["run_id", "count", "reason"], ascending=[True, False, True]).to_csv(
        failure_csv,
        index=False,
    )

    figures_dir = Path(args.figures_dir).resolve()
    if not overview_df.empty:
        _plot_states(overview_df.sort_values("run_id"), figures_dir / "hpc_runs_outcomes_stacked.png")
    if not reason_df.empty:
        _plot_reasons(reason_df, figures_dir / "hpc_failure_reasons_top.png")

    print(f"Wrote {output_csv}")
    print(f"Wrote {failure_csv}")
    print(f"Wrote {figures_dir / 'hpc_runs_outcomes_stacked.png'}")
    print(f"Wrote {figures_dir / 'hpc_failure_reasons_top.png'}")


if __name__ == "__main__":
    main()
