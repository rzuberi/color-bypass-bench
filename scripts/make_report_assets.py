#!/usr/bin/env python3
"""Build tables and diagrams used in README result summaries."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import re
import subprocess
from typing import Iterable

import matplotlib.pyplot as plt
import pandas as pd


def _run_sacct(job_id: str) -> list[dict[str, str]]:
    cmd = [
        "sacct",
        "-j",
        job_id,
        "--format=JobID,State,ExitCode,Elapsed,NodeList",
        "-n",
        "-P",
    ]
    proc = subprocess.run(cmd, check=False, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"sacct failed for job {job_id}: {proc.stderr.strip() or 'unknown error'}")

    rows: list[dict[str, str]] = []
    for raw in proc.stdout.splitlines():
        parts = raw.strip().split("|")
        if len(parts) < 5:
            continue
        rows.append(
            {
                "job_id": parts[0],
                "state": parts[1],
                "exit_code": parts[2],
                "elapsed": parts[3],
                "nodelist": parts[4],
            }
        )
    return rows


def _count_states(rows: Iterable[dict[str, str]], array_job_id: str) -> list[tuple[str, int]]:
    counts: dict[str, int] = {}
    for row in rows:
        job_id = row["job_id"]
        if not job_id.startswith(f"{array_job_id}_"):
            continue
        if "." in job_id:
            continue
        state = row["state"].split()[0]
        counts[state] = counts.get(state, 0) + 1
    return sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))


def _extract_failure_reasons(err_files: list[Path]) -> list[tuple[str, int]]:
    counts: dict[str, int] = {}
    pat = re.compile(r"RuntimeError:\s*(.+)")
    for path in err_files:
        text = path.read_text(encoding="utf-8", errors="replace")
        reason = "Unknown"
        match = pat.search(text)
        if match:
            reason = match.group(1).strip()
        counts[reason] = counts.get(reason, 0) + 1
    return sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _plot_bar(path: Path, labels: list[str], values: list[float], title: str, ylabel: str) -> None:
    plt.figure(figsize=(8, 4.5))
    bars = plt.bar(labels, values, color=["#3b82f6", "#ef4444", "#f59e0b", "#10b981", "#64748b"][: len(labels)])
    plt.title(title)
    plt.ylabel(ylabel)
    plt.xticks(rotation=20, ha="right")
    for bar, value in zip(bars, values):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{value:.3g}", ha="center", va="bottom", fontsize=9)
    plt.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=180)
    plt.close()


def _find_latest_tiny_summary(default_glob_root: Path) -> Path:
    candidates = sorted(default_glob_root.glob("color_bypass_tiny_1turn_*.csv"))
    candidates = [p for p in candidates if not p.name.endswith("_config_snapshot.yaml")]
    if not candidates:
        raise FileNotFoundError("No tiny summary CSV found under results/summaries")
    return candidates[-1]


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate report tables and figures")
    parser.add_argument("--hpc_run_dir", required=True, help="Path to results/<run_id>")
    parser.add_argument("--tiny_summary_csv", default=None, help="Path to tiny baseline summary CSV")
    parser.add_argument("--output_dir", default="docs/figures", help="Directory for generated figure PNGs")
    args = parser.parse_args()

    run_dir = Path(args.hpc_run_dir).resolve()
    output_dir = Path(args.output_dir).resolve()

    env_json = run_dir / "env.json"
    if not env_json.exists():
        raise FileNotFoundError(f"Missing env.json: {env_json}")
    env = pd.read_json(env_json, typ="series")
    job_id = str(env.get("job_id", "")).strip()
    if not job_id:
        raise RuntimeError(f"No job_id in {env_json}")

    sacct_rows = _run_sacct(job_id)
    state_counts = _count_states(sacct_rows, job_id)
    state_rows = [{"state": state, "count": count} for state, count in state_counts]
    _write_csv(run_dir / "summaries" / "job_state_counts.csv", state_rows)

    err_files = sorted((run_dir / "slurm_logs").glob("*.err"))
    reason_counts = _extract_failure_reasons(err_files)
    reason_rows = [{"reason": reason, "count": count} for reason, count in reason_counts]
    _write_csv(run_dir / "summaries" / "failure_reason_counts.csv", reason_rows)

    if state_counts:
        _plot_bar(
            output_dir / "hpc_job_state_counts.png",
            [s for s, _ in state_counts],
            [float(c) for _, c in state_counts],
            "CUDA Array Job State Counts",
            "Shard count",
        )

    tiny_summary_csv = Path(args.tiny_summary_csv) if args.tiny_summary_csv else _find_latest_tiny_summary(Path("results/summaries"))
    tiny_df = pd.read_csv(tiny_summary_csv)
    if tiny_df.empty:
        raise RuntimeError(f"Tiny summary has no rows: {tiny_summary_csv}")

    row = tiny_df.iloc[0]
    rates = {
        "compromise_rate": float(row.get("compromise_rate", 0.0) or 0.0),
        "abstain_rate": float(row.get("abstain_rate", 0.0) or 0.0),
        "invalid_rate": float(row.get("invalid_rate", 0.0) or 0.0),
    }
    _plot_bar(
        output_dir / "tiny_baseline_rates.png",
        list(rates.keys()),
        list(rates.values()),
        "Tiny Baseline Outcome Rates",
        "Rate",
    )

    print(f"Generated: {run_dir / 'summaries' / 'job_state_counts.csv'}")
    print(f"Generated: {run_dir / 'summaries' / 'failure_reason_counts.csv'}")
    print(f"Generated: {output_dir / 'hpc_job_state_counts.png'}")
    print(f"Generated: {output_dir / 'tiny_baseline_rates.png'}")


if __name__ == "__main__":
    main()
