"""Result aggregation and summary-table generation."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import statistics
from typing import Any, Iterable

from .logging_io import write_csv


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    """Read a JSONL file into a list of dictionaries."""

    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _percentile(values: list[float], p: float) -> float:
    if not values:
        return float("nan")
    if len(values) == 1:
        return values[0]

    ordered = sorted(values)
    rank = (len(ordered) - 1) * p
    lower = int(rank)
    upper = min(lower + 1, len(ordered) - 1)
    weight = rank - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _median_and_iqr(values: Iterable[float]) -> tuple[float | None, float | None]:
    data = [float(v) for v in values]
    if not data:
        return None, None
    median = float(statistics.median(data))
    q1 = _percentile(data, 0.25)
    q3 = _percentile(data, 0.75)
    return median, float(q3 - q1)


def compute_summary_rows(episodes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Compute per-(M,H) summary metrics from episode records."""

    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for ep in episodes:
        grouped[(ep["m_model"], ep["h_model"])].append(ep)

    rows: list[dict[str, Any]] = []
    for (m_model, h_model), group in sorted(grouped.items()):
        total = len(group)
        compromised_eps = [ep for ep in group if ep.get("compromised")]
        abstained_eps = [ep for ep in group if ep.get("abstained")]
        invalid_eps = [ep for ep in group if ep.get("had_invalid")]

        compromise_turns = [float(ep["compromise_turn"]) for ep in compromised_eps if ep.get("compromise_turn") is not None]
        all_turns = [float(ep.get("turns_executed", 0)) for ep in group]
        final_distances = [float(ep["final_distance_to_red"]) for ep in group if ep.get("final_distance_to_red") is not None]

        turns_median, turns_iqr = _median_and_iqr(all_turns)
        distance_median, distance_iqr = _median_and_iqr(final_distances)
        avg_turns_to_compromise = (
            sum(compromise_turns) / len(compromise_turns) if compromise_turns else None
        )

        rows.append(
            {
                "m_model": m_model,
                "h_model": h_model,
                "episodes": total,
                "compromise_rate": len(compromised_eps) / total if total else 0.0,
                "abstain_rate": len(abstained_eps) / total if total else 0.0,
                "invalid_rate": len(invalid_eps) / total if total else 0.0,
                "avg_turns_to_compromise": avg_turns_to_compromise,
                "turns_median": turns_median,
                "turns_iqr": turns_iqr,
                "distance_median": distance_median,
                "distance_iqr": distance_iqr,
            }
        )

    return rows


def summarize_results(
    results_dir: str | Path,
    *,
    output_csv: str | Path | None = None,
) -> Path:
    """Aggregate all episode JSONL files under ``results_dir/episodes``."""

    results_root = Path(results_dir)
    episodes_dir = results_root / "episodes"
    episode_files = sorted(episodes_dir.glob("*.jsonl"))

    episodes: list[dict[str, Any]] = []
    for file_path in episode_files:
        episodes.extend(read_jsonl(file_path))

    summary_rows = compute_summary_rows(episodes)
    output_path = (
        Path(output_csv)
        if output_csv is not None
        else results_root / "summaries" / "summary_all_pairings.csv"
    )
    write_csv(output_path, summary_rows)
    return output_path


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Summarize color bypass benchmark results")
    parser.add_argument("--results_dir", required=True, help="Root results directory")
    parser.add_argument(
        "--output_csv",
        default=None,
        help="Optional output CSV path (defaults to results/summaries/summary_all_pairings.csv)",
    )
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    output_path = summarize_results(args.results_dir, output_csv=args.output_csv)
    print(f"Wrote summary to: {output_path}")


if __name__ == "__main__":
    main()
