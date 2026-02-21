#!/usr/bin/env python3
"""Aggregate shard summaries for an HPC run."""

from __future__ import annotations

import argparse
from collections import defaultdict
import csv
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from ollama_color_bypass_bench.logging_io import write_csv


def _to_int(value: object) -> int:
    if value is None:
        return 0
    text = str(value).strip()
    if not text:
        return 0
    return int(float(text))


def _to_float(value: object) -> float:
    if value is None:
        return 0.0
    text = str(value).strip()
    if not text:
        return 0.0
    return float(text)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Aggregate HPC shard summaries")
    parser.add_argument("--results", required=True, help="results/<run_id> directory")
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    run_dir = Path(args.results).resolve()
    shard_summary_files = sorted((run_dir / "shards").glob("shard_*/shard_summary.csv"))
    if not shard_summary_files:
        raise SystemExit(f"No shard_summary.csv files found under: {run_dir / 'shards'}")

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

    shard_files_used = 0
    for summary_file in shard_summary_files:
        with summary_file.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            rows = list(reader)
        if not rows:
            continue
        shard_files_used += 1

        for row in rows:
            key = (
                str(row.get("m_model", "")),
                str(row.get("h_model", "")),
                str(row.get("i_model", "")),
            )
            agg = grouped[key]
            agg["episodes"] += _to_int(row.get("episodes"))
            agg["compromised"] += _to_int(row.get("compromised"))
            agg["abstained"] += _to_int(row.get("abstained"))
            agg["had_invalid"] += _to_int(row.get("had_invalid"))
            agg["turns_total"] += _to_int(row.get("turns_total"))
            agg["compromise_turn_total"] += _to_int(row.get("compromise_turn_total"))
            agg["compromised_with_turn"] += _to_int(row.get("compromised_with_turn"))
            agg["final_distance_sum"] += _to_float(row.get("final_distance_sum"))
            agg["final_distance_count"] += _to_int(row.get("final_distance_count"))

    if not grouped:
        raise SystemExit("Shard summaries were found but contained no rows.")

    run_id = run_dir.name
    per_pair_rows: list[dict[str, object]] = []
    overall = {
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

    for (m_model, h_model, i_model), agg in sorted(grouped.items()):
        episodes = int(agg["episodes"])
        compromised = int(agg["compromised"])
        abstained = int(agg["abstained"])
        had_invalid = int(agg["had_invalid"])
        compromised_with_turn = int(agg["compromised_with_turn"])
        compromise_turn_total = int(agg["compromise_turn_total"])
        final_distance_count = int(agg["final_distance_count"])
        final_distance_sum = float(agg["final_distance_sum"])

        per_pair_rows.append(
            {
                "run_id": run_id,
                "m_model": m_model,
                "h_model": h_model,
                "i_model": i_model,
                "episodes": episodes,
                "compromised": compromised,
                "abstained": abstained,
                "had_invalid": had_invalid,
                "compromise_rate": (compromised / episodes) if episodes else 0.0,
                "abstain_rate": (abstained / episodes) if episodes else 0.0,
                "invalid_rate": (had_invalid / episodes) if episodes else 0.0,
                "avg_turns_to_compromise": (
                    compromise_turn_total / compromised_with_turn
                    if compromised_with_turn
                    else None
                ),
                "avg_final_distance_to_red": (
                    final_distance_sum / final_distance_count
                    if final_distance_count
                    else None
                ),
            }
        )

        overall["episodes"] += episodes
        overall["compromised"] += compromised
        overall["abstained"] += abstained
        overall["had_invalid"] += had_invalid
        overall["turns_total"] += int(agg["turns_total"])
        overall["compromise_turn_total"] += compromise_turn_total
        overall["compromised_with_turn"] += compromised_with_turn
        overall["final_distance_sum"] += final_distance_sum
        overall["final_distance_count"] += final_distance_count

    summaries_dir = run_dir / "summaries"
    summaries_dir.mkdir(parents=True, exist_ok=True)

    per_pair_path = summaries_dir / "per_pair_summary.csv"
    write_csv(per_pair_path, per_pair_rows)

    overall_episodes = int(overall["episodes"])
    overall_compromised = int(overall["compromised"])
    overall_abstained = int(overall["abstained"])
    overall_invalid = int(overall["had_invalid"])
    overall_compromised_with_turn = int(overall["compromised_with_turn"])
    overall_final_distance_count = int(overall["final_distance_count"])

    overall_row = {
        "run_id": run_id,
        "shard_summaries_used": shard_files_used,
        "pairs": len(per_pair_rows),
        "episodes": overall_episodes,
        "compromised": overall_compromised,
        "abstained": overall_abstained,
        "had_invalid": overall_invalid,
        "compromise_rate": (overall_compromised / overall_episodes) if overall_episodes else 0.0,
        "abstain_rate": (overall_abstained / overall_episodes) if overall_episodes else 0.0,
        "invalid_rate": (overall_invalid / overall_episodes) if overall_episodes else 0.0,
        "avg_turns_to_compromise": (
            int(overall["compromise_turn_total"]) / overall_compromised_with_turn
            if overall_compromised_with_turn
            else None
        ),
        "avg_final_distance_to_red": (
            float(overall["final_distance_sum"]) / overall_final_distance_count
            if overall_final_distance_count
            else None
        ),
    }

    overall_path = summaries_dir / "overall_summary.csv"
    write_csv(overall_path, [overall_row])

    print(f"Per-pair summary: {per_pair_path}")
    print(f"Overall summary: {overall_path}")


if __name__ == "__main__":
    main()
