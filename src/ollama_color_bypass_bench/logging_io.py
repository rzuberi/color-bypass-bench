"""I/O helpers for JSONL and CSV benchmark artifacts."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Iterable, Mapping


def ensure_results_tree(results_dir: str | Path) -> dict[str, Path]:
    """Create expected results directory tree if missing."""

    root = Path(results_dir)
    episodes = root / "episodes"
    turns = root / "turns"
    summaries = root / "summaries"
    episodes.mkdir(parents=True, exist_ok=True)
    turns.mkdir(parents=True, exist_ok=True)
    summaries.mkdir(parents=True, exist_ok=True)
    return {"root": root, "episodes": episodes, "turns": turns, "summaries": summaries}


def append_jsonl(path: str | Path, record: Mapping[str, Any]) -> None:
    """Append one JSON record as a line."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=True) + "\n")


def write_jsonl(path: str | Path, records: Iterable[Mapping[str, Any]]) -> None:
    """Write full JSONL file from iterable records."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=True) + "\n")


def write_csv(path: str | Path, rows: list[Mapping[str, Any]]) -> None:
    """Write CSV from homogeneous mapping rows."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not rows:
        output_path.write_text("", encoding="utf-8")
        return

    fieldnames = list(rows[0].keys())
    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(dict(row))
