"""Deterministic matrix sharding helpers for SLURM job arrays."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ..logging_io import write_jsonl
from .matrix import MatrixEntry


@dataclass(frozen=True)
class ShardSpec:
    """Manifest entry for one shard."""

    shard_id: int
    shard_name: str
    shard_dir: Path
    shard_file: Path
    episode_count: int

    def to_manifest_row(self) -> dict[str, object]:
        return {
            "shard_id": self.shard_id,
            "shard_name": self.shard_name,
            "shard_file": str(self.shard_file),
            "episode_count": self.episode_count,
        }


def write_shards(
    *,
    matrix_entries: list[MatrixEntry],
    shards_root: str | Path,
    requested_num_shards: int,
) -> list[ShardSpec]:
    """Split and write matrix entries into deterministic shard input files."""

    if requested_num_shards <= 0:
        raise ValueError("requested_num_shards must be > 0")

    root = Path(shards_root)
    root.mkdir(parents=True, exist_ok=True)

    if not matrix_entries:
        return []

    num_shards = min(requested_num_shards, len(matrix_entries))
    buckets: list[list[MatrixEntry]] = [[] for _ in range(num_shards)]

    for entry in matrix_entries:
        shard_zero_index = entry.episode_index % num_shards
        buckets[shard_zero_index].append(entry)

    specs: list[ShardSpec] = []
    for zero_index, rows in enumerate(buckets):
        shard_id = zero_index + 1
        shard_name = f"shard_{shard_id:04d}"
        shard_dir = root / shard_name
        shard_dir.mkdir(parents=True, exist_ok=True)
        shard_file = shard_dir / "shard_input.jsonl"

        write_jsonl(shard_file, [row.to_dict() for row in rows])

        specs.append(
            ShardSpec(
                shard_id=shard_id,
                shard_name=shard_name,
                shard_dir=shard_dir,
                shard_file=shard_file,
                episode_count=len(rows),
            )
        )

    return specs
