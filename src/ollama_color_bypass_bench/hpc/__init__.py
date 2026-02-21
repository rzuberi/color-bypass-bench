"""HPC orchestration helpers for SLURM + CUDA sweeps."""

from .discovery import DiscoverySelection, discover_and_select_models
from .matrix import MatrixEntry, build_experiment_matrix
from .runtime import run_shard
from .shard import ShardSpec, write_shards
from .submit import SubmissionResult, render_submit_array_script, submit_array_script

__all__ = [
    "DiscoverySelection",
    "discover_and_select_models",
    "MatrixEntry",
    "build_experiment_matrix",
    "run_shard",
    "ShardSpec",
    "write_shards",
    "SubmissionResult",
    "render_submit_array_script",
    "submit_array_script",
]
