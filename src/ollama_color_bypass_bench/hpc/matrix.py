"""Experiment matrix construction for large-scale sweeps."""

from __future__ import annotations

from dataclasses import asdict, dataclass

from ..config import BenchmarkConfig


@dataclass(frozen=True)
class MatrixEntry:
    """One deterministic episode spec in the full experiment matrix."""

    run_id: str
    episode_index: int
    episode_id: str
    m_model: str
    h_model: str
    i_model: str
    task_variant: str
    trial_index: int
    seed: int

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> "MatrixEntry":
        return cls(
            run_id=str(payload["run_id"]),
            episode_index=int(payload["episode_index"]),
            episode_id=str(payload["episode_id"]),
            m_model=str(payload["m_model"]),
            h_model=str(payload["h_model"]),
            i_model=str(payload["i_model"]),
            task_variant=str(payload["task_variant"]),
            trial_index=int(payload["trial_index"]),
            seed=int(payload["seed"]),
        )


def build_experiment_matrix(
    *,
    config: BenchmarkConfig,
    run_id: str,
    m_models: list[str],
    h_models: list[str],
) -> list[MatrixEntry]:
    """Build the full deterministic matrix: (M,H,variant,trial)."""

    entries: list[MatrixEntry] = []
    episode_index = 0

    for m_model in m_models:
        for h_model in h_models:
            for task_variant in config.experiment.task_variants:
                for trial_index in range(config.experiment.n_trials):
                    seed = config.experiment.base_seed + episode_index
                    entries.append(
                        MatrixEntry(
                            run_id=run_id,
                            episode_index=episode_index,
                            episode_id=f"{run_id}_{episode_index:08d}",
                            m_model=m_model,
                            h_model=h_model,
                            i_model=config.models.i_model,
                            task_variant=task_variant,
                            trial_index=trial_index,
                            seed=seed,
                        )
                    )
                    episode_index += 1

    return entries
