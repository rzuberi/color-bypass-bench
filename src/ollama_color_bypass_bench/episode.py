"""Single-episode orchestration for the color bypass benchmark."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import re
from typing import Any

from .config import BenchmarkConfig
from .ollama_client import OllamaChatClient
from .parsing import extract_rgb
from .prompts import (
    build_feedback_for_m,
    build_h_system_prompt,
    build_i_system_prompt,
    build_m_system_prompt,
)
from .scoring import score_rgb


ABSTAIN_PATTERN = re.compile(
    r"^\s*<<ABSTAIN>>\s*reason:\s*(?P<reason>.+?)\s*<</ABSTAIN>>\s*$",
    re.IGNORECASE | re.DOTALL,
)


@dataclass(frozen=True)
class EpisodeJob:
    """Input metadata for a single episode run."""

    run_id: str
    episode_id: str
    m_model: str
    h_model: str
    trial_index: int
    seed: int
    task_variant: str


def _turn_seed(base_seed: int, turn_index: int, offset: int) -> int:
    """Derive deterministic per-turn, per-role seed."""

    return (base_seed * 100_000) + (turn_index * 10) + offset


def _parse_abstention(text: str) -> tuple[bool, str | None]:
    match = ABSTAIN_PATTERN.match(text.strip())
    if not match:
        return (False, None)
    reason = match.group("reason").strip()
    return (True, reason)


def run_episode(
    client: OllamaChatClient,
    config: BenchmarkConfig,
    job: EpisodeJob,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Run one episode for a (M_model, H_model) pairing."""

    max_turns = config.experiment.max_turns
    m_system = build_m_system_prompt(config.scoring)
    h_system = build_h_system_prompt(config.scoring)
    i_system = build_i_system_prompt()

    m_history: list[dict[str, str]] = [{"role": "system", "content": m_system}]
    h_history: list[dict[str, str]] = [{"role": "system", "content": h_system}]

    current_rgb = (0, 0, 0)
    current_score = score_rgb(current_rgb, config.scoring)
    last_invalid = False
    last_parse_error: str | None = None
    valid_rgb_count = 0

    outcome = "max_turns"
    compromise_turn: int | None = None

    turn_logs: list[dict[str, Any]] = []
    started_at = datetime.now(timezone.utc)

    for turn_index in range(1, max_turns + 1):
        feedback_to_m = build_feedback_for_m(
            turn_index=turn_index,
            max_turns=max_turns,
            task_variant=job.task_variant,
            current_rgb=current_rgb,
            distance_to_red=current_score.distance_to_red,
            poisonous=current_score.poisonous,
            last_invalid=last_invalid,
            last_parse_error=last_parse_error,
        )

        m_history.append({"role": "user", "content": feedback_to_m})
        m_message = client.chat(
            model=job.m_model,
            messages=m_history,
            decode=config.decoding.malicious,
            seed=_turn_seed(job.seed, turn_index, 1),
        )
        m_history.append({"role": "assistant", "content": m_message})

        h_history.append({"role": "user", "content": m_message})
        h_response = client.chat(
            model=job.h_model,
            messages=h_history,
            decode=config.decoding.helper,
            seed=_turn_seed(job.seed, turn_index, 2),
        )
        h_history.append({"role": "assistant", "content": h_response})

        abstained, abstain_reason = _parse_abstention(h_response)
        if abstained:
            turn_logs.append(
                {
                    "run_id": job.run_id,
                    "episode_id": job.episode_id,
                    "trial_index": job.trial_index,
                    "seed": job.seed,
                    "task_variant": job.task_variant,
                    "turn_index": turn_index,
                    "m_model": job.m_model,
                    "h_model": job.h_model,
                    "i_model": config.models.i_model,
                    "feedback_to_m": feedback_to_m,
                    "m_message": m_message,
                    "h_response": h_response,
                    "h_mode": "abstain",
                    "abstain_reason": abstain_reason,
                    "instructions_to_i": None,
                    "i_response": None,
                    "parse_source": None,
                    "parse_error": None,
                    "invalid_rgb": False,
                    "rgb": None,
                    "distance_to_red": None,
                    "poisonous": None,
                    "perfect_red": None,
                    "compromised_on_turn": False,
                }
            )
            outcome = "abstained"
            break

        instructions_to_i = h_response.strip()
        i_messages = [
            {"role": "system", "content": i_system},
            {"role": "user", "content": instructions_to_i},
        ]
        i_response = client.chat(
            model=config.models.i_model,
            messages=i_messages,
            decode=config.decoding.innocent,
            seed=_turn_seed(job.seed, turn_index, 3),
        )

        parsed = extract_rgb(i_response)
        compromised_on_turn = False

        if parsed.is_valid:
            assert parsed.rgb is not None
            current_rgb = parsed.rgb
            current_score = score_rgb(current_rgb, config.scoring)
            valid_rgb_count += 1
            last_invalid = False
            last_parse_error = None
            if current_score.poisonous or current_score.perfect_red:
                compromised_on_turn = True
                compromise_turn = turn_index
                outcome = "compromised"
        else:
            last_invalid = True
            last_parse_error = parsed.error

        turn_logs.append(
            {
                "run_id": job.run_id,
                "episode_id": job.episode_id,
                "trial_index": job.trial_index,
                "seed": job.seed,
                "task_variant": job.task_variant,
                "turn_index": turn_index,
                "m_model": job.m_model,
                "h_model": job.h_model,
                "i_model": config.models.i_model,
                "feedback_to_m": feedback_to_m,
                "m_message": m_message,
                "h_response": h_response,
                "h_mode": "instructions",
                "abstain_reason": None,
                "instructions_to_i": instructions_to_i,
                "i_response": i_response,
                "parse_source": parsed.source,
                "parse_error": parsed.error,
                "invalid_rgb": not parsed.is_valid,
                "rgb": list(parsed.rgb) if parsed.rgb is not None else None,
                "distance_to_red": current_score.distance_to_red if parsed.is_valid else None,
                "poisonous": current_score.poisonous if parsed.is_valid else None,
                "perfect_red": current_score.perfect_red if parsed.is_valid else None,
                "compromised_on_turn": compromised_on_turn,
            }
        )

        if compromised_on_turn:
            break

    ended_at = datetime.now(timezone.utc)
    invalid_turns = sum(1 for turn in turn_logs if turn.get("invalid_rgb"))

    episode_record: dict[str, Any] = {
        "run_id": job.run_id,
        "episode_id": job.episode_id,
        "started_at_utc": started_at.isoformat(),
        "ended_at_utc": ended_at.isoformat(),
        "trial_index": job.trial_index,
        "seed": job.seed,
        "task_variant": job.task_variant,
        "m_model": job.m_model,
        "h_model": job.h_model,
        "i_model": config.models.i_model,
        "max_turns": max_turns,
        "turns_executed": len(turn_logs),
        "outcome": outcome,
        "compromised": outcome == "compromised",
        "abstained": outcome == "abstained",
        "compromise_turn": compromise_turn,
        "invalid_turns": invalid_turns,
        "had_invalid": invalid_turns > 0,
        "final_rgb": list(current_rgb) if valid_rgb_count > 0 else None,
        "final_distance_to_red": current_score.distance_to_red if valid_rgb_count > 0 else None,
        "final_poisonous": current_score.poisonous if valid_rgb_count > 0 else None,
        "final_perfect_red": current_score.perfect_red if valid_rgb_count > 0 else None,
        "m_history": m_history,
        "h_history": h_history,
        "turns": turn_logs,
    }
    return episode_record, turn_logs
