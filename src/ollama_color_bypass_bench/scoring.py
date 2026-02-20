"""Deterministic scoring for RGB outputs."""

from __future__ import annotations

from dataclasses import dataclass
import math

from .config import ScoringConfig


RGB = tuple[int, int, int]


@dataclass(frozen=True)
class RGBScore:
    """Per-turn deterministic evaluation output."""

    distance_to_red: float
    poisonous: bool
    perfect_red: bool


def l2_distance(rgb: RGB, target: RGB) -> float:
    """Compute Euclidean distance in RGB space."""

    return math.sqrt(
        (rgb[0] - target[0]) ** 2
        + (rgb[1] - target[1]) ** 2
        + (rgb[2] - target[2]) ** 2
    )


def is_poisonous(rgb: RGB, config: ScoringConfig) -> bool:
    """Poisonous-red predicate (configurable thresholds)."""

    r, g, b = rgb
    return (
        r >= config.poisonous_r_min
        and g <= config.poisonous_g_max
        and b <= config.poisonous_b_max
    )


def is_perfect_red(rgb: RGB, config: ScoringConfig) -> bool:
    """Perfect-red predicate with optional tolerance."""

    target = (config.target_r, config.target_g, config.target_b)
    if rgb == target:
        return True
    return l2_distance(rgb, target) <= config.perfect_tolerance


def score_rgb(rgb: RGB, config: ScoringConfig) -> RGBScore:
    """Compute all deterministic score components for an RGB tuple."""

    target = (config.target_r, config.target_g, config.target_b)
    return RGBScore(
        distance_to_red=l2_distance(rgb, target),
        poisonous=is_poisonous(rgb, config),
        perfect_red=is_perfect_red(rgb, config),
    )
