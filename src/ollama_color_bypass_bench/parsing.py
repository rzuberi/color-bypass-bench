"""Robust deterministic parsing utilities for extracting RGB outputs."""

from __future__ import annotations

from dataclasses import dataclass
import json
import re


RGB = tuple[int, int, int]


JSON_OBJECT_PATTERN = re.compile(r"\{[^{}]*\}", re.DOTALL)
RGB_KEYED_PATTERN = re.compile(
    r"r\s*[:=]\s*(\d{1,3})\D+g\s*[:=]\s*(\d{1,3})\D+b\s*[:=]\s*(\d{1,3})",
    re.IGNORECASE | re.DOTALL,
)
RGB_TUPLE_PATTERN = re.compile(r"\(\s*(\d{1,3})\s*,\s*(\d{1,3})\s*,\s*(\d{1,3})\s*\)")
RGB_BARE_TRIPLE_PATTERN = re.compile(r"\b(\d{1,3})\s*,\s*(\d{1,3})\s*,\s*(\d{1,3})\b")


@dataclass(frozen=True)
class RGBParseResult:
    """Result of deterministic RGB extraction from model text."""

    rgb: RGB | None
    source: str
    raw_match: str | None = None
    error: str | None = None

    @property
    def is_valid(self) -> bool:
        return self.rgb is not None


def _coerce_int(value: object) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, str) and re.fullmatch(r"[+-]?\d+", value.strip()):
        try:
            return int(value.strip())
        except ValueError:
            return None
    return None


def _validate_rgb(r: int, g: int, b: int) -> RGB | None:
    if 0 <= r <= 255 and 0 <= g <= 255 and 0 <= b <= 255:
        return (r, g, b)
    return None


def _try_json_objects(text: str) -> RGBParseResult | None:
    first_out_of_range_error: RGBParseResult | None = None
    for match in JSON_OBJECT_PATTERN.finditer(text):
        raw = match.group(0)
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            continue
        if not isinstance(parsed, dict):
            continue

        if not {"r", "g", "b"}.issubset(parsed.keys()):
            continue

        r_val = _coerce_int(parsed.get("r"))
        g_val = _coerce_int(parsed.get("g"))
        b_val = _coerce_int(parsed.get("b"))
        if r_val is None or g_val is None or b_val is None:
            # Keep scanning for the first JSON object that has integer r/g/b.
            continue

        rgb = _validate_rgb(r_val, g_val, b_val)
        if rgb is None:
            if first_out_of_range_error is None:
                first_out_of_range_error = RGBParseResult(
                    rgb=None,
                    source="json",
                    raw_match=raw,
                    error="JSON object found with integer r/g/b but values were out of range [0,255]",
                )
            continue

        return RGBParseResult(rgb=rgb, source="json", raw_match=raw)

    return first_out_of_range_error


def _try_regex(text: str, pattern: re.Pattern[str], source: str) -> RGBParseResult | None:
    match = pattern.search(text)
    if not match:
        return None

    raw = match.group(0)
    r_val, g_val, b_val = (int(match.group(1)), int(match.group(2)), int(match.group(3)))
    rgb = _validate_rgb(r_val, g_val, b_val)
    if rgb is None:
        return RGBParseResult(
            rgb=None,
            source=source,
            raw_match=raw,
            error="Matched RGB pattern but values were out of range [0,255]",
        )
    return RGBParseResult(rgb=rgb, source=source, raw_match=raw)


def extract_rgb(text: str) -> RGBParseResult:
    """Extract RGB tuple from text using deterministic JSON-first parsing.

    Parsing order:
    1) First JSON object that has integer r/g/b keys.
    2) Regex fallback: ``r=..., g=..., b=...`` style.
    3) Regex fallback: ``(r,g,b)`` tuple.
    4) Regex fallback: bare ``r,g,b`` triple.
    """

    if not text.strip():
        return RGBParseResult(rgb=None, source="none", error="Empty output")

    json_result = _try_json_objects(text)
    if json_result is not None:
        return json_result

    for pattern, source in (
        (RGB_KEYED_PATTERN, "regex_keyed"),
        (RGB_TUPLE_PATTERN, "regex_tuple"),
        (RGB_BARE_TRIPLE_PATTERN, "regex_bare_triple"),
    ):
        regex_result = _try_regex(text, pattern, source)
        if regex_result is not None:
            return regex_result

    return RGBParseResult(
        rgb=None,
        source="none",
        error="No parsable RGB found (expected JSON object or supported regex fallback)",
    )
