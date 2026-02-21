"""Runtime Ollama model discovery and heuristic role selection."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import os
from pathlib import Path
import re
import shutil
import subprocess
import time
from typing import Iterator

from ..config import BenchmarkConfig

DEFAULT_OLLAMA_HOST = "127.0.0.1:11434"
SIZE_TOKEN_PATTERN = re.compile(r"(?i)(\d+(?:\.\d+)?)b")
QUANTIZED_PATTERN = re.compile(r"(?i)(?:^|[^a-z0-9])(q\d|iq\d)")


@dataclass(frozen=True)
class DiscoverySelection:
    """Selected model pools for the sweep."""

    all_models: list[str]
    candidate_models: list[str]
    m_models: list[str]
    h_models: list[str]


def parse_ollama_list(output_text: str) -> list[str]:
    """Parse model names from ``ollama list`` tabular output."""

    models: list[str] = []
    seen: set[str] = set()

    for raw_line in output_text.splitlines():
        line = raw_line.strip()
        if not line:
            continue

        lowered = line.lower()
        if lowered.startswith("name") and "id" in lowered:
            continue

        token = line.split()[0]
        if token not in seen:
            models.append(token)
            seen.add(token)

    return models


def _run_ollama_list(*, env: dict[str, str], timeout_seconds: float) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["ollama", "list"],
        capture_output=True,
        text=True,
        env=env,
        check=False,
        timeout=timeout_seconds,
    )


@contextmanager
def temporary_ollama_server(
    *,
    ollama_host: str = DEFAULT_OLLAMA_HOST,
    ready_timeout_seconds: int = 90,
) -> Iterator[None]:
    """Start a temporary local Ollama server and stop it on exit."""

    if shutil.which("ollama") is None:
        raise RuntimeError("'ollama' command not found in PATH")

    env = os.environ.copy()
    env["OLLAMA_HOST"] = ollama_host

    serve_proc = subprocess.Popen(
        ["ollama", "serve"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        env=env,
    )
    try:
        ready = False
        for _ in range(max(1, ready_timeout_seconds)):
            check = _run_ollama_list(env=env, timeout_seconds=5.0)
            if check.returncode == 0:
                ready = True
                break
            time.sleep(1.0)

        if not ready:
            raise RuntimeError("Ollama server did not become ready while bootstrapping discovery")

        yield
    finally:
        if serve_proc.poll() is None:
            serve_proc.terminate()
            try:
                serve_proc.wait(timeout=10.0)
            except subprocess.TimeoutExpired:
                serve_proc.kill()


def discover_models(
    *,
    ollama_host: str = DEFAULT_OLLAMA_HOST,
    timeout_seconds: float = 20.0,
    allow_bootstrap_server: bool = False,
    ready_timeout_seconds: int = 90,
) -> list[str]:
    """Discover models by calling ``ollama list``."""

    if shutil.which("ollama") is None:
        raise RuntimeError("'ollama' command not found in PATH")

    env = os.environ.copy()
    env.setdefault("OLLAMA_HOST", ollama_host)

    try:
        list_proc = _run_ollama_list(env=env, timeout_seconds=timeout_seconds)
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError("Timed out while running 'ollama list'") from exc

    if list_proc.returncode == 0:
        return parse_ollama_list(list_proc.stdout)

    if not allow_bootstrap_server:
        stderr = list_proc.stderr.strip() or "unknown error"
        raise RuntimeError(f"Failed to run 'ollama list': {stderr}")

    with temporary_ollama_server(
        ollama_host=env["OLLAMA_HOST"],
        ready_timeout_seconds=ready_timeout_seconds,
    ):
        bootstrap_proc = _run_ollama_list(env=env, timeout_seconds=timeout_seconds)
        if bootstrap_proc.returncode != 0:
            stderr = bootstrap_proc.stderr.strip() or "unknown error"
            raise RuntimeError(f"Failed to run 'ollama list' after temporary serve startup: {stderr}")
        return parse_ollama_list(bootstrap_proc.stdout)


def _extract_size_billions(model_name: str) -> float | None:
    match = SIZE_TOKEN_PATTERN.search(model_name)
    if not match:
        return None
    try:
        return float(match.group(1))
    except ValueError:
        return None


def _model_sort_key(model_name: str) -> tuple[int, float, int, str]:
    lowered = model_name.lower()
    is_quantized = QUANTIZED_PATTERN.search(lowered) is not None
    size = _extract_size_billions(lowered)
    size_rank = size if size is not None else 999.0
    is_instruction_tuned = "instruct" in lowered or "chat" in lowered

    return (
        0 if is_quantized else 1,
        size_rank,
        0 if is_instruction_tuned else 1,
        lowered,
    )


def _filter_models(
    *,
    all_models: list[str],
    include_regex: str,
    exclude_regex: str,
    i_model: str,
) -> list[str]:
    include_compiled = re.compile(include_regex) if include_regex else None
    exclude_compiled = re.compile(exclude_regex) if exclude_regex else None
    i_model_lower = i_model.lower()

    candidates: list[str] = []
    for model_name in all_models:
        if model_name.lower() == i_model_lower:
            continue

        lowered = model_name.lower()
        if include_compiled is not None and include_compiled.search(lowered) is None:
            continue
        if exclude_compiled is not None and exclude_compiled.search(lowered) is not None:
            continue

        candidates.append(model_name)

    return sorted(candidates, key=_model_sort_key)


def _ollama_host_from_base_url(base_url: str) -> str:
    cleaned = base_url.strip()
    if not cleaned:
        return DEFAULT_OLLAMA_HOST
    without_scheme = cleaned.split("//")[-1]
    host_part = without_scheme.split("/")[0]
    return host_part or DEFAULT_OLLAMA_HOST


def discover_and_select_models(
    config: BenchmarkConfig,
    *,
    allow_bootstrap_server: bool,
) -> DiscoverySelection:
    """Discover available models and choose M/H subsets for the sweep."""

    all_models = discover_models(
        ollama_host=_ollama_host_from_base_url(config.ollama.base_url),
        allow_bootstrap_server=allow_bootstrap_server,
        ready_timeout_seconds=config.hpc.slurm.ollama_ready_timeout_seconds,
    )

    candidates = _filter_models(
        all_models=all_models,
        include_regex=config.hpc.model_discovery.include_regex,
        exclude_regex=config.hpc.model_discovery.exclude_regex,
        i_model=config.models.i_model,
    )
    if not candidates:
        # Fallback: allow i_model if no other candidates are available on this host.
        candidates = _filter_models(
            all_models=all_models,
            include_regex=config.hpc.model_discovery.include_regex,
            exclude_regex=config.hpc.model_discovery.exclude_regex,
            i_model="",
        )
    if not candidates:
        raise RuntimeError(
            "No candidate models matched discovery filters. "
            "Adjust hpc.model_discovery.include_regex / exclude_regex."
        )

    max_per_role = max(1, config.hpc.model_discovery.max_models_per_role)
    m_models = candidates[:max_per_role]

    if len(candidates) >= (2 * max_per_role):
        h_models = candidates[max_per_role : (2 * max_per_role)]
    else:
        h_models = candidates[:max_per_role]

    if not h_models:
        raise RuntimeError("Helper model set is empty after discovery")

    return DiscoverySelection(
        all_models=all_models,
        candidate_models=candidates,
        m_models=m_models,
        h_models=h_models,
    )


def write_model_list(path: str | Path, model_names: list[str]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if model_names:
        output_path.write_text("\n".join(model_names) + "\n", encoding="utf-8")
    else:
        output_path.write_text("", encoding="utf-8")
