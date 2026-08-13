"""Optimizer acronym resolution for the installed MEALPY version and project custom optimizers."""
from __future__ import annotations

from functools import lru_cache

import mealpy


CUSTOM_OPTIMIZERS = ("MaCRO-DE", "DSADE", "DSADE_AWAD", "DBO")
CUSTOM_OPTIMIZER_ALIASES = {
    "MACRO-DE": "MaCRO-DE",
    "MACRO_DE": "MaCRO-DE",
    "DSADE": "DSADE",
    "DSA-DE": "DSADE",
    "DSA_DE": "DSADE",
    "DSADE_AWAD": "DSADE_AWAD",
    "DSADE-AWAD": "DSADE_AWAD",
    "DBO": "DBO",
}


@lru_cache(maxsize=1)
def _available_optimizer_names() -> dict[str, str]:
    optimizers = mealpy.get_all_optimizers(verbose=False)
    return {optimizer_name.casefold(): optimizer_name for optimizer_name in optimizers.keys()}


def _resolve_custom_optimizer(name: str) -> str | None:
    key = str(name).strip().upper()
    return CUSTOM_OPTIMIZER_ALIASES.get(key)


def resolve_optimizer_name(name: str) -> str:
    """Return the installed MEALPY class name or canonical project custom optimizer name."""
    raw_name = str(name).strip()
    custom_name = _resolve_custom_optimizer(raw_name)
    if custom_name is not None:
        return custom_name

    available_names = _available_optimizer_names()

    if raw_name.casefold() == "dmoa":
        dev_dmoa = available_names.get("devdmoa")
        if dev_dmoa is not None:
            return dev_dmoa

    candidates = [
        raw_name,
        f"Original{raw_name}",
        f"Base{raw_name}",
        f"Dev{raw_name}",
    ]
    for candidate in candidates:
        resolved_name = available_names.get(candidate.casefold())
        if resolved_name is not None:
            return resolved_name

    raise ValueError(f"Unknown MEALPY optimizer '{name}'. Tried: {', '.join(candidates)}")


def optimizer_acronym(name: str) -> str:
    """Return the user-facing acronym/name for plots, tables, and console output."""
    resolved_name = resolve_optimizer_name(name)
    if resolved_name in CUSTOM_OPTIMIZERS:
        return resolved_name
    if resolved_name.startswith("Original") and len(resolved_name) > len("Original"):
        return resolved_name[len("Original"):]
    return resolved_name


def list_available_optimizers() -> str:
    """List installed MEALPY optimizers plus project custom optimizers."""
    rows = []
    for optimizer_name in sorted(_available_optimizer_names().values(), key=str.casefold):
        display_name = optimizer_acronym(optimizer_name)
        rows.append((display_name, optimizer_name))

    width = max((len(display_name) for display_name, _ in rows), default=0)
    lines = [f"{display_name:<{width}} -> {optimizer_name}" for display_name, optimizer_name in rows]
    lines.append("")
    lines.append("Custom:")
    lines.extend(CUSTOM_OPTIMIZERS)
    return "\n".join(lines)
