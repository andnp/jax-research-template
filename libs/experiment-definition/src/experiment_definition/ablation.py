"""Ablation model — named overrides that clone the base search space."""

from __future__ import annotations

from pydantic import BaseModel


class AblationSpec(BaseModel):
    """Represents a named ablation that fixes one or more parameters.

    Args:
        name: Short identifier for the ablation (e.g. "no_gae").
        overrides: Mapping of parameter names to their fixed values.
    """

    name: str
    overrides: dict[str, object]
