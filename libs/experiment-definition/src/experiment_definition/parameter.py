"""Parameter model — a named hyperparameter with JAX-aware metadata."""

from __future__ import annotations

from pydantic import BaseModel

from .component import Component

ParameterValue = int | float | str | bool | None


class ParameterSpec(BaseModel):
    """Specifies a single hyperparameter sweep axis.

    Args:
        name: Parameter name (e.g. "lr").
        values: The list of candidate values.
        is_static: When True this parameter requires JAX recompilation when
            its value changes; the runner will partition PENDING work into
            separate JIT-compiled kernels per unique static configuration.
        component: Optional component scope (None means global).
        conditions: Key/value pairs that must hold for this parameter to be
            included in a configuration (e.g. ``{"use_gae": True}``).
    """

    name: str
    values: list[ParameterValue]
    is_static: bool = False
    component: Component | None = None
    conditions: dict[str, ParameterValue] = {}

    model_config = {"arbitrary_types_allowed": True}
