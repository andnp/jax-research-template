"""Metering pump with a stable-range floor (turndown) and a transport delay along the dose line.

Rates and times share one unit system (for example mol/min and min). `dt` is a parameter rather than a
`step` argument because it fixes the length of the delay history and the interpolation weights, which
must be static; a per-call `dt` could silently disagree with the state it is applied to.
"""

import math
from dataclasses import dataclass

import jax
import jax.numpy as jnp

from process_control._jax_dataclass import jax_dataclass


@dataclass(frozen=True)
class MeteringPumpParams:
    """Stable range [max_output / turndown, max_output]; commands below half the floor stop the pump.

    The rate delivered to the process is the pump output delayed by `delay` along the dose line, with
    `delay` and `dt` in the same time unit.
    """

    max_output: float
    turndown: float
    delay: float
    dt: float

    def __post_init__(self) -> None:
        if self.turndown < 1 or self.delay < 0 or self.dt <= 0:
            raise ValueError("turndown must be at least 1, delay non-negative and dt positive")

    @property
    def min_output(self) -> float:
        return self.max_output / self.turndown

    @property
    def lag_steps(self) -> int:
        return math.floor(self.delay / self.dt)


@jax_dataclass
class MeteringPumpState:
    history: jax.Array  # pump output of the most recent lag_steps + 1 steps, newest first


def reset(params: MeteringPumpParams, output: float | jax.Array = 0.0) -> MeteringPumpState:
    """A pump that has been running at `output` for longer than its line delay."""
    return MeteringPumpState(history=jnp.full(params.lag_steps + 1, jnp.asarray(output, dtype=float)))


def step(state: MeteringPumpState, command: jax.Array, params: MeteringPumpParams) -> tuple[MeteringPumpState, jax.Array, jax.Array]:
    """Advance one step, returning the pump output and the rate delivered to the process, averaged over the step.

    For a piecewise-constant output the delayed signal averaged over a step is exact:
    frac * u[k-m-1] + (1 - frac) * u[k-m] with delay = (m + frac) * dt.
    """
    floor = params.min_output
    output = jnp.where(command < 0.5 * floor, 0.0, jnp.clip(command, floor, params.max_output))
    m = params.lag_steps
    frac = params.delay / params.dt - m
    history = jnp.concatenate([output[None], state.history])[: m + 2]
    delivered = frac * history[m + 1] + (1.0 - frac) * history[m] if frac > 0 else history[m]
    return MeteringPumpState(history=history[: m + 1]), output, delivered
