"""Actuators whose output follows a bounded command through a chain of first-order lags.

Times are in hours. The BSM1 aeration preset follows Alex et al. (2008), Sec. 7.4 and Table 15:
https://www2.iea.lth.se/publications/Reports/LTH-IEA-7229.pdf
"""

from dataclasses import dataclass

import jax
import jax.numpy as jnp

from process_control._jax_dataclass import jax_dataclass
from process_control.lag_chain import advance_lag_chain, time_constant_for_t90


@dataclass(frozen=True)
class LaggedActuatorParams:
    """Command bounds and a chain of `order` lags reaching 90 % of a step after `t90`."""

    t90: float
    order: int
    low: float
    high: float

    @property
    def time_constant(self) -> float:
        return time_constant_for_t90(self.t90, self.order)


BSM1_KLA_ACTUATOR = LaggedActuatorParams(t90=4.0 / 60.0, order=2, low=0.0, high=360.0 / 24.0)


@jax_dataclass
class LaggedActuatorState:
    stages: jax.Array


def reset(output: jax.Array, params: LaggedActuatorParams) -> LaggedActuatorState:
    """An actuator at rest at `output`."""
    return LaggedActuatorState(stages=jnp.full(params.order, jnp.asarray(output)))


def step(state: LaggedActuatorState, command: jax.Array, params: LaggedActuatorParams, dt: jax.Array) -> tuple[LaggedActuatorState, jax.Array]:
    """Advance over dt with the command held, returning the realised output."""
    target = jnp.clip(command, params.low, params.high)
    stages = advance_lag_chain(state.stages, target, params.time_constant, dt)
    return LaggedActuatorState(stages=stages), stages[-1]
