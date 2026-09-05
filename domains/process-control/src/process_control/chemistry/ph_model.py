from dataclasses import dataclass

import jax
import jax.numpy as jnp


@dataclass(frozen=True)
class PhModelParams:
    sensitivity: float = 0.01  # base_excess units at which pH transitions most rapidly


def compute_ph(base_excess: jax.Array, params: PhModelParams) -> jax.Array:
    """Map base excess concentration to pH via a smooth titration curve.

    Uses a scaled tanh approximation of a strong acid / strong base titration.
    Returns pH in (0, 14).

    - base_excess > 0: basic region, pH > 7
    - base_excess < 0: acidic region, pH < 7
    - sensitivity controls the width of the neutrality transition zone
    """
    return jnp.array(7.0) + jnp.array(7.0) * jnp.tanh(base_excess / params.sensitivity)
