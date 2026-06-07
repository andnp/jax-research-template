import jax
import jax.numpy as jnp

from process_control._jax_dataclass import jax_dataclass


@jax_dataclass
class SignalBus:
    flow: jax.Array
    outlet_residual: jax.Array

    @staticmethod
    def create(flow: float = 0.0, outlet_residual: float = 0.0) -> "SignalBus":
        return SignalBus(
            flow=jnp.array(flow),
            outlet_residual=jnp.array(outlet_residual),
        )
