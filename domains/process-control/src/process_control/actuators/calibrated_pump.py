from dataclasses import dataclass
from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax.typing import ArrayLike


class PumpDelivery(NamedTuple):
    """The command as asked, the speed the pump actually ran, and the volume it delivered in L/h."""

    command_rpm: jax.Array
    realized_rpm: jax.Array
    flow_l_per_h: jax.Array


@dataclass(frozen=True)
class CalibratedPump:
    """Positive-displacement dosing pump whose delivery is linear in speed, one calibration per head.

    Heads are alternative pumps for one duty, such as duty and standby. The linear policy assumes the
    head, tubing and backpressure regime in which it was calibrated, so calibrations are parameter
    leaves to be updated from flow tests. Speed limits are per head, in RPM.
    """

    names: tuple[str, ...]
    liters_per_hour_per_rpm: jax.Array
    min_rpm: jax.Array
    max_rpm: jax.Array

    def __post_init__(self):
        if len(set(self.names)) != len(self.names):
            raise ValueError(f"head names must be unique: {self.names}")

    def head_index(self, name: str):
        return self.names.index(name)

    def deliver(self, command_rpm: ArrayLike, head: ArrayLike, permitted: ArrayLike = True):
        """Gate and limit the command to a realized speed, then convert that speed to a flow.

        A closed interlock or a non-positive command stops the pump. Otherwise the command is clipped
        to the head's speed range. Conversion uses only the realized speed.
        """
        command = jnp.asarray(command_rpm)
        index = jnp.asarray(head, dtype=int)
        limited = jnp.clip(command, self.min_rpm[index], self.max_rpm[index])
        realized = jnp.where(jnp.asarray(permitted) & (command > 0.0), limited, 0.0)
        return PumpDelivery(command, realized, realized * self.liters_per_hour_per_rpm[index])


jax.tree_util.register_dataclass(
    CalibratedPump,
    data_fields=["liters_per_hour_per_rpm", "min_rpm", "max_rpm"],
    meta_fields=["names"],
)
