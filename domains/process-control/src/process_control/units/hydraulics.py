"""Level geometry and a finite sharp-crested rectangular weir.

A positive head stores liquid above the crest; the crest is not a maximum volume. Geometry and
discharge calibration are installation specific and have no defaults here. The simplified law omits
the head and approach corrections of standard practice; see the USBR Water Measurement Manual,
rectangular weirs: https://www.usbr.gov/tsc/techreferences/mands/wmm/chap07_06.html
"""

import math
from typing import Protocol

import jax
import jax.numpy as jnp

from process_control._jax_dataclass import jax_dataclass

GRAVITY_M_PER_S2 = 9.80665
SECONDS_PER_HOUR = 3600.0


class LevelGeometry(Protocol):
    """Liquid level above the datum (m) as a function of stored volume (m³)."""

    def level_m(self, volume_m3: jax.Array) -> jax.Array: ...


@jax_dataclass
class PrismaticGeometry:
    """Constant plan area (m²), so level is volume over area."""

    area_m2: jax.Array | float

    def level_m(self, volume_m3: jax.Array) -> jax.Array:
        return volume_m3 / self.area_m2


@jax_dataclass
class RectangularWeir:
    """Crest height above the level datum (m), crest width (m) and discharge coefficient."""

    crest_m: jax.Array | float
    width_m: jax.Array | float
    discharge_coefficient: jax.Array | float

    def flow_m3_per_h(self, level_m: jax.Array) -> jax.Array:
        """Q = 3600 (2/3) C_d b_w sqrt(2 g) max(h - h_c, 0)^(3/2), with metres and seconds inside the law."""
        head = jnp.maximum(level_m - self.crest_m, 0.0)
        per_second = (2.0 / 3.0) * self.discharge_coefficient * self.width_m * math.sqrt(2.0 * GRAVITY_M_PER_S2) * head**1.5
        return SECONDS_PER_HOUR * per_second


def weir_overflow_m3_per_h(volume_m3: jax.Array, geometry: LevelGeometry, weir: RectangularWeir) -> jax.Array:
    """Overflow of a stored volume over a weir, in m³/h."""
    return weir.flow_m3_per_h(geometry.level_m(volume_m3))
