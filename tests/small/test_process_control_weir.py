"""Level geometry and the finite rectangular weir law."""

import jax
import jax.numpy as jnp
import pytest
from process_control.units.hydraulics import GRAVITY_M_PER_S2, PrismaticGeometry, RectangularWeir, weir_overflow_m3_per_h

GEOMETRY = PrismaticGeometry(area_m2=4.0)


def _flow(volume: float, crest: float = 1.0, width: float = 0.5, coefficient: float = 0.6) -> float:
    return float(weir_overflow_m3_per_h(jnp.asarray(volume), GEOMETRY, RectangularWeir(crest, width, coefficient)))


class TestWeirLaw:
    @pytest.mark.parametrize("volume", [0.0, 2.0, 4.0])
    def test_no_head_at_or_below_the_crest_gives_no_flow(self, volume: float):
        """Level at or below the crest (volume at most 4 m3 for a 1 m crest on 4 m2) discharges nothing."""
        assert _flow(volume) == 0.0

    def test_matches_the_closed_form_for_a_known_head(self):
        """Q = 3600 (2/3) C_d b sqrt(2g) h^1.5 for 0.25 m of head, in m3/h."""
        expected = 3600.0 * (2.0 / 3.0) * 0.6 * 0.5 * (2.0 * GRAVITY_M_PER_S2) ** 0.5 * 0.25**1.5
        assert _flow(5.0) == pytest.approx(expected)

    def test_flow_scales_with_head_to_the_three_halves(self):
        """Quadrupling the head multiplies the flow by eight."""
        assert _flow(4.0 + 4.0 * 0.4) == pytest.approx(_flow(4.0 + 4.0 * 0.1) * 8.0)

    def test_flow_is_linear_in_width_and_coefficient(self):
        """Doubling the crest width or the discharge coefficient doubles the flow."""
        base = _flow(5.0)
        assert _flow(5.0, width=1.0) == pytest.approx(2.0 * base)
        assert _flow(5.0, coefficient=1.2) == pytest.approx(2.0 * base)

    def test_flow_rises_monotonically_with_volume(self):
        """More stored liquid never reduces the overflow."""
        flows = [_flow(volume) for volume in (3.0, 4.0, 4.5, 5.0, 8.0)]
        assert flows == sorted(flows)
        assert flows[-1] > flows[0]

    def test_unit_check_one_metre_head_over_one_metre_crest(self):
        """With C_d = 1 and b = 1 m, 1 m of head gives (2/3) sqrt(2g) m3/s, which is that times 3600 per hour."""
        flow = float(RectangularWeir(0.0, 1.0, 1.0).flow_m3_per_h(jnp.asarray(1.0)))
        assert flow == pytest.approx(3600.0 * 2.0 / 3.0 * 4.42869, rel=1e-5)

    def test_law_is_jittable_and_differentiable_away_from_the_crest(self):
        """The overflow is a pure array function whose volume sensitivity is positive above the crest."""

        def overflow(volume: jax.Array) -> jax.Array:
            return weir_overflow_m3_per_h(volume, GEOMETRY, RectangularWeir(1.0, 0.5, 0.6))

        slope = jax.jit(jax.grad(overflow))(jnp.asarray(5.0))
        assert float(slope) > 0.0


class TestGeometry:
    def test_prismatic_level_is_volume_over_area(self):
        """A 4 m2 tank holds 1 m of liquid at 4 m3, and level scales linearly with volume."""
        assert float(GEOMETRY.level_m(jnp.asarray(4.0))) == pytest.approx(1.0)
        assert float(GEOMETRY.level_m(jnp.asarray(10.0))) == pytest.approx(2.5)
