import jax.numpy as jnp
from process_control.chemistry.precipitation import PrecipitationParams, precipitate


class TestPrecipitation:
    def test_no_dose_no_removal(self) -> None:
        params = PrecipitationParams()
        eff, consumed = precipitate(jnp.array(5.0), jnp.array(0.0), params)
        assert float(eff) == 5.0
        assert float(consumed) == 0.0

    def test_high_dose_removes_most(self) -> None:
        params = PrecipitationParams()
        eff, _ = precipitate(jnp.array(5.0), jnp.array(50.0), params)
        assert float(eff) < 0.5

    def test_floor_respected(self) -> None:
        params = PrecipitationParams(p_min=0.05)
        eff, _ = precipitate(jnp.array(5.0), jnp.array(1000.0), params)
        assert float(eff) >= 0.05

    def test_monod_saturation(self) -> None:
        params = PrecipitationParams()
        # With a moderate dose, absolute removal is limited when P is low
        # due to Monod half-saturation. At higher P, absolute removal increases.
        eff_low, _ = precipitate(jnp.array(0.2), jnp.array(5.0), params)
        eff_high, _ = precipitate(jnp.array(5.0), jnp.array(5.0), params)
        removed_low = 0.2 - float(eff_low)
        removed_high = 5.0 - float(eff_high)
        assert removed_low < removed_high
