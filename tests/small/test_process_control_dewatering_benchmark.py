import jax
import jax.numpy as jnp
from process_control.units.dewatering import DewateringParams
from process_control.units.dewatering import reset as dw_reset
from process_control.units.dewatering import step as dw_step


class TestDewateringUnitCapture:
    def test_peak_capture_reaches_max_capture(self) -> None:
        """At the optimum dose ratio sqrt(k*ki) the solids captured equal max_capture of the feed."""
        params = DewateringParams()
        state = dw_reset(params, jax.random.PRNGKey(0))
        feed_tss = jnp.array(10000.0)
        optimum_ratio = float(jnp.sqrt(params.k_polymer_capture * params.ki_polymer_capture))
        dose = jnp.array(optimum_ratio * float(feed_tss) / 1000.0)

        new_state, *_ = dw_step(state, feed_tss, jnp.array(20.0), dose, jnp.array(0.5), params, jnp.array(0.05))

        solids_in = float(feed_tss) * 20.0 * 0.05 / 1e3
        assert abs(float(new_state.cake_produced) / solids_in - params.max_capture) < 1e-4
