import jax.numpy as jnp
from process_control.chemistry.ph_model import PhModelParams, compute_ph


class TestPhModel:
    def test_zero_excess_is_neutral(self) -> None:
        """Zero base excess maps to neutral pH."""
        result = compute_ph(jnp.array(0.0), PhModelParams())

        assert jnp.allclose(result, 7.0)

    def test_ph_is_finite_bounded_and_monotone(self) -> None:
        """The titration curve stays physical while increasing with base excess."""
        excess = jnp.array([-1.0, -0.1, 0.0, 0.1, 1.0])

        result = compute_ph(excess, PhModelParams(sensitivity=0.5))

        assert jnp.all(jnp.isfinite(result))
        assert jnp.all((result > 0.0) & (result < 14.0))
        assert jnp.all(jnp.diff(result) > 0.0)

    def test_sensitivity_widens_the_neutrality_transition(self) -> None:
        """Larger sensitivity reduces pH displacement for the same excess."""
        excess = jnp.array(0.1)
        sharp = compute_ph(excess, PhModelParams(sensitivity=0.01))
        wide = compute_ph(excess, PhModelParams(sensitivity=1.0))

        assert jnp.abs(wide - 7.0) < jnp.abs(sharp - 7.0)
