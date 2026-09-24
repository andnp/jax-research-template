import math

import jax
import jax.numpy as jnp
from process_control.chemistry.ph_model import PhModelParams, compute_ph


class TestPhModel:
    def test_matches_strong_acid_base_closed_form_without_buffer(self) -> None:
        """With no carbonate buffer, pH follows the exact strong acid/base charge balance.

        The reference values are computed in double-precision Python floats:
        float32 catastrophic cancellation in (-BE + sqrt(BE**2 + 4*Kw))/2 would
        otherwise swamp the tiny Kw = 1e-14 term.
        """
        values = [-0.01, -0.001, 0.001, 0.01]
        kw = 1e-14
        expected = [-math.log10((-be + math.sqrt(be**2 + 4.0 * kw)) / 2.0) for be in values]

        result = compute_ph(jnp.array(values), PhModelParams(buffer_capacity=0.0))

        assert jnp.allclose(result, jnp.array(expected), atol=1e-3)

    def test_ph_is_finite_bounded_and_monotone(self) -> None:
        """The titration curve stays physical while increasing with net base excess."""
        excess = jnp.array([-0.02, -0.01, -0.001, 0.0, 0.001, 0.01, 0.02])

        result = compute_ph(excess, PhModelParams())

        assert jnp.all(jnp.isfinite(result))
        assert jnp.all((result > 0.0) & (result < 14.0))
        assert jnp.all(jnp.diff(result) > 0.0)

    def test_buffer_capacity_resists_ph_change(self) -> None:
        """A larger carbonate buffer needs more net base to reach the same pH shift."""
        excess = jnp.array(0.002)
        low_buffer = compute_ph(excess, PhModelParams(buffer_capacity=0.001))
        high_buffer = compute_ph(excess, PhModelParams(buffer_capacity=0.02))

        assert high_buffer < low_buffer

    def test_jit_and_vmap_compatible(self) -> None:
        """The Newton solve uses only fixed-count, data-independent iteration."""
        excess = jnp.array([-0.01, 0.0, 0.01])
        params = PhModelParams()

        def f(x: jax.Array) -> jax.Array:
            return compute_ph(x, params)

        assert jnp.allclose(jax.jit(f)(excess), compute_ph(excess, params))
        assert jnp.allclose(jax.vmap(f)(excess), compute_ph(excess, params))
