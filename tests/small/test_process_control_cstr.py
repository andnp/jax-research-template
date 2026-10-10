import jax
import jax.numpy as jnp
from process_control.units.cstr import CSTRParams, CSTRState, step


class TestCSTR:
    def test_reaches_exact_steady_state_in_one_large_step(self) -> None:
        """The exact exponential update lands exactly on steady state, unlike explicit Euler."""
        params = CSTRParams(volume=100.0)
        state = CSTRState(concentration=jnp.array(0.0))
        flow = jnp.array(10.0)
        inlet_concentration = jnp.array(2.0)
        addition_rate = jnp.array(0.0)
        steady_state = inlet_concentration + addition_rate / flow

        result = step(state, flow, inlet_concentration, addition_rate, params, dt=jnp.array(1000.0))

        assert jnp.allclose(result.concentration, steady_state, atol=1e-6)

    def test_stable_when_dt_far_exceeds_time_constant(self) -> None:
        """A dt many multiples of tau does not overshoot or blow up (unlike explicit Euler)."""
        params = CSTRParams(volume=10.0)
        state = CSTRState(concentration=jnp.array(5.0))
        flow = jnp.array(10.0)  # tau = volume / flow = 1.0
        inlet_concentration = jnp.array(1.0)
        addition_rate = jnp.array(0.0)

        result = step(state, flow, inlet_concentration, addition_rate, params, dt=jnp.array(50.0))

        assert jnp.isfinite(result.concentration)
        assert jnp.allclose(result.concentration, inlet_concentration, atol=1e-9)

    def test_addition_rate_shifts_steady_state(self) -> None:
        """A positive addition_rate raises the steady-state concentration above the inlet value."""
        params = CSTRParams(volume=50.0)
        state = CSTRState(concentration=jnp.array(0.0))
        flow = jnp.array(5.0)
        inlet_concentration = jnp.array(0.0)
        addition_rate = jnp.array(2.5)
        expected_steady_state = inlet_concentration + addition_rate / flow

        result = step(state, flow, inlet_concentration, addition_rate, params, dt=jnp.array(500.0))

        assert jnp.allclose(result.concentration, expected_steady_state, atol=1e-6)

    def test_jit_compatible(self) -> None:
        """The closed-form update has no data-dependent control flow."""
        params = CSTRParams(volume=100.0)
        state = CSTRState(concentration=jnp.array(1.0))

        def f(flow: jax.Array, cin: jax.Array, add: jax.Array, dt: jax.Array) -> CSTRState:
            return step(state, flow, cin, add, params, dt)

        result = jax.jit(f)(jnp.array(10.0), jnp.array(2.0), jnp.array(0.5), jnp.array(1.0))

        assert jnp.isfinite(result.concentration)

    def test_stacked_invariants_mix_independently_with_per_component_sources(self) -> None:
        """A [S, Ct]-style vector state reaches each component's own steady state, so reaction invariants can share one CSTR."""
        params = CSTRParams(volume=10.0)
        state = CSTRState(concentration=jnp.array([0.0, 4.0]))
        inlet_concentration = jnp.array([-3.0, 5.0])
        addition_rate = jnp.array([2.0, 0.0])
        flow = jnp.array(5.0)

        result = step(state, flow, inlet_concentration, addition_rate, params, dt=jnp.array(1000.0))

        assert result.concentration.shape == (2,)
        assert jnp.allclose(result.concentration, inlet_concentration + addition_rate / flow, atol=1e-6)
