import jax
import jax.numpy as jnp
from process_control.benchmarks.ph_neutralization import (
    PhNeutralizationBenchmarkConfig,
    make_ph_neutralization_benchmark,
)


class TestPhNeutralizationBenchmark:
    def test_reset_returns_expected_shapes(self) -> None:
        """reset produces a 4D observation matching the documented contract."""
        reset_fn, _step_fn = make_ph_neutralization_benchmark(PhNeutralizationBenchmarkConfig())

        _state, obs = reset_fn(jax.random.PRNGKey(0))

        assert obs.shape == (4,)
        assert jnp.all(jnp.isfinite(obs))

    def test_step_is_jittable_and_preserves_shapes(self) -> None:
        """The plant is a plain JAX pytree, so jax.jit over reset+step must work."""
        config = PhNeutralizationBenchmarkConfig()
        reset_fn, step_fn = make_ph_neutralization_benchmark(config)
        state, _obs = reset_fn(jax.random.PRNGKey(0))

        jit_step = jax.jit(step_fn)
        new_state, obs, reward, done, info = jit_step(state, jnp.array(0.5), jax.random.PRNGKey(1))

        assert obs.shape == (4,)
        assert reward.shape == ()
        assert bool(done) is False
        assert new_state.step_count == 1
        assert jnp.all(jnp.isfinite(info["true_ph"]))

    def test_default_pi_loop_reaches_target_ph(self) -> None:
        """Driving the pump with the built-in PI dose brings pH to the 7.0 setpoint.

        Uses constant nominal flow/acid load (no diurnal or noise disturbance)
        so convergence reflects the control loop, not disturbance rejection.
        """
        config = PhNeutralizationBenchmarkConfig(
            diurnal_amplitude=0.0,
            mean_flow=50.0,
            min_flow=50.0,
            max_flow=50.0,
            acid_amplitude=0.0,
            acid_drift_scale=0.0,
            drift_scale=0.0,
            ph_noise_std=0.0,
            ph_lag_coefficient=0.0,
        )
        reset_fn, step_fn = make_ph_neutralization_benchmark(config)
        jit_step = jax.jit(step_fn)
        state, _obs = reset_fn(jax.random.PRNGKey(0))

        step_key = jax.random.PRNGKey(1)
        final_ph = jnp.array(0.0)
        # Drive the pump with the controller's own PI dose (pi_dose) each step.
        dose = jnp.array(config.pi_ff)
        for _ in range(200):
            state, _obs, _reward, _done, info = jit_step(state, dose, step_key)
            dose = info["pi_dose"]
            final_ph = info["true_ph"]

        assert jnp.abs(final_ph - config.target_ph) < 0.1
