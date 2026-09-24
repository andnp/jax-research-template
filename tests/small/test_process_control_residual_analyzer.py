import jax
import jax.numpy as jnp
from process_control.sensors.residual_analyzer import ResidualAnalyzerParams, ResidualAnalyzerState, step


def test_analyzer_reports_initial_value_before_first_sample() -> None:
    """An analyzer holds its initial measured value until sampling begins."""
    initial_value = jnp.array(4.5)
    params = ResidualAnalyzerParams(noise_std=1.0, lag_coefficient=0.5, sample_period=2)
    state = ResidualAnalyzerState.create(initial_value)

    _, reading = step(state, jnp.array(9.0), params, jax.random.PRNGKey(0))

    assert jnp.allclose(reading, initial_value)
