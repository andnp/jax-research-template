import jax
import jax.numpy as jnp
import pytest
from process_control.actuators.lagged import BSM1_KLA_ACTUATOR, LaggedActuatorParams, reset, step

MINUTE = 1.0 / 60.0


class TestBSM1Aeration:
    def test_stage_time_constant_matches_the_report(self) -> None:
        """BSM1 models aeration as two lags of T = T90 / 3.89 = 1.03 min."""
        assert BSM1_KLA_ACTUATOR.time_constant / MINUTE == pytest.approx(1.03, rel=2e-3)

    def test_output_is_ninety_percent_through_a_step_after_four_minutes(self) -> None:
        """A kLa step from rest is 90 % realised after the 4-minute response time."""
        _, output = step(reset(jnp.array(0.0), BSM1_KLA_ACTUATOR), jnp.array(10.0), BSM1_KLA_ACTUATOR, jnp.array(4.0 * MINUTE))

        assert jnp.allclose(output, 9.0, atol=1e-4)

    def test_command_is_clipped_to_the_handle_limits(self) -> None:
        """Commands beyond 360 d⁻¹ (15 h⁻¹) settle at the limit, and negative commands at zero."""
        state = reset(jnp.array(5.0), BSM1_KLA_ACTUATOR)

        _, high = step(state, jnp.array(50.0), BSM1_KLA_ACTUATOR, jnp.array(1.0))
        _, low = step(state, jnp.array(-3.0), BSM1_KLA_ACTUATOR, jnp.array(1.0))

        assert jnp.allclose(high, 15.0)
        assert jnp.allclose(low, 0.0)


class TestLaggedActuator:
    def test_actuator_at_rest_holds_its_output(self) -> None:
        """Commanding the current output leaves the actuator where it is."""
        params = LaggedActuatorParams(t90=0.1, order=3, low=-1.0, high=1.0)

        _, output = step(reset(jnp.array(0.4), params), jnp.array(0.4), params, jnp.array(0.05))

        assert jnp.allclose(output, 0.4)

    def test_higher_order_delays_the_early_response(self) -> None:
        """With equal T90, more stages respond later at first but meet at T90."""
        first = LaggedActuatorParams(t90=1.0, order=1, low=0.0, high=1.0)
        fourth = LaggedActuatorParams(t90=1.0, order=4, low=0.0, high=1.0)

        _, early_first = step(reset(jnp.array(0.0), first), jnp.array(1.0), first, jnp.array(0.1))
        _, early_fourth = step(reset(jnp.array(0.0), fourth), jnp.array(1.0), fourth, jnp.array(0.1))
        _, late_fourth = step(reset(jnp.array(0.0), fourth), jnp.array(1.0), fourth, jnp.array(1.0))

        assert early_fourth < early_first
        assert jnp.allclose(late_fourth, 0.9, atol=1e-5)

    def test_vmap_over_commands(self) -> None:
        """A batch of commands advances in one jitted call, each matching its unbatched step."""

        def actuate(command: jax.Array) -> jax.Array:
            return step(reset(jnp.array(0.0), BSM1_KLA_ACTUATOR), command, BSM1_KLA_ACTUATOR, jnp.array(MINUTE))[1]

        outputs = jax.jit(jax.vmap(actuate))(jnp.array([0.0, 6.0, 30.0]))

        assert outputs.shape == (3,)
        assert jnp.allclose(outputs[1], actuate(jnp.array(6.0)))
        assert jnp.allclose(outputs[2], actuate(jnp.array(15.0)))
