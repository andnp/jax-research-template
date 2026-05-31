import jax
import jax.numpy as jnp

from process_control.actuators.dosing_system import (
    DosingSystemParams,
    DosingSystemState,
    reset,
    step,
)


class TestDosingSystem:
    def test_reset_creates_valid_state(self) -> None:
        state = reset(7.0, 50.0, jax.random.PRNGKey(0))
        assert float(state.sensor_value) == 7.0
        assert float(state.pump_output) == 50.0
        assert float(state.pi_integral) == 0.0
        assert float(state.sensor_drift) == 0.0

    def test_step_returns_correct_shapes(self) -> None:
        params = DosingSystemParams()
        state = reset(7.0, 50.0, jax.random.PRNGKey(0))
        dt = jnp.array(0.02)

        new_state, sensed, pump = step(state, jnp.array(9.5), jnp.array(7.0), params, dt, jax.random.PRNGKey(1))
        assert sensed.shape == ()
        assert pump.shape == ()
        assert new_state.sensor_value.shape == ()

    def test_pi_drives_pump_toward_setpoint(self) -> None:
        """When true PV is below setpoint, PI should ramp pump up."""
        params = DosingSystemParams(
            sensor_noise_std=0.0, sensor_lag=0.0, sensor_drift_rate=0.0,
            kp=5.0, ki=1.0, ff=0.0, output_min=0.0, output_max=125.0,
            max_integral=200.0, max_ramp_rate=1000.0,
        )
        state = reset(5.0, 0.0, jax.random.PRNGKey(0))
        dt = jnp.array(0.02)

        # Setpoint=9.5, PV=5.0, large error → pump should increase
        new_state, _, pump = step(state, jnp.array(9.5), jnp.array(5.0), params, dt, jax.random.PRNGKey(1))
        assert float(pump) > 10.0  # PI drives pump up from 0

    def test_ramp_limiting_constrains_pump_changes(self) -> None:
        """Pump speed changes should be limited by max_ramp_rate."""
        params = DosingSystemParams(
            sensor_noise_std=0.0, sensor_lag=0.0, sensor_drift_rate=0.0,
            kp=100.0, ki=0.0, ff=0.0, output_min=0.0, output_max=125.0,
            max_integral=200.0, max_ramp_rate=10.0,  # 10 units/hour
        )
        state = reset(5.0, 50.0, jax.random.PRNGKey(0))
        dt = jnp.array(0.1)  # 0.1 hour

        # Huge kp drives command to max, but ramp limits to 10*0.1=1 unit change
        new_state, _, pump = step(state, jnp.array(9.5), jnp.array(5.0), params, dt, jax.random.PRNGKey(1))
        assert abs(float(pump) - 51.0) < 0.01

    def test_sensor_lag_smooths_readings(self) -> None:
        """Sensor lag should smooth out sudden PV changes."""
        params = DosingSystemParams(
            sensor_noise_std=0.0, sensor_lag=0.95, sensor_drift_rate=0.0,
            kp=0.0, ki=0.0, ff=50.0,
        )
        state = reset(7.0, 50.0, jax.random.PRNGKey(0))
        dt = jnp.array(0.02)

        # True PV jumps to 10.0, but sensor should lag behind
        _, sensed, _ = step(state, jnp.array(9.5), jnp.array(10.0), params, dt, jax.random.PRNGKey(1))
        assert float(sensed) < 8.0  # should be close to 7.0 due to 0.95 lag

    def test_ideal_params_pass_through(self) -> None:
        """Zero noise, zero lag, high gains → pump closely tracks command."""
        params = DosingSystemParams(
            sensor_noise_std=0.0, sensor_lag=0.0, sensor_drift_rate=0.0,
            kp=10.0, ki=0.0, ff=0.0, output_min=0.0, output_max=125.0,
            max_integral=200.0, max_ramp_rate=1e6,
        )
        state = reset(9.5, 0.0, jax.random.PRNGKey(0))
        dt = jnp.array(0.02)

        # PV matches setpoint exactly → error=0 → pump=ff=0
        _, sensed, pump = step(state, jnp.array(9.5), jnp.array(9.5), params, dt, jax.random.PRNGKey(1))
        assert abs(float(sensed) - 9.5) < 0.01
        assert abs(float(pump) - 0.0) < 0.01

    def test_integrator_accumulates_persistent_error(self) -> None:
        """Sustained error should cause integrator to wind up."""
        params = DosingSystemParams(
            sensor_noise_std=0.0, sensor_lag=0.0, sensor_drift_rate=0.0,
            kp=0.0, ki=10.0, ff=0.0, output_min=0.0, output_max=125.0,
            max_integral=200.0, max_ramp_rate=1e6,
        )
        state = reset(5.0, 0.0, jax.random.PRNGKey(0))
        dt = jnp.array(0.1)

        # Run 20 steps with constant error of 4.5
        for i in range(20):
            state, _, pump = step(state, jnp.array(9.5), jnp.array(5.0), params, dt, jax.random.PRNGKey(i))

        # Integral should have accumulated: 4.5 * 0.1 * 20 = 9.0
        # Pump = ki * integral = 10 * 9.0 = 90.0
        assert float(pump) > 50.0  # substantial pump output from integral

    def test_jit_compatible(self) -> None:
        params = DosingSystemParams()
        state = reset(7.0, 50.0, jax.random.PRNGKey(0))
        dt = jnp.array(0.02)

        jitted = jax.jit(lambda s, sp, pv, k: step(s, sp, pv, params, dt, k))
        new_state, sensed, pump = jitted(state, jnp.array(9.5), jnp.array(7.0), jax.random.PRNGKey(1))
        assert sensed.shape == ()
        assert pump.shape == ()
