import jax
import jax.numpy as jnp
from process_control.actuators.dose_pump import DosePumpParams
from process_control.actuators.dose_pump import reset as dose_pump_reset
from process_control.actuators.dose_pump import step as dose_pump_step
from process_control.benchmarks.chlorine import ChlorineBenchmarkConfig, make_chlorine_benchmark
from process_control.controllers.pi_controller import PIControllerParams, PIControllerState
from process_control.controllers.pi_controller import step as pi_step
from process_control.transport import Transport
from process_control.units.contact_basin import ContactBasinParams
from process_control.units.contact_basin import reset as basin_reset
from process_control.units.contact_basin import step as basin_step


class TestChlorineBenchmarkResetAndStep:
    def test_reset_returns_state_and_obs(self) -> None:
        config = ChlorineBenchmarkConfig()
        reset_fn, _step_fn = make_chlorine_benchmark(config)
        key = jax.random.PRNGKey(42)

        state, obs = reset_fn(key)

        assert obs.shape == (4,)
        assert state.step_count == 0

    def test_step_returns_correct_shapes(self) -> None:
        config = ChlorineBenchmarkConfig()
        reset_fn, step_fn = make_chlorine_benchmark(config)
        key = jax.random.PRNGKey(42)
        k1, k2 = jax.random.split(key)

        state, _ = reset_fn(k1)
        action = jnp.array(2.0)
        new_state, obs, reward, _done, info = step_fn(state, action, k2)

        assert obs.shape == (4,)
        assert reward.shape == ()
        assert new_state.step_count == 1
        assert "pi_dose" in info
        assert "outlet_residual" in info

    def test_multi_step_execution(self) -> None:
        config = ChlorineBenchmarkConfig()
        reset_fn, step_fn = make_chlorine_benchmark(config)
        key = jax.random.PRNGKey(0)
        k1, key = jax.random.split(key)
        state, _ = reset_fn(k1)

        for _ in range(10):
            k_step, key = jax.random.split(key)
            state, _obs, _reward, _done, _info = step_fn(state, jnp.array(2.5), k_step)

        assert state.step_count == 10


class TestChlorineBenchmarkDeterministic:
    def test_same_seed_same_trajectory(self) -> None:
        config = ChlorineBenchmarkConfig()
        reset_fn, step_fn = make_chlorine_benchmark(config)

        def run_trajectory(seed: int) -> tuple[list[jax.Array], list[jax.Array]]:
            key = jax.random.PRNGKey(seed)
            k1, key = jax.random.split(key)
            state, obs0 = reset_fn(k1)
            observations = [obs0]
            rewards = []
            for _ in range(5):
                k_step, key = jax.random.split(key)
                state, obs, reward, _, _ = step_fn(state, jnp.array(2.0), k_step)
                observations.append(obs)
                rewards.append(reward)
            return observations, rewards

        obs_a, rew_a = run_trajectory(123)
        obs_b, rew_b = run_trajectory(123)

        for a, b in zip(obs_a, obs_b, strict=True):
            assert jnp.allclose(a, b)
        for a, b in zip(rew_a, rew_b, strict=True):
            assert jnp.allclose(a, b)


class TestContactBasinAdvection:
    def test_dose_appears_at_outlet(self) -> None:
        params = ContactBasinParams(total_volume=100.0, n_segments=5, tau=1000.0)
        key = jax.random.PRNGKey(0)
        state = basin_reset(params, key)
        dt = jnp.array(1.0)

        dosed_transport = Transport.create(flow=20.0, chlorine_residual=5.0, demand=0.0)
        zero_transport = Transport.create(flow=20.0, chlorine_residual=0.0, demand=0.0)

        state, _ = basin_step(state, dosed_transport, params, dt, key)

        max_outlet = 0.0
        for _ in range(20):
            state, outlet = basin_step(state, zero_transport, params, dt, key)
            max_outlet = max(max_outlet, float(outlet))

        assert max_outlet > 0.1


class TestDosePumpSaturation:
    def test_dose_clamped_to_max(self) -> None:
        params = DosePumpParams(max_dose=5.0, min_dose=0.0, max_ramp_rate=100.0)
        key = jax.random.PRNGKey(0)
        state = dose_pump_reset(key)
        dt = jnp.array(1.0)

        _, realized = dose_pump_step(state, jnp.array(10.0), params, dt)

        assert jnp.allclose(realized, jnp.array(5.0))

    def test_dose_clamped_to_min(self) -> None:
        params = DosePumpParams(max_dose=5.0, min_dose=0.5, max_ramp_rate=100.0)
        key = jax.random.PRNGKey(0)
        state = dose_pump_reset(key)
        dt = jnp.array(1.0)

        _, realized = dose_pump_step(state, jnp.array(-1.0), params, dt)

        assert jnp.allclose(realized, jnp.array(0.5))

    def test_ramp_rate_limiting(self) -> None:
        params = DosePumpParams(max_dose=5.0, min_dose=0.0, max_ramp_rate=1.0)
        key = jax.random.PRNGKey(0)
        state = dose_pump_reset(key)
        dt = jnp.array(1.0)

        _new_state, realized = dose_pump_step(state, jnp.array(5.0), params, dt)

        assert jnp.allclose(realized, jnp.array(1.0))


class TestPIControllerTracksSetpoint:
    def test_converges_toward_setpoint(self) -> None:
        params = PIControllerParams(
            kp=0.1, ki=0.1, ff=3.0,
            output_min=0.0, output_max=10.0,
            max_integral=100.0,
        )
        state = PIControllerState.create()
        dt = jnp.array(0.25)
        setpoint = jnp.array(1.5)
        measurement = jnp.array(0.5)

        outputs = []
        for _ in range(20):
            state, output = pi_step(state, measurement, setpoint, params, dt)
            outputs.append(float(output))

        assert outputs[-1] > outputs[0]
        assert state.integral > 0.0


class TestJITCompatibility:
    def test_step_is_jittable(self) -> None:
        config = ChlorineBenchmarkConfig()
        reset_fn, step_fn = make_chlorine_benchmark(config)

        jit_step = jax.jit(step_fn)

        key = jax.random.PRNGKey(99)
        k1, k2 = jax.random.split(key)
        state, _ = reset_fn(k1)
        action = jnp.array(2.0)

        new_state, obs, _reward, _done, _info = jit_step(state, action, k2)

        assert obs.shape == (4,)
        assert new_state.step_count == 1
