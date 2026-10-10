import dataclasses
from collections.abc import Iterator

import jax
import jax.numpy as jnp
import pytest
from process_control.benchmarks.bsm1 import (
    HANDLE_LIMITS,
    BSM1Config,
    BSM1State,
    Observation,
    action_space,
    bsm1_combined_control,
    bsm1_long_term,
    bsm1_recycle_control,
    bsm1_reference_closed_loop,
    bsm1_sludge_control,
    bsm1_supervisory_control,
    make_bsm1_benchmark,
)
from process_control.scenarios.tabular_influent import NO_RANDOMIZATION, Uniform

CONSTANT_INFLUENT = dataclasses.replace(NO_RANDOMIZATION, profile_weights=(1.0, 0.0, 0.0, 0.0))
QUANTITIES = ("s_o_3", "s_o_5", "s_no_2", "effluent_s_nh", "mlss", "underflow_tss", "blanket_height", "temperature")
IDEAL = tuple(Observation(quantity, 1.0) for quantity in QUANTITIES)
STEPS_PER_DAY = 96
Q0 = 18446.0 / 24.0

type Rollout = tuple[jax.Array, dict[str, jax.Array]]


@pytest.fixture(scope="module", autouse=True)
def float64() -> Iterator[None]:
    with jax.enable_x64(True):
        yield


def _testable(config: BSM1Config) -> BSM1Config:
    return dataclasses.replace(config, observations=IDEAL, influent=CONSTANT_INFLUENT)


def _rollout(config: BSM1Config, actions: jax.Array) -> Rollout:
    """One day from one reset for each row of `actions`."""
    reset, step = make_bsm1_benchmark(config)
    start, _ = reset(jax.random.key(0))
    keys = jax.random.split(jax.random.key(1), STEPS_PER_DAY)

    def run(action: jax.Array) -> Rollout:
        def body(state: BSM1State, key: jax.Array) -> tuple[BSM1State, Rollout]:
            state, observation, _, _, info = step(state, action, key)
            return state, (observation, info)

        return jax.lax.scan(body, start, keys)[1]

    return jax.jit(jax.vmap(run))(actions)


def _late_mean(rollout: Rollout, row: int, quantity: str) -> jax.Array:
    """Mean of an exact observation over the last quarter of the day."""
    observations, _ = rollout
    return jnp.mean(observations[row, -STEPS_PER_DAY // 4 :, QUANTITIES.index(quantity)])


class TestRecycleControl:
    def test_internal_recycle_returns_nitrate_to_the_anoxic_zone(self) -> None:
        """A larger agent-set Qa carries more nitrate into reactor 2 and costs more pumping energy."""
        config = bsm1_recycle_control()
        actions = jnp.array([[HANDLE_LIMITS["q_a"][1], Q0], [5000.0 / 24.0, Q0]])
        rollout = _rollout(_testable(config), actions)

        assert action_space(config.handles).names == ("q_a", "q_r")
        assert _late_mean(rollout, 0, "s_no_2") > _late_mean(rollout, 1, "s_no_2") + 1.0
        assert jnp.mean(rollout[1]["pumping_energy"][0]) > jnp.mean(rollout[1]["pumping_energy"][1])


class TestCombinedControl:
    def test_agent_drives_aeration_and_both_recycles(self) -> None:
        """Aeration raises reactor 3 oxygen and a larger return sludge flow thins the underflow."""
        config = bsm1_combined_control()
        actions = jnp.array([[15.0, 5.0, 2000.0, 1.5 * Q0], [3.0, 5.0, 2000.0, 0.5 * Q0]])
        rollout = _rollout(_testable(config), actions)

        assert action_space(config.handles).names == ("kla_34", "kla_5", "q_a", "q_r")
        assert _late_mean(rollout, 0, "s_o_3") > _late_mean(rollout, 1, "s_o_3")
        assert _late_mean(rollout, 0, "underflow_tss") < _late_mean(rollout, 1, "underflow_tss")


class TestSludgeControl:
    def test_wastage_and_return_sludge_move_the_solids(self) -> None:
        """Wasting at the limit draws down the mixed liquor and the blanket; a weak return flow lets the blanket rise."""
        config = bsm1_sludge_control()
        q_w_max = HANDLE_LIMITS["q_w"][1]
        actions = jnp.array([[Q0, q_w_max], [Q0, 0.0], [0.4 * Q0, 0.0]])
        rollout = _rollout(_testable(config), actions)

        assert action_space(config.handles).names == ("q_r", "q_w")
        assert _late_mean(rollout, 0, "mlss") < 0.95 * _late_mean(rollout, 1, "mlss")
        assert _late_mean(rollout, 0, "blanket_height") < 0.9 * _late_mean(rollout, 1, "blanket_height")
        assert _late_mean(rollout, 2, "blanket_height") > _late_mean(rollout, 1, "blanket_height")

    def test_hydraulic_overload_raises_the_blanket(self) -> None:
        """Doubling the influent flow at a fixed return flow loads the settler with solids and lifts the blanket."""
        config = _testable(bsm1_sludge_control())
        overloaded = dataclasses.replace(config, influent=dataclasses.replace(CONSTANT_INFLUENT, flow_scale=Uniform(2.0, 2.0)))
        action = jnp.array([[Q0, 0.0]])

        nominal_blanket = _late_mean(_rollout(config, action), 0, "blanket_height")
        overloaded_blanket = _late_mean(_rollout(overloaded, action), 0, "blanket_height")

        assert overloaded_blanket > 1.5 * nominal_blanket


class TestSupervisoryControl:
    @pytest.mark.parametrize(("oxygen", "nitrate"), [(1.0, 0.5), (3.0, 2.0)])
    def test_controllers_track_agent_setpoints(self, oxygen: float, nitrate: float) -> None:
        """The BSM1 controllers bring reactor 5 oxygen and reactor 2 nitrate to the setpoints the agent chooses."""
        config = bsm1_supervisory_control()
        rollout = _rollout(_testable(config), jnp.array([[oxygen, nitrate]]))

        assert action_space(config.handles).names == ("kla_5_setpoint", "q_a_setpoint")
        assert jnp.allclose(_late_mean(rollout, 0, "s_o_5"), oxygen, atol=0.1)
        assert jnp.allclose(_late_mean(rollout, 0, "s_no_2"), nitrate, atol=0.2)


class TestReferenceClosedLoop:
    def test_default_controllers_need_no_agent(self) -> None:
        """The BSM1 default strategy runs without actions and holds both controlled variables."""
        config = bsm1_reference_closed_loop("constant", sensor_noise=False)
        rollout = _rollout(dataclasses.replace(config, observations=IDEAL), jnp.zeros((1, 0)))

        assert action_space(config.handles).names == ()
        assert jnp.allclose(_late_mean(rollout, 0, "s_o_5"), 2.0, atol=0.02)
        assert jnp.allclose(_late_mean(rollout, 0, "s_no_2"), 1.0, atol=0.05)

    def test_rejects_unknown_weather(self) -> None:
        """Only the BSM1 weather profiles can drive the reference."""
        with pytest.raises(ValueError, match="weather"):
            bsm1_reference_closed_loop("drought")


class TestLongTerm:
    def test_episodes_sample_the_seasons(self) -> None:
        """Long-term episodes start at different points of the 10-20 °C annual cycle."""
        reset, step = make_bsm1_benchmark(bsm1_long_term())
        action = jnp.array([10.0, 3.5, 16.0])

        @jax.jit
        def first_temperature(key: jax.Array) -> jax.Array:
            state, _ = reset(key)
            return step(state, action, key)[4]["temperature"]

        temperatures = jnp.array([first_temperature(jax.random.key(seed)) for seed in range(8)])

        assert jnp.all((temperatures >= 10.0) & (temperatures <= 20.0))
        assert jnp.max(temperatures) - jnp.min(temperatures) > 4.0
