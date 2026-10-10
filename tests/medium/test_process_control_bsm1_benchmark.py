import dataclasses
from collections.abc import Iterator

import jax
import jax.numpy as jnp
import pytest
from process_control.benchmarks.bsm1_flowsheet import (
    HANDLE_LIMITS,
    BSM1Config,
    BSM1Handles,
    BSM1State,
    Fixed,
    Observation,
    OperatingPoint,
    make_bsm1_benchmark,
)
from process_control.plants.activated_sludge import BSM1_OPEN_LOOP_REACTORS
from process_control.scenarios.tabular_influent import NO_RANDOMIZATION

CONSTANT_INFLUENT = dataclasses.replace(NO_RANDOMIZATION, profile_weights=(1.0, 0.0, 0.0, 0.0))
QUANTITIES = ("s_o_3", "s_o_5", "s_no_2", "s_nh_5", "effluent_s_nh", "mlss", "underflow_tss", "blanket_height")
# Exact observations, so tests read the plant rather than one noise realisation.
IDEAL = tuple(Observation(quantity, 1.0) for quantity in QUANTITIES)
STEPS_PER_DAY = 96
Q_W = 385.0 / 24.0
KLA_MAX = HANDLE_LIMITS["kla_5"][1]

type Rollout = tuple[BSM1State, tuple[jax.Array, jax.Array, dict[str, jax.Array]]]
Q_W_MAX = HANDLE_LIMITS["q_w"][1]


@pytest.fixture(scope="module", autouse=True)
def float64() -> Iterator[None]:
    with jax.enable_x64(True):
        yield


def _rollout(config: BSM1Config, actions: jax.Array, steps: int = STEPS_PER_DAY, states: BSM1State | None = None) -> Rollout:
    """Roll the benchmark out once per row of `actions` from one reset, or from each of `states`."""
    reset, step = make_bsm1_benchmark(config)
    if states is None:
        state, _ = reset(jax.random.key(0))
        states = _batched(state, actions.shape[0])
    keys = jax.random.split(jax.random.key(1), steps)

    def run(action: jax.Array, start: BSM1State):
        def body(state: BSM1State, key: jax.Array):
            state, observation, reward, _, info = step(state, action, key)
            return state, (observation, reward, info)

        return jax.lax.scan(body, start, keys)

    return jax.jit(jax.vmap(run))(actions, states)


def _batched(state: BSM1State, size: int) -> BSM1State:
    def stack(leaf: jax.Array) -> jax.Array:
        return jnp.stack([leaf] * size)

    return jax.tree.map(stack, state)


def _observed(observations: jax.Array, quantity: str) -> jax.Array:
    return observations[..., QUANTITIES.index(quantity)]


class TestReset:
    def test_starts_at_the_open_loop_steady_state(self) -> None:
        """Reset observes the Table 6 plant, as float32, identically for the same key."""
        reset, _ = make_bsm1_benchmark(BSM1Config(observations=IDEAL))
        _, observation = reset(jax.random.key(0))
        _, again = reset(jax.random.key(0))

        assert observation.dtype == jnp.float32
        assert observation.shape == (len(QUANTITIES),)
        assert jnp.allclose(_observed(observation, "s_o_5"), BSM1_OPEN_LOOP_REACTORS[4][7], rtol=1e-4)
        assert jnp.array_equal(observation, again)

    def test_operating_point_holds_without_a_transient(self) -> None:
        """Every handle fixed at the open-loop point under constant influent keeps the plant where it started."""
        operating = OperatingPoint().values()
        handles = BSM1Handles(**{name: Fixed(value) for name, value in operating.items()})
        config = BSM1Config(handles=handles, observations=IDEAL, influent=CONSTANT_INFLUENT)
        (end, (observations, _, info)) = _rollout(config, jnp.zeros((1, 0)))

        reactors = jnp.stack([end.plant[f"reactor_{index}"][0] for index in range(1, 6)])
        assert jnp.allclose(reactors, jnp.array(BSM1_OPEN_LOOP_REACTORS), rtol=0.01, atol=1e-3)
        assert jnp.allclose(info["effluent_tss"], 12.5, rtol=0.01)
        assert not jnp.any(info["infeasible_hydraulics"])

    def test_needs_float64(self) -> None:
        """Building the benchmark in float32 is refused rather than silently stalling the slow states."""
        with jax.enable_x64(False), pytest.raises(RuntimeError, match="float64"):
            make_bsm1_benchmark(BSM1Config())


@pytest.fixture(scope="module")
def default_runs(float64: None) -> Rollout:
    """One day of the default problem under constant influent, for each of a set of actions."""
    actions = jnp.array(
        [
            [10.0, 3.5, Q_W],
            [KLA_MAX, KLA_MAX, Q_W],
            [2.0, 2.0, Q_W],
            [2.0 * KLA_MAX, 2.0 * KLA_MAX, 2.0 * Q_W_MAX],
            [KLA_MAX, KLA_MAX, Q_W_MAX],
            [10.0, 3.5, 0.0],
        ]
    )
    return _rollout(BSM1Config(observations=IDEAL, influent=CONSTANT_INFLUENT), actions)


class TestDefaultProblem:
    def test_aeration_trades_energy_for_effluent_quality(self, default_runs: Rollout) -> None:
        """Harder aeration raises oxygen and aeration energy while lowering effluent ammonia and EQ."""
        _, (observations, _, info) = default_runs
        high, low = 1, 2

        assert jnp.mean(_observed(observations[high], "s_o_3")) > jnp.mean(_observed(observations[low], "s_o_3"))
        assert jnp.mean(info["aeration_energy"][high]) > jnp.mean(info["aeration_energy"][low])
        assert jnp.mean(info["effluent_s_nh"][high]) < jnp.mean(info["effluent_s_nh"][low])
        assert jnp.mean(info["effluent_quality"][high]) < jnp.mean(info["effluent_quality"][low])

    def test_reward_weighs_quality_against_cost(self, default_runs: Rollout) -> None:
        """Neither the cheapest nor the most aerated action wins: the moderate one earns more than both."""
        _, (_, rewards, _) = default_runs
        moderate, high, low = 0, 1, 2

        assert rewards.dtype == jnp.float32
        assert jnp.mean(rewards[moderate]) > jnp.mean(rewards[high])
        assert jnp.mean(rewards[moderate]) > jnp.mean(rewards[low])

    def test_actions_beyond_the_limits_are_clipped(self, default_runs: Rollout) -> None:
        """Commands above the Table 15 limits act exactly like the limits."""
        _, (observations, rewards, _) = default_runs

        assert jnp.array_equal(observations[3], observations[4])
        assert jnp.array_equal(rewards[3], rewards[4])

    def test_wastage_follows_the_qw_action(self, default_runs: Rollout) -> None:
        """After the first interval no wastage removes no sludge, while the open-loop Qw wastes Qw times the Table 8 underflow solids."""
        _, (_, _, info) = default_runs

        assert jnp.allclose(info["sludge_wastage"][5, 1:], 0.0)
        assert jnp.allclose(info["sludge_wastage"][0, 0], 6394.0 * 385.0 / 1000.0, rtol=0.01)

    def test_nitrate_controller_holds_its_setpoint(self, default_runs: Rollout) -> None:
        """Qa on the BSM1 nitrate controller brings reactor 2 nitrate to 1 g N/m³ within a day."""
        _, (observations, _, info) = default_runs

        assert jnp.allclose(jnp.mean(_observed(observations[0, -STEPS_PER_DAY // 4 :], "s_no_2")), 1.0, atol=0.1)
        assert info["q_a"][0, -1] < 55338.0 / 24.0


class TestTimeScales:
    def test_agent_steps_compose_control_intervals(self) -> None:
        """Fifteen one-interval agent steps advance the plant exactly like one fifteen-interval step."""
        action = jnp.array([[12.0, 4.0, Q_W]])
        fine = BSM1Config(observations=IDEAL, influent=CONSTANT_INFLUENT, intervals_per_step=1)
        coarse = dataclasses.replace(fine, intervals_per_step=15)

        fine_end, _ = _rollout(fine, action, steps=15)
        coarse_end, _ = _rollout(coarse, action, steps=1)

        assert all(jnp.allclose(fine_end.plant[name], coarse_end.plant[name], rtol=1e-9) for name in fine_end.plant)


class TestTemperature:
    def test_influent_temperature_is_the_profile_temperature(self, default_runs: Rollout) -> None:
        """By default the reactors run at the 15 °C of the BSM1 influent files."""
        _, (_, _, info) = default_runs

        assert jnp.allclose(info["temperature"], 15.0)
