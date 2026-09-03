"""Medium tests for the shared training loop's episode-boundary contract.

Every expected number here is a literal taken from the design's measured reference run.
The point of the file is that the loop's arithmetic is pinned from outside, so nothing
below recomputes it.

The toy environment's observation is its step counter as float32, so the true final
state of an episode and the post-reset state are observably different values. That is
what makes the boundary assertions falsifiable: an agent handed the post-reset
observation as its bootstrap sees ``0.0`` where it should see the cutoff value.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import cast

import chex
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from rl_components.agent_protocol import AgentStep
from rl_components.env_protocol import EnvReset, EnvSpec, EnvStep, TruncationPolicy
from rl_components.loop import run
from rl_components.structs import chex_struct
from rl_components.timestep import Timestep

GAMMA = 0.9
NEVER = -1


@chex_struct(frozen=True)
class ToyState:
    """Toy environment state: the number of steps taken this episode."""

    counter: jax.Array


def _unit_reward(counter: jax.Array) -> jax.Array:
    del counter
    return jnp.ones((), jnp.float32)


class ToyEnv:
    """A counter that pays a reward a step and ends the episode at configured counts.

    Args:
        terminate_at: Counter value reported as ``terminated``. ``NEVER`` disables it.
        truncate_at: Counter value reported as ``truncated``. ``NEVER`` disables it.
        truncation_policy: The policy the environment declares on its spec.
        reward_of: Injected per-step reward, applied to the post-step counter. The
            default pays 1.0 a step; a step-dependent reward makes the reward metric
            falsifiable against a constant.
    """

    def __init__(
        self,
        *,
        terminate_at: int = NEVER,
        truncate_at: int = NEVER,
        truncation_policy: TruncationPolicy = "bootstrap",
        reward_of: Callable[[jax.Array], jax.Array] = _unit_reward,
    ) -> None:
        self.terminate_at = terminate_at
        self.truncate_at = truncate_at
        self.truncation_policy = truncation_policy
        self.reward_of = reward_of

    def spec(self, params: None = None) -> EnvSpec:
        del params
        return EnvSpec(
            id="toy-counter",
            observation_shape=(1,),
            action_shape=(),
            observation_dtype=jnp.float32,
            action_dtype=jnp.int32,
            num_actions=2,
            truncation_policy=self.truncation_policy,
        )

    def reset(self, key: chex.PRNGKey, params: None = None) -> EnvReset[jax.Array, ToyState]:
        del key, params
        return EnvReset(observation=jnp.zeros((1,), jnp.float32), state=ToyState(counter=jnp.zeros((), jnp.int32)))

    def step(self, key: chex.PRNGKey, state: ToyState, action: jax.Array, params: None = None) -> EnvStep[jax.Array, ToyState]:
        del key, action, params
        counter = state.counter + jnp.ones((), jnp.int32)
        return EnvStep(
            observation=counter.astype(jnp.float32)[None],
            state=ToyState(counter=counter),
            reward=self.reward_of(counter),
            terminated=counter == jnp.asarray(self.terminate_at, jnp.int32),
            truncated=counter == jnp.asarray(self.truncate_at, jnp.int32),
            info={},
        )


@chex_struct(frozen=True)
class SeedState:
    """State of the environment whose episode length is drawn at reset."""

    counter: jax.Array
    limit: jax.Array


class SeedLengthEnv:
    """A counter whose terminal count is drawn from the reset key.

    Two runs seeded differently therefore have different episode structure, which is
    what the batching test needs to observe.
    """

    def spec(self, params: None = None) -> EnvSpec:
        del params
        return EnvSpec(
            id="toy-seed-length",
            observation_shape=(1,),
            action_shape=(),
            observation_dtype=jnp.float32,
            action_dtype=jnp.int32,
            num_actions=2,
        )

    def reset(self, key: chex.PRNGKey, params: None = None) -> EnvReset[jax.Array, SeedState]:
        del params
        limit = jax.random.randint(key, (), 2, 6, dtype=jnp.int32)
        return EnvReset(
            observation=jnp.zeros((1,), jnp.float32),
            state=SeedState(counter=jnp.zeros((), jnp.int32), limit=limit),
        )

    def step(self, key: chex.PRNGKey, state: SeedState, action: jax.Array, params: None = None) -> EnvStep[jax.Array, SeedState]:
        del key, action, params
        counter = state.counter + jnp.ones((), jnp.int32)
        return EnvStep(
            observation=counter.astype(jnp.float32)[None],
            state=SeedState(counter=counter, limit=state.limit),
            reward=jnp.ones((), jnp.float32),
            terminated=counter >= state.limit,
            truncated=jnp.zeros((), jnp.bool_),
            info={},
        )


class KeyEchoEnv:
    """A counter whose observation echoes a draw from the key it was handed.

    The echo makes the loop's key stream observable through agent metrics, so two runs
    with different episode structure can be compared step for step.
    """

    def __init__(self, *, terminate_at: int) -> None:
        self.terminate_at = terminate_at

    def spec(self, params: None = None) -> EnvSpec:
        del params
        return EnvSpec(
            id="toy-key-echo",
            observation_shape=(1,),
            action_shape=(),
            observation_dtype=jnp.float32,
            action_dtype=jnp.int32,
            num_actions=2,
        )

    def reset(self, key: chex.PRNGKey, params: None = None) -> EnvReset[jax.Array, ToyState]:
        del params
        return EnvReset(observation=jax.random.uniform(key, (1,), jnp.float32), state=ToyState(counter=jnp.zeros((), jnp.int32)))

    def step(self, key: chex.PRNGKey, state: ToyState, action: jax.Array, params: None = None) -> EnvStep[jax.Array, ToyState]:
        del action, params
        counter = state.counter + jnp.ones((), jnp.int32)
        return EnvStep(
            observation=jax.random.uniform(key, (1,), jnp.float32),
            state=ToyState(counter=counter),
            reward=jnp.ones((), jnp.float32),
            terminated=counter == jnp.asarray(self.terminate_at, jnp.int32),
            truncated=jnp.zeros((), jnp.bool_),
            info={},
        )


@chex_struct(frozen=True)
class RecordAgentState:
    """Agent state: the number of transitions this agent has seen closed."""

    closed: jax.Array


class RecordAgent:
    """Records the timestep it was handed and counts the transitions it closed."""

    def init(self, key: chex.PRNGKey, spec: EnvSpec) -> RecordAgentState:
        del key, spec
        return RecordAgentState(closed=jnp.zeros((), jnp.int32))

    def step(self, state: RecordAgentState, timestep: Timestep[jax.Array], step_index: jax.Array) -> AgentStep[RecordAgentState, jax.Array]:
        closes = (step_index > jnp.zeros((), jnp.int32)).astype(jnp.int32)
        return AgentStep(
            state=RecordAgentState(closed=state.closed + closes),
            action=jnp.zeros((), jnp.int32),
            metrics={
                "seen/bootstrap_observation": timestep.bootstrap_observation[0],
                "seen/observation": timestep.observation[0],
                "seen/reward": timestep.reward,
                "seen/discount": timestep.discount,
                "seen/episode_end": timestep.episode_end,
            },
        )


class BootstrapAgent:
    """Computes a real one-step bootstrap target from the timestep it was handed.

    Recording the delivered fields and consuming the right one are independent
    failures: an agent that bootstraps from ``Timestep.observation`` passes every
    assertion about ``bootstrap_observation``. This agent therefore consumes the field,
    and the test pins the resulting number.

    Args:
        value_of: Injected value function, applied to the bootstrap observation.
    """

    def __init__(self, value_of: Callable[[jax.Array], jax.Array]) -> None:
        self.value_of = value_of

    def init(self, key: chex.PRNGKey, spec: EnvSpec) -> RecordAgentState:
        del key, spec
        return RecordAgentState(closed=jnp.zeros((), jnp.int32))

    def step(self, state: RecordAgentState, timestep: Timestep[jax.Array], step_index: jax.Array) -> AgentStep[RecordAgentState, jax.Array]:
        target = timestep.reward + timestep.discount * self.value_of(timestep.bootstrap_observation)
        return AgentStep(
            state=RecordAgentState(closed=state.closed + (step_index > jnp.zeros((), jnp.int32)).astype(jnp.int32)),
            action=jnp.zeros((), jnp.int32),
            metrics={"seen/target": target[0]},
        )


class LoopNamespaceAgent:
    """An ill-behaved agent that writes into the loop's reserved metric namespace."""

    def init(self, key: chex.PRNGKey, spec: EnvSpec) -> RecordAgentState:
        del key, spec
        return RecordAgentState(closed=jnp.zeros((), jnp.int32))

    def step(self, state: RecordAgentState, timestep: Timestep[jax.Array], step_index: jax.Array) -> AgentStep[RecordAgentState, jax.Array]:
        del timestep, step_index
        return AgentStep(state=state, action=jnp.zeros((), jnp.int32), metrics={"loop/reward": jnp.zeros((), jnp.float32)})


def _metric(metrics: dict[str, jax.Array], key: str) -> np.ndarray:
    return np.asarray(metrics[key])


class TestBoundaryObservations:
    def test_termination_delivers_the_true_final_observation(self) -> None:
        _, metrics = run(RecordAgent(), ToyEnv(terminate_at=3), jax.random.key(0), steps=9, gamma=GAMMA)

        np.testing.assert_allclose(_metric(metrics, "seen/bootstrap_observation"), [0, 1, 2, 3, 1, 2, 3, 1, 2], rtol=1e-6)
        np.testing.assert_allclose(_metric(metrics, "seen/observation"), [0, 1, 2, 0, 1, 2, 0, 1, 2], rtol=1e-6)

    def test_truncation_delivers_the_true_final_observation(self) -> None:
        _, metrics = run(RecordAgent(), ToyEnv(truncate_at=3), jax.random.key(0), steps=7, gamma=GAMMA)

        np.testing.assert_allclose(_metric(metrics, "seen/bootstrap_observation"), [0, 1, 2, 3, 1, 2, 3], rtol=1e-6)
        np.testing.assert_allclose(_metric(metrics, "seen/observation"), [0, 1, 2, 0, 1, 2, 0], rtol=1e-6)


class TestBootstrapTargetConsumption:
    """The agent must be able to compute a correct target, not merely see the fields."""

    def test_truncation_target_uses_the_final_state_not_the_post_reset_state(self) -> None:
        _, metrics = run(BootstrapAgent(lambda observation: 10.0 * observation), ToyEnv(truncate_at=3), jax.random.key(0), steps=5, gamma=GAMMA)

        targets = _metric(metrics, "seen/target")
        np.testing.assert_allclose(targets, [0.0, 10.0, 19.0, 28.0, 10.0], rtol=1e-6)
        assert targets[3] == pytest.approx(28.0, rel=1e-6)
        assert targets[3] != pytest.approx(1.0, rel=1e-6)

    def test_termination_target_drops_the_bootstrap_entirely(self) -> None:
        _, metrics = run(BootstrapAgent(lambda observation: 10.0 * observation), ToyEnv(terminate_at=3), jax.random.key(0), steps=5, gamma=GAMMA)

        np.testing.assert_allclose(_metric(metrics, "seen/target"), [0.0, 10.0, 19.0, 1.0, 10.0], rtol=1e-6)


class TestDiscountSemantics:
    def test_termination_kills_the_bootstrap(self) -> None:
        _, metrics = run(RecordAgent(), ToyEnv(terminate_at=3), jax.random.key(0), steps=9, gamma=GAMMA)

        np.testing.assert_allclose(_metric(metrics, "loop/discount"), [0.9, 0.9, 0.0, 0.9, 0.9, 0.0, 0.9, 0.9, 0.0], rtol=1e-6)
        np.testing.assert_allclose(_metric(metrics, "seen/discount"), [0.0, 0.9, 0.9, 0.0, 0.9, 0.9, 0.0, 0.9, 0.9], rtol=1e-6)
        np.testing.assert_array_equal(_metric(metrics, "loop/episode_end"), [False, False, True, False, False, True, False, False, True])

    def test_truncation_keeps_the_bootstrap_under_the_bootstrap_policy(self) -> None:
        _, metrics = run(RecordAgent(), ToyEnv(truncate_at=3), jax.random.key(0), steps=7, gamma=GAMMA)

        np.testing.assert_allclose(_metric(metrics, "loop/discount"), [0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9], rtol=1e-6)
        np.testing.assert_array_equal(_metric(metrics, "loop/episode_end"), [False, False, True, False, False, True, False])
        np.testing.assert_array_equal(_metric(metrics, "loop/terminated"), [False] * 7)
        np.testing.assert_array_equal(_metric(metrics, "loop/truncated"), [False, False, True, False, False, True, False])

    def test_truncation_kills_the_bootstrap_under_the_terminate_policy(self) -> None:
        _, metrics = run(RecordAgent(), ToyEnv(truncate_at=3), jax.random.key(0), steps=5, gamma=GAMMA, truncation_policy="terminate")

        np.testing.assert_allclose(_metric(metrics, "loop/discount"), [0.9, 0.9, 0.0, 0.9, 0.9], rtol=1e-6)
        np.testing.assert_array_equal(_metric(metrics, "loop/episode_end"), [False, False, True, False, False])

    def test_termination_dominates_a_same_step_truncation(self) -> None:
        _, metrics = run(RecordAgent(), ToyEnv(terminate_at=3, truncate_at=3), jax.random.key(0), steps=5, gamma=GAMMA)

        np.testing.assert_allclose(_metric(metrics, "loop/discount"), [0.9, 0.9, 0.0, 0.9, 0.9], rtol=1e-6)
        np.testing.assert_array_equal(_metric(metrics, "loop/terminated"), [False, False, True, False, False])
        np.testing.assert_array_equal(_metric(metrics, "loop/truncated"), [False] * 5)


class TestEpisodeCutoff:
    def test_cutoff_fires_on_the_intended_step_as_a_truncation(self) -> None:
        _, metrics = run(RecordAgent(), ToyEnv(), jax.random.key(0), steps=5, gamma=GAMMA, episode_cutoff=3)

        np.testing.assert_array_equal(_metric(metrics, "loop/episode_end"), [False, False, True, False, False])
        np.testing.assert_array_equal(_metric(metrics, "loop/truncated"), [False, False, True, False, False])
        np.testing.assert_array_equal(_metric(metrics, "loop/terminated"), [False] * 5)
        np.testing.assert_allclose(_metric(metrics, "loop/discount"), [0.9, 0.9, 0.9, 0.9, 0.9], rtol=1e-6)
        np.testing.assert_allclose(_metric(metrics, "seen/bootstrap_observation"), [0, 1, 2, 3, 1], rtol=1e-6)
        np.testing.assert_allclose(_metric(metrics, "seen/observation"), [0, 1, 2, 0, 1], rtol=1e-6)

    def test_a_same_step_termination_dominates_the_cutoff(self) -> None:
        _, metrics = run(RecordAgent(), ToyEnv(terminate_at=3), jax.random.key(0), steps=5, gamma=GAMMA, episode_cutoff=3)

        np.testing.assert_allclose(_metric(metrics, "loop/discount"), [0.9, 0.9, 0.0, 0.9, 0.9], rtol=1e-6)
        np.testing.assert_array_equal(_metric(metrics, "loop/terminated"), [False, False, True, False, False])
        np.testing.assert_array_equal(_metric(metrics, "loop/truncated"), [False] * 5)

    def test_a_non_positive_cutoff_never_fires(self) -> None:
        _, metrics = run(RecordAgent(), ToyEnv(), jax.random.key(0), steps=5, gamma=GAMMA, episode_cutoff=NEVER)

        np.testing.assert_array_equal(_metric(metrics, "loop/episode_end"), [False] * 5)


class TestIndexing:
    def test_the_agent_closes_one_fewer_transition_than_the_horizon(self) -> None:
        final_state, metrics = run(RecordAgent(), ToyEnv(terminate_at=3), jax.random.key(0), steps=9, gamma=GAMMA)

        assert int(final_state.agent_state.closed) == 8
        assert _metric(metrics, "loop/reward").shape == (9,)

    def test_the_first_iteration_closes_nothing(self) -> None:
        _, metrics = run(RecordAgent(), ToyEnv(terminate_at=3), jax.random.key(0), steps=9, gamma=GAMMA)

        assert _metric(metrics, "seen/reward")[0] == pytest.approx(0.0)
        assert _metric(metrics, "seen/discount")[0] == pytest.approx(0.0)
        assert not bool(_metric(metrics, "seen/episode_end")[0])


class TestCompilation:
    def test_jit_agrees_with_the_eager_trace(self) -> None:
        env = ToyEnv(terminate_at=3)
        eager_state, eager_metrics = run(RecordAgent(), env, jax.random.key(0), steps=9, gamma=GAMMA)
        jitted_state, jitted_metrics = jax.jit(lambda key: run(RecordAgent(), env, key, steps=9, gamma=GAMMA))(jax.random.key(0))

        assert int(jitted_state.agent_state.closed) == int(eager_state.agent_state.closed)
        assert set(jitted_metrics) == set(eager_metrics)
        for key in eager_metrics:
            np.testing.assert_allclose(_metric(jitted_metrics, key), _metric(eager_metrics, key), rtol=1e-6)


class TestBatching:
    def test_vmap_over_seeds_stacks_metrics_and_diverges_per_seed(self) -> None:
        num_seeds = 4
        steps = 9
        env = SeedLengthEnv()
        batched = jax.jit(jax.vmap(lambda key: run(RecordAgent(), env, key, steps=steps, gamma=GAMMA)))

        _, metrics = batched(jax.random.split(jax.random.key(0), num_seeds))

        for key in metrics:
            assert _metric(metrics, key).shape == (num_seeds, steps), key
        boundaries = _metric(metrics, "loop/episode_end")
        first_boundary_index = np.argmax(boundaries, axis=1)
        assert boundaries.any(axis=1).all()
        assert len(set(first_boundary_index.tolist())) > 1


class TestKeyStream:
    def test_the_key_stream_does_not_depend_on_episode_structure(self) -> None:
        steps = 6
        _, with_boundary = run(RecordAgent(), KeyEchoEnv(terminate_at=3), jax.random.key(0), steps=steps, gamma=GAMMA)
        _, without_boundary = run(RecordAgent(), KeyEchoEnv(terminate_at=NEVER), jax.random.key(0), steps=steps, gamma=GAMMA)

        assert _metric(with_boundary, "loop/episode_end").any()
        assert not _metric(without_boundary, "loop/episode_end").any()
        np.testing.assert_array_equal(
            _metric(with_boundary, "seen/bootstrap_observation"),
            _metric(without_boundary, "seen/bootstrap_observation"),
        )


class TestRewardMetric:
    def test_the_reward_metric_carries_each_step_s_environment_reward(self) -> None:
        """Catches a ``loop/reward`` replaced by any constant, zero included.

        The ramp differs step to step and across the boundary, so no single value
        satisfies the expected array.
        """
        env = ToyEnv(terminate_at=3, reward_of=lambda counter: counter.astype(jnp.float32) * 2.0)

        _, metrics = run(RecordAgent(), env, jax.random.key(0), steps=9, gamma=GAMMA)

        np.testing.assert_allclose(_metric(metrics, "loop/reward"), [2, 4, 6, 2, 4, 6, 2, 4, 6], rtol=1e-6)


class TestEpisodeStatistics:
    def test_episode_totals_are_impulses_at_each_boundary(self) -> None:
        _, metrics = run(RecordAgent(), ToyEnv(terminate_at=3), jax.random.key(0), steps=9, gamma=GAMMA)

        np.testing.assert_allclose(_metric(metrics, "loop/episode_return"), [0, 0, 3, 0, 0, 3, 0, 0, 3], rtol=1e-6)
        np.testing.assert_array_equal(_metric(metrics, "loop/episode_length"), [0, 0, 3, 0, 0, 3, 0, 0, 3])

    def test_a_horizon_ending_mid_episode_keeps_the_partial_totals_in_the_state(self) -> None:
        final_state, metrics = run(RecordAgent(), ToyEnv(), jax.random.key(0), steps=4, gamma=GAMMA)

        assert int(final_state.episode_length) == 4
        assert float(final_state.episode_return) == pytest.approx(4.0)
        np.testing.assert_array_equal(_metric(metrics, "loop/episode_end"), [False] * 4)
        np.testing.assert_allclose(_metric(metrics, "loop/episode_return"), [0, 0, 0, 0], rtol=1e-6)
        np.testing.assert_array_equal(_metric(metrics, "loop/episode_length"), [0, 0, 0, 0])


class TestValidation:
    def test_a_missing_override_defers_to_the_spec_policy(self) -> None:
        env = ToyEnv(truncate_at=3, truncation_policy="terminate")

        _, metrics = run(RecordAgent(), env, jax.random.key(0), steps=5, gamma=GAMMA)

        np.testing.assert_allclose(_metric(metrics, "loop/discount"), [0.9, 0.9, 0.0, 0.9, 0.9], rtol=1e-6)

    def test_an_invalid_spec_policy_is_rejected_when_env_spec_is_built(self) -> None:
        env = ToyEnv(truncate_at=3, truncation_policy=cast(TruncationPolicy, "bootstrapp"))

        with pytest.raises(ValueError, match="truncation_policy"):
            run(RecordAgent(), env, jax.random.key(0), steps=5, gamma=GAMMA)

    def test_an_invalid_override_policy_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="truncation_policy"):
            run(
                RecordAgent(),
                ToyEnv(truncate_at=3),
                jax.random.key(0),
                steps=5,
                gamma=GAMMA,
                truncation_policy=cast(TruncationPolicy, "terminatee"),
            )

    def test_an_agent_metric_in_the_loop_namespace_is_rejected_at_trace_time(self) -> None:
        with pytest.raises(ValueError, match="loop/"):
            run(LoopNamespaceAgent(), ToyEnv(), jax.random.key(0), steps=3, gamma=GAMMA)
