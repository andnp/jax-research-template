"""Medium tests for opt-in environment state preservation at loop boundaries."""

from __future__ import annotations

from dataclasses import dataclass, replace

import chex
import jax
import jax.numpy as jnp
import numpy as np
from rl_components.agent_protocol import AgentStep
from rl_components.env_protocol import BoundaryResetProtocol, EnvReset, EnvSpec, EnvStep
from rl_components.gymnax_bridge import GymnaxEnv, make_gymnax_env
from rl_components.loop import LoopState, run
from rl_components.structs import chex_struct
from rl_components.timestep import Timestep


@chex_struct(frozen=True)
class _CounterState:
    episode_step: jax.Array
    total_step: jax.Array
    previous_boundary_step: jax.Array


class _CounterEnv:
    def __init__(self, *, terminates: bool = True) -> None:
        self.terminates = terminates

    def spec(self, params: None = None) -> EnvSpec:
        del params
        return EnvSpec(
            id="boundary-counter",
            observation_shape=(1,),
            action_shape=(),
            observation_dtype=jnp.float32,
            action_dtype=jnp.int32,
            num_actions=2,
        )

    def reset(self, key: chex.PRNGKey, params: None = None) -> EnvReset[jax.Array, _CounterState]:
        del key, params
        state = _CounterState(
            episode_step=jnp.array(0),
            total_step=jnp.array(0),
            previous_boundary_step=jnp.array(-1),
        )
        return EnvReset(observation=jnp.zeros((1,), jnp.float32), state=state)

    def step(
        self,
        key: chex.PRNGKey,
        state: _CounterState,
        action: jax.Array,
        params: None = None,
    ) -> EnvStep[jax.Array, _CounterState]:
        del key, action, params
        episode_step = state.episode_step + 1
        total_step = state.total_step + 1
        return EnvStep(
            observation=episode_step.astype(jnp.float32)[None],
            state=_CounterState(
                episode_step=episode_step,
                total_step=total_step,
                previous_boundary_step=state.previous_boundary_step,
            ),
            reward=jnp.ones((), jnp.float32),
            terminated=(episode_step == 2) if self.terminates else jnp.array(False),
            truncated=jnp.array(False),
            info={},
        )


class _PreservingCounterEnv(_CounterEnv):
    def reset_after_episode(
        self,
        key: chex.PRNGKey,
        previous_state: _CounterState,
        params: None = None,
    ) -> EnvReset[jax.Array, _CounterState]:
        del key, params
        state = _CounterState(
            episode_step=jnp.array(0),
            total_step=previous_state.total_step,
            previous_boundary_step=previous_state.total_step,
        )
        return EnvReset(observation=jnp.zeros((1,), jnp.float32), state=state)


@chex_struct(frozen=True)
class _AgentState:
    count: jax.Array


class _ZeroAgent:
    def init(self, key: chex.PRNGKey, spec: EnvSpec) -> _AgentState:
        del key, spec
        return _AgentState(count=jnp.array(0))

    def step(
        self,
        state: _AgentState,
        timestep: Timestep[jax.Array],
        step_index: jax.Array,
    ) -> AgentStep[_AgentState, jax.Array]:
        del timestep, step_index
        return AgentStep(
            state=_AgentState(count=state.count + 1),
            action=jnp.array(0, jnp.int32),
            metrics={},
        )


def _run(env: _CounterEnv, *, episode_cutoff: int = -1) -> tuple[LoopState[_CounterState, _AgentState, jax.Array], dict[str, jax.Array]]:
    return jax.jit(lambda key: run(_ZeroAgent(), env, key, steps=5, gamma=0.99, episode_cutoff=episode_cutoff))(jax.random.key(0))


def test_boundary_reset_receives_prior_state_and_preserves_cumulative_clock_under_jit_vmap() -> None:
    """Boundary resets receive prior state and retain its cumulative clock across episodes."""
    env = _PreservingCounterEnv()

    batched_run = jax.jit(jax.vmap(lambda key: run(_ZeroAgent(), env, key, steps=5, gamma=0.99)))
    final_states, metrics = batched_run(jax.random.split(jax.random.key(0), 2))

    assert isinstance(env, BoundaryResetProtocol)
    np.testing.assert_array_equal(final_states.env_state.total_step, [5, 5])
    np.testing.assert_array_equal(final_states.env_state.previous_boundary_step, [4, 4])
    np.testing.assert_array_equal(metrics["loop/episode_end"], [[False, True, False, True, False]] * 2)


def test_boundary_reset_preserves_state_through_loop_truncation() -> None:
    """A loop cutoff still uses the opt-in reset and preserves the terminal state."""
    final_state, metrics = _run(_PreservingCounterEnv(terminates=False), episode_cutoff=2)

    np.testing.assert_array_equal(metrics["loop/truncated"], [False, True, False, True, False])
    assert int(final_state.env_state.total_step) == 5


def test_environment_without_boundary_protocol_keeps_ordinary_reset_behavior() -> None:
    """Environments without the opt-in protocol continue to reset normally at boundaries."""
    env = _CounterEnv()

    final_state, _ = _run(env)

    assert not isinstance(env, BoundaryResetProtocol)
    assert int(final_state.env_state.total_step) == 1
    assert int(final_state.env_state.previous_boundary_step) == -1


@dataclass(frozen=True)
class _Params:
    max_steps_in_episode: int = 3

    def replace(self, **changes: object) -> _Params:
        return replace(self, **changes)


class _ForwardingEnv:
    default_params = _Params()

    def reset_after_episode(
        self,
        key: chex.PRNGKey,
        previous_state: jax.Array,
        params: _Params,
    ) -> EnvReset[jax.Array, jax.Array]:
        del key
        assert params.max_steps_in_episode == 4
        return EnvReset(observation=previous_state + 1, state=previous_state)


def test_gymnax_adapter_forwards_opt_in_boundary_reset_under_jit() -> None:
    """Gymnax wrappers forward boundary resets while preserving the deferred limit."""
    ordinary = GymnaxEnv(_ForwardingEnv())
    env = make_gymnax_env(_ForwardingEnv())

    assert not isinstance(ordinary, BoundaryResetProtocol)
    assert isinstance(env, BoundaryResetProtocol)

    reset = jax.jit(lambda state: env.reset_after_episode(jax.random.key(0), state))(jnp.array(7))

    assert int(reset.observation) == 8
    assert int(reset.state) == 7
