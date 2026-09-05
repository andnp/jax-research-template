"""Medium tests for the brax adapter running through the shared training loop under jit.

These used to drive the brax adapter through GymnaxCompatibilityBridge, now deleted.
The bridge itself is not the coverage that matters here: it is the brax adapter -- real
physics, and the action-normalization wrapper composed with it -- working correctly
under jit. Both are retargeted to run through rl_components.loop.run directly, which is
what every agent actually drives the adapter through now.
"""

from typing import cast

import chex
import jax
import jax.numpy as jnp
import pytest
from rl_components.action_normalization import ActionNormalizationWrapper
from rl_components.agent_protocol import AgentStep
from rl_components.brax import BraxConfig, make_brax_adapter
from rl_components.env_protocol import EnvProtocol
from rl_components.loop import run
from rl_components.structs import chex_struct
from rl_components.timestep import Timestep

STEPS = 2


@chex_struct(frozen=True)
class _ZeroAgentState:
    unused: jax.Array


class _ZeroActionAgent:
    """Always acts zero; enough to drive an env through the loop under jit."""

    def __init__(self, action_shape: tuple[int, ...], action_dtype: jnp.dtype) -> None:
        self._action_shape = action_shape
        self._action_dtype = action_dtype

    def init(self, key: chex.PRNGKey, spec: object) -> _ZeroAgentState:
        del key, spec
        return _ZeroAgentState(unused=jnp.zeros((), jnp.int32))

    def step(
        self,
        state: _ZeroAgentState,
        timestep: Timestep[jax.Array],
        step_index: jax.Array,
    ) -> AgentStep[_ZeroAgentState, jax.Array]:
        del timestep, step_index
        return AgentStep(state=state, action=jnp.zeros(self._action_shape, self._action_dtype), metrics={})


class TestBraxAdapterThroughLoopJIT:
    def test_real_brax_adapter_runs_through_loop_under_jit(self) -> None:
        pytest.importorskip("brax")

        adapter = cast(
            EnvProtocol[jax.Array, object, jax.Array, None],
            make_brax_adapter(BraxConfig(env_name="inverted_pendulum")),
        )
        spec = adapter.spec()
        agent = _ZeroActionAgent(spec.action_shape, jnp.dtype(spec.action_dtype))

        def train_fn(key: jax.Array):
            return run(agent, adapter, key, steps=STEPS, gamma=0.99)

        final_state, metrics = jax.jit(train_fn)(jax.random.key(0))

        assert final_state.timestep.observation.shape == spec.observation_shape
        assert metrics["loop/reward"].shape == (STEPS,)
        assert metrics["loop/terminated"].shape == (STEPS,)
        assert metrics["loop/truncated"].shape == (STEPS,)

    def test_normalized_brax_adapter_runs_through_loop_under_jit(self) -> None:
        pytest.importorskip("brax")

        adapter = cast(
            EnvProtocol[jax.Array, object, jax.Array, None],
            make_brax_adapter(BraxConfig(env_name="inverted_pendulum")),
        )
        normalized_adapter = cast(
            EnvProtocol[jax.Array, object, jax.Array, None],
            ActionNormalizationWrapper(adapter),
        )
        spec = normalized_adapter.spec()
        agent = _ZeroActionAgent(spec.action_shape, jnp.dtype(spec.action_dtype))

        def train_fn(key: jax.Array):
            return run(agent, normalized_adapter, key, steps=STEPS, gamma=0.99)

        final_state, metrics = jax.jit(train_fn)(jax.random.key(0))

        assert spec.action_low is not None
        assert spec.action_high is not None
        assert jnp.allclose(spec.action_low, -jnp.ones(spec.action_shape, dtype=spec.action_dtype))
        assert jnp.allclose(spec.action_high, jnp.ones(spec.action_shape, dtype=spec.action_dtype))
        assert final_state.timestep.observation.shape == spec.observation_shape
        assert final_state.env_state.obs.shape == spec.observation_shape
        assert metrics["loop/reward"].shape == (STEPS,)
        assert metrics["loop/terminated"].shape == (STEPS,)
        assert metrics["loop/truncated"].shape == (STEPS,)
