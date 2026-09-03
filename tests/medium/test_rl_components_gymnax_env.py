"""Medium tests for the Gymnax driven adapter.

The two properties that matter are the two Gymnax fuses this adapter undoes: the
auto-reset inside ``Environment.step``, and the time limit folded into ``is_terminal``.
Both are checked against a real Gymnax environment rather than a stub, because both are
implemented in Gymnax's base class and a stub would only restate the assertion.
"""

from __future__ import annotations

from typing import Protocol, cast

import gymnax
import jax
import jax.numpy as jnp
from rl_components.env_protocol import EnvSpec
from rl_components.gymnax_bridge import make_gymnax_env


class _CartPoleState(Protocol):
    time: jax.Array


class _CartPoleParams(Protocol):
    max_steps_in_episode: int


class _CartPole(Protocol):
    def get_obs(self, state: _CartPoleState) -> jax.Array: ...

    def step(
        self,
        key: jax.Array,
        state: _CartPoleState,
        action: jax.Array,
        params: _CartPoleParams,
    ) -> tuple[jax.Array, _CartPoleState, jax.Array, jax.Array, dict[str, jax.Array]]: ...


def _cartpole(limit: int | None = None) -> tuple[_CartPole, _CartPoleParams]:
    raw, params = gymnax.make("CartPole-v1")
    if limit is not None:
        params = params.replace(max_steps_in_episode=limit)
    return cast(_CartPole, raw), cast(_CartPoleParams, params)


class TestGymnaxEnvSpec:
    def test_spec_describes_the_discrete_action_space(self) -> None:
        raw, params = _cartpole()
        env = make_gymnax_env(raw)

        spec = env.spec(params)

        assert spec == EnvSpec(
            id="gymnax:CartPole-v1",
            observation_shape=(4,),
            action_shape=(),
            observation_dtype=jnp.dtype(jnp.float32),
            action_dtype=jnp.dtype(jnp.int32),
            num_actions=2,
        )

    def test_spec_falls_back_to_the_default_params(self) -> None:
        raw, params = _cartpole()
        env = make_gymnax_env(raw)

        assert env.spec() == env.spec(params)


class TestGymnaxEnvBoundaries:
    def test_the_time_limit_is_a_truncation_not_a_termination(self) -> None:
        """The step that reaches ``max_steps_in_episode`` must keep its bootstrap."""
        raw, params = _cartpole(3)
        env = make_gymnax_env(raw)
        key = jax.random.key(0)

        state = env.reset(key, params).state
        flags = []
        for _ in range(3):
            step = jax.jit(env.step)(key, state, jnp.int32(0), params)
            flags.append((bool(step.terminated), bool(step.truncated)))
            state = step.state

        assert flags == [(False, False), (False, False), (False, True)]

    def test_the_terminal_observation_survives_the_boundary(self) -> None:
        """Gymnax's fused ``step`` would replace this observation with a reset one."""
        raw, params = _cartpole(1)
        env = make_gymnax_env(raw)
        key = jax.random.key(0)

        reset = env.reset(key, params)
        adapted = env.step(key, reset.state, jnp.int32(0), params)
        fused_observation, _, _, fused_done, _ = raw.step(key, reset.state, jnp.int32(0), params)

        assert bool(fused_done)
        assert bool(adapted.truncated)
        assert not jnp.allclose(adapted.observation, fused_observation)
        assert jnp.allclose(adapted.observation, raw.get_obs(adapted.state))

    def test_a_real_termination_is_not_reported_as_a_truncation(self) -> None:
        """Driving the pole past its angle threshold must kill the bootstrap."""
        raw, params = _cartpole(500)
        env = make_gymnax_env(raw)
        key = jax.random.key(0)

        state = env.reset(key, params).state
        step = jax.jit(env.step)(key, state, jnp.int32(0), params)
        while not (step.terminated | step.truncated):
            step = jax.jit(env.step)(key, step.state, jnp.int32(0), params)

        assert bool(step.terminated)
        assert not bool(step.truncated)
        assert int(step.state.time) < params.max_steps_in_episode
