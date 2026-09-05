"""No ``EnvProtocol`` adapter may reset inside ``step``; the boundary belongs to the loop.

This is the invariant the whole adapter migration rests on. :func:`rl_components.loop.run`
reads the observation a transition actually reached, stores it as
``Timestep.bootstrap_observation``, and only then performs its own reset. An adapter that
fuses a reset into ``step`` -- which is what Gymnax's ``Environment.step`` and Brax's
``AutoResetWrapper`` do natively -- hands the loop the *next* episode's first observation
in place of the terminal one. Training continues, and every terminal transition bootstraps
from the start of the following episode.

Each adapter here is driven to a boundary and asked two questions a fused reset cannot
answer correctly:

- is the returned observation the terminal one, rather than what ``reset`` produces?
- does the returned *state* carry the step that was just taken, rather than a fresh one?

The second question is the sharper of the two, because an observation check alone can be
satisfied by accident whenever a fused reset draws a different key than the test's own
``reset`` call.

Only ``GymnaxEnv`` was covered before, by its own boundary tests. For the other adapters
the invariant was invisible: a fusion that replaces just the boundary observation --
precisely the value the loop stores as ``bootstrap_observation`` -- passed every existing
test of ``BraxAdapter``, ``MujocoPlaygroundAdapter`` and ``ProcessControlEnv``, because
those tests read the returned flags rather than the returned observation.

``PythonEnvBridge`` is deliberately absent: its emulator lives in mutable Python, so
``tests/medium/test_rl_envs_python_env_bridge.py`` pins the same invariant more
strongly still, by counting the real resets the environment performs.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, cast

import chex
import gymnax
import jax
import jax.numpy as jnp
import pytest
import rl_envs.brax as brax_module
from rl_components.gymnax_bridge import make_gymnax_env
from rl_components.process_control_bridge import ProcessControlEnv
from rl_envs.brax import BraxAdapter, BraxConfig
from rl_envs.frame_stack import FrameStackWrapper
from rl_envs.mujoco_playground_bridge import MujocoPlaygroundAdapter

RESET_OBSERVATION = -99.0
"""A value no double's ``step`` ever produces, so a fused reset is unmistakable."""


@dataclass(frozen=True)
class Boundary:
    """What one adapter returned from the step that ended an episode.

    Attributes:
        observation: The observation the boundary ``step`` returned.
        reset_observation: What ``reset`` returns, which the above must not equal.
        ended: Whether the step really was a boundary, guarding against a vacuous pass.
        steps_taken: The step count carried by the returned state; ``0`` after a reset.
    """

    observation: jax.Array
    reset_observation: jax.Array
    ended: bool
    steps_taken: int


def _gymnax_boundary() -> Boundary:
    """Real CartPole with a one-step limit, so the first step is already the boundary."""
    raw, params = gymnax.make("CartPole-v1")
    params = params.replace(max_steps_in_episode=1)
    env = make_gymnax_env(raw)
    key = jax.random.key(0)

    reset = env.reset(key, params)
    step = env.step(key, reset.state, jnp.int32(0), params)

    return Boundary(
        observation=step.observation,
        reset_observation=env.reset(key, params).observation,
        ended=bool(step.terminated | step.truncated),
        steps_taken=int(step.state.time),
    )


@dataclass(frozen=True)
class _DoubleState:
    """A state that says how many steps produced it, which a reset cannot fake."""

    obs: jax.Array
    reward: jax.Array
    done: jax.Array
    info: dict[str, jax.Array]
    steps_taken: jax.Array


def _double_reset_state() -> _DoubleState:
    return _DoubleState(
        obs=jnp.full((2,), RESET_OBSERVATION, dtype=jnp.float32),
        reward=jnp.array(0.0, dtype=jnp.float32),
        done=jnp.array(0.0, dtype=jnp.float32),
        info={},
        steps_taken=jnp.array(0, dtype=jnp.int32),
    )


def _double_step_state(previous: _DoubleState) -> _DoubleState:
    steps_taken = previous.steps_taken + 1
    return _DoubleState(
        obs=jnp.full((2,), 1.0, dtype=jnp.float32) * steps_taken.astype(jnp.float32),
        reward=jnp.array(1.0, dtype=jnp.float32),
        done=jnp.array(1.0, dtype=jnp.float32),
        info={},
        steps_taken=steps_taken,
    )


class _EndingPlaygroundEnv:
    """A playground env that terminates on its first step."""

    action_size = 1
    observation_size = 2
    mj_model: Any = None

    def reset(self, rng: jax.Array) -> _DoubleState:
        del rng
        return _double_reset_state()

    def step(self, state: _DoubleState, action: jax.Array) -> _DoubleState:
        del action
        return _double_step_state(state)


def _playground_boundary() -> Boundary:
    env = MujocoPlaygroundAdapter(cast(Any, _EndingPlaygroundEnv()))
    key = jax.random.key(0)

    reset = env.reset(key)
    step = env.step(key, reset.state, jnp.zeros((1,), dtype=jnp.float32))

    return Boundary(
        observation=step.observation,
        reset_observation=env.reset(key).observation,
        ended=bool(step.terminated | step.truncated),
        steps_taken=int(cast(_DoubleState, step.state).steps_taken),
    )


class _EndingBraxEnv:
    """A Brax env that reports ``done`` on its first step."""

    action_size = 1
    observation_size = 2
    sys: Any = None

    def reset(self, key: chex.PRNGKey) -> _DoubleState:
        del key
        return _double_reset_state()

    def step(self, state: _DoubleState, action: jax.Array) -> _DoubleState:
        del action
        return _double_step_state(state)


def _brax_boundary(monkeypatch: pytest.MonkeyPatch) -> Boundary:
    def make_ending_env(config: BraxConfig):
        return _EndingBraxEnv()

    monkeypatch.setattr(brax_module, "_make_brax_env", make_ending_env)
    env = BraxAdapter(BraxConfig(env_name="fake"))
    key = jax.random.key(0)

    reset = env.reset(key)
    step = env.step(key, reset.state, jnp.zeros((1,), dtype=jnp.float32))

    return Boundary(
        observation=step.observation,
        reset_observation=env.reset(key).observation,
        ended=bool(step.terminated | step.truncated),
        steps_taken=int(cast(_DoubleState, step.state).steps_taken),
    )


def _process_control_env() -> ProcessControlEnv[jax.Array]:
    """A benchmark whose fused ``done`` fires on the first step."""

    def reset_fn(key: chex.PRNGKey) -> tuple[jax.Array, jax.Array]:
        del key
        return jnp.array(0, dtype=jnp.int32), jnp.full((2,), RESET_OBSERVATION, dtype=jnp.float32)

    def step_fn(
        state: jax.Array, action: jax.Array, key: chex.PRNGKey
    ) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, dict[str, jax.Array]]:
        del action, key
        steps_taken = state + 1
        return (
            steps_taken,
            jnp.full((2,), 1.0, dtype=jnp.float32) * steps_taken.astype(jnp.float32),
            jnp.array(1.0, dtype=jnp.float32),
            jnp.array(True),
            {},
        )

    return ProcessControlEnv(reset_fn, step_fn, obs_dim=2, action_dim=1)


def _process_control_boundary() -> Boundary:
    env = _process_control_env()
    key = jax.random.key(0)

    reset = env.reset(key)
    step = env.step(key, reset.state, jnp.zeros((1,), dtype=jnp.float32))

    return Boundary(
        observation=step.observation,
        reset_observation=env.reset(key).observation,
        ended=bool(step.terminated | step.truncated),
        steps_taken=int(step.state),
    )


def _frame_stack_boundary() -> Boundary:
    """Stacking must roll the terminal frame in, not refill from a fresh episode."""
    env = FrameStackWrapper(_process_control_env(), n_frames=2)
    key = jax.random.key(0)

    reset = env.reset(key)
    step = env.step(key, reset.state, jnp.zeros((1,), dtype=jnp.float32))

    return Boundary(
        observation=step.observation,
        reset_observation=env.reset(key).observation,
        ended=bool(step.terminated | step.truncated),
        steps_taken=int(step.state.inner_state),
    )


def _ignore_monkeypatch_gymnax(_monkeypatch: pytest.MonkeyPatch):
    return _gymnax_boundary()


def _ignore_monkeypatch_playground(_monkeypatch: pytest.MonkeyPatch):
    return _playground_boundary()


def _ignore_monkeypatch_process_control(_monkeypatch: pytest.MonkeyPatch):
    return _process_control_boundary()


def _ignore_monkeypatch_frame_stack(_monkeypatch: pytest.MonkeyPatch):
    return _frame_stack_boundary()


@pytest.mark.parametrize(
    "build_boundary",
    [
        pytest.param(_ignore_monkeypatch_gymnax, id="GymnaxEnv"),
        pytest.param(_brax_boundary, id="BraxAdapter"),
        pytest.param(_ignore_monkeypatch_playground, id="MujocoPlaygroundAdapter"),
        pytest.param(_ignore_monkeypatch_process_control, id="ProcessControlEnv"),
        pytest.param(_ignore_monkeypatch_frame_stack, id="FrameStackWrapper"),
    ],
)
def test_step_returns_the_terminal_state_not_a_fresh_one(
    build_boundary: Callable[[pytest.MonkeyPatch], Boundary],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    boundary = build_boundary(monkeypatch)

    assert boundary.ended, "the case must actually reach a boundary, or it proves nothing"
    assert boundary.steps_taken == 1, "step returned a reset state, so the transition was lost"
    assert not jnp.allclose(boundary.observation, boundary.reset_observation), (
        "step returned the post-reset observation, so the terminal one never reaches the loop"
    )
