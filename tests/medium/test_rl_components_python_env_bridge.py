"""Medium tests for PythonEnvBridge terminal observation semantics and boundary ownership."""

from __future__ import annotations

from typing import Literal, override

import gymnasium
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from rl_components.agent_protocol import AgentStep
from rl_components.env_protocol import EnvSpec
from rl_components.frame_stack import FrameStackWrapper
from rl_components.loop import run
from rl_components.python_env_bridge import PythonEnvBridge
from rl_components.timestep import Timestep

EPISODE_LENGTH = 2
"""The fake env ends its episode on its second step, so a short run crosses two boundaries."""

GAMMA = 0.9


class _FakeALE:
    def __init__(self, env: "_FakeGymALE") -> None:
        self._env = env

    def cloneState(self) -> tuple[int, int]:
        return self._env._episode_start, self._env._episode_steps

    def restoreState(self, state: object, /) -> None:
        if not isinstance(state, tuple) or len(state) != 2 or not all(isinstance(value, int) for value in state):
            raise TypeError("unexpected fake ALE state")
        self._env._episode_start, self._env._episode_steps = state


class _FakeGymALE(gymnasium.Env[np.ndarray, int]):
    """An ALE-shaped env whose observations encode which episode and step produced them.

    ``reset`` moves the episode base to ``10 * reset_count``, so the post-reset observation
    of every episode differs from the terminal observation of the previous one and the two
    are never confusable. It also counts its own resets, which is how a test can assert who
    performed them.
    """

    observation_space = gymnasium.spaces.Box(0, 255, shape=(1,), dtype=np.uint8)
    action_space = gymnasium.spaces.Discrete(2)

    def __init__(self, end_kind: Literal["terminated", "truncated"]) -> None:
        self.ale = _FakeALE(self)
        self._end_kind = end_kind
        self.reset_count = 0
        self.step_count = 0
        self._episode_start = 0
        self._episode_steps = 0

    @override
    def reset(self, *, seed: int | None = None, options: dict[str, object] | None = None) -> tuple[np.ndarray, dict[str, object]]:
        del seed, options
        self.reset_count += 1
        self._episode_start = self.reset_count * 10
        self._episode_steps = 0
        return np.array([self._episode_start], dtype=np.uint8), {}

    @override
    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict[str, object]]:
        self.step_count += 1
        self._episode_steps += 1
        observation = np.array([self._episode_start + self._episode_steps], dtype=np.uint8)
        done = self._episode_steps == EPISODE_LENGTH
        return (
            observation,
            float(action),
            done if self._end_kind == "terminated" else False,
            done if self._end_kind == "truncated" else False,
            {},
        )


class _ConstantActionAgent:
    """A do-nothing agent, so a run measures the environment's boundary behaviour alone."""

    def init(self, key: jax.Array, spec: EnvSpec) -> jax.Array:
        del key
        return jnp.zeros(tuple(spec.action_shape), dtype=jnp.dtype(spec.action_dtype))

    def step(
        self,
        state: jax.Array,
        timestep: Timestep[jax.Array],
        step_index: jax.Array,
    ) -> AgentStep[jax.Array, jax.Array]:
        del step_index
        return AgentStep(
            state=state,
            action=state,
            metrics={"bootstrap_frame": timestep.bootstrap_observation[-1, 0].astype(jnp.int32)},
        )


@pytest.mark.parametrize("end_kind", ["terminated", "truncated"])
def test_the_bridge_returns_the_true_terminal_observation(
    end_kind: Literal["terminated", "truncated"],
) -> None:
    """The bridge no longer resets at a boundary, so its observation is never post-reset.

    The fused reset it used to perform returned ``40`` -- the second episode's first frame
    -- from the step that terminated the first episode, which is the value asserted against
    here. The emulator now stays on the terminal observation until someone resets it.
    """
    gym_env = _FakeGymALE(end_kind)
    bridge = PythonEnvBridge(lambda: gym_env)
    reset = bridge.reset(jax.random.key(0))
    resets_after_construction = gym_env.reset_count

    first = bridge.step(jax.random.key(1), reset.state, jnp.asarray(0, dtype=jnp.int32))
    terminal = bridge.step(jax.random.key(2), first.state, jnp.asarray(0, dtype=jnp.int32))

    assert int(first.observation[0]) == 31
    assert bool(first.terminated or first.truncated) is False
    assert int(terminal.observation[0]) == 32, "the terminal observation must not be replaced by a post-reset one"
    assert bool(terminal.terminated or terminal.truncated) is True
    assert gym_env.reset_count == resets_after_construction, "step must not reset the environment"
    assert terminal.info == {}


def test_the_loop_resets_the_frame_stacked_bridge_once_per_boundary() -> None:
    """The reset callback must fire exactly on the boundary iterations and nowhere else.

    ``PythonEnvBridge`` keeps its emulator in mutable Python behind an ordered
    ``io_callback``, outside the pytree, so the loop's ``lax.cond`` is load-bearing rather
    than an optimisation: a select would discard the returned value while the side effect
    stood, resetting the emulator behind JAX's back on every step. Counting the fake env's
    own resets is the only way to observe which of the two actually happened.
    """
    steps = 6
    gym_env = _FakeGymALE("terminated")
    env = FrameStackWrapper(PythonEnvBridge(lambda: gym_env), n_frames=2)
    resets_before = gym_env.reset_count

    _final_state, metrics = jax.jit(
        lambda key: run(_ConstantActionAgent(), env, key, steps=steps, gamma=GAMMA)
    )(jax.random.key(0))

    boundaries = int(jnp.sum(metrics["loop/episode_end"]))
    loop_resets = gym_env.reset_count - resets_before
    assert boundaries == steps // EPISODE_LENGTH == 3
    assert gym_env.step_count == steps
    assert loop_resets == 1 + boundaries, "one initial reset plus exactly one per boundary"


def test_the_loop_supplies_the_post_reset_observation_the_bridge_no_longer_does() -> None:
    """The two halves of the boundary: the bridge's terminal frame, the loop's fresh one.

    ``bootstrap_observation`` must carry the terminal frame the transition reached, while
    the observation the agent acts from next must be the loop's post-reset frame. The fused
    reset collapsed both onto the post-reset value, which is what made a terminal
    transition bootstrap from the start of the next episode.
    """
    steps = 5
    gym_env = _FakeGymALE("terminated")
    env = FrameStackWrapper(PythonEnvBridge(lambda: gym_env), n_frames=2)

    final_state, metrics = jax.jit(
        lambda key: run(_ConstantActionAgent(), env, key, steps=steps, gamma=GAMMA)
    )(jax.random.key(0))

    episode_end = metrics["loop/episode_end"].tolist()
    # Frames the agent bootstrapped from, shifted by one: a metric at index i describes the
    # transition the agent sees closed at i + 1.
    bootstrap_frames = metrics["bootstrap_frame"].tolist()
    assert episode_end == [False, True, False, True, False]
    assert bootstrap_frames[2] == 32, "the closed terminal transition must bootstrap from the terminal frame"
    assert bootstrap_frames[3] == 41, "the step after a boundary must start from the loop's post-reset frame"
    assert int(final_state.timestep.observation[-1, 0]) == 51


def test_frame_stack_rolls_across_a_boundary_instead_of_refilling() -> None:
    """The wrapper's own fused refill is gone, so a boundary step keeps the older frame."""
    gym_env = _FakeGymALE("terminated")
    env = FrameStackWrapper(PythonEnvBridge(lambda: gym_env), n_frames=2)
    reset = env.reset(jax.random.key(0))
    # The bridge probes the env at construction, so this episode's base is not the first.
    base = 10 * gym_env.reset_count

    first = env.step(jax.random.key(1), reset.state, jnp.asarray(0, dtype=jnp.int32))
    terminal = env.step(jax.random.key(2), first.state, jnp.asarray(0, dtype=jnp.int32))

    assert reset.observation[:, 0].tolist() == [base, base], "reset stacks the fresh observation n_frames times"
    assert first.observation[:, 0].tolist() == [base, base + 1]
    assert terminal.observation[:, 0].tolist() == [base + 1, base + 2], "a refill would have produced two copies of the last frame"
