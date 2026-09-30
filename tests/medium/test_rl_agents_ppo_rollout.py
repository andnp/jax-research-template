"""Behavioural gate: PPO's rollout slots, and when the update fires.

Two properties this port turns on cannot be seen in the returns, and both are silent
cross-episode corruption that still trains:

1. **Slot `i` records the transition CLOSED at iteration `i`.** The acting half of the slot
   comes from what the agent carried in, and the closing half from the incoming `Timestep`.
   A slot that paired iteration `i`'s reward with iteration `i`'s observation would describe
   a transition that never happened.
2. **The boundary slot's `bootstrap_value` is the value of the TRUE state reached**, never
   of the post-reset observation the agent goes on to act from. Those two are the same array
   on every step except a boundary, which is exactly where it matters.

`PPOAgent.step` sees no environment, so both are asserted by driving it with hand-built
timesteps and comparing against the network applied by hand -- no environment, no RNG in the
expectations, and no re-implementation of anything inside the agent.

The update schedule is gated separately, through `rl_components.loop.run`, because "the
window is full" is a statement about `step_index` over the whole run rather than about one
call. Two more properties are gated the same way, because both are decided by the shape of
what the agent carries rather than by anything a caller passes: the continuous branch, which
`step` selects from the rank of its own stored action since it never sees a spec, and
observation normalization, whose statistics now advance once per loop iteration.
"""

from __future__ import annotations

from typing import override

import jax
import jax.numpy as jnp
from rl_agents.ppo import PPOAgent, PPOAgentState, PPOConfig
from rl_components.env_protocol import EnvReset, EnvSpec, EnvStep
from rl_components.loop import run
from rl_components.timestep import Timestep

_NUM_STEPS = 4
_GAMMA = 0.99
_SEED = 0


def _select_second_slot(window: jax.Array):
    return window[1]

_SPEC = EnvSpec(
    id="toy:ppo-rollout",
    observation_shape=(2,),
    action_shape=(),
    observation_dtype=jnp.dtype(jnp.float32),
    action_dtype=jnp.dtype(jnp.int32),
    num_actions=2,
)

_FIRST_OBSERVATION = jnp.array([1.0, 2.0], dtype=jnp.float32)
_SECOND_OBSERVATION = jnp.array([3.0, 5.0], dtype=jnp.float32)
_TERMINAL_OBSERVATION = jnp.array([-7.0, 11.0], dtype=jnp.float32)
_RESET_OBSERVATION = jnp.array([0.5, -0.25], dtype=jnp.float32)


def _config(reward_scale: float = 1.0) -> PPOConfig:
    """A rollout short enough that a handful of steps spans a whole window."""
    return PPOConfig(NUM_STEPS=_NUM_STEPS, NUM_MINIBATCHES=1, UPDATE_EPOCHS=1, REWARD_SCALE=reward_scale)


def _timestep(
    observation: jax.Array,
    *,
    reward: float,
    discount: float,
    episode_end: bool,
    bootstrap_observation: jax.Array | None = None,
) -> Timestep[jax.Array]:
    return Timestep(
        reward=jnp.asarray(reward, jnp.float32),
        discount=jnp.asarray(discount, jnp.float32),
        bootstrap_observation=observation if bootstrap_observation is None else bootstrap_observation,
        episode_end=jnp.asarray(episode_end, jnp.bool_),
        observation=observation,
    )


def _value_of(state: PPOAgentState, observation: jax.Array) -> jax.Array:
    """The critic's estimate for one observation, applied outside the agent."""
    _, value = state.train_state.apply_fn(state.train_state.params, observation)
    return jnp.asarray(value, jnp.float32)


class TestRolloutSlotLayout:
    def test_the_first_slot_of_the_run_is_the_only_invalid_one(self) -> None:
        """`valid` is `step_index > 0`, which is false on exactly one iteration ever."""
        agent = PPOAgent(_config())
        state = agent.init(jax.random.key(_SEED), _SPEC)

        first = agent.step(state, _timestep(_FIRST_OBSERVATION, reward=0.0, discount=0.0, episode_end=False), jnp.int32(0))
        second = agent.step(
            first.state, _timestep(_SECOND_OBSERVATION, reward=1.0, discount=_GAMMA, episode_end=False), jnp.int32(1)
        )

        assert not bool(second.state.rollout.valid[0])
        assert bool(second.state.rollout.valid[1])

    def test_a_slot_pairs_the_carried_action_with_the_incoming_reward(self) -> None:
        """The acting half comes from the agent's own state, the closing half from the timestep."""
        agent = PPOAgent(_config(reward_scale=3.0))
        state = agent.init(jax.random.key(_SEED), _SPEC)

        first = agent.step(state, _timestep(_FIRST_OBSERVATION, reward=0.0, discount=0.0, episode_end=False), jnp.int32(0))
        second = agent.step(
            first.state, _timestep(_SECOND_OBSERVATION, reward=2.0, discount=_GAMMA, episode_end=False), jnp.int32(1)
        )
        slot = jax.tree_util.tree_map(_select_second_slot, second.state.rollout)

        assert jnp.array_equal(slot.obs, _FIRST_OBSERVATION), "the slot must hold the state the action was chosen from"
        assert jnp.array_equal(slot.action, first.action), "the slot must hold the action that opened the transition"
        assert jnp.array_equal(slot.value, _value_of(state, _FIRST_OBSERVATION))
        assert jnp.array_equal(slot.reward, jnp.asarray(6.0, jnp.float32)), "REWARD_SCALE must reach the stored reward"
        assert jnp.array_equal(slot.discount, jnp.asarray(_GAMMA, jnp.float32))
        assert not bool(slot.episode_end)

    def test_a_boundary_slot_bootstraps_from_the_true_final_state(self) -> None:
        """The post-reset observation must not be valued as the state the transition reached.

        The fused auto-reset this port removes made those two the same array. Here they are
        deliberately different, and only one of them can be right.
        """
        agent = PPOAgent(_config())
        state = agent.init(jax.random.key(_SEED), _SPEC)

        first = agent.step(state, _timestep(_FIRST_OBSERVATION, reward=0.0, discount=0.0, episode_end=False), jnp.int32(0))
        boundary = agent.step(
            first.state,
            _timestep(
                _RESET_OBSERVATION,
                reward=1.0,
                discount=0.0,
                episode_end=True,
                bootstrap_observation=_TERMINAL_OBSERVATION,
            ),
            jnp.int32(1),
        )
        slot = jax.tree_util.tree_map(_select_second_slot, boundary.state.rollout)

        assert jnp.array_equal(slot.bootstrap_value, _value_of(state, _TERMINAL_OBSERVATION))
        assert not jnp.array_equal(_value_of(state, _TERMINAL_OBSERVATION), _value_of(state, _RESET_OBSERVATION)), (
            "the two observations must be distinguishable or the assertion above is vacuous"
        )
        assert bool(slot.episode_end), "the trace must be cut at the boundary"
        assert jnp.array_equal(slot.discount, jnp.zeros((), jnp.float32))
        assert jnp.array_equal(boundary.state.last_obs, _RESET_OBSERVATION), (
            "the next action must be selected from the post-reset observation"
        )


class ToyEpisodeEnv:
    """A two-action environment that terminates every third step; state is the step count."""

    def spec(self, params: None = None) -> EnvSpec:
        del params
        return _SPEC

    def reset(self, key: jax.Array, params: None = None) -> EnvReset[jax.Array, jax.Array]:
        del key, params
        return EnvReset(observation=_RESET_OBSERVATION, state=jnp.zeros((), jnp.int32))

    def step(self, key: jax.Array, state: jax.Array, action: jax.Array, params: None = None) -> EnvStep[jax.Array, jax.Array]:
        del key, params
        next_state = state + jnp.ones((), jnp.int32)
        terminated = next_state % 3 == 0
        observation = jnp.stack([next_state.astype(jnp.float32), action.astype(jnp.float32)])
        return EnvStep(
            observation=observation,
            state=jnp.where(terminated, jnp.zeros((), jnp.int32), next_state),
            reward=jnp.ones((), jnp.float32),
            terminated=terminated,
            truncated=jnp.zeros((), jnp.bool_),
            info={},
        )


def _drive(steps: int) -> tuple[PPOAgentState, dict[str, jax.Array]]:
    agent = PPOAgent(_config())

    def train_fn(key: jax.Array):
        return run(agent, ToyEpisodeEnv(), key, steps=steps, gamma=_GAMMA)

    final_state, metrics = jax.jit(train_fn)(jax.random.key(_SEED))
    return final_state.agent_state, metrics


class TestUpdateSchedule:
    def test_the_update_fires_once_the_window_fills(self) -> None:
        """A run one step short of a full window must leave every parameter untouched.

        This is the negative control that separates "the update ran" from "the update ran
        and happened to change nothing": the two runs share a seed and an environment, so
        the only difference is whether the window ever filled.
        """
        filled, filled_metrics = _drive(_NUM_STEPS)
        short, short_metrics = _drive(_NUM_STEPS - 1)

        moved = [
            not bool(jnp.array_equal(before, after))
            for before, after in zip(
                jax.tree_util.tree_leaves(short.train_state.params),
                jax.tree_util.tree_leaves(filled.train_state.params),
                strict=True,
            )
        ]
        assert any(moved), "the update never fired on the iteration that filled the window"
        assert jnp.all(short_metrics["total_loss"] == 0.0), "a loss was reported without an update"
        assert int(jnp.sum(filled_metrics["total_loss"] != 0.0)) == 1, "exactly one update fits in one window"

    def test_the_metric_schema_is_fixed_and_zero_before_an_update(self) -> None:
        """Every iteration reports the same four keys, as zero placeholders when idle."""
        _, metrics = _drive(_NUM_STEPS)
        agent_metrics = {
            key: value for key, value in metrics.items() if not key.startswith(("loop/", "env/"))
        }

        assert set(agent_metrics) == {"total_loss", "value_loss", "policy_loss", "entropy"}
        for key, value in agent_metrics.items():
            assert value.shape == (_NUM_STEPS,), key
            assert value.dtype == jnp.float32, key
            assert jnp.all(jnp.isfinite(value)), key
            assert jnp.all(value[:-1] == 0.0), f"{key} was nonzero on an iteration that took no gradient step"

    def test_every_slot_after_the_first_window_is_valid(self) -> None:
        """`step_index` counts the run, so validity is not a per-rollout artefact."""
        first_window, _ = _drive(_NUM_STEPS)
        second_window, _ = _drive(2 * _NUM_STEPS)

        assert int(jnp.sum(first_window.rollout.valid)) == _NUM_STEPS - 1
        assert bool(jnp.all(second_window.rollout.valid))

    def test_terminal_slots_store_a_zero_discount(self) -> None:
        """The agent must fuse the loop's coefficient into the slot, not merely observe it."""
        window, metrics = _drive(3 * _NUM_STEPS)

        assert int(jnp.sum(metrics["loop/terminated"])) >= 2, "the toy environment never terminated"
        assert bool(jnp.any(window.rollout.discount == 0.0)), "no terminal transition reached the window"
        assert bool(jnp.any(window.rollout.discount > 0.0)), "every transition was recorded as terminal"
        assert bool(jnp.all(window.rollout.episode_end == (window.rollout.discount == 0.0))), (
            "with no truncation in this environment the two boundary quantities must agree"
        )

    def test_a_truncated_slot_keeps_its_bootstrap_while_cutting_the_trace(self) -> None:
        """``discount == gamma`` AND ``episode_end`` true in the same slot.

        This is the one case a single fused flag cannot represent, and the only regime that
        distinguishes the loop's two boundary quantities: everywhere else they agree, so
        ``test_terminal_slots_store_a_zero_discount`` above is satisfied by an agent that
        derives either one from the other.

        Defends against both fusions. Deriving ``episode_end`` as ``discount == 0.0`` makes a
        truncation look like a continuation and leaks the GAE trace across the boundary, which
        is the cross-episode leakage this port exists to remove. Zeroing ``discount`` wherever
        ``episode_end`` is set costs a truncation the bootstrap it is entitled to, which is the
        pre-port fused ``not_done`` defect verbatim. Both survived the whole PPO suite before
        this test existed, including the golden gate and the learning regression.
        """
        agent = PPOAgent(_config())

        def train_fn(key: jax.Array):
            return run(agent, ToyEpisodeEnv(), key, steps=3 * _NUM_STEPS, gamma=_GAMMA, episode_cutoff=2)

        final_state, metrics = jax.jit(train_fn)(jax.random.key(_SEED))
        rollout = final_state.agent_state.rollout

        # The cutoff fires at step 2, before the environment's own step-3 termination, so the
        # boundary is a truncation rather than a termination.
        assert int(jnp.sum(metrics["loop/truncated"])) >= 2, "the cutoff never fired"
        truncated_slots = rollout.episode_end & (rollout.discount > 0.0)
        assert bool(jnp.any(truncated_slots)), "no truncated transition reached the window"
        assert bool(jnp.all(jnp.where(truncated_slots, rollout.discount == jnp.asarray(_GAMMA, jnp.float32), True))), (
            "a truncated slot must keep the full discount while still breaking the trace"
        )


class ToyContinuousEnv:
    """A one-dimensional continuous environment, normalized to the box a tanh policy needs."""

    def spec(self, params: None = None) -> EnvSpec:
        del params
        return EnvSpec(
            id="toy:ppo-continuous",
            observation_shape=(2,),
            action_shape=(1,),
            observation_dtype=jnp.dtype(jnp.float32),
            action_dtype=jnp.dtype(jnp.float32),
            action_low=-jnp.ones((1,), jnp.float32),
            action_high=jnp.ones((1,), jnp.float32),
        )

    def reset(self, key: jax.Array, params: None = None) -> EnvReset[jax.Array, jax.Array]:
        del key, params
        return EnvReset(observation=jnp.zeros((2,), jnp.float32), state=jnp.zeros((), jnp.int32))

    def step(self, key: jax.Array, state: jax.Array, action: jax.Array, params: None = None) -> EnvStep[jax.Array, jax.Array]:
        del key, params
        next_state = state + jnp.ones((), jnp.int32)
        return EnvStep(
            observation=jnp.stack([next_state.astype(jnp.float32), action[0]]),
            state=next_state,
            reward=jnp.ones((), jnp.float32) - 0.1 * jnp.sum(jnp.square(action)),
            terminated=jnp.zeros((), jnp.bool_),
            truncated=jnp.zeros((), jnp.bool_),
            info={},
        )


class LargeMagnitudeEnv(ToyEpisodeEnv):
    """`ToyEpisodeEnv` with observations six orders of magnitude larger."""

    @override
    def reset(self, key: jax.Array, params: None = None) -> EnvReset[jax.Array, jax.Array]:
        reset = super().reset(key, params)
        return EnvReset(observation=reset.observation * 1e6, state=reset.state)

    @override
    def step(self, key: jax.Array, state: jax.Array, action: jax.Array, params: None = None) -> EnvStep[jax.Array, jax.Array]:
        env_step = super().step(key, state, action, params)
        return EnvStep(
            observation=env_step.observation * 1e6,
            state=env_step.state,
            reward=env_step.reward,
            terminated=env_step.terminated,
            truncated=env_step.truncated,
            info=env_step.info,
        )


class TestContinuousPolicy:
    def test_the_continuous_branch_trains_and_stays_finite(self) -> None:
        """`step` reads the policy's kind off the rank of its own stored action.

        A discrete policy returns one log-probability per step and a continuous one returns
        one per action dimension, which has to be summed. There is no spec in `step` to ask,
        so the inference is from `last_action`, and getting it wrong on a one-dimensional
        action space is invisible -- the sum of a length-one vector is the vector -- which is
        why this environment is driven end to end rather than probed for the flag.
        """
        agent = PPOAgent(_config())

        def train_fn(key: jax.Array):
            return run(agent, ToyContinuousEnv(), key, steps=2 * _NUM_STEPS, gamma=_GAMMA)

        final_state, metrics = jax.jit(train_fn)(jax.random.key(_SEED))

        assert final_state.agent_state.last_action.shape == (1,)
        assert int(jnp.sum(metrics["total_loss"] != 0.0)) == 2
        for leaf in jax.tree_util.tree_leaves(final_state.agent_state.train_state.params):
            assert bool(jnp.all(jnp.isfinite(leaf)))


class TestObservationNormalization:
    def test_large_magnitude_observations_stay_finite(self) -> None:
        """Normalization is the whole reason a 1e6 observation does not blow the update up."""
        agent = PPOAgent(PPOConfig(NUM_STEPS=_NUM_STEPS, NUM_MINIBATCHES=1, UPDATE_EPOCHS=1, NORMALIZE_OBSERVATIONS=True))
        steps = 2 * _NUM_STEPS

        def train_fn(key: jax.Array):
            return run(agent, LargeMagnitudeEnv(), key, steps=steps, gamma=_GAMMA)

        final_state, _ = jax.jit(train_fn)(jax.random.key(_SEED))

        for leaf in jax.tree_util.tree_leaves(final_state.agent_state.train_state.params):
            assert bool(jnp.all(jnp.isfinite(leaf)))
        assert bool(jnp.all(jnp.abs(final_state.agent_state.rollout.obs) <= PPOConfig().OBS_NORM_CLIP))

    def test_the_statistics_advance_once_per_iteration(self) -> None:
        """One sample per iteration, from the observation the agent acts from.

        The pre-port loop took one per environment step plus the reset observation, which is
        one more, because a run of `N` iterations closes only `N - 1` transitions.
        """
        steps = 3 * _NUM_STEPS
        agent = PPOAgent(PPOConfig(NUM_STEPS=_NUM_STEPS, NUM_MINIBATCHES=1, UPDATE_EPOCHS=1, NORMALIZE_OBSERVATIONS=True))

        def train_fn(key: jax.Array):
            return run(agent, ToyEpisodeEnv(), key, steps=steps, gamma=_GAMMA)

        final_state, _ = jax.jit(train_fn)(jax.random.key(_SEED))

        assert float(final_state.agent_state.obs_norm_state.observation_count) == float(steps)
