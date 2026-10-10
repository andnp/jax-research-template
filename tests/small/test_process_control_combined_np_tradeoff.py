"""Cost-compliance trade-off of the combined N+P benchmark.

Rewards are euros per step, averaged over the last three days of a six-day
rollout (nitrifier washout makes shorter horizons unrepresentative) on shared
seeds, so policies are compared on identical noise.
"""

import itertools
from collections.abc import Callable

import jax
import jax.numpy as jnp
from process_control.benchmarks.combined_np import CombinedNPConfig, CombinedNPState, make_combined_np_benchmark

_SEEDS = jnp.arange(3)
_DAYS = 6
_AVERAGED_DAYS = 3

# Best constant action [kla, recycle, Fe, carbon] from a grid search over
# kla {.55,.6,.65,.7}, recycle 1, Fe {.5,.6,.7,.8,.9}, carbon {.25,.5} (10-day, 6-seed rollouts).
_BEST_CONSTANT = jnp.array([0.65, 1.0, 0.9, 0.25])


def _mean_rewards(policy: Callable[[jax.Array], jax.Array]) -> jax.Array:
    """Per-seed mean reward over the final averaged days under an observation-to-action policy."""
    config = CombinedNPConfig()
    reset, step = make_combined_np_benchmark(config)
    n_steps = _DAYS * config.steps_per_day

    def rollout(seed: jax.Array):
        state, obs = reset(jax.random.PRNGKey(seed))

        def body(carry: tuple, key: jax.Array):
            state, obs = carry
            state, obs, reward, _done, _info = step(state, policy(obs), key)
            return (state, obs), reward

        keys = jax.random.split(jax.random.PRNGKey(seed + 100), n_steps)
        _, rewards = jax.lax.scan(body, (state, obs), keys)
        return rewards[-_AVERAGED_DAYS * config.steps_per_day :].mean()

    return jax.jit(jax.vmap(rollout))(_SEEDS)


def _constant(action: jax.Array) -> Callable[[jax.Array], jax.Array]:
    return lambda _obs: action


def _feedback(obs: jax.Array) -> jax.Array:
    """Proportional loops on the effluent analyzers around the constant's operating point."""
    po4, nh4, no3 = obs[0] * 10.0, obs[1] * 35.0, obs[2] * 20.0
    kla = 0.6 + 0.08 * (nh4 - 3.0)
    fe = 0.9 + 0.8 * (po4 - 0.8)
    carbon = 0.25 + 0.1 * (no3 - 6.5)
    return jnp.clip(jnp.array([kla, 1.0, fe, carbon]), 0.0, 1.0)


class TestConstantActionsAreNotOptimal:
    def test_tuned_constant_beats_zero_mid_and_max_actions(self) -> None:
        """No endpoint or midpoint constant is competitive with a tuned interior one."""
        best = float(_mean_rewards(_constant(_BEST_CONSTANT)).mean())

        for action in (jnp.zeros(4), jnp.full(4, 0.5), jnp.ones(4)):
            assert best > float(_mean_rewards(_constant(action)).mean()) + 0.05


class TestObservationsMatter:
    def test_feedback_policy_beats_tuned_constant_on_every_shared_seed(self) -> None:
        """Load follows flow, so a policy acting on analyzer readings out-earns the best constant."""
        constant = _mean_rewards(_constant(_BEST_CONSTANT))
        feedback = _mean_rewards(_feedback)

        assert bool(jnp.all(feedback > constant))
        assert float((feedback - constant).mean()) > 0.03


def _one_day_rewards(action: jax.Array) -> jax.Array:
    config = CombinedNPConfig()
    reset, step = make_combined_np_benchmark(config)

    def body(state: CombinedNPState, key: jax.Array):
        state, _obs, reward, _done, _info = step(state, action, key)
        return state, reward

    state, _ = reset(jax.random.PRNGKey(0))
    _, rewards = jax.lax.scan(body, state, jax.random.split(jax.random.PRNGKey(1), config.steps_per_day))
    return rewards


class TestRewardIsBounded:
    def test_one_day_reward_stays_within_bounds_at_action_corners(self) -> None:
        """Over one default-config day, every corner of the action space costs money but stays finite."""
        for corner in itertools.product([0.0, 1.0], repeat=4):
            rewards = _one_day_rewards(jnp.array(corner))

            assert bool(jnp.all(rewards < 0.0))
            assert bool(jnp.all(rewards > -50.0))
