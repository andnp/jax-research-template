"""One compiled Rainbow kernel must train a batch of hyperparameter arms independently."""

from __future__ import annotations

import dataclasses

import jax
import jax.numpy as jnp
from rl_agents.rainbow import RainbowAgent, RainbowAgentState, RainbowConfig, RainbowHypers, rainbow_hypers
from rl_components.env_protocol import EnvSpec
from rl_components.timestep import Timestep

OBSERVATION_SHAPE = (4, 84, 84, 1)
STEPS = 10
"""Long enough that the three-step window fills, replay clears its warmup, and the
learn branch fires several times."""

SPEC = EnvSpec(
    id="toy-atari",
    observation_shape=OBSERVATION_SHAPE,
    action_shape=(),
    observation_dtype=jnp.dtype(jnp.uint8),
    action_dtype=jnp.dtype(jnp.int32),
    num_actions=3,
)


def _config() -> RainbowConfig:
    return RainbowConfig(
        REPLAY_CAPACITY=16,
        MIN_REPLAY_CAPACITY_FRACTION=0.25,
        BATCH_SIZE=4,
        LEARN_PERIOD_FRAMES=4,
        TARGET_NETWORK_UPDATE_PERIOD_FRAMES=8,
    )


def _run_arm(agent: RainbowAgent, hypers: RainbowHypers, key: jax.Array):
    """Drive one arm's ``step`` directly, with the swept hypers pushed into its state.

    The agent object is static under ``jit``, so a sweep cannot vary anything closed over
    it; the traced hypers ride the state pytree instead, which is exactly what ``vmap``
    can map over.
    """
    state = dataclasses.replace(agent.init(key, SPEC), hypers=hypers)

    def _one_step(state: RainbowAgentState, step_index: jax.Array):
        observation = jnp.full(OBSERVATION_SHAPE, step_index + 1, dtype=jnp.uint8)
        timestep = Timestep(
            reward=jnp.float32(1.0),
            discount=jnp.float32(0.9),
            bootstrap_observation=observation,
            episode_end=jnp.bool_(False),
            observation=observation,
        )
        out = agent.step(state, timestep, step_index)
        return out.state, out.metrics["loss"]

    state, _ = jax.lax.scan(_one_step, state, jnp.arange(STEPS, dtype=jnp.int32))
    return state.train_state.params


def _stack_leaves(*leaves: jax.Array):
    return jnp.stack(leaves)


def _sweep(arms: list[RainbowHypers]):
    agent = RainbowAgent(_config())
    key = jax.random.key(0)
    keys = jnp.stack([key] * len(arms))
    batched = jax.tree_util.tree_map(_stack_leaves, *arms)

    def run_hypers_key(hypers: RainbowHypers, k: jax.Array):
        return _run_arm(agent, hypers, k)

    return jax.jit(jax.vmap(run_hypers_key))(batched, keys)


class TestRainbowVmapZone:
    def test_batched_learning_rates_train_independent_arms(self) -> None:
        """A single vmapped kernel must give each learning-rate arm its own weights.

        Both arms share one key, so identical weights would mean the swept rate never
        reached the kernel -- the failure that existed while ``LEARNING_RATE`` was closed
        over the ``RainbowConfig`` baked into the optimizer at construction.
        """
        base = rainbow_hypers(_config())
        arms = [dataclasses.replace(base, LEARNING_RATE=jnp.float32(rate)) for rate in (1e-1, 1e-4)]

        leaves = jax.tree_util.tree_leaves(_sweep(arms))

        assert leaves, "expected at least one parameter leaf"
        for leaf in leaves:
            assert leaf.shape[0] == len(arms)
        assert not jnp.allclose(leaves[0][0], leaves[0][1])

    def test_shared_hypers_train_identical_arms(self) -> None:
        """Two arms given the same hypers and key must land on identical weights.

        Guards the negative direction: divergence has to come from the swept value, not
        from the batching itself.
        """
        base = rainbow_hypers(_config())
        arms = [base, base]

        leaf = jax.tree_util.tree_leaves(_sweep(arms))[0]

        assert jnp.allclose(leaf[0], leaf[1])

    def test_batched_atom_supports_train_independent_arms(self) -> None:
        """Two arms sweeping the C51 support endpoints sharing a key must diverge too.

        ``V_MIN``/``V_MAX`` do not reach the loss directly: ``step`` rebuilds the atom
        support from them on every iteration, and the projection of the target
        distribution onto that support is the only path from the endpoints to a gradient.
        A derived quantity computed once at trace-setup time from the closed-over config
        rather than from the traced hypers is precisely how a sweep silently shares one
        arm's value across every arm, so a batched support is the case worth pinning now
        that the swept discount this test used to sweep belongs to the loop.
        """
        base = rainbow_hypers(_config())
        arms = [
            dataclasses.replace(base, V_MIN=jnp.float32(-1.0), V_MAX=jnp.float32(1.0)),
            dataclasses.replace(base, V_MIN=jnp.float32(-100.0), V_MAX=jnp.float32(100.0)),
        ]

        leaves = jax.tree_util.tree_leaves(_sweep(arms))

        assert leaves, "expected at least one parameter leaf"
        assert not jnp.allclose(leaves[0][0], leaves[0][1])
