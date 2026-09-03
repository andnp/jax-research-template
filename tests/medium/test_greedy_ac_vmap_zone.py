"""One compiled GAC kernel must train a batch of hyperparameter arms independently."""

import dataclasses

import jax
import jax.numpy as jnp
from rl_agents.greedy_ac import GACAgent, GACAgentState, GACConfig, GACHypers, gac_hypers
from rl_components.env_protocol import EnvSpec
from rl_components.timestep import Timestep

SPEC = EnvSpec(id="toy-continuous", observation_shape=(2,), action_shape=(1,), action_dtype=jnp.float32)


def _config() -> GACConfig:
    return GACConfig(
        BATCH_SIZE=2,
        BUFFER_SIZE=64,
        HIDDEN_SIZE=8,
        LEARNING_STARTS=0,
        TRAIN_FREQUENCY=1,
        NUM_SAMPLES=4,
        NUM_RAND_ACTIONS=2,
    )


def _run_arm(agent: GACAgent, base_hypers: GACHypers, learning_rate: jax.Array, key: jax.Array):
    hypers = dataclasses.replace(base_hypers, LR=learning_rate)
    state = dataclasses.replace(agent.init(key, SPEC), hypers=hypers)

    def one_step(state: GACAgentState, step_index: jax.Array):
        timestep = Timestep(
            observation=jnp.ones((2,), jnp.float32) * (step_index + 1),
            reward=jnp.float32(1.0),
            discount=jnp.float32(1.0),
            bootstrap_observation=jnp.ones((2,), jnp.float32),
            episode_end=jnp.bool_(False),
        )
        out = agent.step(state, timestep, step_index)
        return out.state, out.metrics["critic_loss"]

    state, _ = jax.lax.scan(one_step, state, jnp.arange(8))
    return state.critic_state.params


class TestVmapZone:
    def test_batched_learning_rates_train_independent_arms(self) -> None:
        """A single vmapped kernel must give each learning-rate arm its own critic weights.

        Both arms share one key, so identical weights would mean the swept rate
        never reached the kernel -- the failure that existed while hyperparameters
        were closed over the agent object.
        """
        config = _config()
        agent = GACAgent(config)
        base_hypers = gac_hypers(config)
        learning_rates = jnp.array([1e-1, 1e-5], dtype=jnp.float32)
        key = jax.random.key(0)
        keys = jnp.stack([key, key])

        params = jax.jit(jax.vmap(lambda lr, k: _run_arm(agent, base_hypers, lr, k)))(learning_rates, keys)

        leaves = jax.tree_util.tree_leaves(params)
        assert leaves, "expected at least one parameter leaf"
        for leaf in leaves:
            assert leaf.shape[0] == learning_rates.shape[0]

        fast, slow = leaves[0][0], leaves[0][1]
        assert not jnp.allclose(fast, slow)

    def test_shared_hypers_train_identical_arms(self) -> None:
        """Two arms given the same rate and key must land on identical critic weights.

        Guards the negative direction: divergence has to come from the swept
        value, not from the batching itself.
        """
        config = _config()
        agent = GACAgent(config)
        base_hypers = gac_hypers(config)
        learning_rates = jnp.array([1e-3, 1e-3], dtype=jnp.float32)
        key = jax.random.key(0)
        keys = jnp.stack([key, key])

        params = jax.jit(jax.vmap(lambda lr, k: _run_arm(agent, base_hypers, lr, k)))(learning_rates, keys)

        leaf = jax.tree_util.tree_leaves(params)[0]
        assert jnp.allclose(leaf[0], leaf[1])
