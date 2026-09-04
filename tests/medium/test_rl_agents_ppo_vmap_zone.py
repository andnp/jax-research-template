"""One compiled kernel must train a batch of PPO hyperparameter arms independently.

`TestAgentVmapZone` drives `PPOAgent`, where the property has a second job: the swept values
ride the agent STATE rather than the agent object, because the object is static under `jit`
and cannot carry a traced value at all.
"""

import dataclasses

import jax
import jax.numpy as jnp
from rl_agents.ppo import PPOAgent, PPOAgentState, PPOHypers, ppo_hypers
from rl_components.env_protocol import EnvSpec
from rl_components.timestep import Timestep
from rl_components.types import PPOConfig

_AGENT_SPEC = EnvSpec(
    id="toy:ppo-vmap",
    observation_shape=(2,),
    action_shape=(),
    observation_dtype=jnp.dtype(jnp.float32),
    action_dtype=jnp.dtype(jnp.int32),
    num_actions=2,
)
_AGENT_NUM_STEPS = 4


def _run_agent_arm(agent: PPOAgent, base_hypers: PPOHypers, learning_rate: jax.Array, key: jax.Array):
    hypers = dataclasses.replace(base_hypers, LEARNING_RATE=learning_rate)
    state = dataclasses.replace(agent.init(key, _AGENT_SPEC), hypers=hypers)

    def one_step(state: PPOAgentState, step_index: jax.Array):
        observation = jnp.ones((2,), jnp.float32) * (step_index + 1).astype(jnp.float32)
        timestep = Timestep(
            reward=jnp.float32(1.0),
            discount=jnp.float32(0.99),
            bootstrap_observation=observation,
            episode_end=jnp.bool_(False),
            observation=observation,
        )
        out = agent.step(state, timestep, step_index)
        return out.state, out.metrics["total_loss"]

    state, _ = jax.lax.scan(one_step, state, jnp.arange(2 * _AGENT_NUM_STEPS, dtype=jnp.int32))
    return state.train_state.params


class TestAgentVmapZone:
    def test_batched_learning_rates_train_independent_arms(self) -> None:
        """A single vmapped kernel must give each learning-rate arm its own weights.

        Both arms share one key, so identical weights would mean the swept rate never
        reached the kernel -- which is what happens the moment a hyperparameter is read off
        the static agent object instead of the traced state.
        """
        config = PPOConfig(NUM_STEPS=_AGENT_NUM_STEPS, NUM_MINIBATCHES=1, UPDATE_EPOCHS=1)
        agent = PPOAgent(config)
        base_hypers = ppo_hypers(config)
        learning_rates = jnp.array([1e-1, 1e-4], dtype=jnp.float32)
        key = jax.random.key(0)
        keys = jnp.stack([key, key])

        params = jax.jit(jax.vmap(lambda lr, k: _run_agent_arm(agent, base_hypers, lr, k)))(learning_rates, keys)

        leaves = jax.tree_util.tree_leaves(params)
        assert leaves, "expected at least one parameter leaf"
        for leaf in leaves:
            assert leaf.shape[0] == learning_rates.shape[0]
        assert not jnp.allclose(leaves[0][0], leaves[0][1])

    def test_shared_hypers_train_identical_arms(self) -> None:
        """Two arms given the same rate and key must land on identical weights.

        Guards the negative direction: divergence has to come from the swept value, not
        from the batching itself.
        """
        config = PPOConfig(NUM_STEPS=_AGENT_NUM_STEPS, NUM_MINIBATCHES=1, UPDATE_EPOCHS=1)
        agent = PPOAgent(config)
        base_hypers = ppo_hypers(config)
        learning_rates = jnp.array([1e-3, 1e-3], dtype=jnp.float32)
        key = jax.random.key(0)
        keys = jnp.stack([key, key])

        params = jax.jit(jax.vmap(lambda lr, k: _run_agent_arm(agent, base_hypers, lr, k)))(learning_rates, keys)

        leaf = jax.tree_util.tree_leaves(params)[0]
        assert jnp.allclose(leaf[0], leaf[1])
