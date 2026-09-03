"""One compiled kernel must train a batch of hyperparameter arms independently."""

import jax
import jax.numpy as jnp
from rl_agents.dqn import DQNAgent, DQNConfig
from rl_agents.replay_q_agent import replay_q_hypers
from rl_components.env_protocol import EnvSpec
from rl_components.timestep import Timestep

SPEC = EnvSpec(id="toy-discrete", observation_shape=(2,), action_shape=(), num_actions=2)


def _run_arm(agent: DQNAgent, base_hypers, learning_rate: jax.Array, key: jax.Array):
    state = agent.init(key, SPEC).replace(hypers=base_hypers.replace(LR=learning_rate))

    def one_step(state, step_index):
        timestep = Timestep(
            observation=jnp.ones((2,), jnp.float32) * (step_index + 1),
            reward=jnp.float32(1.0),
            discount=jnp.float32(1.0),
            bootstrap_observation=jnp.ones((2,), jnp.float32),
            episode_end=jnp.bool_(False),
        )
        out = agent.step(state, timestep, step_index)
        return out.state, out.metrics["loss"]

    state, _ = jax.lax.scan(one_step, state, jnp.arange(8))
    return state.train_state.params


class TestVmapZone:
    def test_batched_learning_rates_train_independent_arms(self) -> None:
        """A single vmapped kernel must give each learning-rate arm its own weights.

        Both arms share one key, so identical weights would mean the swept rate
        never reached the kernel -- the failure that existed while hyperparameters
        were closed over the agent object.
        """
        config = DQNConfig(LEARNING_STARTS=0, TRAIN_FREQUENCY=1, BATCH_SIZE=2, BUFFER_SIZE=64)
        agent = DQNAgent(config)
        base_hypers = replay_q_hypers(config)
        learning_rates = jnp.array([1e-1, 1e-4], dtype=jnp.float32)
        key = jax.random.key(0)
        keys = jnp.stack([key, key])

        params = jax.jit(jax.vmap(lambda lr, key: _run_arm(agent, base_hypers, lr, key)))(learning_rates, keys)

        leaves = jax.tree_util.tree_leaves(params)
        assert leaves, "expected at least one parameter leaf"
        for leaf in leaves:
            assert leaf.shape[0] == learning_rates.shape[0]

        fast, slow = leaves[0][0], leaves[0][1]
        assert not jnp.allclose(fast, slow)
        assert jnp.linalg.norm(fast) > jnp.linalg.norm(slow)

    def test_shared_hypers_train_identical_arms(self) -> None:
        """Two arms given the same rate and key must land on identical weights.

        Guards the negative direction: divergence has to come from the swept
        value, not from the batching itself.
        """
        config = DQNConfig(LEARNING_STARTS=0, TRAIN_FREQUENCY=1, BATCH_SIZE=2, BUFFER_SIZE=64)
        agent = DQNAgent(config)
        base_hypers = replay_q_hypers(config)
        learning_rates = jnp.array([1e-3, 1e-3], dtype=jnp.float32)
        key = jax.random.key(0)
        keys = jnp.stack([key, key])

        params = jax.jit(jax.vmap(lambda lr, k: _run_arm(agent, base_hypers, lr, k)))(learning_rates, keys)

        leaf = jax.tree_util.tree_leaves(params)[0]
        assert jnp.allclose(leaf[0], leaf[1])
