from typing import TYPE_CHECKING

import flax.linen as nn
import gymnax
import gymnax.wrappers
import jax
import jax.numpy as jnp
import optax
from chex import dataclass
from flax.training.train_state import TrainState
from rl_components.buffers import ReplayBuffer


@dataclass(frozen=True)
class DQNConfig:
    LR: float = 3e-4
    BUFFER_SIZE: int = 100_000
    BATCH_SIZE: int = 64
    TOTAL_TIMESTEPS: int = 200_000
    LEARNING_STARTS: int = 1_000
    TRAIN_FREQUENCY: int = 1
    TARGET_NETWORK_FREQUENCY: int = 1_000
    GAMMA: float = 0.99
    TAU: float = 1.0  # Soft update
    EPSILON_START: float = 1.0
    EPSILON_END: float = 0.05
    EPSILON_FRACTION: float = 0.5
    ENV_NAME: str = "MountainCar-v0"
    SEED: int = 42

    if TYPE_CHECKING:
        def __init__(
            self,
            *,
            LR: float = 3e-4,
            BUFFER_SIZE: int = 100_000,
            BATCH_SIZE: int = 64,
            TOTAL_TIMESTEPS: int = 200_000,
            LEARNING_STARTS: int = 1_000,
            TRAIN_FREQUENCY: int = 1,
            TARGET_NETWORK_FREQUENCY: int = 1_000,
            GAMMA: float = 0.99,
            TAU: float = 1.0,
            EPSILON_START: float = 1.0,
            EPSILON_END: float = 0.05,
            EPSILON_FRACTION: float = 0.5,
            ENV_NAME: str = "MountainCar-v0",
            SEED: int = 42,
        ) -> None: ...


class QNetwork(nn.Module):
    action_dim: int

    if TYPE_CHECKING:
        def apply(
            self,
            variables: object,
            x: jax.Array,
            *,
            rngs: object | None = None,
        ) -> jax.Array: ...

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        x = nn.Dense(64)(x)
        x = nn.relu(x)
        x = nn.Dense(64)(x)
        x = nn.relu(x)
        x = nn.Dense(self.action_dim)(x)
        return x


def make_train(config: DQNConfig):
    env, env_params = gymnax.make(config.ENV_NAME)
    env = gymnax.wrappers.LogWrapper(env)

    def train(rng):
        # INIT NETWORK
        network = QNetwork(env.action_space(env_params).n)
        rng, _rng = jax.random.split(rng)
        init_x = jnp.zeros(env.observation_space(env_params).shape)
        params = network.init(_rng, init_x)

        target_params = params

        tx = optax.adam(config.LR)
        train_state = TrainState.create(
            apply_fn=network.apply,
            params=params,
            tx=tx,
        )

        # INIT BUFFER
        buffer = ReplayBuffer(
            config.BUFFER_SIZE,
            env.observation_space(env_params).shape,
            (),  # action_shape for discrete is empty
            jnp.int32,  # action_dtype
        )
        buffer_state = buffer.init()

        # INIT ENV
        rng, _rng = jax.random.split(rng)
        obsv, env_state = env.reset(_rng, env_params)  # type: ignore[not-iterable, too-many-positional-arguments]  # gymnax JitWrapped

        def _update_step(runner_state, t):
            train_state, target_params, buffer_state, env_state, last_obs, rng = runner_state

            # EPSILON GREEDY
            epsilon = jnp.maximum(
                config.EPSILON_END,
                config.EPSILON_START
                - (config.EPSILON_START - config.EPSILON_END)
                * (t / (config.TOTAL_TIMESTEPS * config.EPSILON_FRACTION)),
            )

            rng, _rng_action, _rng_step = jax.random.split(rng, 3)
            q_values = network.apply(train_state.params, last_obs)
            greedy_action = jnp.argmax(q_values)
            random_action = jax.random.randint(_rng_action, (), 0, env.action_space(env_params).n)
            chose_random = jax.random.uniform(_rng_action, ()) < epsilon
            action = jnp.where(chose_random, random_action, greedy_action)

            # STEP ENV
            obsv, env_state, reward, done, info = env.step(_rng_step, env_state, action, env_params)  # type: ignore[not-iterable, too-many-positional-arguments]  # gymnax JitWrapped

            # ADD TO BUFFER
            # ReplayBuffer.add expects vectorized inputs, let's update it or wrap it
            # For "One Agent, One World", we can just expand dims here.
            buffer_state = buffer.add(
                buffer_state,
                last_obs[None, ...],
                action[None, ...],
                reward[None, ...],
                obsv[None, ...],
                done[None, ...]
            )

            # TRAIN
            def _do_train(train_state, target_params, buffer_state, rng):
                rng, _rng = jax.random.split(rng)
                obs, actions, rewards, next_obs, dones = buffer.sample(buffer_state, _rng, config.BATCH_SIZE)

                def _loss_fn(params, target_params, obs, actions, rewards, next_obs, dones):
                    q_values = network.apply(params, obs)
                    q_action = jnp.take_along_axis(q_values, actions[:, None], axis=-1).squeeze()

                    next_q_values = network.apply(target_params, next_obs)
                    next_q_max = jnp.max(next_q_values, axis=-1)
                    target = rewards + config.GAMMA * next_q_max * (1.0 - dones)

                    loss = jnp.mean(jnp.square(q_action - jax.lax.stop_gradient(target)))
                    return loss

                grad_fn = jax.value_and_grad(_loss_fn)
                loss, grads = grad_fn(train_state.params, target_params, obs, actions, rewards, next_obs, dones)
                train_state = train_state.apply_gradients(grads=grads)
                return train_state, loss

            can_train = (t > config.LEARNING_STARTS) & (t % config.TRAIN_FREQUENCY == 0)
            train_state, loss = jax.lax.cond(
                can_train,
                lambda: _do_train(train_state, target_params, buffer_state, rng),
                lambda: (train_state, 0.0),
            )

            # UPDATE TARGET
            should_update_target = t % config.TARGET_NETWORK_FREQUENCY == 0
            target_params = jax.lax.cond(
                should_update_target,
                lambda: jax.tree_util.tree_map(
                    lambda tp, p: config.TAU * p + (1.0 - config.TAU) * tp,
                    target_params,
                    train_state.params,
                ),
                lambda: target_params,
            )

            runner_state = (train_state, target_params, buffer_state, env_state, obsv, rng)
            return runner_state, info

        # RUNNER
        runner_state = (train_state, target_params, buffer_state, env_state, obsv, rng)
        runner_state, metrics = jax.lax.scan(
            _update_step, runner_state, jnp.arange(config.TOTAL_TIMESTEPS)
        )
        return {"runner_state": runner_state, "metrics": metrics}

    return train
