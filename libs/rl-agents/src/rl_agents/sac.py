from typing import TYPE_CHECKING, Protocol, cast

import distrax
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
class SACConfig:
    LR: float = 3e-4
    BUFFER_SIZE: int = 100_000
    BATCH_SIZE: int = 256
    TOTAL_TIMESTEPS: int = 1_000_000
    LEARNING_STARTS: int = 5_000
    TRAIN_FREQUENCY: int = 1
    GAMMA: float = 0.99
    TAU: float = 0.005
    ALPHA: float = 0.2
    TARGET_ENTROPY: float | None = None
    ENV_NAME: str = "MountainCarContinuous-v0"
    SEED: int = 42

    if TYPE_CHECKING:
        def __init__(
            self,
            *,
            LR: float = 3e-4,
            BUFFER_SIZE: int = 100_000,
            BATCH_SIZE: int = 256,
            TOTAL_TIMESTEPS: int = 1_000_000,
            LEARNING_STARTS: int = 5_000,
            TRAIN_FREQUENCY: int = 1,
            GAMMA: float = 0.99,
            TAU: float = 0.005,
            ALPHA: float = 0.2,
            TARGET_ENTROPY: float | None = None,
            ENV_NAME: str = "MountainCarContinuous-v0",
            SEED: int = 42,
        ) -> None: ...


class Critic(nn.Module):
    if TYPE_CHECKING:
        def apply(
            self,
            variables: object,
            x: jax.Array,
            a: jax.Array,
            *,
            rngs: object | None = None,
        ) -> jax.Array: ...

    @nn.compact
    def __call__(self, x: jnp.ndarray, a: jnp.ndarray) -> jnp.ndarray:
        x = jnp.concatenate([x, a], axis=-1)
        x = nn.Dense(256)(x)
        x = nn.relu(x)
        x = nn.Dense(256)(x)
        x = nn.relu(x)
        x = nn.Dense(1)(x)
        return jnp.squeeze(x, axis=-1)


class Actor(nn.Module):
    action_dim: int

    if TYPE_CHECKING:
        def apply(
            self,
            variables: object,
            x: jax.Array,
            *,
            rngs: object | None = None,
        ) -> tuple[jax.Array, jax.Array]: ...

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
        x = nn.Dense(256)(x)
        x = nn.relu(x)
        x = nn.Dense(256)(x)
        x = nn.relu(x)
        mean = nn.Dense(self.action_dim)(x)
        log_std = nn.Dense(self.action_dim)(x)
        log_std = jnp.clip(log_std, -20, 2)
        return mean, log_std

    def sample(self, params, x, rng):
        mean, log_std = self.apply(params, x)
        std = jnp.exp(log_std)
        normal = distrax.Normal(mean, std)
        x_t = normal.sample(seed=rng)
        y_t = jnp.tanh(x_t)
        action = y_t

        # Log prob adjustment for Tanh
        log_prob = normal.log_prob(x_t)
        log_prob -= jnp.log(1.0 - y_t**2 + 1e-6)
        log_prob = jnp.sum(log_prob, axis=-1)
        return action, log_prob


class _ObservationSpace(Protocol):
    shape: tuple[int, ...]


class _ActionSpace(Protocol):
    shape: tuple[int, ...]


class _EnvLike(Protocol):
    def observation_space(self, params: object | None = None) -> _ObservationSpace: ...

    def action_space(self, params: object | None = None) -> _ActionSpace: ...

    def reset(self, key: jax.Array, params: object | None = None) -> tuple[jax.Array, object]: ...

    def step(
        self,
        key: jax.Array,
        state: object,
        action: jax.Array,
        params: object | None = None,
    ) -> tuple[jax.Array, object, jax.Array, jax.Array, dict[str, jax.Array]]: ...


def _resolve_env(
    config: SACConfig,
    env: object | None,
    env_params: object | None,
) -> tuple[_EnvLike, object | None]:
    if env is not None:
        return cast(_EnvLike, env), env_params

    resolved_env, resolved_env_params = gymnax.make(config.ENV_NAME)
    return cast(_EnvLike, gymnax.wrappers.LogWrapper(resolved_env)), resolved_env_params


def make_train(config: SACConfig, env: object | None = None, env_params: object | None = None):
    env, env_params = _resolve_env(config, env, env_params)

    def train(rng):
        # INIT NETWORKS
        rng, _rng_actor, _rng_critic = jax.random.split(rng, 3)
        action_dim = env.action_space(env_params).shape[0]
        obs_dim = env.observation_space(env_params).shape

        actor = Actor(action_dim)
        actor_params = actor.init(_rng_actor, jnp.zeros(obs_dim))
        actor_state = TrainState.create(apply_fn=actor.apply, params=actor_params, tx=optax.adam(config.LR))

        critic = Critic()
        rng, _rng_critic = jax.random.split(rng)
        critic_params = jax.vmap(critic.init, in_axes=(0, None, None))(
            jax.random.split(_rng_critic, 2), jnp.zeros(obs_dim), jnp.zeros((action_dim,))
        )
        critic_target_params = critic_params
        critic_state = TrainState.create(apply_fn=critic.apply, params=critic_params, tx=optax.adam(config.LR))

        # Automatic Entropy Tuning
        if config.TARGET_ENTROPY is None:
            target_entropy = -float(action_dim)
        else:
            target_entropy = config.TARGET_ENTROPY

        log_alpha = jnp.zeros(1)
        alpha_state = TrainState.create(apply_fn=None, params=log_alpha, tx=optax.adam(config.LR))

        # INIT BUFFER
        action_shape = env.action_space(env_params).shape
        buffer = ReplayBuffer(config.BUFFER_SIZE, obs_dim, action_shape, jnp.float32)
        buffer_state = buffer.init()

        # INIT ENV
        rng, _rng = jax.random.split(rng)
        obsv, env_state = env.reset(_rng, env_params)

        def _update_step(runner_state, t):
            (
                actor_state,
                critic_state,
                critic_target_params,
                alpha_state,
                buffer_state,
                env_state,
                last_obs,
                rng,
            ) = runner_state

            # SELECT ACTION
            rng, _rng = jax.random.split(rng)

            def _random_action():
                return jax.random.uniform(_rng, (action_dim,), minval=-1, maxval=1)

            def _policy_action():
                action, _ = actor.sample(actor_state.params, last_obs, _rng)
                return action

            action = jax.lax.cond(
                t < config.LEARNING_STARTS,
                _random_action,
                _policy_action,
            )

            # STEP ENV
            rng, _rng = jax.random.split(rng)
            obsv, env_state, reward, done, info = env.step(_rng, env_state, action, env_params)

            # ADD TO BUFFER
            buffer_state = buffer.add(
                buffer_state,
                last_obs[None, ...],
                action[None, ...],
                reward[None, ...],
                obsv[None, ...],
                done[None, ...],
            )

            # TRAIN
            def _do_train(actor_state, critic_state, critic_target_params, alpha_state, buffer_state, rng):
                rng, _rng = jax.random.split(rng)
                obs, actions, rewards, next_obs, dones = buffer.sample(buffer_state, _rng, config.BATCH_SIZE)

                alpha = jnp.exp(alpha_state.params[0])

                # CRITIC UPDATE
                def _critic_loss_fn(critic_params, actor_params, critic_target_params, alpha, obs, actions, rewards, next_obs, dones, rng):
                    rng, _rng = jax.random.split(rng)
                    next_actions, next_log_probs = jax.vmap(actor.sample, in_axes=(None, 0, 0))(
                        actor_params, next_obs, jax.random.split(_rng, config.BATCH_SIZE)
                    )

                    # Twin Q targets
                    next_q_values = jax.vmap(critic.apply, in_axes=(0, None, None))(critic_target_params, next_obs, next_actions)
                    next_q_min = jnp.min(next_q_values, axis=0)
                    target_q = rewards + config.GAMMA * (1.0 - dones) * (next_q_min - alpha * next_log_probs)

                    def _single_critic_loss(params):
                        q = critic.apply(params, obs, actions)
                        return jnp.mean(jnp.square(q - jax.lax.stop_gradient(target_q)))

                    loss = jnp.mean(jax.vmap(_single_critic_loss)(critic_params))
                    return loss

                grad_fn = jax.value_and_grad(_critic_loss_fn)
                critic_loss, critic_grads = grad_fn(
                    critic_state.params, actor_state.params, critic_target_params, alpha, obs, actions, rewards, next_obs, dones, rng
                )
                critic_state = critic_state.apply_gradients(grads=critic_grads)

                # ACTOR UPDATE
                def _actor_loss_fn(actor_params, critic_params, alpha, obs, rng):
                    rng, _rng = jax.random.split(rng)
                    new_actions, log_probs = jax.vmap(actor.sample, in_axes=(None, 0, 0))(
                        actor_params, obs, jax.random.split(_rng, config.BATCH_SIZE)
                    )

                    q_values = jax.vmap(critic.apply, in_axes=(0, None, None))(critic_params, obs, new_actions)
                    q_min = jnp.min(q_values, axis=0)

                    loss = jnp.mean(alpha * log_probs - q_min)
                    return loss

                grad_fn = jax.value_and_grad(_actor_loss_fn)
                actor_loss, actor_grads = grad_fn(actor_state.params, critic_state.params, alpha, obs, rng)
                actor_state = actor_state.apply_gradients(grads=actor_grads)

                # ALPHA UPDATE
                def _alpha_loss_fn(log_alpha, actor_params, obs, rng):
                    rng, _rng = jax.random.split(rng)
                    _, log_probs = jax.vmap(actor.sample, in_axes=(None, 0, 0))(actor_params, obs, jax.random.split(_rng, config.BATCH_SIZE))
                    loss = -jnp.mean(log_alpha * (log_probs + target_entropy))
                    return loss

                grad_fn = jax.value_and_grad(_alpha_loss_fn)
                alpha_loss, alpha_grads = grad_fn(alpha_state.params, actor_state.params, obs, rng)
                alpha_state = alpha_state.apply_gradients(grads=alpha_grads)

                # TARGET UPDATE
                critic_target_params = jax.tree_util.tree_map(
                    lambda tp, p: config.TAU * p + (1.0 - config.TAU) * tp,
                    critic_target_params,
                    critic_state.params,
                )

                return actor_state, critic_state, critic_target_params, alpha_state

            can_train = (t > config.LEARNING_STARTS) & (t % config.TRAIN_FREQUENCY == 0)
            actor_state, critic_state, critic_target_params, alpha_state = jax.lax.cond(
                can_train,
                lambda: _do_train(actor_state, critic_state, critic_target_params, alpha_state, buffer_state, rng),
                lambda: (actor_state, critic_state, critic_target_params, alpha_state),
            )

            runner_state = (actor_state, critic_state, critic_target_params, alpha_state, buffer_state, env_state, obsv, rng)
            return runner_state, info

        # RUNNER
        runner_state = (actor_state, critic_state, critic_target_params, alpha_state, buffer_state, env_state, obsv, rng)
        runner_state, metrics = jax.lax.scan(_update_step, runner_state, jnp.arange(config.TOTAL_TIMESTEPS))
        return {"runner_state": runner_state, "metrics": metrics}

    return train
