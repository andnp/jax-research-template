from typing import Any, Callable, NamedTuple, cast

import flax.linen as nn
import jax
import jax.numpy as jnp
import optax
from flax.training.train_state import TrainState
from flax.typing import VariableDict
from jax_nn.typed_module import TypedApply
from rl_components.buffers import ReplayBuffer, ReplayBufferState
from rl_components.gym_env import ContinuousActionSpace, GymEnv
from rl_components.structs import chex_struct


@chex_struct(frozen=True, kw_only=True)
class TD3Config:
    LR: float = 3e-4
    BUFFER_SIZE: int = 100_000
    BATCH_SIZE: int = 256
    TOTAL_TIMESTEPS: int = 1_000_000
    LEARNING_STARTS: int = 25_000
    TRAIN_FREQUENCY: int = 1
    GAMMA: float = 0.99
    TAU: float = 0.005
    POLICY_DELAY: int = 2
    EXPLORATION_NOISE: float = 0.1
    POLICY_NOISE: float = 0.2
    NOISE_CLIP: float = 0.5
    ENV_NAME: str = "MountainCarContinuous-v0"
    SEED: int = 42


class Critic(nn.Module):
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

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        x = nn.Dense(256)(x)
        x = nn.relu(x)
        x = nn.Dense(256)(x)
        x = nn.relu(x)
        x = nn.Dense(self.action_dim)(x)
        return jnp.tanh(x)


def _critic_apply(module: Critic, variables: VariableDict, x: jax.Array, a: jax.Array) -> jax.Array:
    return cast(jax.Array, module.apply(variables, x, a))


def _actor_apply(module: Actor, variables: VariableDict, x: jax.Array) -> jax.Array:
    return cast(jax.Array, module.apply(variables, x))


class RunnerState(NamedTuple):
    actor_state: TrainState
    critic_state: TrainState
    critic_target_params: VariableDict
    actor_target_params: VariableDict
    buffer_state: ReplayBufferState
    env_state: object
    last_obs: jax.Array
    rng: jax.Array


def make_train(config: TD3Config, env: GymEnv[ContinuousActionSpace], env_params: object | None = None) -> Callable[[jax.Array], dict[str, Any]]:
    def train(rng: jax.Array) -> dict[str, Any]:
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

        actor_target_params = actor_params

        # INIT BUFFER
        action_shape = env.action_space(env_params).shape
        buffer = ReplayBuffer(config.BUFFER_SIZE, obs_dim, action_shape, jnp.float32)
        buffer_state = buffer.init()

        # INIT ENV
        rng, _rng = jax.random.split(rng)
        obsv, env_state = env.reset(_rng, env_params)

        def _update_step(runner_state: RunnerState, t: jax.Array) -> tuple[RunnerState, dict[str, jax.Array]]:
            (
                actor_state,
                critic_state,
                critic_target_params,
                actor_target_params,
                buffer_state,
                env_state,
                last_obs,
                rng,
            ) = runner_state

            # SELECT ACTION
            rng, _rng_action, _rng_noise = jax.random.split(rng, 3)

            def _random_action() -> jax.Array:
                return jax.random.uniform(_rng_action, (action_dim,), minval=-1, maxval=1)

            def _policy_action() -> jax.Array:
                action = _actor_apply(actor, actor_state.params, last_obs)
                noise = jax.random.normal(_rng_noise, action.shape) * config.EXPLORATION_NOISE
                return jnp.clip(action + noise, -1.0, 1.0)

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
            def _do_train(actor_state: TrainState, critic_state: TrainState, critic_target_params: VariableDict, actor_target_params: VariableDict, buffer_state: ReplayBufferState, rng: jax.Array) -> tuple[TrainState, TrainState, VariableDict, VariableDict]:
                rng, _rng = jax.random.split(rng)
                obs, actions, rewards, next_obs, dones = buffer.sample(buffer_state, _rng, config.BATCH_SIZE)

                # CRITIC UPDATE
                def _critic_loss_fn(critic_params: VariableDict, actor_target_params: VariableDict, critic_target_params: VariableDict, obs: jax.Array, actions: jax.Array, rewards: jax.Array, next_obs: jax.Array, dones: jax.Array, rng: jax.Array) -> jax.Array:
                    rng, _rng_noise = jax.random.split(rng)

                    # Target policy smoothing
                    next_actions = _actor_apply(actor, actor_target_params, next_obs)
                    noise = jax.random.normal(_rng_noise, next_actions.shape) * config.POLICY_NOISE
                    noise = jnp.clip(noise, -config.NOISE_CLIP, config.NOISE_CLIP)
                    next_actions = jnp.clip(next_actions + noise, -1.0, 1.0)

                    # Twin Q targets
                    next_q_values = jax.vmap(
                        lambda params, o, a: _critic_apply(critic, params, o, a),
                        in_axes=(0, None, None),
                    )(critic_target_params, next_obs, next_actions)
                    next_q_min = jnp.min(next_q_values, axis=0)
                    target_q = rewards + config.GAMMA * (1.0 - dones) * next_q_min

                    def _single_critic_loss(params: VariableDict) -> jax.Array:
                        q = _critic_apply(critic, params, obs, actions)
                        return jnp.mean(jnp.square(q - jax.lax.stop_gradient(target_q)))

                    loss = jnp.mean(jax.vmap(_single_critic_loss)(critic_params))
                    return loss

                grad_fn = jax.value_and_grad(_critic_loss_fn)
                critic_loss, critic_grads = grad_fn(
                    critic_state.params, actor_target_params, critic_target_params, obs, actions, rewards, next_obs, dones, rng
                )
                critic_state = critic_state.apply_gradients(grads=critic_grads)

                # DELAYED ACTOR UPDATE
                def _update_actor(actor_state: TrainState, critic_state: TrainState, critic_target_params: VariableDict, actor_target_params: VariableDict) -> tuple[TrainState, VariableDict, VariableDict]:
                    def _actor_loss_fn(actor_params: VariableDict, critic_params: VariableDict, obs: jax.Array) -> jax.Array:
                        new_actions = _actor_apply(actor, actor_params, obs)
                        q_values = jax.vmap(
                            lambda params, o, a: _critic_apply(critic, params, o, a),
                            in_axes=(0, None, None),
                        )(critic_params, obs, new_actions)
                        # Use Q1 only
                        q1 = q_values[0]
                        return -jnp.mean(q1)

                    grad_fn = jax.value_and_grad(_actor_loss_fn)
                    actor_loss, actor_grads = grad_fn(actor_state.params, critic_state.params, obs)
                    actor_state = actor_state.apply_gradients(grads=actor_grads)

                    # TARGET UPDATES (both critic and actor targets)
                    critic_target_params = jax.tree_util.tree_map(
                        lambda tp, p: config.TAU * p + (1.0 - config.TAU) * tp,
                        critic_target_params,
                        critic_state.params,
                    )
                    actor_target_params = jax.tree_util.tree_map(
                        lambda tp, p: config.TAU * p + (1.0 - config.TAU) * tp,
                        actor_target_params,
                        actor_state.params,
                    )
                    return actor_state, critic_target_params, actor_target_params

                def _skip_actor(actor_state: TrainState, critic_state: TrainState, critic_target_params: VariableDict, actor_target_params: VariableDict) -> tuple[TrainState, VariableDict, VariableDict]:
                    return actor_state, critic_target_params, actor_target_params

                should_update_actor = (t % config.POLICY_DELAY == 0)
                actor_state, critic_target_params, actor_target_params = jax.lax.cond(
                    should_update_actor,
                    _update_actor,
                    _skip_actor,
                    actor_state,
                    critic_state,
                    critic_target_params,
                    actor_target_params,
                )

                return actor_state, critic_state, critic_target_params, actor_target_params

            can_train = (t > config.LEARNING_STARTS) & (t % config.TRAIN_FREQUENCY == 0)
            actor_state, critic_state, critic_target_params, actor_target_params = jax.lax.cond(
                can_train,
                lambda: _do_train(actor_state, critic_state, critic_target_params, actor_target_params, buffer_state, rng),
                lambda: (actor_state, critic_state, critic_target_params, actor_target_params),
            )

            runner_state = RunnerState(actor_state=actor_state, critic_state=critic_state, critic_target_params=critic_target_params, actor_target_params=actor_target_params, buffer_state=buffer_state, env_state=env_state, last_obs=obsv, rng=rng)
            return runner_state, info

        # RUNNER
        runner_state = RunnerState(actor_state=actor_state, critic_state=critic_state, critic_target_params=critic_target_params, actor_target_params=actor_target_params, buffer_state=buffer_state, env_state=env_state, last_obs=obsv, rng=rng)
        runner_state, metrics = jax.lax.scan(_update_step, runner_state, jnp.arange(config.TOTAL_TIMESTEPS))
        return {"runner_state": runner_state, "metrics": metrics}

    return train
