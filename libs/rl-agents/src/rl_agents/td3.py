from functools import partial
from typing import Callable, NamedTuple, TypedDict

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

    def apply(
        self,
        variables: object,
        x: jax.Array,
        a: jax.Array,
        *,
        rngs: object | None = None,
    ) -> jax.Array:
        return super().apply(variables, x, a, rngs=rngs)


class Actor(TypedApply[jax.Array], nn.Module):
    action_dim: int

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        x = nn.Dense(256)(x)
        x = nn.relu(x)
        x = nn.Dense(256)(x)
        x = nn.relu(x)
        x = nn.Dense(self.action_dim)(x)
        return jnp.tanh(x)


def _soft_update(tau: float, target: VariableDict, online: VariableDict) -> VariableDict:
    return jax.tree_util.tree_map(lambda tp, p: tau * p + (1.0 - tau) * tp, target, online)


def _critic_loss(
    critic_params: VariableDict,
    actor_target_params: VariableDict,
    critic_target_params: VariableDict,
    obs: jax.Array,
    actions: jax.Array,
    rewards: jax.Array,
    next_obs: jax.Array,
    discounts: jax.Array,
    rng: jax.Array,
    *,
    actor: Actor,
    critic: Critic,
    config: "TD3Config",
) -> jax.Array:
    rng, _rng_noise = jax.random.split(rng)
    next_actions = actor.apply(actor_target_params, next_obs)
    noise = jax.random.normal(_rng_noise, next_actions.shape) * config.POLICY_NOISE
    noise = jnp.clip(noise, -config.NOISE_CLIP, config.NOISE_CLIP)
    next_actions = jnp.clip(next_actions + noise, -1.0, 1.0)

    next_q = jax.vmap(
        critic.apply,
        in_axes=(0, None, None),
    )(critic_target_params, next_obs, next_actions)
    target_q = rewards + discounts * jnp.min(next_q, axis=0)

    def _single(p: VariableDict) -> jax.Array:
        q = critic.apply(p, obs, actions)
        return jnp.mean(jnp.square(q - jax.lax.stop_gradient(target_q)))

    return jnp.mean(jax.vmap(_single)(critic_params))


def _actor_loss(
    actor_params: VariableDict,
    critic_params: VariableDict,
    obs: jax.Array,
    *,
    actor: Actor,
    critic: Critic,
) -> jax.Array:
    new_actions = actor.apply(actor_params, obs)
    q_values = jax.vmap(
        lambda p, o, a: critic.apply(p, o, a),
        in_axes=(0, None, None),
    )(critic_params, obs, new_actions)
    return -jnp.mean(q_values[0])


class RunnerState(NamedTuple):
    actor_state: TrainState
    critic_state: TrainState
    critic_target_params: VariableDict
    actor_target_params: VariableDict
    buffer_state: ReplayBufferState
    env_state: object
    last_obs: jax.Array
    rng: jax.Array


class TD3TrainOutput(TypedDict):
    runner_state: RunnerState
    metrics: dict[str, jax.Array]


def make_train(config: TD3Config, env: GymEnv[ContinuousActionSpace], env_params: object | None = None) -> Callable[[jax.Array], TD3TrainOutput]:
    action_dim = env.action_space(env_params).shape[0]
    obs_dim = env.observation_space(env_params).shape
    action_shape = env.action_space(env_params).shape

    actor = Actor(action_dim)
    critic = Critic()
    buffer = ReplayBuffer(config.BUFFER_SIZE, obs_dim, action_shape, jnp.float32)

    _bound_critic_loss = partial(_critic_loss, actor=actor, critic=critic, config=config)
    _bound_actor_loss = partial(_actor_loss, actor=actor, critic=critic)

    def train(rng: jax.Array) -> TD3TrainOutput:
        # INIT NETWORKS
        rng, _rng_actor, _rng_critic = jax.random.split(rng, 3)
        actor_params = actor.init(_rng_actor, jnp.zeros(obs_dim))
        actor_state = TrainState.create(apply_fn=actor.apply, params=actor_params, tx=optax.adam(config.LR))

        rng, _rng_critic = jax.random.split(rng)
        critic_params = jax.vmap(critic.init, in_axes=(0, None, None))(jax.random.split(_rng_critic, 2), jnp.zeros(obs_dim), jnp.zeros((action_dim,)))
        critic_state = TrainState.create(apply_fn=critic.apply, params=critic_params, tx=optax.adam(config.LR))
        critic_target_params = critic_params
        actor_target_params = actor_params

        # INIT BUFFER & ENV
        buffer_state = buffer.init()
        rng, _rng = jax.random.split(rng)
        obsv, env_state = env.reset(_rng, env_params)

        def _do_train(
            actor_state: TrainState,
            critic_state: TrainState,
            critic_target_params: VariableDict,
            actor_target_params: VariableDict,
            buffer_state: ReplayBufferState,
            rng: jax.Array,
            t: jax.Array,
        ) -> tuple[TrainState, TrainState, VariableDict, VariableDict, jax.Array, jax.Array]:
            rng, _rng, _rng_cl = jax.random.split(rng, 3)
            obs, actions, rewards, next_obs, discounts = buffer.sample(buffer_state, _rng, config.BATCH_SIZE)

            critic_loss, critic_grads = jax.value_and_grad(_bound_critic_loss)(
                critic_state.params,
                actor_target_params,
                critic_target_params,
                obs,
                actions,
                rewards,
                next_obs,
                discounts,
                _rng_cl,
            )
            critic_state = critic_state.apply_gradients(grads=critic_grads)

            def _update_actor(
                actor_state: TrainState,
                critic_state: TrainState,
                critic_target_params: VariableDict,
                actor_target_params: VariableDict,
                obs: jax.Array,
            ) -> tuple[TrainState, VariableDict, VariableDict, jax.Array]:
                actor_loss, actor_grads = jax.value_and_grad(_bound_actor_loss)(actor_state.params, critic_state.params, obs)
                actor_state = actor_state.apply_gradients(grads=actor_grads)
                return (
                    actor_state,
                    _soft_update(config.TAU, critic_target_params, critic_state.params),
                    _soft_update(config.TAU, actor_target_params, actor_state.params),
                    actor_loss,
                )

            def _skip_actor(
                actor_state: TrainState,
                critic_state: TrainState,
                critic_target_params: VariableDict,
                actor_target_params: VariableDict,
                obs: jax.Array,
            ) -> tuple[TrainState, VariableDict, VariableDict, jax.Array]:
                return actor_state, critic_target_params, actor_target_params, jnp.array(0.0)

            actor_state, critic_target_params, actor_target_params, actor_loss = jax.lax.cond(
                t % config.POLICY_DELAY == 0,
                _update_actor,
                _skip_actor,
                actor_state,
                critic_state,
                critic_target_params,
                actor_target_params,
                obs,
            )
            return actor_state, critic_state, critic_target_params, actor_target_params, critic_loss, actor_loss

        def _skip_train(
            actor_state: TrainState,
            critic_state: TrainState,
            critic_target_params: VariableDict,
            actor_target_params: VariableDict,
            buffer_state: ReplayBufferState,
            rng: jax.Array,
            t: jax.Array,
        ) -> tuple[TrainState, TrainState, VariableDict, VariableDict, jax.Array, jax.Array]:
            return actor_state, critic_state, critic_target_params, actor_target_params, jnp.array(0.0), jnp.array(0.0)

        def _update_step(runner_state: RunnerState, t: jax.Array) -> tuple[RunnerState, dict[str, jax.Array]]:
            actor_state, critic_state, critic_target_params, actor_target_params, buffer_state, env_state, last_obs, rng = runner_state

            # SELECT ACTION
            rng, _rng_action, _rng_noise = jax.random.split(rng, 3)

            def _random_action() -> jax.Array:
                return jax.random.uniform(_rng_action, (action_dim,), minval=-1, maxval=1)

            def _policy_action() -> jax.Array:
                action = actor.apply(actor_state.params, last_obs)
                noise = jax.random.normal(_rng_noise, action.shape) * config.EXPLORATION_NOISE
                return jnp.clip(action + noise, -1.0, 1.0)

            action = jax.lax.cond(t < config.LEARNING_STARTS, _random_action, _policy_action)

            # STEP ENV
            rng, _rng = jax.random.split(rng)
            obsv, env_state, reward, done, info = env.step(_rng, env_state, action, env_params)
            discount = config.GAMMA * (1.0 - done)

            # ADD TO BUFFER
            buffer_state = buffer.add(
                buffer_state,
                last_obs[None, ...],
                action[None, ...],
                reward[None, ...],
                obsv[None, ...],
                discount[None, ...],
            )

            # TRAIN
            can_train = (t > config.LEARNING_STARTS) & (t % config.TRAIN_FREQUENCY == 0)
            actor_state, critic_state, critic_target_params, actor_target_params, critic_loss, actor_loss = jax.lax.cond(
                can_train,
                _do_train,
                _skip_train,
                actor_state,
                critic_state,
                critic_target_params,
                actor_target_params,
                buffer_state,
                rng,
                t,
            )

            runner_state = RunnerState(
                actor_state=actor_state,
                critic_state=critic_state,
                critic_target_params=critic_target_params,
                actor_target_params=actor_target_params,
                buffer_state=buffer_state,
                env_state=env_state,
                last_obs=obsv,
                rng=rng,
            )
            info = dict(info)
            info["critic_loss"] = critic_loss
            info["actor_loss"] = actor_loss
            return runner_state, info

        runner_state = RunnerState(
            actor_state=actor_state,
            critic_state=critic_state,
            critic_target_params=critic_target_params,
            actor_target_params=actor_target_params,
            buffer_state=buffer_state,
            env_state=env_state,
            last_obs=obsv,
            rng=rng,
        )
        runner_state, metrics = jax.lax.scan(_update_step, runner_state, jnp.arange(config.TOTAL_TIMESTEPS))
        return {"runner_state": runner_state, "metrics": metrics}

    return train
