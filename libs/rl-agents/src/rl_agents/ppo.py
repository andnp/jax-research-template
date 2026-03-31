import math
from typing import NamedTuple, Protocol, cast

import jax
import jax.numpy as jnp
import optax
from flax.training.train_state import TrainState
from rl_components.networks import ActorCritic, ContinuousActorCritic
from rl_components.types import PPOConfig


class Transition(NamedTuple):
    done: jax.Array
    action: jax.Array
    value: jax.Array
    reward: jax.Array
    log_prob: jax.Array
    obs: jax.Array
    info: dict[str, jax.Array]


class _ObservationNormState(NamedTuple):
    observation_count: jax.Array
    mean: jax.Array
    m2: jax.Array


class _ObservationSpace(Protocol):
    shape: tuple[int, ...]


class _ActionSpace(Protocol):
    n: int


class _ContinuousActionSpace(Protocol):
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


def _is_discrete_action_space(action_space: object) -> bool:
    return hasattr(action_space, "n")


def _continuous_action_dim(action_space: _ContinuousActionSpace) -> int:
    if len(action_space.shape) != 1:
        raise ValueError(f"continuous PPO expects a flat action shape, got {action_space.shape}")
    return action_space.shape[0]


def _sum_action_event_terms(values: object, *, is_continuous: bool) -> jax.Array:
    values_array = jnp.asarray(values)
    if not is_continuous:
        return values_array
    return jnp.sum(values_array, axis=-1)


def _empty_observation_norm_state(obs: jax.Array) -> _ObservationNormState:
    obs_array = jnp.asarray(obs, dtype=jnp.float32)
    zeros = jnp.zeros_like(obs_array)
    return _ObservationNormState(observation_count=jnp.array(0.0, dtype=jnp.float32), mean=zeros, m2=zeros)


def _update_observation_norm_state(state: _ObservationNormState, obs: jax.Array) -> _ObservationNormState:
    obs_array = jnp.asarray(obs, dtype=jnp.float32)
    observation_count = state.observation_count + jnp.array(1.0, dtype=jnp.float32)
    delta = obs_array - state.mean
    mean = state.mean + delta / observation_count
    delta2 = obs_array - mean
    m2 = state.m2 + delta * delta2
    return _ObservationNormState(observation_count=observation_count, mean=mean, m2=m2)


def _init_observation_norm_state(obs: jax.Array) -> _ObservationNormState:
    return _update_observation_norm_state(_empty_observation_norm_state(obs), obs)


def _normalize_observation(state: _ObservationNormState, obs: jax.Array, *, eps: float, clip: float) -> jax.Array:
    obs_array = jnp.asarray(obs, dtype=jnp.float32)
    variance = jnp.where(
        state.observation_count > 0.0,
        state.m2 / state.observation_count,
        jnp.ones_like(state.mean),
    )
    normalized_obs = (obs_array - state.mean) / jnp.sqrt(variance + jnp.asarray(eps, dtype=obs_array.dtype))
    clip_value = jnp.asarray(clip, dtype=obs_array.dtype)
    return jnp.clip(normalized_obs, -clip_value, clip_value)


def _maybe_update_observation_norm_state(state: _ObservationNormState, obs: jax.Array, *, enabled: bool) -> _ObservationNormState:
    if not enabled:
        return state
    return _update_observation_norm_state(state, obs)


def _maybe_normalize_observation(
    obs: jax.Array,
    state: _ObservationNormState,
    *,
    eps: float,
    clip: float,
    enabled: bool,
) -> jax.Array:
    if not enabled:
        return obs
    return _normalize_observation(state, obs, eps=eps, clip=clip)


def make_train(config: PPOConfig, env: object, env_params: object | None = None):
    env = cast(_EnvLike, env)
    if not math.isfinite(config.REWARD_SCALE) or config.REWARD_SCALE <= 0.0:
        raise ValueError(f"REWARD_SCALE must be finite and > 0, got {config.REWARD_SCALE!r}")

    normalize_observations = config.NORMALIZE_OBSERVATIONS
    reward_scale = jnp.asarray(config.REWARD_SCALE, dtype=jnp.float32)

    def train(rng):
        # INIT NETWORK
        action_space = env.action_space(env_params)
        continuous_actions = not _is_discrete_action_space(action_space)
        if continuous_actions:
            continuous_action_space = cast(_ContinuousActionSpace, action_space)
            network = ContinuousActorCritic(_continuous_action_dim(continuous_action_space), activation="tanh")
        else:
            discrete_action_space = cast(_ActionSpace, action_space)
            network = ActorCritic(discrete_action_space.n, activation="tanh")
        rng, _rng = jax.random.split(rng)
        init_x = jnp.zeros(env.observation_space(env_params).shape)
        network_params = network.init(_rng, init_x)

        tx = optax.chain(
            optax.clip_by_global_norm(config.MAX_GRAD_NORM),
            optax.adam(config.LR, eps=1e-5),
        )
        train_state = TrainState.create(
            apply_fn=network.apply,
            params=network_params,
            tx=tx,
        )

        # INIT ENV
        rng, _rng = jax.random.split(rng)
        obsv, env_state = env.reset(_rng, env_params)
        obs_norm_state = _init_observation_norm_state(obsv)

        # TRAIN LOOP
        def _update_step(runner_state, unused):
            train_state, env_state, last_obs, obs_norm_state, rng = runner_state

            # COLLECT TRAJECTORIES
            def _env_step(runner_state, unused):
                train_state, env_state, last_obs, obs_norm_state, rng = runner_state
                normalized_obs = _maybe_normalize_observation(
                    last_obs,
                    obs_norm_state,
                    eps=config.OBS_NORM_EPS,
                    clip=config.OBS_NORM_CLIP,
                    enabled=normalize_observations,
                )

                # SELECT ACTION
                rng, _rng = jax.random.split(rng)
                policy, value = network.apply(train_state.params, normalized_obs)
                action = policy.sample(seed=_rng)
                log_prob = _sum_action_event_terms(policy.log_prob(action), is_continuous=continuous_actions)

                # STEP ENV
                rng, _rng = jax.random.split(rng)
                obsv, env_state, reward, done, info = env.step(_rng, env_state, action, env_params)
                scaled_reward = reward * reward_scale
                obs_norm_state = _maybe_update_observation_norm_state(obs_norm_state, obsv, enabled=normalize_observations)
                transition = Transition(done, action, value, scaled_reward, log_prob, normalized_obs, info)
                runner_state = (train_state, env_state, obsv, obs_norm_state, rng)
                return runner_state, transition

            runner_state, traj_batch = jax.lax.scan(_env_step, runner_state, None, config.NUM_STEPS)

            # CALCULATE ADVANTAGE
            train_state, env_state, last_obs, obs_norm_state, rng = runner_state
            normalized_last_obs = _maybe_normalize_observation(
                last_obs,
                obs_norm_state,
                eps=config.OBS_NORM_EPS,
                clip=config.OBS_NORM_CLIP,
                enabled=normalize_observations,
            )
            _, last_val = network.apply(train_state.params, normalized_last_obs)

            def _calculate_gae(traj_batch, last_val):
                def _get_advantages(gae_and_next_value, transition):
                    gae, next_value = gae_and_next_value
                    done, value, reward = (
                        transition.done,
                        transition.value,
                        transition.reward,
                    )
                    not_done = 1.0 - done
                    delta = reward + config.GAMMA * next_value * not_done - value
                    gae = delta + config.GAMMA * config.GAE_LAMBDA * not_done * gae
                    return (gae, value), gae

                _, advantages = jax.lax.scan(
                    _get_advantages,
                    (jnp.zeros_like(last_val), last_val),
                    traj_batch,
                    reverse=True,
                )
                return advantages, advantages + traj_batch.value

            advantages, targets = _calculate_gae(traj_batch, last_val)

            # UPDATE NETWORK
            def _update_epoch(update_state, unused):
                def _update_minbatch(train_state, batch_info):
                    traj_batch, advantages, targets = batch_info

                    def _loss_fn(params, traj_batch, advantages, targets):
                        # RERUN NETWORK
                        policy, value = network.apply(params, traj_batch.obs)
                        log_prob = _sum_action_event_terms(policy.log_prob(traj_batch.action), is_continuous=continuous_actions)

                        # CALCULATE VALUE LOSS
                        value_pred_clipped = traj_batch.value + (value - traj_batch.value).clip(-config.CLIP_EPS, config.CLIP_EPS)
                        value_losses = jnp.square(value - targets)
                        value_losses_clipped = jnp.square(value_pred_clipped - targets)
                        value_loss = 0.5 * jnp.maximum(value_losses, value_losses_clipped).mean()

                        # CALCULATE POLICY LOSS
                        ratio = jnp.exp(log_prob - traj_batch.log_prob)
                        gae = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
                        loss_pc1 = ratio * gae
                        loss_pc2 = (
                            jnp.clip(
                                ratio,
                                1.0 - config.CLIP_EPS,
                                1.0 + config.CLIP_EPS,
                            )
                            * gae
                        )
                        policy_loss = -jnp.minimum(loss_pc1, loss_pc2).mean()

                        # CALCULATE ENTROPY LOSS
                        entropy_loss = _sum_action_event_terms(policy.entropy(), is_continuous=continuous_actions).mean()

                        loss = policy_loss + config.VF_COEF * value_loss - config.ENT_COEF * entropy_loss
                        return loss, (value_loss, policy_loss, entropy_loss)

                    grad_fn = jax.value_and_grad(_loss_fn, has_aux=True)
                    loss, grads = grad_fn(train_state.params, traj_batch, advantages, targets)

                    train_state = train_state.apply_gradients(grads=grads)
                    return train_state, loss

                train_state, traj_batch, advantages, targets, rng = update_state
                rng, _rng = jax.random.split(rng)
                batch_size = config.NUM_STEPS
                permutation = jax.random.permutation(_rng, batch_size)
                batch = (traj_batch, advantages, targets)
                shuffled_batch = jax.tree_util.tree_map(
                    lambda x: jnp.take(x, permutation, axis=0).reshape([config.NUM_MINIBATCHES, -1] + list(x.shape[1:])),
                    batch,
                )
                train_state, losses = jax.lax.scan(_update_minbatch, train_state, shuffled_batch)
                update_state = (train_state, traj_batch, advantages, targets, rng)
                return update_state, losses

            update_state = (train_state, traj_batch, advantages, targets, rng)
            update_state, loss_info = jax.lax.scan(_update_epoch, update_state, None, config.UPDATE_EPOCHS)
            train_state = update_state[0]
            metric = jax.tree_util.tree_map(lambda x: x.mean(), traj_batch.info)
            metric["returned_episode_returns"] = traj_batch.info["returned_episode_returns"].mean()

            runner_state = (train_state, env_state, last_obs, obs_norm_state, update_state[-1])
            return runner_state, metric

        num_updates = config.TOTAL_TIMESTEPS // config.NUM_STEPS
        runner_state = (train_state, env_state, obsv, obs_norm_state, rng)
        runner_state, metrics = jax.lax.scan(_update_step, runner_state, None, num_updates)
        return {"runner_state": runner_state, "metrics": metrics}

    return train
