from typing import NamedTuple, Protocol, cast

import gymnax
import gymnax.wrappers
import jax
import jax.numpy as jnp
import optax
from flax.training.train_state import TrainState
from rl_components.networks import ActorCritic
from rl_components.types import PPOConfig


class Transition(NamedTuple):
    done: jax.Array
    action: jax.Array
    value: jax.Array
    reward: jax.Array
    log_prob: jax.Array
    obs: jax.Array
    info: dict[str, jax.Array]


class _ObservationSpace(Protocol):
    shape: tuple[int, ...]


class _ActionSpace(Protocol):
    n: int


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
    config: PPOConfig,
    env: object | None,
    env_params: object | None,
) -> tuple[_EnvLike, object | None]:
    if env is not None:
        return cast(_EnvLike, env), env_params

    resolved_env, resolved_env_params = gymnax.make(config.ENV_NAME)
    return cast(_EnvLike, gymnax.wrappers.LogWrapper(resolved_env)), resolved_env_params


def make_train(config: PPOConfig, env: object | None = None, env_params: object | None = None):
    env, env_params = _resolve_env(config, env, env_params)

    def train(rng):
        # INIT NETWORK
        network = ActorCritic(env.action_space(env_params).n, activation="tanh")
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

        # TRAIN LOOP
        def _update_step(runner_state, unused):
            train_state, env_state, last_obs, rng = runner_state

            # COLLECT TRAJECTORIES
            def _env_step(runner_state, unused):
                train_state, env_state, last_obs, rng = runner_state

                # SELECT ACTION
                rng, _rng = jax.random.split(rng)
                probs, value = network.apply(train_state.params, last_obs)
                action = probs.sample(seed=_rng)
                log_prob = jnp.asarray(probs.log_prob(action))

                # STEP ENV
                rng, _rng = jax.random.split(rng)
                obsv, env_state, reward, done, info = env.step(_rng, env_state, action, env_params)
                transition = Transition(done, action, value, reward, log_prob, last_obs, info)
                runner_state = (train_state, env_state, obsv, rng)
                return runner_state, transition

            runner_state, traj_batch = jax.lax.scan(_env_step, runner_state, None, config.NUM_STEPS)

            # CALCULATE ADVANTAGE
            train_state, env_state, last_obs, rng = runner_state
            _, last_val = network.apply(train_state.params, last_obs)

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
                        probs, value = network.apply(params, traj_batch.obs)
                        log_prob = probs.log_prob(traj_batch.action)

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
                        entropy_loss = probs.entropy().mean()

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

            runner_state = (train_state, env_state, last_obs, update_state[-1])
            return runner_state, metric

        num_updates = config.TOTAL_TIMESTEPS // config.NUM_STEPS
        runner_state = (train_state, env_state, obsv, rng)
        runner_state, metrics = jax.lax.scan(_update_step, runner_state, None, num_updates)
        return {"runner_state": runner_state, "metrics": metrics}

    return train
