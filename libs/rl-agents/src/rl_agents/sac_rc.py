"""Soft Actor-Critic with Regularized Corrections (SAC-RC) for continuous-action control.

The continuous-action analogue of ``rl_agents.qrc``: SAC's twin-critic
ensemble gets the same TDRC/QRC gradient-TD correction QRC applies to DQN
(Ghiassian et al., "Gradient Temporal-Difference Learning with Regularized
Corrections", ICML 2020, arXiv:2007.00611).

Two design decisions carried over from ``qrc.py``:

- The h-head reads stop-gradiented trunk features — it is a linear probe on
  frozen features and never shapes the trunk/q-head representation.
- There is no target network. SAC-RC is pure gradient TD: every bootstrap
  uses the online critic parameters, and the ``gamma * sg(h) * bootstrap``
  term in the loss supplies the gradient correction a target network would
  otherwise stand in for. Stop-gradienting that bootstrap would silently
  collapse SAC-RC back into plain semi-gradient SAC with an inert extra head.

The squared losses carry a ``0.5`` that TDRC's published form omits. That is
a house convention, but it scales the data terms and not ``BETA``'s
regulariser, so a given ``BETA`` is twice as strong relative to the data as
the same number in the published sweeps.

The actor loss and the entropy-coefficient (alpha) update are unchanged from
``rl_agents.sac``; only the critic loss and the twin-critic network gain the
h-head and correction term.
"""

from typing import Callable, NamedTuple, TypedDict, cast

import flax.linen as nn
import jax
import jax.numpy as jnp
import optax
from flax.training.train_state import TrainState
from flax.typing import VariableDict
from rl_components.buffers import ReplayBuffer, ReplayBufferState
from rl_components.gym_env import ContinuousActionSpace, GymEnv
from rl_components.structs import chex_struct

from rl_agents.sac import Actor


@chex_struct(frozen=True, kw_only=True)
class SACRCConfig:
    LR: float = 3e-4
    BUFFER_SIZE: int = 100_000
    BATCH_SIZE: int = 256
    TOTAL_TIMESTEPS: int = 1_000_000
    LEARNING_STARTS: int = 5_000
    TRAIN_FREQUENCY: int = 1
    GAMMA: float = 0.99
    ALPHA: float = 0.2
    TARGET_ENTROPY: float | None = None
    BETA: float = 1.0
    ENV_NAME: str = "MountainCarContinuous-v0"
    SEED: int = 42


class SACRCCritic(nn.Module):
    """Twin-critic trunk (matching ``sac.Critic``) plus a QRC-style h-head.

    The h-head reads ``jax.lax.stop_gradient``-ed trunk features, and is
    zero-initialised and bias-free, exactly as in ``qrc.QRCNetwork`` — it is
    a linear probe on frozen features that starts at zero and never shapes
    the trunk. The trunk and q-head match ``sac.Critic`` (default init).
    """

    @nn.compact
    def __call__(self, x: jnp.ndarray, a: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
        xa = jnp.concatenate([x, a], axis=-1)
        phi = nn.Dense(256)(xa)
        phi = nn.relu(phi)
        phi = nn.Dense(256)(phi)
        phi = nn.relu(phi)
        q = nn.Dense(1, name="q_head")(phi)
        h = nn.Dense(
            1,
            kernel_init=nn.initializers.zeros,
            use_bias=False,
            name="h_head",
        )(jax.lax.stop_gradient(phi))
        return jnp.squeeze(q, axis=-1), jnp.squeeze(h, axis=-1)


def _critic_apply(module: SACRCCritic, variables: VariableDict, x: jax.Array, a: jax.Array) -> tuple[jax.Array, jax.Array]:
    return cast(tuple[jax.Array, jax.Array], module.apply(variables, x, a))


def sac_rc_loss(
    q: jax.Array,
    h: jax.Array,
    reward: jax.Array,
    discount: jax.Array,
    next_q_min: jax.Array,
    next_log_prob: jax.Array,
    alpha: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Per-transition, per-ensemble-member SAC-RC loss.

    ``bootstrap`` is the soft value ``min_j Q_j(s', a') - alpha * log pi(a'|s')``
    computed from the ONLINE critic (no target network). ``target`` stop-
    gradients that bootstrap for the semi-gradient TD term, but the
    correction term ``gamma * sg(h) * bootstrap`` reuses the un-stopped
    ``bootstrap`` so its gradient reaches the online parameters that produced
    it: differentiating it w.r.t. those parameters gives
    ``gamma * delta_hat * grad(bootstrap)``, the term that distinguishes
    gradient TD from semi-gradient TD and removes the divergence-inducing
    off-policy bootstrapping bias that a target network is normally used to
    paper over.
    """
    target = jax.lax.stop_gradient(reward + discount * (next_q_min - alpha * next_log_prob))
    delta = target - q
    delta_hat = h

    v_loss = 0.5 * delta**2 + discount * jax.lax.stop_gradient(delta_hat) * (next_q_min - alpha * next_log_prob)
    h_loss = 0.5 * (jax.lax.stop_gradient(delta) - delta_hat) ** 2

    return v_loss, h_loss, delta


def sac_rc_loss_batch(
    critic_params: VariableDict,
    critic: SACRCCritic,
    actor_params: VariableDict,
    actor: Actor,
    alpha: jax.Array,
    obs: jax.Array,
    actions: jax.Array,
    rewards: jax.Array,
    next_obs: jax.Array,
    discounts: jax.Array,
    rng: jax.Array,
    beta: float,
) -> jax.Array:
    """Mean SAC-RC critic loss over a minibatch, plus L2 regularisation on the h-heads.

    ``next_q_min`` is shared across ensemble members (the min is over the
    ensemble), each member's own ``q``/``h`` are read from its own params via
    ``vmap``, matching the twin-critic structure of ``sac.py``.
    """
    next_actions, next_log_probs = jax.vmap(actor.sample, in_axes=(None, 0, 0))(
        actor_params, next_obs, jax.random.split(rng, obs.shape[0])
    )

    next_q_values = jax.vmap(
        lambda params, o, a: _critic_apply(critic, params, o, a)[0], in_axes=(0, None, None)
    )(critic_params, next_obs, next_actions)
    next_q_min = jnp.min(next_q_values, axis=0)

    def _single_critic_loss(params: VariableDict) -> jax.Array:
        q, h = _critic_apply(critic, params, obs, actions)
        v_loss, h_loss, _ = jax.vmap(sac_rc_loss, in_axes=(0, 0, 0, 0, 0, 0, None))(
            q, h, rewards, discounts, next_q_min, next_log_probs, alpha
        )
        h_reg = sum(jnp.sum(jnp.square(p)) for p in jax.tree.leaves(params["params"]["h_head"]))
        return jnp.mean(v_loss) + jnp.mean(h_loss) + beta * h_reg

    return jnp.mean(jax.vmap(_single_critic_loss)(critic_params))


class RunnerState(NamedTuple):
    actor_state: TrainState
    critic_state: TrainState
    alpha_state: TrainState
    buffer_state: ReplayBufferState
    env_state: object
    last_obs: jax.Array
    rng: jax.Array


class SACRCTrainOutput(TypedDict):
    runner_state: RunnerState
    metrics: dict[str, jax.Array]


def make_train(config: SACRCConfig, env: GymEnv[ContinuousActionSpace], env_params: object | None = None) -> Callable[[jax.Array], SACRCTrainOutput]:
    def train(rng: jax.Array) -> SACRCTrainOutput:
        # INIT NETWORKS
        rng, _rng_actor, _rng_critic = jax.random.split(rng, 3)
        action_dim = env.action_space(env_params).shape[0]
        obs_dim = env.observation_space(env_params).shape

        actor = Actor(action_dim)
        actor_params = actor.init(_rng_actor, jnp.zeros(obs_dim))
        actor_state = TrainState.create(apply_fn=actor.apply, params=actor_params, tx=optax.adam(config.LR))

        critic = SACRCCritic()
        rng, _rng_critic = jax.random.split(rng)
        critic_params = jax.vmap(critic.init, in_axes=(0, None, None))(
            jax.random.split(_rng_critic, 2), jnp.zeros(obs_dim), jnp.zeros((action_dim,))
        )
        critic_state = TrainState.create(apply_fn=critic.apply, params=critic_params, tx=optax.adam(config.LR))

        # Automatic Entropy Tuning
        if config.TARGET_ENTROPY is None:
            target_entropy = -float(action_dim)
        else:
            target_entropy = config.TARGET_ENTROPY

        log_alpha = {"log_alpha": jnp.log(jnp.array([config.ALPHA]))}
        alpha_state = TrainState.create(apply_fn=None, params=log_alpha, tx=optax.adam(config.LR))

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
                alpha_state,
                buffer_state,
                env_state,
                last_obs,
                rng,
            ) = runner_state

            # SELECT ACTION
            rng, _rng = jax.random.split(rng)

            def _random_action() -> jax.Array:
                return jax.random.uniform(_rng, (action_dim,), minval=-1, maxval=1)

            def _policy_action() -> jax.Array:
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
            def _do_train(
                actor_state: TrainState,
                critic_state: TrainState,
                alpha_state: TrainState,
                buffer_state: ReplayBufferState,
                rng: jax.Array,
            ) -> tuple[TrainState, TrainState, TrainState, jax.Array, jax.Array]:
                rng, _rng = jax.random.split(rng)
                obs, actions, rewards, next_obs, discounts = buffer.sample(buffer_state, _rng, config.BATCH_SIZE)

                log_alpha_arr = cast(jax.Array, alpha_state.params["log_alpha"])
                alpha = jnp.exp(log_alpha_arr[0])

                # CRITIC UPDATE
                rng, _rng = jax.random.split(rng)

                def _critic_loss_fn(critic_params: VariableDict) -> jax.Array:
                    return sac_rc_loss_batch(
                        critic_params, critic, actor_state.params, actor, alpha,
                        obs, actions, rewards, next_obs, discounts, _rng, config.BETA,
                    )

                critic_loss, critic_grads = jax.value_and_grad(_critic_loss_fn)(critic_state.params)
                critic_state = critic_state.apply_gradients(grads=critic_grads)

                # ACTOR UPDATE
                def _actor_loss_fn(actor_params: VariableDict, critic_params: VariableDict, alpha: jax.Array, obs: jax.Array, rng: jax.Array) -> jax.Array:
                    rng, _rng = jax.random.split(rng)
                    new_actions, log_probs = jax.vmap(actor.sample, in_axes=(None, 0, 0))(
                        actor_params, obs, jax.random.split(_rng, config.BATCH_SIZE)
                    )

                    q_values = jax.vmap(
                        lambda params, obs_value, action: _critic_apply(critic, params, obs_value, action)[0], in_axes=(0, None, None)
                    )(critic_params, obs, new_actions)
                    q_min = jnp.min(q_values, axis=0)

                    loss = jnp.mean(alpha * log_probs - q_min)
                    return loss

                grad_fn = jax.value_and_grad(_actor_loss_fn)
                actor_loss, actor_grads = grad_fn(actor_state.params, critic_state.params, alpha, obs, rng)
                actor_state = actor_state.apply_gradients(grads=actor_grads)

                # ALPHA UPDATE
                def _alpha_loss_fn(alpha_params: VariableDict, actor_params: VariableDict, obs: jax.Array, rng: jax.Array) -> jax.Array:
                    rng, _rng = jax.random.split(rng)
                    _, log_probs = jax.vmap(actor.sample, in_axes=(None, 0, 0))(actor_params, obs, jax.random.split(_rng, config.BATCH_SIZE))
                    loss = -jnp.mean(alpha_params["log_alpha"] * (log_probs + target_entropy))
                    return loss

                grad_fn = jax.value_and_grad(_alpha_loss_fn)
                alpha_loss, alpha_grads = grad_fn(alpha_state.params, actor_state.params, obs, rng)
                alpha_state = alpha_state.apply_gradients(grads=alpha_grads)

                return actor_state, critic_state, alpha_state, critic_loss, actor_loss

            can_train = (t > config.LEARNING_STARTS) & (t % config.TRAIN_FREQUENCY == 0)
            actor_state, critic_state, alpha_state, critic_loss, actor_loss = jax.lax.cond(
                can_train,
                lambda: _do_train(actor_state, critic_state, alpha_state, buffer_state, rng),
                lambda: (actor_state, critic_state, alpha_state, jnp.array(0.0), jnp.array(0.0)),
            )

            runner_state = RunnerState(
                actor_state=actor_state, critic_state=critic_state,
                alpha_state=alpha_state,
                buffer_state=buffer_state, env_state=env_state, last_obs=obsv, rng=rng,
            )
            info = dict(info)
            info["critic_loss"] = critic_loss
            info["actor_loss"] = actor_loss
            return runner_state, info

        # RUNNER
        runner_state = RunnerState(
            actor_state=actor_state, critic_state=critic_state,
            alpha_state=alpha_state,
            buffer_state=buffer_state, env_state=env_state, last_obs=obsv, rng=rng,
        )
        runner_state, metrics = jax.lax.scan(_update_step, runner_state, jnp.arange(config.TOTAL_TIMESTEPS))
        return {"runner_state": runner_state, "metrics": metrics}

    return train
