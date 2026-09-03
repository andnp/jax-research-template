"""Q-learning with Regularized Corrections (QRC) for discrete-action control.

Ghiassian et al., "Gradient Temporal-Difference Learning with Regularized
Corrections", ICML 2020 (arXiv:2007.00611). QRC is the nonlinear control
variant of TDRC: an auxiliary h-head corrects the semi-gradient bias that can
make DQN-style bootstrapping diverge.

Two design decisions carried over from the reference implementation:

- The h-head reads stop-gradiented trunk features (``addHead(..., grad=False)``
  in the reference code) — it is a linear probe on frozen features and never
  shapes the trunk representation.
- There is no target network. QRC is pure gradient TD: the bootstrap uses the
  online parameters, and the ``gamma * sg(h) * v_next`` term in the loss
  supplies the gradient correction a target network would otherwise stand in
  for.
"""

from typing import Callable, NamedTuple, TypedDict

import flax.linen as nn
import jax
import jax.numpy as jnp
import optax
from flax.training.train_state import TrainState
from flax.typing import VariableDict
from jax_nn.typed_module import TypedApply
from rl_components.buffers import ReplayBuffer, ReplayBufferState
from rl_components.gym_env import DiscreteActionSpace, GymEnv
from rl_components.structs import chex_struct


@chex_struct(frozen=True, kw_only=True)
class QRCConfig:
    LR: float = 3e-4
    BUFFER_SIZE: int = 100_000
    BATCH_SIZE: int = 64
    TOTAL_TIMESTEPS: int = 200_000
    LEARNING_STARTS: int = 1_000
    TRAIN_FREQUENCY: int = 1
    GAMMA: float = 0.99
    EPSILON_START: float = 1.0
    EPSILON_END: float = 0.05
    EPSILON_FRACTION: float = 0.5
    BETA: float = 1.0
    ENV_NAME: str = "CartPole-v1"
    SEED: int = 42


class QRCNetwork(TypedApply[tuple[jax.Array, jax.Array]], nn.Module):
    """Shared trunk with zero-initialised linear q- and h-heads.

    The h-head reads ``jax.lax.stop_gradient``-ed trunk features, so only the
    q-head's loss shapes the trunk; the h-head is a linear probe on frozen
    features.
    """

    action_dim: int

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
        phi = nn.Dense(64)(x)
        phi = nn.relu(phi)
        phi = nn.Dense(64)(phi)
        phi = nn.relu(phi)
        q = nn.Dense(
            self.action_dim,
            kernel_init=nn.initializers.zeros,
            bias_init=nn.initializers.zeros,
            name="q_head",
        )(phi)
        h = nn.Dense(
            self.action_dim,
            kernel_init=nn.initializers.zeros,
            use_bias=False,
            name="h_head",
        )(jax.lax.stop_gradient(phi))
        return q, h


def expected_action_value(q_next: jax.Array, epsilon: jax.Array | float) -> jax.Array:
    """Epsilon-greedy expected next-state value, uniform over the argmax set.

    The greedy distribution is stop-gradiented, so the bootstrap contributes
    gradient only through the action values themselves.
    """
    n_actions = q_next.shape[-1]
    greedy = (q_next == q_next.max()).astype(q_next.dtype)
    pi = greedy / greedy.sum()
    pi = (1.0 - epsilon) * pi + epsilon / n_actions
    pi = jax.lax.stop_gradient(pi)
    return q_next.dot(pi)


def qrc_loss(
    q: jax.Array,
    h: jax.Array,
    action: jax.Array,
    reward: jax.Array,
    gamma: jax.Array,
    q_next: jax.Array,
    epsilon: jax.Array | float,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Per-transition QRC loss (Ghiassian et al. 2020, eqs. 8-9).

    ``gamma * sg(delta_hat) * v_next`` is the gradient-TD correction term:
    differentiating it w.r.t. the online parameters yields
    ``gamma * delta_hat * grad(v_next)``, which is exactly the term that
    distinguishes gradient TD from semi-gradient TD and removes the bias
    responsible for divergence under off-policy bootstrapping (e.g. DQN).
    """
    v_next = expected_action_value(q_next, epsilon)
    target = jax.lax.stop_gradient(reward + gamma * v_next)

    delta = target - q[action]
    delta_hat = h[action]

    v_loss = 0.5 * delta**2 + gamma * jax.lax.stop_gradient(delta_hat) * v_next
    h_loss = 0.5 * (jax.lax.stop_gradient(delta) - delta_hat) ** 2

    return v_loss, h_loss, delta


def qrc_loss_batch(
    params: VariableDict,
    network: QRCNetwork,
    obs: jax.Array,
    actions: jax.Array,
    rewards: jax.Array,
    next_obs: jax.Array,
    discounts: jax.Array,
    epsilon: jax.Array | float,
    beta: float,
) -> jax.Array:
    """Mean QRC loss over a minibatch, plus L2 regularisation on the h-head.

    Regularising only the h-head (Ghiassian et al. 2020, §3.2) keeps the
    correction term bounded without biasing the q-head's value estimates.
    """
    q, h = network.apply(params, obs)
    q_next, _ = network.apply(params, next_obs)

    v_loss, h_loss, _ = jax.vmap(qrc_loss, in_axes=(0, 0, 0, 0, 0, 0, None))(
        q, h, actions, rewards, discounts, q_next, epsilon
    )

    h_reg = sum(jnp.sum(jnp.square(p)) for p in jax.tree.leaves(params["params"]["h_head"]))
    return jnp.mean(v_loss) + jnp.mean(h_loss) + beta * h_reg


class RunnerState(NamedTuple):
    train_state: TrainState
    buffer_state: ReplayBufferState
    env_state: object
    last_obs: jax.Array
    rng: jax.Array


class QRCTrainOutput(TypedDict):
    runner_state: RunnerState
    metrics: dict[str, jax.Array]


def make_train(config: QRCConfig, env: GymEnv[DiscreteActionSpace], env_params: object | None = None) -> Callable[[jax.Array], QRCTrainOutput]:
    def train(rng: jax.Array) -> QRCTrainOutput:
        # INIT NETWORK
        observation_shape = tuple(env.observation_space(env_params).shape)
        action_dim = env.action_space(env_params).n
        network = QRCNetwork(action_dim)
        rng, _rng = jax.random.split(rng)
        init_x = jnp.zeros(observation_shape, dtype=env.observation_space(env_params).dtype)
        params = network.init(_rng, init_x)

        tx = optax.adam(config.LR)
        train_state = TrainState.create(apply_fn=network.apply, params=params, tx=tx)

        # INIT BUFFER
        buffer = ReplayBuffer(config.BUFFER_SIZE, observation_shape, (), jnp.int32)
        buffer_state = buffer.init()

        # INIT ENV
        rng, _rng = jax.random.split(rng)
        obsv, env_state = env.reset(_rng, env_params)

        def _update_step(runner_state: RunnerState, t: jax.Array) -> tuple[RunnerState, dict[str, jax.Array]]:
            train_state, buffer_state, env_state, last_obs, rng = runner_state

            # EPSILON GREEDY
            epsilon = jnp.maximum(
                config.EPSILON_END,
                config.EPSILON_START
                - (config.EPSILON_START - config.EPSILON_END)
                * (t / (config.TOTAL_TIMESTEPS * config.EPSILON_FRACTION)),
            )

            rng, _rng_action, _rng_step = jax.random.split(rng, 3)
            q_values, _ = network.apply(train_state.params, last_obs)
            greedy_action = jnp.argmax(q_values)
            random_action = jax.random.randint(_rng_action, (), 0, action_dim)
            chose_random = jax.random.uniform(_rng_action, ()) < epsilon
            action = jnp.where(chose_random, random_action, greedy_action)

            # STEP ENV
            obsv, env_state, reward, done, info = env.step(_rng_step, env_state, action, env_params)
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
            def _do_train(train_state: TrainState, buffer_state: ReplayBufferState, rng: jax.Array) -> tuple[TrainState, jax.Array]:
                rng, _rng = jax.random.split(rng)
                obs, actions, rewards, next_obs, discounts = buffer.sample(buffer_state, _rng, config.BATCH_SIZE)

                def _loss_fn(params: VariableDict) -> jax.Array:
                    return qrc_loss_batch(params, network, obs, actions, rewards, next_obs, discounts, epsilon, config.BETA)

                loss, grads = jax.value_and_grad(_loss_fn)(train_state.params)
                train_state = train_state.apply_gradients(grads=grads)
                return train_state, loss

            can_train = (t > config.LEARNING_STARTS) & (t % config.TRAIN_FREQUENCY == 0)
            train_state, loss = jax.lax.cond(
                can_train,
                lambda: _do_train(train_state, buffer_state, rng),
                lambda: (train_state, 0.0),
            )

            runner_state = RunnerState(train_state=train_state, buffer_state=buffer_state, env_state=env_state, last_obs=obsv, rng=rng)
            return runner_state, info

        # RUNNER
        runner_state = RunnerState(train_state=train_state, buffer_state=buffer_state, env_state=env_state, last_obs=obsv, rng=rng)
        runner_state, metrics = jax.lax.scan(
            _update_step, runner_state, jnp.arange(config.TOTAL_TIMESTEPS)
        )
        return {"runner_state": runner_state, "metrics": metrics}

    return train
