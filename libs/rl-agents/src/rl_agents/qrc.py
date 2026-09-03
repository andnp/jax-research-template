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

That second decision is why :class:`QRCAgent` cannot be a
:class:`rl_agents.replay_q_agent.ReplayQAgent`, despite being a discrete-action
replay agent with an epsilon schedule: it has no ``TAU``, no target-network
frequency and no ``target_params``, and its loss needs the current ``epsilon``,
which no target rule in that family takes. It also has no eligibility trace to
reset at an episode boundary -- the ``h``-head is an ordinary ``nn.Dense``
living in ``TrainState.params`` and is persistent by design.

:class:`QRCAgent` is the port; ``make_train`` below it is the private training
loop it replaces. This module has no out-of-repository caller, so
``make_train``, :class:`RunnerState`, :class:`QRCTrainOutput` and
``QRCConfig.GAMMA`` can go as soon as the in-repository callers do. **Removal
condition:** the ``make_train`` drivers in ``tests/`` are the last of them.

``GAMMA`` survives only for that legacy path. The discount belongs to
:func:`rl_components.loop.run`, which stores a per-transition coefficient in
replay, and :func:`qrc_loss_batch` reads that coefficient straight through as
its per-transition ``gamma``; :class:`QRCAgent` never reads ``config.GAMMA``.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import NamedTuple, TypedDict

import chex
import flax.linen as nn
import jax
import jax.numpy as jnp
import optax
from flax.training.train_state import TrainState
from flax.typing import VariableDict
from jax_nn.typed_module import TypedApply
from rl_components.agent_protocol import AgentStep
from rl_components.buffers import ReplayBuffer, ReplayBufferState
from rl_components.env_protocol import EnvSpec
from rl_components.gym_env import DiscreteActionSpace, GymEnv
from rl_components.structs import chex_struct
from rl_components.timestep import Timestep

type QRCApplyFn = Callable[..., tuple[jax.Array, jax.Array]]
"""``QRCNetwork.apply``, as reached through ``TrainState.apply_fn``."""


@chex_struct(frozen=True, kw_only=True)
class QRCConfig:
    LR: float = 3e-4
    BUFFER_SIZE: int = 100_000
    BATCH_SIZE: int = 64
    TOTAL_TIMESTEPS: int = 200_000
    LEARNING_STARTS: int = 1_000
    TRAIN_FREQUENCY: int = 1
    GAMMA: float = 0.99  # Legacy make_train only; see the module docstring.
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
    obs: jax.Array,
    actions: jax.Array,
    rewards: jax.Array,
    next_obs: jax.Array,
    discounts: jax.Array,
    epsilon: jax.Array | float,
    beta: float,
    *,
    apply_fn: QRCApplyFn,
) -> jax.Array:
    """Mean QRC loss over a minibatch, plus L2 regularisation on the h-head.

    Regularising only the h-head (Ghiassian et al. 2020, §3.2) keeps the
    correction term bounded without biasing the q-head's value estimates.

    ``discounts`` is the per-transition bootstrap coefficient the loop computed
    and is passed straight through as each transition's ``gamma``, so a terminal
    row's ``0.0`` removes the bootstrap and the correction term with it.

    The network arrives as ``apply_fn`` rather than as a module, because the
    agent's ``step`` reaches it through ``TrainState.apply_fn`` -- a static field
    -- and cannot construct a module from an action count it only carries as a
    traced leaf.
    """
    q, h = apply_fn(params, obs)
    q_next, _ = apply_fn(params, next_obs)

    v_loss, h_loss, _ = jax.vmap(qrc_loss, in_axes=(0, 0, 0, 0, 0, 0, None))(
        q, h, actions, rewards, discounts, q_next, epsilon
    )

    h_reg = sum(jnp.sum(jnp.square(p)) for p in jax.tree.leaves(params["params"]["h_head"]))
    return jnp.mean(v_loss) + jnp.mean(h_loss) + beta * h_reg


@chex_struct(frozen=True)
class QRCAgentState:
    """Everything :class:`QRCAgent` carries between loop iterations.

    There is no ``target_params`` field and no eligibility trace, and both absences are
    the algorithm: the bootstrap reads the online parameters and the ``h``-head's
    correction supplies what a target network would damp, while the ``h``-head itself is
    an ordinary parameter inside ``train_state`` and is meant to persist across episodes.

    Attributes:
        train_state: Online parameters, optimizer state, and the static ``apply_fn``
            through which ``step`` reaches the network.
        buffer_state: Replay contents, and the only record of the buffer's geometry.
        last_obs: Observation the pending transition started from. Zero-primed at
            ``init``; never read before the first insertion, which is guarded on
            ``step_index > 0``.
        last_action: Action that opened the pending transition, zero-primed likewise.
        num_actions: Size of the discrete action space, as an int32 leaf so that random
            exploration can sample against a traced bound.
        key: PRNG key for sampling minibatches and exploring.
    """

    train_state: TrainState
    buffer_state: ReplayBufferState
    last_obs: jax.Array
    last_action: jax.Array
    num_actions: jax.Array
    key: chex.PRNGKey


class QRCAgent:
    """Q-learning with Regularized Corrections over a discrete action space."""

    def __init__(self, config: QRCConfig) -> None:
        """Bind the configuration. The object is static under ``jit``.

        Args:
            config: Hyperparameters. Read at trace time only, so every field may be a
                Python scalar. ``GAMMA`` is not read at all.
        """
        self.config = config

    def init(self, key: chex.PRNGKey, spec: EnvSpec) -> QRCAgentState:
        """Build parameters, optimizer, replay buffer and the zero-primed pending slots.

        Args:
            key: PRNG key for parameter initialization.
            spec: The environment description, and the only place the observation shape
                comes from. Must be discrete: the bootstrap is an expectation over a
                finite action set.

        Returns:
            The initial agent state.

        Raises:
            ValueError: If ``spec`` describes a continuous action space.
        """
        if spec.num_actions is None:
            raise ValueError(f"QRC requires a discrete action space, got spec {spec.id!r} with none")

        observation_shape = tuple(spec.observation_shape)
        observation_dtype = jnp.dtype(spec.observation_dtype)
        action_dtype = jnp.dtype(spec.action_dtype)
        network = QRCNetwork(spec.num_actions)

        params_key, carry_key = jax.random.split(key)
        buffer = ReplayBuffer(
            capacity=self.config.BUFFER_SIZE,
            obs_shape=observation_shape,
            action_shape=tuple(spec.action_shape),
            action_dtype=action_dtype,
            obs_dtype=observation_dtype,
        )

        return QRCAgentState(
            train_state=TrainState.create(
                apply_fn=network.apply,
                params=network.init(params_key, jnp.zeros(observation_shape, dtype=observation_dtype)),
                tx=optax.adam(self.config.LR),
            ),
            buffer_state=buffer.init(),
            last_obs=jnp.zeros(observation_shape, dtype=observation_dtype),
            last_action=jnp.zeros(tuple(spec.action_shape), dtype=action_dtype),
            num_actions=jnp.asarray(spec.num_actions, dtype=jnp.int32),
            key=carry_key,
        )

    def step(
        self,
        state: QRCAgentState,
        timestep: Timestep[jax.Array],
        step_index: jax.Array,
    ) -> AgentStep[QRCAgentState, jax.Array]:
        """Close the pending transition, learn from replay, then act.

        The order is normative. Insertion uses ``timestep.bootstrap_observation`` and not
        ``timestep.observation``: at an episode boundary the latter is the post-reset
        observation, and bootstrapping from it is the defect this port removes. There is
        no target-network synchronisation step, and nothing to reset at a boundary.

        One epsilon serves both roles it has in this algorithm, as it must: the
        exploration rate and the policy the bootstrap takes its expectation under are the
        same number, and computing them separately is how the two drift apart.

        Args:
            state: The agent state from the previous iteration.
            timestep: This iteration's view of the environment.
            step_index: Transitions closed so far. Iteration ``0`` closes none, so both
                insertion and learning are guarded above it.

        Returns:
            The next state, the action to apply, and a fixed three-key metric schema.
            ``loss`` is a zero placeholder on the iterations that do not learn, so the
            pytree returned to ``lax.scan`` is identical on every iteration.
        """
        config = self.config
        buffer = ReplayBuffer.from_state(state.buffer_state)
        carry_key, sample_key, explore_key, action_key = jax.random.split(state.key, 4)

        buffer_state = jax.lax.cond(
            step_index > 0,
            lambda: buffer.add(
                state.buffer_state,
                state.last_obs[None, ...],
                state.last_action[None, ...],
                timestep.reward[None, ...],
                timestep.bootstrap_observation[None, ...],
                timestep.discount[None, ...],
            ),
            lambda: state.buffer_state,
        )

        epsilon = jnp.maximum(
            jnp.asarray(config.EPSILON_END, jnp.float32),
            jnp.asarray(config.EPSILON_START, jnp.float32)
            - (config.EPSILON_START - config.EPSILON_END)
            * (step_index / (config.TOTAL_TIMESTEPS * config.EPSILON_FRACTION)),
        ).astype(jnp.float32)

        can_train = (step_index > config.LEARNING_STARTS) & (step_index % config.TRAIN_FREQUENCY == 0)
        train_state, loss = jax.lax.cond(
            can_train,
            lambda: self._learn(state.train_state, buffer_state, buffer, sample_key, epsilon),
            lambda: (state.train_state, jnp.zeros((), jnp.float32)),
        )

        q_values, _ = train_state.apply_fn(train_state.params, timestep.observation)
        chose_random = jax.random.uniform(explore_key, ()) < epsilon
        random_action = jax.random.randint(action_key, (), 0, state.num_actions)
        action = jnp.where(chose_random, random_action, jnp.argmax(q_values)).astype(state.last_action.dtype)

        return AgentStep(
            state=QRCAgentState(
                train_state=train_state,
                buffer_state=buffer_state,
                last_obs=timestep.observation,
                last_action=action,
                num_actions=state.num_actions,
                key=carry_key,
            ),
            action=action,
            metrics={
                "loss": loss,
                "epsilon": epsilon,
                "q_max": jnp.max(q_values).astype(jnp.float32),
            },
        )

    def _learn(
        self,
        train_state: TrainState,
        buffer_state: ReplayBufferState,
        buffer: ReplayBuffer,
        key: chex.PRNGKey,
        epsilon: jax.Array,
    ) -> tuple[TrainState, jax.Array]:
        """Take one gradient step on a replay minibatch."""
        obs, actions, rewards, next_obs, discounts = buffer.sample(buffer_state, key, self.config.BATCH_SIZE)

        loss, grads = jax.value_and_grad(
            lambda params: qrc_loss_batch(
                params,
                obs,
                actions,
                rewards,
                next_obs,
                discounts,
                epsilon,
                self.config.BETA,
                apply_fn=train_state.apply_fn,
            )
        )(train_state.params)
        return train_state.apply_gradients(grads=grads), loss.astype(jnp.float32)


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
                    return qrc_loss_batch(
                        params, obs, actions, rewards, next_obs, discounts, epsilon, config.BETA, apply_fn=network.apply
                    )

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
