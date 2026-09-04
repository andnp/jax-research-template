"""The action-bounds rejection, exercised on the path that actually drives training.

``tests/small/test_rl_agents_continuous_actions.py`` builds an :class:`EnvSpec` eagerly
and so proves only that :func:`continuous_action_dim` can reject. Production never calls
it that way: :func:`rl_components.loop.run` reads ``env.spec`` and calls ``agent.init``
inside the caller's ``jit``, where a staged bound makes the comparison unresolvable and
the rejection silently degrades into acceptance. That is the whole failure this guard
exists to prevent, so it has to be pinned through a real jitted ``loop.run``.

Gymnax's Pendulum declares ``Box(-2, 2)``, which is the case that trains happily while
the tanh policy reaches only the middle half of the real torque range.
"""

from __future__ import annotations

import gymnax
import jax
import jax.numpy as jnp
import pytest
from rl_agents.sac import SACAgent, SACConfig
from rl_components import loop
from rl_components.action_normalization import make_action_normalization_wrapper
from rl_components.env_protocol import EnvProtocol
from rl_components.gymnax_bridge import make_gymnax_env


def _agent() -> SACAgent:
    return SACAgent(SACConfig(LEARNING_STARTS=2, BUFFER_SIZE=64, BATCH_SIZE=4))


def _run_jitted(env: EnvProtocol, params: object) -> dict[str, jax.Array]:
    def driver(key: jax.Array) -> dict[str, jax.Array]:
        _, metrics = loop.run(_agent(), env, key, steps=4, gamma=0.99, env_params=params)
        return metrics

    return jax.jit(driver)(jax.random.key(0))


class TestUnnormalizedBoundsRejectedUnderJit:
    def test_a_bare_unnormalized_environment_is_rejected(self) -> None:
        raw, params = gymnax.make("Pendulum-v1")

        with pytest.raises(ValueError, match=r"\[-1, 1\]"):
            _run_jitted(make_gymnax_env(raw), params)

    def test_the_normalization_wrapper_makes_the_same_environment_acceptable(self) -> None:
        """The rejection names this wrapper as the remedy, so the remedy must work."""
        raw, params = gymnax.make("Pendulum-v1")

        metrics = _run_jitted(make_action_normalization_wrapper(make_gymnax_env(raw)), params)

        assert bool(jnp.all(jnp.isfinite(metrics["critic_loss"])))

    def test_an_already_normalized_environment_is_accepted(self) -> None:
        raw, params = gymnax.make("MountainCarContinuous-v0")

        metrics = _run_jitted(make_gymnax_env(raw), params)

        assert bool(jnp.all(jnp.isfinite(metrics["critic_loss"])))

    def test_bounds_derived_from_traced_params_stay_unverifiable(self) -> None:
        """Tracing ``env_params`` makes the bounds genuinely unknowable at trace time.

        Gymnax reads Pendulum's torque limit off ``params``, so this is the documented
        case the guard cannot judge. It must fall through to acceptance rather than
        reject every environment whose bounds it merely failed to read.
        """
        raw, params = gymnax.make("Pendulum-v1")
        env = make_gymnax_env(raw)

        def driver(key: jax.Array, traced_params: object) -> dict[str, jax.Array]:
            _, metrics = loop.run(_agent(), env, key, steps=4, gamma=0.99, env_params=traced_params)
            return metrics

        metrics = jax.jit(driver)(jax.random.key(0), params)

        assert bool(jnp.all(jnp.isfinite(metrics["critic_loss"])))
