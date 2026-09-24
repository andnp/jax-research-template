"""Tests for optimizer hyperparameter injection."""

import jax.numpy as jnp
import optax
import pytest
from flax.training.train_state import TrainState
from flax.typing import VariableDict
from rl_components.optim import with_hyperparams


def _apply_fn(params: VariableDict, values: jnp.ndarray) -> jnp.ndarray:
    return values


def test_with_hyperparams_updates_plain_injected_state() -> None:
    """A plain injected optimizer state receives traced hyperparameters."""
    state = TrainState.create(
        apply_fn=_apply_fn,
        params={"weight": jnp.array(1.0)},
        tx=optax.inject_hyperparams(optax.sgd)(learning_rate=0.1),
    )

    updated = with_hyperparams(state, learning_rate=jnp.array(0.2))

    assert jnp.allclose(updated.opt_state.hyperparams["learning_rate"], 0.2)


def test_with_hyperparams_updates_last_chained_state() -> None:
    """A chained optimizer updates the injected transform at its tail."""
    state = TrainState.create(
        apply_fn=_apply_fn,
        params={"weight": jnp.array(1.0)},
        tx=optax.chain(optax.clip_by_global_norm(1.0), optax.inject_hyperparams(optax.sgd)(learning_rate=0.1)),
    )

    updated = with_hyperparams(state, learning_rate=jnp.array(0.2))

    assert jnp.allclose(updated.opt_state[-1].hyperparams["learning_rate"], 0.2)


def test_with_hyperparams_rejects_non_injected_state() -> None:
    """Reject optimizer states without an injected hyperparameter transform."""
    state = TrainState.create(
        apply_fn=_apply_fn,
        params={"weight": jnp.array(1.0)},
        tx=optax.sgd(learning_rate=0.1),
    )

    with pytest.raises(TypeError, match="injected hyperparameters"):
        with_hyperparams(state, learning_rate=jnp.array(0.2))
