"""Optimizer-state helpers used by traced agent hyperparameters."""

from __future__ import annotations

import jax
import optax
from flax.training.train_state import TrainState


def with_hyperparams(train_state: TrainState, **hyperparams: jax.Array) -> TrainState:
    """Replace injected optimizer hyperparameters in plain or chained state."""
    opt_state = train_state.opt_state
    if isinstance(opt_state, (optax.InjectHyperparamsState, optax.InjectStatefulHyperparamsState)):
        updated_state = opt_state._replace(hyperparams={**opt_state.hyperparams, **hyperparams})
        return train_state.replace(opt_state=updated_state)
    if isinstance(opt_state, tuple) and isinstance(opt_state[-1], (optax.InjectHyperparamsState, optax.InjectStatefulHyperparamsState)):
        updated_state = opt_state[-1]._replace(hyperparams={**opt_state[-1].hyperparams, **hyperparams})
        return train_state.replace(opt_state=(*opt_state[:-1], updated_state))
    raise TypeError("train_state optimizer must use injected hyperparameters")
