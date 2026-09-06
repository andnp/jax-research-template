"""Small public API tests for jax-eval."""

from __future__ import annotations

import jax_eval
from jax_eval.representations import dead_neurons, feature_drift


def test_representation_kernels_use_direct_submodule_imports() -> None:
    """Keep the package root empty while exposing the canonical submodule paths."""
    assert "dead_neurons" not in vars(jax_eval)
    assert "feature_drift" not in vars(jax_eval)
    assert callable(dead_neurons)
    assert callable(feature_drift)
