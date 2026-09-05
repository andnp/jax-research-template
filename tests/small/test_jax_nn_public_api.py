"""Small (unit) tests for jax_nn public API surface."""

from __future__ import annotations


class TestPublicAPI:
    def test_submodule_imports(self) -> None:
        from jax_nn.initializers import output_orthogonal, stable_orthogonal

        for obj in (
            output_orthogonal,
            stable_orthogonal,
        ):
            assert obj is not None
