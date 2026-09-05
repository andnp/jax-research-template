"""Small tests for jax_utils.linalg — stable rank of a feature covariance matrix."""

import jax
import jax.numpy as jnp
from jax_utils.linalg import stable_rank


class TestStableRank:
    def test_one_hot_features_lose_one_rank_to_mean_centering(self) -> None:
        features = jnp.eye(4)
        assert jnp.allclose(stable_rank(features), 3.0)

    def test_collapsed_features_have_stable_rank_near_one(self) -> None:
        base = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0])
        features = jnp.stack([base, base, base], axis=1)
        assert jnp.allclose(stable_rank(features), 1.0, atol=1e-3)

    def test_stable_rank_is_jittable(self) -> None:
        features = jnp.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [-1.0, 1.0]])
        jitted = jax.jit(stable_rank)
        assert jnp.allclose(jitted(features), stable_rank(features))
