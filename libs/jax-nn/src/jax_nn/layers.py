"""Specialized Flax Linen layers for RL."""

from __future__ import annotations

import flax.linen as nn
import jax
import jax.numpy as jnp
from jax import Array


class NoisyLinear(nn.Module):
    """Noisy linear layer with factored Gaussian noise (Fortunato et al., 2017).

    Implements:

    .. math::

        y = (W + \\sigma_W \\odot \\varepsilon_W) \\cdot x
            + (b + \\sigma_b \\odot \\varepsilon_b)

    where :math:`\\varepsilon` are factored noise samples regenerated every
    forward pass. Factored noise reduces parameter count from
    :math:`p \\cdot q` to :math:`p + q` noise samples by computing
    :math:`\\varepsilon_W = f(\\varepsilon_p) \\otimes f(\\varepsilon_q)`
    where :math:`f(x) = \\operatorname{sign}(x) \\sqrt{|x|}`.

    Uses ``self.make_rng('noise')`` — callers must supply
    ``rngs={'params': ..., 'noise': ...}`` during ``init`` and ``apply``.

    Attributes:
        features: Number of output features.
        sigma_init: Initial value for all :math:`\\sigma` parameters.
            Default 0.5 per Fortunato et al.
        dtype: Computation and parameter dtype.
    """

    features: int
    sigma_init: float = 0.5
    dtype: jnp.dtype = jnp.float32

    @nn.compact
    def __call__(self, x: Array) -> Array:
        in_features = x.shape[-1]
        input_scale = 1.0 / jnp.sqrt(jnp.float32(in_features))

        mu_w = self.param(
            "mu_w",
            nn.initializers.uniform(scale=input_scale),
            (in_features, self.features),
        )
        mu_b = self.param(
            "mu_b",
            nn.initializers.uniform(scale=input_scale),
            (self.features,),
        )
        sigma_w = self.param(
            "sigma_w",
            nn.initializers.constant(self.sigma_init / jnp.sqrt(jnp.float32(in_features))),
            (in_features, self.features),
        )
        sigma_b = self.param(
            "sigma_b",
            nn.initializers.constant(self.sigma_init / jnp.sqrt(jnp.float32(in_features))),
            (self.features,),
        )

        key = self.make_rng("noise")
        key_p, key_q = jax.random.split(key)
        eps_p = _factored_noise(key_p, in_features)
        eps_q = _factored_noise(key_q, self.features)
        eps_w = jnp.outer(eps_p, eps_q)
        eps_b = eps_q

        w = jnp.asarray(mu_w + sigma_w * eps_w, dtype=self.dtype)
        b = jnp.asarray(mu_b + sigma_b * eps_b, dtype=self.dtype)
        return x @ w + b


def _factored_noise(key: Array, size: int) -> Array:
    """Generate factored noise: :math:`f(x) = \\operatorname{sign}(x) \\sqrt{|x|}`."""
    raw = jax.random.normal(key, (size,))
    return jnp.sign(raw) * jnp.sqrt(jnp.abs(raw))
