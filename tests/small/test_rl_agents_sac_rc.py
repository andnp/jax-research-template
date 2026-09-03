"""Small tests for rl_agents.sac_rc — config validation and network shape/init contracts."""

import jax
import jax.numpy as jnp
from rl_agents.sac_rc import SACRCConfig, SACRCCritic


class TestSACRCConfig:
    def test_defaults(self) -> None:
        cfg = SACRCConfig()
        assert cfg.LR == 3e-4
        assert cfg.BUFFER_SIZE == 100_000
        assert cfg.BATCH_SIZE == 256
        assert cfg.GAMMA == 0.99
        assert cfg.ALPHA == 0.2
        assert cfg.BETA == 1.0

    def test_custom_config(self) -> None:
        cfg = SACRCConfig(LR=1e-3, BUFFER_SIZE=1000, ENV_NAME="Pendulum-v1", BETA=0.5)
        assert cfg.LR == 1e-3
        assert cfg.BUFFER_SIZE == 1000
        assert cfg.ENV_NAME == "Pendulum-v1"
        assert cfg.BETA == 0.5

    def test_no_target_network_knobs(self) -> None:
        """SAC-RC has no target network, so it must not carry a Polyak TAU knob."""
        cfg = SACRCConfig()
        assert not hasattr(cfg, "TAU")


class TestSACRCCritic:
    def test_output_shape(self) -> None:
        net = SACRCCritic()
        obs = jnp.zeros((8,))
        action = jnp.zeros((2,))
        params = net.init(jax.random.key(0), obs, action)
        q, h = net.apply(params, jnp.ones((8,)), jnp.ones((2,)))
        assert q.shape == ()
        assert h.shape == ()

    def test_batch_output_shape(self) -> None:
        net = SACRCCritic()
        params = net.init(jax.random.key(0), jnp.zeros((4,)), jnp.zeros((2,)))
        q, h = net.apply(params, jnp.ones((10, 4)), jnp.ones((10, 2)))
        assert q.shape == (10,)
        assert h.shape == (10,)

    def test_h_head_is_zero_at_init(self) -> None:
        """The h-head is zero-initialised and bias-free, so h must start at zero."""
        net = SACRCCritic()
        params = net.init(jax.random.key(0), jnp.zeros((8,)), jnp.zeros((2,)))
        _, h = net.apply(
            params,
            jax.random.normal(jax.random.key(1), (8,)),
            jax.random.normal(jax.random.key(2), (2,)),
        )
        assert jnp.allclose(h, 0.0)

    def test_h_head_has_no_bias_param(self) -> None:
        net = SACRCCritic()
        params = net.init(jax.random.key(0), jnp.zeros((8,)), jnp.zeros((2,)))
        assert "bias" not in params["params"]["h_head"]

    def test_ensemble_init_gives_independent_members(self) -> None:
        """The twin-critic ensemble (vmapped init) must give distinct q-heads per member."""
        net = SACRCCritic()
        keys = jax.random.split(jax.random.key(0), 2)
        params = jax.vmap(net.init, in_axes=(0, None, None))(keys, jnp.zeros((4,)), jnp.zeros((2,)))
        kernel = params["params"]["q_head"]["kernel"]
        assert kernel.shape[0] == 2
        assert not jnp.allclose(kernel[0], kernel[1])
