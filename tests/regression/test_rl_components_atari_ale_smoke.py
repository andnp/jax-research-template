import jax
import jax.numpy as jnp
from rl_components.atari_ale import AleAtariConfig, make_atari_adapter


def test_ale_adapter_reset_and_step_shapes():
    config = AleAtariConfig(game="Pong", frame_stack=4, frame_skip=4)
    env = make_atari_adapter(config)

    spec = env.spec()
    assert spec.num_actions is not None and spec.num_actions > 0

    reset_out = env.reset(jax.random.key(0))
    # With frame_stack=4 and screen_size=84, grayscale=True: (4, 84, 84, 1)
    assert reset_out.observation.shape == (4, 84, 84, 1)

    # All frames on reset should be identical (filled with the first obs)
    assert jnp.all(reset_out.observation[0] == reset_out.observation[-1])

    action = jnp.array(0, dtype=jnp.int32)
    step_out = env.step(jax.random.key(1), reset_out.state, action)
    assert step_out.observation.shape == (4, 84, 84, 1)
    assert step_out.reward.shape == ()
    assert step_out.terminated.shape == ()
    assert step_out.truncated.shape == ()
