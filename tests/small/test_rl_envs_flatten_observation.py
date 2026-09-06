import chex
import jax
import jax.numpy as jnp
from rl_components.env_protocol import EnvReset, EnvSpec, EnvStep
from rl_envs.flatten_observation import FlattenObservationWrapper
from rl_envs.frame_stack import FrameStackWrapper


class _SequenceEnv:
    def spec(self, params: None = None):
        del params
        return EnvSpec(
            id="sequence",
            observation_shape=(2,),
            action_shape=(1,),
            action_dtype=jnp.float32,
            action_low=jnp.array([-1.0], dtype=jnp.float32),
            action_high=jnp.array([1.0], dtype=jnp.float32),
            truncation_policy="terminate",
        )

    def reset(self, key: chex.PRNGKey, params: None = None):
        del key, params
        return EnvReset(observation=jnp.array([1.0, 2.0]), state=jnp.array(0, dtype=jnp.int32))

    def step(self, key: chex.PRNGKey, state: jax.Array, action: jax.Array, params: None = None):
        del key, action, params
        next_state = state + 1
        return EnvStep(
            observation=jnp.asarray([next_state + 1, next_state + 2], dtype=jnp.float32),
            state=next_state,
            reward=jnp.array(1.0),
            terminated=jnp.array(False),
            truncated=jnp.array(False),
            info={},
        )


def test_flattens_stacked_observations_for_vector_policies() -> None:
    """Flattening preserves frame order while exposing a one-dimensional observation."""
    environment = FlattenObservationWrapper(FrameStackWrapper(_SequenceEnv(), n_frames=2))

    reset = environment.reset(jax.random.key(0))
    step = environment.step(jax.random.key(1), reset.state, jnp.zeros((1,), dtype=jnp.float32))

    assert environment.spec().observation_shape == (4,)
    assert reset.observation.shape == (4,)
    assert reset.observation.tolist() == [1.0, 2.0, 1.0, 2.0]
    assert step.observation.tolist() == [1.0, 2.0, 2.0, 3.0]


def test_preserves_action_metadata_and_truncation_policy() -> None:
    """Flattening changes only observation metadata and values."""
    environment = FlattenObservationWrapper(FrameStackWrapper(_SequenceEnv(), n_frames=3))
    spec = environment.spec()

    assert spec.id == "sequence"
    assert spec.action_shape == (1,)
    assert spec.action_low is not None
    assert spec.action_high is not None
    assert jnp.allclose(spec.action_low, jnp.array([-1.0], dtype=jnp.float32))
    assert jnp.allclose(spec.action_high, jnp.array([1.0], dtype=jnp.float32))
    assert spec.truncation_policy == "terminate"
