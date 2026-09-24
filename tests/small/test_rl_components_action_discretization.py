import chex
import jax
import jax.numpy as jnp
import pytest
from rl_components.action_discretization import DiscretizeAction
from rl_components.env_protocol import EnvProtocol, EnvReset, EnvSpec, EnvStep, TruncationPolicy


class _DiscreteEnv:
    def __init__(self, *, num_actions: int = 2, truncation_policy: TruncationPolicy = "bootstrap") -> None:
        self._spec = EnvSpec(
            id="discrete",
            observation_shape=(1,),
            action_shape=(),
            action_dtype=jnp.int32,
            num_actions=num_actions,
            truncation_policy=truncation_policy,
        )

    def spec(self, params: None = None) -> EnvSpec:
        del params
        return self._spec

    def reset(self, key: chex.PRNGKey, params: None = None) -> EnvReset[jax.Array, jax.Array]:
        del key, params
        return EnvReset(observation=jnp.zeros((1,), dtype=jnp.float32), state=jnp.int32(0))

    def step(
        self,
        key: chex.PRNGKey,
        state: jax.Array,
        action: jax.Array,
        params: None = None,
    ) -> EnvStep[jax.Array, jax.Array]:
        del key, params
        return EnvStep(
            observation=jnp.zeros((1,), dtype=jnp.float32),
            state=state,
            reward=jnp.float32(0.0),
            terminated=jnp.bool_(False),
            truncated=jnp.bool_(False),
            info={"action": action},
        )


def test_spec_exposes_the_unit_interval_and_preserves_truncation_policy() -> None:
    """The continuous-facing spec keeps the discrete environment's boundary semantics.

    Reset still returns the wrapped environment's state.
    """
    env: EnvProtocol[jax.Array, jax.Array, jax.Array, None] = _DiscreteEnv(truncation_policy="terminate")
    wrapper = DiscretizeAction(env)

    spec = wrapper.spec()
    reset = wrapper.reset(jax.random.key(0))

    assert spec.id == "discrete"
    assert spec.action_shape == (1,)
    assert spec.num_actions is None
    assert spec.action_dtype == jnp.dtype(jnp.float32)
    assert spec.action_low is not None and spec.action_high is not None
    assert jnp.array_equal(spec.action_low, jnp.array([-1.0], dtype=jnp.float32))
    assert jnp.array_equal(spec.action_high, jnp.array([1.0], dtype=jnp.float32))
    assert spec.truncation_policy == "terminate"
    assert jnp.array_equal(reset.state, jnp.int32(0))


@pytest.mark.parametrize(("action", "expected"), [(-0.1, 0), (0.0, 1), (0.1, 1)])
def test_step_maps_negative_and_nonnegative_actions_to_the_two_choices(action: float, expected: int) -> None:
    """The zero threshold maps tanh outputs to the two discrete actions.

    Zero belongs to the nonnegative action.
    """
    env: EnvProtocol[jax.Array, jax.Array, jax.Array, None] = _DiscreteEnv()
    wrapper = DiscretizeAction(env)

    transition = wrapper.step(
        jax.random.key(0),
        jnp.int32(0),
        jnp.array([action], dtype=jnp.float32),
    )

    assert jnp.array_equal(transition.info["action"], jnp.array(expected, dtype=jnp.int32))


def test_spec_rejects_environments_with_more_than_two_actions() -> None:
    """The binary mapping refuses action spaces it cannot represent.

    Three possible actions cannot be represented by the sign threshold.
    """
    env: EnvProtocol[jax.Array, jax.Array, jax.Array, None] = _DiscreteEnv(num_actions=3)

    with pytest.raises(ValueError, match="two-action"):
        DiscretizeAction(env).spec()
