import jax
import jax.numpy as jnp
import pytest
from rl_components.action_normalization import make_action_normalization_wrapper
from rl_components.env_protocol import EnvProtocol
from rl_envs.env_registry import ENV_FAMILIES, make_env_factory, resolve_env


def test_process_control_family_resolves_a_working_env() -> None:
    """process_control:<name> resolves to a steppable EnvProtocol with no env params."""
    env, env_params = resolve_env("process_control:chlorine")

    assert env_params is None
    spec = env.spec()
    assert spec.observation_shape == (spec.observation_shape[0],)
    assert spec.action_shape == (1,)

    reset = env.reset(jax.random.PRNGKey(0))
    assert reset.observation.shape == spec.observation_shape
    step = env.step(jax.random.PRNGKey(1), reset.state, jnp.zeros(spec.action_shape, dtype=spec.action_dtype))
    assert step.observation.shape == spec.observation_shape


def test_gymnax_family_resolves_a_working_env() -> None:
    """gymnax:<id> resolves to a steppable EnvProtocol carrying env params."""
    env, env_params = resolve_env("gymnax:CartPole-v1")

    assert env_params is not None
    spec = env.spec(env_params)
    assert spec.observation_shape == (4,)
    assert spec.num_actions == 2

    reset = env.reset(jax.random.PRNGKey(0), env_params)
    assert reset.observation.shape == (4,)
    step = env.step(jax.random.PRNGKey(1), reset.state, jnp.array(0), env_params)
    assert step.observation.shape == (4,)


def test_wrapper_hook_is_applied_to_the_resolved_env() -> None:
    """The wrapper hook wraps the resolved env before it is returned."""
    env, _ = resolve_env("process_control:chlorine", wrapper=make_action_normalization_wrapper)

    spec = env.spec()
    assert spec.action_low is not None
    assert spec.action_high is not None
    assert jnp.array_equal(spec.action_low, jnp.array([-1.0], dtype=jnp.float32))
    assert jnp.array_equal(spec.action_high, jnp.array([1.0], dtype=jnp.float32))


def test_overrides_reach_the_process_control_builder() -> None:
    """Keyword overrides pass through to the named benchmark's config."""
    env, _ = resolve_env("process_control:h2s_scrubber", ph_sp_min=8.5, ph_sp_max=10.5)

    spec = env.spec()
    assert spec.action_low is not None
    assert float(spec.action_low[0]) == pytest.approx(8.5)
    assert spec.action_high is not None
    assert float(spec.action_high[0]) == pytest.approx(10.5)


def test_unknown_family_raises_value_error_naming_known_families() -> None:
    with pytest.raises(ValueError, match="process_control"):
        resolve_env("nonexistent:thing")


def test_missing_colon_raises_value_error() -> None:
    with pytest.raises(ValueError, match="namespaced"):
        resolve_env("chlorine")


def test_env_families_covers_process_control_and_gymnax() -> None:
    assert set(ENV_FAMILIES) == {"process_control", "gymnax"}


class TestMakeEnvFactory:
    def test_reads_env_name_from_the_config_mapping(self) -> None:
        """The factory reads ``env_name`` out of the mapping it is handed."""
        factory = make_env_factory()
        env, env_params = factory({"env_name": "process_control:chlorine", "seed": 0})

        assert env_params is None
        assert isinstance(env, EnvProtocol)

    def test_raises_type_error_when_env_name_is_absent(self) -> None:
        factory = make_env_factory()
        with pytest.raises(TypeError, match="env_name"):
            factory({"seed": 0})

    def test_raises_type_error_when_env_name_is_not_a_string(self) -> None:
        factory = make_env_factory()
        with pytest.raises(TypeError, match="env_name"):
            factory({"env_name": 1})
