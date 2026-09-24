from collections.abc import Mapping

import jax
import jax.numpy as jnp
import pytest
from experiment_definition.parameter import ParameterValue
from rl_agents.batch_trainer import EnvFactory
from rl_components.action_normalization import make_action_normalization_wrapper
from rl_components.env_protocol import EnvProtocol
from rl_envs.env_registry import (
    ParameterAccessor,
    environment_builder,
    gymnax_builder,
    make_env_factory,
    process_control_builder,
)
from rl_envs.process_control_bridge import make_process_control_env


@environment_builder
def _scrubber_builder(
    parameters: ParameterAccessor,
) -> tuple[EnvProtocol[jax.Array, object, jax.Array, object], object | None]:
    env = make_process_control_env(
        "h2s_scrubber",
        ph_sp_min=parameters.get("ph_sp_min", float),
        ph_sp_max=parameters.get_or("ph_sp_max", float, 10.5),
    )
    return env, None


def test_registered_builder_reads_per_run_parameters() -> None:
    """The selected builder receives typed values from this run's config.

    An omitted optional bound uses the builder's declared default.
    """
    factory = make_env_factory({"scrubber": _scrubber_builder})
    env, env_params = factory({"env_name": "scrubber", "ph_sp_min": 8})

    assert env_params is None
    spec = env.spec()
    assert spec.action_low is not None
    assert float(spec.action_low[0]) == pytest.approx(8.0)
    assert spec.action_high is not None
    assert float(spec.action_high[0]) == pytest.approx(10.5)


def test_builder_returns_its_wrapped_environment() -> None:
    """The factory preserves wrappers applied by a project-owned builder.

    The returned environment keeps the normalized action bounds.
    """
    factory = make_env_factory(
        {
            "chlorine": process_control_builder(
                "chlorine",
                wrapper=make_action_normalization_wrapper,
            ),
        }
    )
    env, env_params = factory({"env_name": "chlorine"})

    assert env_params is None
    spec = env.spec()
    assert spec.action_low is not None
    assert spec.action_high is not None
    assert jnp.array_equal(spec.action_low, jnp.array([-1.0], dtype=jnp.float32))
    assert jnp.array_equal(spec.action_high, jnp.array([1.0], dtype=jnp.float32))


def test_process_control_builder_passes_configured_values() -> None:
    """The process-control constructor forwards only selected parameters.

    A scenario selected for this run appears in the returned environment identity.
    """
    factory = make_env_factory(
        {
            "scrubber": process_control_builder(
                "h2s_scrubber",
                passthrough_keys=("scenario",),
            ),
        }
    )
    env, env_params = factory({"env_name": "scrubber", "scenario": "nominal"})

    assert env_params is None
    assert env.spec().id == "process_control:h2s_scrubber:nominal"


def test_factory_accepts_heterogeneous_environment_builders() -> None:
    """One typed factory dispatches to builders with distinct state types.

    Two Gymnax tasks produce different runtime state types alongside a process-control
    environment, without casts at the factory call site.
    """
    factory: EnvFactory = make_env_factory(
        {
            "cartpole": gymnax_builder("CartPole-v1"),
            "pendulum": gymnax_builder("Pendulum-v1", normalized=True),
            "chlorine": process_control_builder("chlorine"),
            "scrubber": _scrubber_builder,
        }
    )

    cartpole, cartpole_params = factory({"env_name": "cartpole"})
    pendulum, pendulum_params = factory({"env_name": "pendulum"})
    chlorine, chlorine_params = factory({"env_name": "chlorine"})
    scrubber, scrubber_params = factory({"env_name": "scrubber", "ph_sp_min": 8})

    cartpole_state = cartpole.reset(jax.random.key(0), cartpole_params).state
    pendulum_state = pendulum.reset(jax.random.key(1), pendulum_params).state
    assert type(cartpole_state) is not type(pendulum_state)

    assert chlorine_params is None
    assert chlorine.spec().id == "process_control:chlorine"
    assert scrubber_params is None
    scrubber_spec = scrubber.spec()
    assert scrubber_spec.action_low is not None
    assert float(scrubber_spec.action_low[0]) == pytest.approx(8.0)

    assert pendulum_params is not None
    spec = pendulum.spec(pendulum_params)
    assert spec.action_low is not None
    assert spec.action_high is not None
    assert jnp.array_equal(spec.action_low, jnp.array([-1.0], dtype=jnp.float32))
    assert jnp.array_equal(spec.action_high, jnp.array([1.0], dtype=jnp.float32))


@pytest.mark.parametrize(
    ("parameters", "error", "message"),
    [
        ({}, ValueError, "env_name"),
        ({"env_name": 1}, TypeError, "env_name"),
        ({"env_name": "missing"}, ValueError, "chlorine"),
    ],
)
def test_factory_rejects_invalid_environment_config(
    parameters: Mapping[str, ParameterValue], error: type[Exception], message: str
) -> None:
    """Malformed config fails before a project builder is selected.

    Missing names, wrong types, and unknown builder keys have distinct errors.
    """
    factory = make_env_factory({"chlorine": process_control_builder("chlorine")})

    with pytest.raises(error, match=message):
        factory(parameters)
