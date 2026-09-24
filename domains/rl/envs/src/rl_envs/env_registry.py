"""Build environment factories from project-owned environment builders."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Protocol, cast

import gymnax
import jax
from experiment_definition.parameter import ParameterValue
from research_runner.types import get_parameter
from rl_components.env_protocol import EnvProtocol

from rl_envs.gymnax_bridge import (
    GymnaxParams,
    GymnaxState,
    make_gymnax_env,
    make_normalized_gymnax_env,
)
from rl_envs.process_control_bridge import make_process_control_env


def _resolve_process_control(env: str, **overrides: object) -> tuple[EnvProtocol, None]:
    return make_process_control_env(env, **overrides), None


def _resolve_gymnax(env: str, **overrides: object) -> tuple[EnvProtocol, object]:
    raw_env, env_params = gymnax.make(env, **overrides)
    return make_gymnax_env(raw_env), env_params


ENV_FAMILIES: Mapping[str, Callable[..., tuple[EnvProtocol, object | None]]] = {
    "process_control": _resolve_process_control,
    "gymnax": _resolve_gymnax,
}


# Retain this resolver until the remaining project test migrates to project-owned builders.
def resolve_env(
    name: str,
    *,
    wrapper: Callable[[EnvProtocol], EnvProtocol] | None = None,
    **overrides: object,
) -> tuple[EnvProtocol, object | None]:
    """Resolve ``"<family>:<env>"`` to an environment, applying ``wrapper`` if given."""
    if ":" not in name:
        raise ValueError(f"env name must be namespaced as '<family>:<env>', got {name!r}; known families: {sorted(ENV_FAMILIES)}")
    family, env = name.split(":", 1)
    builder = ENV_FAMILIES.get(family)
    if builder is None:
        raise ValueError(f"unknown environment family {family!r}; known families: {sorted(ENV_FAMILIES)}")
    env_instance, env_params = builder(env, **overrides)
    if wrapper is not None:
        env_instance = wrapper(env_instance)
    return env_instance, env_params


class ParameterAccessor(Protocol):
    """Read a run parameter with an explicit value type."""

    def get[T](self, name: str, expected_type: type[T]) -> T: ...

    def get_or[T](self, name: str, expected_type: type[T], default: T) -> T: ...

    def get_value(self, name: str) -> ParameterValue: ...


class EnvironmentBuilder[EnvStateT, EnvParamsT](Protocol):
    """Build one environment while preserving its state and parameter types."""

    def __call__(
        self,
        parameters: ParameterAccessor,
    ) -> tuple[EnvProtocol[jax.Array, EnvStateT, jax.Array, EnvParamsT], EnvParamsT | None]: ...


class ErasedEnvironmentBuilder(Protocol):
    """Build an environment behind the batch trainer's erased state boundary."""

    def __call__(
        self,
        parameters: ParameterAccessor,
    ) -> tuple[EnvProtocol[jax.Array, object, jax.Array, object], object | None]: ...


def environment_builder[S, P](builder: EnvironmentBuilder[S, P]) -> ErasedEnvironmentBuilder:
    """Erase a typed builder's state and parameter types for the batch trainer."""

    def erased_builder(
        parameters: ParameterAccessor,
    ) -> tuple[EnvProtocol[jax.Array, object, jax.Array, object], object | None]:
        env, env_params = builder(parameters)
        # The trainer feeds a state and params only back to the environment that produced them.
        return cast(EnvProtocol[jax.Array, object, jax.Array, object], env), env_params

    return erased_builder


class _MappingParameterAccessor:
    def __init__(self, parameters: Mapping[str, ParameterValue]) -> None:
        self._parameters = parameters

    def get[T](self, name: str, expected_type: type[T]) -> T:
        return get_parameter(self._parameters, name, expected_type)

    def get_or[T](self, name: str, expected_type: type[T], default: T) -> T:
        if name not in self._parameters:
            return default
        return self.get(name, expected_type)

    def get_value(self, name: str) -> ParameterValue:
        try:
            return self._parameters[name]
        except KeyError as error:
            raise ValueError(f"missing parameter {name!r}") from error


def gymnax_builder[EnvStateT: GymnaxState, EnvParamsT: GymnaxParams](
    env_id: str,
    *,
    normalized: bool = False,
    wrapper: Callable[
        [EnvProtocol[jax.Array, EnvStateT, jax.Array, EnvParamsT]],
        EnvProtocol[jax.Array, EnvStateT, jax.Array, EnvParamsT],
    ]
    | None = None,
) -> ErasedEnvironmentBuilder:
    """Build a Gymnax environment, optionally normalizing actions or applying a wrapper."""

    def builder(
        parameters: ParameterAccessor,
    ) -> tuple[EnvProtocol[jax.Array, EnvStateT, jax.Array, EnvParamsT], EnvParamsT | None]:
        del parameters
        raw_env, env_params = gymnax.make(env_id)
        env = make_normalized_gymnax_env(raw_env) if normalized else make_gymnax_env(raw_env)
        if wrapper is not None:
            env = wrapper(env)
        return env, env_params

    return environment_builder(builder)


def process_control_builder(
    env_id: str,
    *,
    passthrough_keys: tuple[str, ...] = (),
    wrapper: Callable[
        [EnvProtocol[jax.Array, object, jax.Array, object]],
        EnvProtocol[jax.Array, object, jax.Array, object],
    ]
    | None = None,
) -> ErasedEnvironmentBuilder:
    """Build a process-control environment with selected config values and a wrapper."""

    def builder(
        parameters: ParameterAccessor,
    ) -> tuple[EnvProtocol[jax.Array, object, jax.Array, object], None]:
        overrides = {key: parameters.get_value(key) for key in passthrough_keys}
        env = make_process_control_env(env_id, **overrides)
        if wrapper is not None:
            env = wrapper(env)
        return env, None

    return environment_builder(builder)


def make_env_factory(
    builders: Mapping[str, ErasedEnvironmentBuilder],
) -> Callable[
    [Mapping[str, ParameterValue]],
    tuple[EnvProtocol[jax.Array, object, jax.Array, object], object | None],
]:
    """Dispatch each run's typed environment parameters to its project builder."""

    def factory(
        parameters: Mapping[str, ParameterValue],
    ) -> tuple[EnvProtocol[jax.Array, object, jax.Array, object], object | None]:
        accessor = _MappingParameterAccessor(parameters)
        env_name = accessor.get("env_name", str)
        try:
            builder = builders[env_name]
        except KeyError as error:
            raise ValueError(f"unknown environment {env_name!r}; known environments: {sorted(builders)}") from error
        return builder(accessor)

    return factory
