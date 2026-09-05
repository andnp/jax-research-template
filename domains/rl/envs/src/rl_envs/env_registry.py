"""Resolve environments by namespaced name (``"<family>:<env>"``)."""

from __future__ import annotations

from collections.abc import Callable, Mapping

import gymnax
from experiment_definition import ParameterValue
from rl_components.env_protocol import EnvProtocol

from rl_envs.gymnax_bridge import make_gymnax_env
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


def make_env_factory(
    *,
    wrapper: Callable[[EnvProtocol], EnvProtocol] | None = None,
    **overrides: object,
) -> Callable[[Mapping[str, ParameterValue]], tuple[EnvProtocol, object | None]]:
    """Build a batch-trainer ``env_factory`` that reads ``"env_name"`` from its config mapping."""

    def factory(static_config: Mapping[str, ParameterValue]) -> tuple[EnvProtocol, object | None]:
        env_name = static_config.get("env_name")
        if not isinstance(env_name, str):
            raise TypeError(f"env_name must be text, got {env_name!r}")
        return resolve_env(env_name, wrapper=wrapper, **overrides)

    return factory
