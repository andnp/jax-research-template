"""One :class:`~research_runner.BatchTrainer` for every algorithm that declares its own hypers.

Each ``*_batch.py`` module used to hand-restate three facts an algorithm already owns:
which fields are traceable (its ``*Hypers`` struct), whether each is ``int`` or ``float``
(its ``Config`` dataclass's annotation), and each field's default (``Config()``). An
:class:`AlgorithmSpec` names the three classes and this module derives the rest, including
``Config(**{name.upper(): value})`` construction -- viable with no per-algorithm mapper
because every ``Config`` field is the upcase of its experiment parameter name.

Some fields are swept without ever being traced by ``step``: they are read once, by
``init``, which runs inside the vmapped region just as much as ``step`` does, so a
per-arm value works there too. An algorithm declares these via ``INIT_SWEPT_FIELDS``,
alongside ``REPORTED_METRICS``, and :attr:`AlgorithmSpec.parameter_schema` unions them in.

``gamma`` never appears on a ``Config`` or a ``Hypers`` struct: it is a parameter of
:func:`rl_components.loop.run` alone, so every spec's dynamic keys add it by hand.

Two exceptions this module encodes rather than derives:

- ``greedy_ac.GACConfig.ACTOR_PERCENTILE`` and ``UNIFORM_WEIGHT`` must never be swept:
  both size an array through a Python ``int()`` cast (see ``GACHypers``'s own docstring),
  so tracing them would need a redesign. They are excluded by name as a backstop, in case
  a future ``Hypers`` change reintroduces them.
- ``H_REGULARIZATION`` means different things per algorithm -- GAC's is a numeric weight,
  QRC's a categorical ``"adamw"``/``"l2"`` optimizer choice -- which only matters once a
  mixed-algorithm dispatch (``batch_registry.py``) has to reconcile the two. That reconciliation
  is out of scope here; each algorithm's own derivation is unambiguous in isolation.
- PPO reports its return curve under ``"returned_episode_returns"``, held from
  ``loop/episode_return`` by :func:`rl_components.metrics.episode_return_curve`, rather than
  under ``"reward"`` from ``loop/reward``: PPO is on-policy and its horizon spans many
  episodes, so a per-step reward is not the return anyone wants plotted. It also returns its
  final policy parameters as an artifact, which no other algorithm here does. Both are
  encoded by name (``spec.name == "ppo"``) rather than derived, matching how the other named
  exceptions in this module are handled.

SAC is no longer absent: ``SACConfig.LR`` was renamed to ``LEARNING_RATE``, so the uniform
upcase rule now holds for it too.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from types import UnionType
from typing import Any, Literal, TypeAliasType, cast, get_args, get_origin, get_type_hints

import jax
from experiment_definition import ParameterValue
from research_runner import BatchTrainer, BoundBatch, ExecutionContext, ExecutionResult, ParameterSchema
from rl_components.batch import run_batch
from rl_components.env_protocol import EnvProtocol
from rl_components.loop import run
from rl_components.metrics import episode_return_curve, project_metrics

EnvFactory = Callable[
    [Mapping[str, ParameterValue]],
    tuple[EnvProtocol[jax.Array, object, jax.Array, object], object | None],
]

GAMMA = "gamma"
ENV_NAME = "env_name"

# Cast through Python int() to size an array in greedy_ac's proposal/ranking path (see
# GACHypers's own docstring); sweeping them would need a redesign that is out of scope.
_NEVER_DYNAMIC = frozenset({"actor_percentile", "uniform_weight", "total_timesteps"})
"""Hypers fields that must stay static however they are declared.

``actor_percentile`` and ``uniform_weight`` are unsweepable by necessity. ``total_timesteps``
rides ``ReplayQHypers``/``QRCHypers`` because the epsilon schedule divides by it, but it is
also the ``lax.scan`` length ``run`` is given, which has to be a Python int -- a traced value
there fails to compile. The scan boundary wins, so it is resolved statically.
"""


@dataclass(frozen=True, slots=True)
class AlgorithmSpec:
    """The three classes an algorithm already declares, from which everything else follows."""

    name: str
    agent_cls: type
    config_cls: type
    hypers_cls: type

    @property
    def parameter_schema(self) -> ParameterSchema:
        fields = get_type_hints(self.hypers_cls)
        dynamic_keys = frozenset(name.lower() for name in fields) - _NEVER_DYNAMIC
        init_swept = frozenset(name.lower() for name in self.agent_cls.INIT_SWEPT_FIELDS)
        return ParameterSchema(dynamic_keys=dynamic_keys | init_swept | {GAMMA})


def _resolve_alias(hint: object) -> object:
    """Unwrap a PEP 695 ``type X = ...`` alias (e.g. ``qrc.HRegularization``).

    ``get_type_hints`` leaves such an alias as an opaque ``TypeAliasType`` rather than
    inlining its value, so ``get_origin`` sees neither ``Literal`` nor anything else
    useful without this.
    """
    while isinstance(hint, TypeAliasType):
        hint = hint.__value__
    return hint


def _scalar_type(hint: object) -> type | None:
    """The plain ``int``/``float``/``str``/``bool`` a ``Config`` field's annotation reduces to."""
    hint = _resolve_alias(hint)
    if hint in (int, float, str, bool):
        return cast(type, hint)
    origin = get_origin(hint)
    if origin is Literal:
        return str
    if origin is UnionType:
        args = [arg for arg in get_args(hint) if arg is not type(None)]
        if len(args) == 1:
            return _scalar_type(args[0])
    return None


_MISSING = object()


def _config_kwargs(config_cls: type, batch: BoundBatch, dynamic: Mapping[str, jax.Array]) -> dict[str, object]:
    defaults = config_cls()
    kwargs: dict[str, object] = {}
    for field_name, hint in get_type_hints(config_cls).items():
        key = field_name.lower()
        if key == "env_name":
            kwargs[field_name] = batch.static(ENV_NAME, str)
            continue
        if key in dynamic:
            kwargs[field_name] = dynamic[key]
            continue
        # An `X | None` field (e.g. SAC's TARGET_ENTROPY) explicitly set to `None` in the
        # static config means "derive it at init", and must reach the agent as `None` rather
        # than fail `static_or`'s type coercion -- which does not accept `None` for a `float`
        # field. This is distinct from the key being absent, which falls through to the
        # field's own default below (also `None` for these fields, so the outcome agrees).
        if batch.static_config.get(key, _MISSING) is None:
            kwargs[field_name] = None
            continue
        default = getattr(defaults, field_name)
        resolved_hint = _resolve_alias(hint)
        scalar_type = _scalar_type(hint)
        value = default if scalar_type is None else batch.static_or(key, scalar_type, default)
        if get_origin(resolved_hint) is Literal:
            allowed = get_args(resolved_hint)
            if value not in allowed:
                raise ValueError(f"unsupported {config_cls.__name__}.{field_name}: {value!r}; expected one of {allowed}")
        kwargs[field_name] = value
    return kwargs


_PPO = "ppo"
"""The one algorithm whose report shape genuinely differs -- see the module docstring."""


class GenericBatchTrainer:
    """Bind an :class:`AlgorithmSpec` to an environment factory and run its seeded batch."""

    def __init__(self, spec: AlgorithmSpec, env_factory: EnvFactory, *, extra_metric_names: tuple[str, ...] = ()) -> None:
        self._spec = spec
        self.parameter_schema = spec.parameter_schema
        self._env_factory = env_factory
        self._extra_metric_names = extra_metric_names

    def __call__(self, context: ExecutionContext) -> ExecutionResult:
        spec = self._spec
        batch = context.bind(self.parameter_schema)
        env, env_params = self._env_factory(batch.static_config)
        integer_keys = {name.lower() for name, hint in get_type_hints(spec.hypers_cls).items() if hint is jax.Array} & {
            name for name, hint in get_type_hints(spec.config_cls).items() if hint is int
        }
        dynamic_axes = {
            key: batch.dynamic(key, int if key in integer_keys else float)
            for key in self.parameter_schema.dynamic_keys
            if key in batch.dynamic_axes
        }
        is_ppo = spec.name == _PPO

        def train_one(dynamic: Mapping[str, jax.Array], key: jax.Array):
            config = spec.config_cls(**_config_kwargs(spec.config_cls, batch, dynamic))
            final_state, metrics = run(
                spec.agent_cls(config),
                env,
                key,
                steps=cast(int, config.TOTAL_TIMESTEPS),
                gamma=cast(float, dynamic.get(GAMMA, batch.static_or(GAMMA, float, 0.99))),
                env_params=env_params,
            )
            reported: tuple[str, ...] = spec.agent_cls.REPORTED_METRICS
            if is_ppo:
                curves = {
                    "returned_episode_returns": episode_return_curve(cast(Mapping[str, jax.Array], metrics)),
                    **{name: metrics[name] for name in reported},
                    **{name: metrics[name] for name in self._extra_metric_names},
                }
                return cast(Any, final_state).agent_state.train_state.params, curves
            projected = project_metrics(metrics)
            return {
                "reward": projected["reward"],
                **{name: projected[name] for name in reported},
                **{name: metrics[name] for name in self._extra_metric_names},
            }

        result = run_batch(train_one, dynamic_axes, batch.seeds)
        if is_ppo:
            final_params, curves = cast(tuple[object, Mapping[str, jax.Array]], result)
            artifacts: dict[str, object] = {"final_policy_params": final_params}
        else:
            curves = cast(Mapping[str, jax.Array], result)
            artifacts = {}
        return ExecutionResult(
            metadata={
                "algorithm": spec.name,
                "env_name": batch.static(ENV_NAME, str),
                "seed_values": list(batch.seeds),
            },
            metric_curves={name: values.tolist() for name, values in curves.items()},
            artifacts=artifacts,
        )


def make_batch_trainer(spec: AlgorithmSpec, env_factory: EnvFactory, *, extra_metric_names: tuple[str, ...] = ()) -> BatchTrainer:
    return GenericBatchTrainer(spec, env_factory, extra_metric_names=extra_metric_names)
