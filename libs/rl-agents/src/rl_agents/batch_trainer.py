"""One :class:`~research_runner.BatchTrainer` for every algorithm that declares its own hypers.

Each ``*_batch.py`` module used to hand-restate three facts an algorithm already owns:
which fields are traceable (its ``*Hypers`` struct), whether each is ``int`` or ``float``
(its ``Config`` dataclass's annotation), and each field's default (``Config()``). An
:class:`AlgorithmSpec` names the three classes and this module derives the rest, including
``Config(**{name.upper(): value})`` construction -- viable with no per-algorithm mapper
because a prior rename made every ``Config`` field the upcase of its experiment parameter
name, with GAC's ``ACTOR_LR`` the one field this module's own dynamic-key derivation
(below) resolves consistently as ``actor_lr``, not the hand-written ``actor_learning_rate``.

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

SAC is deliberately absent: ``SACConfig.LR`` has not been renamed to ``LEARNING_RATE``,
so the uniform upcase rule does not hold for it yet.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from types import UnionType
from typing import Literal, cast, get_args, get_origin, get_type_hints

import jax
from experiment_definition import ParameterValue
from research_runner import BatchTrainer, BoundBatch, ExecutionContext, ExecutionResult, ParameterSchema
from rl_components.batch import run_batch
from rl_components.env_protocol import EnvProtocol
from rl_components.loop import run
from rl_components.metrics import project_metrics

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
        return ParameterSchema(dynamic_keys=dynamic_keys | {GAMMA})


def _scalar_type(hint: object) -> type | None:
    """The plain ``int``/``float``/``str``/``bool`` a ``Config`` field's annotation reduces to."""
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
        default = getattr(defaults, field_name)
        scalar_type = _scalar_type(hint)
        value = default if scalar_type is None else batch.static_or(key, scalar_type, default)
        kwargs[field_name] = cast(hint, value) if get_origin(hint) is Literal else value
    return kwargs


class GenericBatchTrainer:
    """Bind an :class:`AlgorithmSpec` to an environment factory and run its seeded batch."""

    def __init__(self, spec: AlgorithmSpec, env_factory: EnvFactory) -> None:
        self._spec = spec
        self.parameter_schema = spec.parameter_schema
        self._env_factory = env_factory

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

        def train_one(dynamic: Mapping[str, jax.Array], key: jax.Array):
            config = spec.config_cls(**_config_kwargs(spec.config_cls, batch, dynamic))
            _, metrics = run(
                spec.agent_cls(config),
                env,
                key,
                steps=cast(int, config.TOTAL_TIMESTEPS),
                gamma=cast(float, dynamic.get(GAMMA, batch.static_or(GAMMA, float, 0.99))),
                env_params=env_params,
            )
            projected = project_metrics(metrics)
            reported: tuple[str, ...] = spec.agent_cls.REPORTED_METRICS
            return {"reward": projected["reward"], **{name: projected[name] for name in reported}}

        metrics = run_batch(train_one, dynamic_axes, batch.seeds)
        return ExecutionResult(
            metadata={
                "algorithm": spec.name,
                "env_name": batch.static(ENV_NAME, str),
                "seed_values": list(batch.seeds),
            },
            metric_curves={name: values.tolist() for name, values in metrics.items()},
        )


def make_batch_trainer(spec: AlgorithmSpec, env_factory: EnvFactory) -> BatchTrainer:
    return GenericBatchTrainer(spec, env_factory)
