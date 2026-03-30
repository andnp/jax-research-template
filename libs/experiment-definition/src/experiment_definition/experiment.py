"""Fluent builder for experiment definitions.

Usage::

    from experiment_definition import Experiment, Component

    exp = Experiment("Policy Gradient Ablations")

    ppo = Component(name="PPO", path="libs/rl-agents/src/rl_agents/ppo.py", type="ALGO")
    env = Component(name="CartPole", path="libs/rl-components/src/rl_components/envs.py", type="ENV")

    exp.add_parameter("seed", range(5))
    exp.add_parameter("gamma", [0.99])

    with exp.for_component(ppo):
        exp.add_parameter("lr", [1e-3, 3e-4])
        exp.add_parameter("use_gae", [True, False])
        with exp.when(use_gae=True):
            exp.add_parameter("gae_lambda", [0.9, 0.95])

    exp.add_ablation("no_gae", {"use_gae": False})
    exp.add_metric("reward", type="float", frequency="per_episode")
    exp.sync("experiments.sqlite")
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Generator, Iterable

from .ablation import AblationSpec
from .component import Component
from .metric import MetricFrequency, MetricSpec, MetricType
from .parameter import ParameterSpec, ParameterValue


@dataclass
class _ExperimentState:
    """Plain container holding all accumulated definition data."""

    name: str
    description: str | None
    components: list[Component] = field(default_factory=list)
    parameters: list[ParameterSpec] = field(default_factory=list)
    metrics: list[MetricSpec] = field(default_factory=list)
    ablations: list[AblationSpec] = field(default_factory=list)


class Experiment:
    """Fluent builder for declaring an experiment's search space and metrics.

    Args:
        name: Human-readable experiment name.
        description: Optional longer description stored in the DB.

    Example::

        exp = Experiment("My Sweep", description="Testing LR schedules")
        exp.add_parameter("lr", [1e-3, 3e-4])
        exp.add_metric("reward", type="float", frequency="per_episode")
        exp.sync("experiments.sqlite")
    """

    def __init__(self, name: str, description: str | None = None) -> None:
        self._state = _ExperimentState(name=name, description=description)
        self._component_scope: Component | None = None
        self._condition_stack: list[dict[str, ParameterValue]] = []

    # ── Parameter API ─────────────────────────────────────────────────────────

    def add_parameter(
        self,
        name: str,
        values: Iterable[ParameterValue],
        *,
        is_static: bool = False,
    ) -> "Experiment":
        """Add a hyperparameter sweep axis.

        Args:
            name: Parameter name.
            values: Iterable of candidate values.
            is_static: When ``True`` the parameter triggers JAX recompilation;
                the runner groups PENDING work into separate JIT kernels per
                unique static configuration.

        Returns:
            ``self`` for optional chaining.
        """
        conditions: dict[str, ParameterValue] = {}
        for frame in self._condition_stack:
            conditions.update(frame)

        # Register the component if not yet tracked
        if self._component_scope and self._component_scope not in self._state.components:
            self._state.components.append(self._component_scope)

        spec = ParameterSpec(
            name=name,
            values=list(values),
            is_static=is_static,
            component=self._component_scope,
            conditions=conditions,
        )
        self._state.parameters.append(spec)
        return self

    # ── Scoping context managers ──────────────────────────────────────────────

    @contextmanager
    def for_component(self, component: Component) -> Generator[None, None, None]:
        """Scope subsequent ``add_parameter`` calls to a specific component.

        Args:
            component: The component to scope parameters to.

        Yields:
            Nothing — use as ``with exp.for_component(ppo):``.
        """
        if component not in self._state.components:
            self._state.components.append(component)
        previous = self._component_scope
        self._component_scope = component
        try:
            yield
        finally:
            self._component_scope = previous

    @contextmanager
    def when(self, **conditions: ParameterValue) -> Generator[None, None, None]:
        """Add conditional parameters triggered only when ``conditions`` match.

        Args:
            **conditions: Key/value pairs that must all hold in a configuration
                for the enclosed parameters to be active.

        Yields:
            Nothing — use as ``with exp.when(use_gae=True):``.
        """
        self._condition_stack.append(dict(conditions))
        try:
            yield
        finally:
            self._condition_stack.pop()

    # ── Ablations ─────────────────────────────────────────────────────────────

    def add_ablation(self, name: str, overrides: dict[str, ParameterValue]) -> "Experiment":
        """Register a named ablation that clones the search space with fixed overrides.

        Args:
            name: Short identifier (e.g. ``"no_gae"``).
            overrides: Parameter values to fix for this ablation.

        Returns:
            ``self`` for optional chaining.
        """
        self._state.ablations.append(AblationSpec(name=name, overrides=overrides))
        return self

    # ── Metrics ───────────────────────────────────────────────────────────────

    def add_metric(
        self,
        name: str,
        *,
        type: str,  # noqa: A002
        frequency: str,
    ) -> "Experiment":
        """Whitelist a metric for collection by the instrumentation layer.

        Metrics not declared here are ignored by the collector, saving compute
        and disk space.

        Args:
            name: Identifier used by the collector (e.g. ``"reward"``).
            type: Scalar data type — one of ``"float"``, ``"int"``, ``"bool"``.
            frequency: Collection cadence — one of ``"per_episode"``,
                ``"per_update"``, ``"eval_only"``.

        Returns:
            ``self`` for optional chaining.
        """
        self._state.metrics.append(
            MetricSpec(
                name=name,
                type=MetricType(type),
                frequency=MetricFrequency(frequency),
            )
        )
        return self

    # ── Persistence ───────────────────────────────────────────────────────────

    def sync(self, db_path: Path | str) -> None:
        """Persist the experiment definition to a SQLite database.

        Creates the file and schema (ADR 008) if they do not exist.
        Existing experiments with the same name are reopened additively:
        matching metadata is reused, matching logical runs are left intact,
        and newly declared runs are appended.

        Args:
            db_path: Path to the target SQLite file.
        """
        from .db import sync_to_db

        sync_to_db(db_path, self._state)
