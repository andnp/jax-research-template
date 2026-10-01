"""Synthetic experiment builders for analysis tests."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from experiment_definition.component import ComponentType
from experiment_definition.db import DatabaseManager
from experiment_definition.parameter import ParameterValue
from research_instrument.scalar import write_execution_scalars, write_seed_batch_scalars
from research_instrument.seed_batch import write_seed_batch_curve
from research_instrument.sqlite_backend import open_metrics_connection


@dataclass(frozen=True)
class SyntheticComponentVersion:
    """Component version metadata for a synthetic experiment."""

    name: str
    component_type: ComponentType
    code_snapshot_hash: str
    version_id: int | None = None
    version_number: int | None = None


@dataclass(frozen=True)
class SyntheticRunSpec:
    """Run data; component overrides are ordered as algorithm and environment IDs."""

    seed: int
    hyperparameters: Mapping[str, ParameterValue]
    curve_metrics: Mapping[str, Sequence[float]] = field(default_factory=dict)
    run_scalars: Mapping[str, float] = field(default_factory=dict)
    ablation: str | None = None
    component_version_ids: tuple[int | None, int | None] | None = None


@dataclass(frozen=True)
class SyntheticRun:
    """Database identity of a synthetic run."""

    run_id: int
    execution_id: int
    seed: int


class SyntheticExperiment:
    """Build an experiment database and persist metrics through core writers."""

    def __init__(
        self,
        root: Path,
        name: str,
        *,
        component_versions: Sequence[SyntheticComponentVersion] | None = None,
    ) -> None:
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)
        self.name = name
        self.db_path = self.root / "experiments.sqlite"
        self.metrics_path = self.root / "metrics.sqlite"
        self._runs: dict[int, SyntheticRun] = {}

        versions = (
            component_versions
            if component_versions is not None
            else (
                SyntheticComponentVersion("ppo", ComponentType.ALGO, "algo-hash"),
                SyntheticComponentVersion("cartpole", ComponentType.ENV, "env-hash"),
            )
        )
        self.component_version_ids: dict[SyntheticComponentVersion, int] = {}
        component_ids: dict[tuple[str, ComponentType], int] = {}
        versions_by_type: dict[ComponentType, list[int]] = {}
        with DatabaseManager(self.db_path) as database:
            database.initialize()
            self.experiment_id = database.add_experiment(name)
            for version in versions:
                component_key = (version.name, version.component_type)
                component_id = component_ids.get(component_key)
                if component_id is None:
                    component_id = database.add_component(version.name, version.component_type.value)
                    component_ids[component_key] = component_id
                version_id = self._add_component_version(database, component_id, version)
                self.component_version_ids[version] = version_id
                versions_by_type.setdefault(version.component_type, []).append(version_id)

        self.algorithm_version_id = next(iter(versions_by_type.get(ComponentType.ALGO, [])), None)
        self.environment_version_id = next(iter(versions_by_type.get(ComponentType.ENV, [])), None)
        self._default_component_version_ids = (self.algorithm_version_id, self.environment_version_id)

        metrics_connection = open_metrics_connection(self.metrics_path)
        metrics_connection.close()

    def add_execution(
        self,
        runs: Sequence[SyntheticRunSpec],
        *,
        status: str = "COMPLETED",
        record_artifacts: bool = True,
        start_time: str | None = None,
        execution_scalars: Mapping[str, float] | None = None,
    ) -> tuple[SyntheticRun, ...]:
        """Create an execution, its runs, and any curve or scalar observations."""
        identities: list[SyntheticRun] = []
        with DatabaseManager(self.db_path) as database:
            execution_id = database.add_execution()
            for run in runs:
                version_ids = run.component_version_ids or self._default_component_version_ids
                hyper_id = database.add_hyperparam_config({**run.hyperparameters, "seed": run.seed})
                run_id = database.add_run(
                    self.experiment_id,
                    version_ids[0],
                    version_ids[1],
                    hyper_id,
                    run.seed,
                    run.ablation,
                )
                database.link_execution_run(execution_id, run_id)
                identity = SyntheticRun(run_id, execution_id, run.seed)
                identities.append(identity)
                self._runs[run_id] = identity

            if record_artifacts:
                database.record_execution_artifacts(
                    execution_id,
                    str(self.root / "executions" / str(execution_id)),
                    metadata={"metrics_db_path": str(self.metrics_path)},
                )
            database.update_execution_status(execution_id, status, start_time=start_time)

        for run, identity in zip(runs, identities, strict=True):
            for metric_name, values in run.curve_metrics.items():
                write_seed_batch_curve(
                    self.metrics_path,
                    [values],
                    metric_name=metric_name,
                    experiment_id=self.experiment_id,
                    execution_id=identity.execution_id,
                    run_ids=[identity.run_id],
                    seed_ids=[identity.seed],
                )
            self.write_run_scalars(identity.run_id, run.run_scalars)

        if execution_scalars:
            write_execution_scalars(
                self.metrics_path,
                execution_scalars,
                experiment_id=self.experiment_id,
                execution_id=execution_id,
            )
        return tuple(identities)

    def write_run_scalars(self, run_id: int, metrics: Mapping[str, float]) -> None:
        """Write scalar values for a run already created by this builder."""
        run = self._runs[run_id]
        write_seed_batch_scalars(
            self.metrics_path,
            {name: (value,) for name, value in metrics.items()},
            experiment_id=self.experiment_id,
            execution_id=run.execution_id,
            run_ids=[run.run_id],
            seed_ids=[run.seed],
        )

    @staticmethod
    def _add_component_version(
        database: DatabaseManager,
        component_id: int,
        component: SyntheticComponentVersion,
    ) -> int:
        if component.version_id is None and component.version_number is None:
            return database.add_component_version(component_id, component.code_snapshot_hash)

        version_number = component.version_number
        if version_number is None:
            row = database.conn.execute(
                "SELECT COALESCE(MAX(version_number), 0) + 1 FROM ComponentVersions WHERE component_id = ?",
                (component_id,),
            ).fetchone()
            if row is None:
                raise RuntimeError("Unable to determine the next component version number.")
            version_number = int(row[0])

        if component.version_id is None:
            cursor = database.conn.execute(
                "INSERT INTO ComponentVersions(component_id, version_number, code_snapshot_hash) "
                "VALUES (?, ?, ?)",
                (component_id, version_number, component.code_snapshot_hash),
            )
        else:
            # Explicit IDs let tests select historical versions with VersionSelection.exact.
            cursor = database.conn.execute(
                "INSERT INTO ComponentVersions(id, component_id, version_number, code_snapshot_hash) "
                "VALUES (?, ?, ?, ?)",
                (component.version_id, component_id, version_number, component.code_snapshot_hash),
            )
        database.conn.commit()
        if cursor.lastrowid is None:
            raise RuntimeError("Insert failed: no component version ID returned.")
        return int(cursor.lastrowid)
