"""Small (< 1 ms) unit tests for the experiment-definition schema module.

These tests exercise pure logic: DDL string contents, constant values, and
the structure of ALL_DDL — no I/O, no SQLite connection required.
"""

import pytest
from experiment_definition.schema import (
    ALL_DDL,
    COMPONENT_TYPES,
    CREATE_COMPONENT_VERSIONS,
    CREATE_COMPONENTS,
    CREATE_EXECUTION_RUNS,
    CREATE_EXECUTIONS,
    CREATE_EXPERIMENTS,
    CREATE_HYPERPARAM_CONFIGS,
    CREATE_LATEST_VERSIONS_VIEW,
    CREATE_METRIC_SPECS,
    CREATE_PARAMETER_SPECS,
    CREATE_RUNS,
    DEFAULT_DB_NAME,
    EXECUTION_STATUSES,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------


def test_component_types_complete() -> None:
    assert set(COMPONENT_TYPES) == {"ALGO", "ENV", "WRAPPER"}


def test_execution_statuses_complete() -> None:
    assert set(EXECUTION_STATUSES) == {"PENDING", "RUNNING", "COMPLETED", "FAILED", "INVALID"}


# ---------------------------------------------------------------------------
# DDL string contents — verify key structural keywords are present
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "ddl, expected_fragments",
    [
        (CREATE_COMPONENTS, ["Components", "AUTOINCREMENT", "ALGO", "ENV", "WRAPPER"]),
        (CREATE_COMPONENT_VERSIONS, ["ComponentVersions", "version_number", "code_snapshot_hash", "UNIQUE(component_id, version_number)"]),
        (CREATE_HYPERPARAM_CONFIGS, ["HyperparamConfigs", "hash", "json_blob", "vmap_zone_json"]),
        (CREATE_EXPERIMENTS, ["Experiments", "created_at"]),
        (CREATE_RUNS, ["Runs", "algo_version_id", "env_version_id", "hyper_id", "seed", "ablation", "UNIQUE("]),
        (CREATE_EXECUTIONS, ["Executions", "PENDING", "RUNNING", "COMPLETED", "FAILED", "INVALID", "jax_config_json", "git_commit", "git_diff_blob"]),
        (CREATE_EXECUTION_RUNS, ["ExecutionRuns", "execution_id", "run_id", "PRIMARY KEY"]),
        (CREATE_LATEST_VERSIONS_VIEW, ["ComponentLatestVersions", "MAX(version_number)", "GROUP BY component_id"]),
        (CREATE_PARAMETER_SPECS, ["ParameterSpecs", "experiment_id", "values_json", "is_static", "conditions_json"]),
        (CREATE_METRIC_SPECS, ["MetricSpecs", "experiment_id", "name", "type", "frequency", "UNIQUE(experiment_id, name)"]),
    ],
)
def test_ddl_contains_expected_fragments(ddl: str, expected_fragments: list[str]) -> None:
    for fragment in expected_fragments:
        assert fragment in ddl, f"Expected {fragment!r} in DDL:\n{ddl}"


def test_all_ddl_is_non_empty_list() -> None:
    assert isinstance(ALL_DDL, list)
    assert len(ALL_DDL) > 0


def test_all_ddl_covers_all_seven_tables() -> None:
    """Every required table DDL must appear in ALL_DDL."""
    required = [
        "Components",
        "ComponentVersions",
        "HyperparamConfigs",
        "Experiments",
        "Runs",
        "Executions",
        "ExecutionRuns",
        "ParameterSpecs",
        "MetricSpecs",
    ]
    combined = "\n".join(ALL_DDL)
    for table in required:
        assert table in combined, f"Table {table!r} missing from ALL_DDL"


def test_all_ddl_includes_latest_versions_view() -> None:
    combined = "\n".join(ALL_DDL)
    assert "ComponentLatestVersions" in combined


def test_all_ddl_entries_are_strings() -> None:
    for stmt in ALL_DDL:
        assert isinstance(stmt, str) and stmt.strip(), "ALL_DDL contains an empty or non-string entry"


def test_foreign_keys_declared_in_runs() -> None:
    """Runs table must reference both ComponentVersions and HyperparamConfigs."""
    assert "REFERENCES ComponentVersions(id)" in CREATE_RUNS
    assert "REFERENCES HyperparamConfigs(id)" in CREATE_RUNS
    assert "REFERENCES Experiments(id)" in CREATE_RUNS


def test_vmap_zone_json_in_hyperparam_configs() -> None:
    """JAX vmap-zone metadata column must be present in HyperparamConfigs."""
    assert "vmap_zone_json" in CREATE_HYPERPARAM_CONFIGS


def test_jax_config_json_in_executions() -> None:
    """JAX runtime metadata column must be present in Executions."""
    assert "jax_config_json" in CREATE_EXECUTIONS


def test_default_db_name_is_sqlite_file() -> None:
    assert DEFAULT_DB_NAME == "experiments.sqlite"
