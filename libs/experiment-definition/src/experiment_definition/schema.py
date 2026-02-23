"""
SQL DDL for the master experiment database.

Schema design follows ADR 008 (Relational Experiment Schema) and ADR 007
(Declarative Experiment Management). All tables use INTEGER PRIMARY KEY
for SQLite rowid aliasing (fast lookups, no UUID overhead in hot paths).
"""

# ---------------------------------------------------------------------------
# Enum-like string constants referenced in DDL CHECK constraints
# ---------------------------------------------------------------------------

COMPONENT_TYPES = ("ALGO", "ENV", "WRAPPER")
EXECUTION_STATUSES = ("PENDING", "RUNNING", "COMPLETED", "FAILED", "INVALID")

# ---------------------------------------------------------------------------
# Table DDL
# ---------------------------------------------------------------------------

CREATE_COMPONENTS = """
CREATE TABLE IF NOT EXISTS Components (
    id   INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT    NOT NULL UNIQUE,
    type TEXT    NOT NULL CHECK(type IN ('ALGO', 'ENV', 'WRAPPER'))
);
"""

# version_number is per-component (not global). It is managed by the
# DatabaseManager.add_component_version() method, which sets it to
# MAX(version_number) + 1 for the same component_id.
CREATE_COMPONENT_VERSIONS = """
CREATE TABLE IF NOT EXISTS ComponentVersions (
    id                 INTEGER PRIMARY KEY AUTOINCREMENT,
    component_id       INTEGER NOT NULL REFERENCES Components(id),
    version_number     INTEGER NOT NULL,
    created_at         TEXT    NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
    code_snapshot_hash TEXT    NOT NULL,
    notes              TEXT,
    UNIQUE(component_id, version_number)
);
"""

# vmap_zone_json stores JAX-native batching metadata:
#   {"static_keys": ["arch", "env_name"], "dynamic_keys": ["lr", "seed"]}
# static_keys require recompilation; dynamic_keys are vmap-able at runtime.
CREATE_HYPERPARAM_CONFIGS = """
CREATE TABLE IF NOT EXISTS HyperparamConfigs (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    hash           TEXT    NOT NULL UNIQUE,
    json_blob      TEXT    NOT NULL,
    vmap_zone_json TEXT
);
"""

CREATE_EXPERIMENTS = """
CREATE TABLE IF NOT EXISTS Experiments (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    name        TEXT    NOT NULL UNIQUE,
    description TEXT,
    created_at  TEXT    NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
);
"""

# A Run is the *intent*: a unique (experiment, algo_version, env_version,
# hypers, seed) combination. It is satisfied by one or more Executions.
CREATE_RUNS = """
CREATE TABLE IF NOT EXISTS Runs (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    experiment_id   INTEGER NOT NULL REFERENCES Experiments(id),
    algo_version_id INTEGER NOT NULL REFERENCES ComponentVersions(id),
    env_version_id  INTEGER NOT NULL REFERENCES ComponentVersions(id),
    hyper_id        INTEGER NOT NULL REFERENCES HyperparamConfigs(id),
    seed            INTEGER NOT NULL,
    UNIQUE(experiment_id, algo_version_id, env_version_id, hyper_id, seed)
);
"""

# An Execution is the *physical event*: a single process (possibly covering
# many logical Runs via jax.vmap). jax_config_json stores runtime metadata:
#   {"jax_version": "0.4.x", "platform": "gpu", "devices": 4, "flags": {...}}
CREATE_EXECUTIONS = """
CREATE TABLE IF NOT EXISTS Executions (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    status          TEXT NOT NULL DEFAULT 'PENDING'
                         CHECK(status IN ('PENDING', 'RUNNING', 'COMPLETED', 'FAILED', 'INVALID')),
    hostname        TEXT,
    start_time      TEXT,
    end_time        TEXT,
    git_commit      TEXT,
    git_diff_blob   TEXT,
    jax_config_json TEXT
);
"""

# Many logical Runs can be satisfied by one Execution (via vmap batching).
CREATE_EXECUTION_RUNS = """
CREATE TABLE IF NOT EXISTS ExecutionRuns (
    execution_id INTEGER NOT NULL REFERENCES Executions(id),
    run_id       INTEGER NOT NULL REFERENCES Runs(id),
    PRIMARY KEY (execution_id, run_id)
);
"""

# ---------------------------------------------------------------------------
# Index DDL  (queried frequently in the runner hot path)
# ---------------------------------------------------------------------------

CREATE_INDEXES = [
    "CREATE INDEX IF NOT EXISTS idx_cv_component_id ON ComponentVersions(component_id);",
    "CREATE INDEX IF NOT EXISTS idx_cv_hash         ON ComponentVersions(code_snapshot_hash);",
    "CREATE INDEX IF NOT EXISTS idx_runs_experiment  ON Runs(experiment_id);",
    "CREATE INDEX IF NOT EXISTS idx_execruns_run    ON ExecutionRuns(run_id);",
    "CREATE INDEX IF NOT EXISTS idx_execruns_exec   ON ExecutionRuns(execution_id);",
]

# ---------------------------------------------------------------------------
# View: "latest" version pointer per component
#
# Resolves `latest` to the highest version_number for each component.
# Researchers and analysis scripts query this view instead of hard-coding
# version numbers, so plots automatically reflect the most recent valid work.
# ---------------------------------------------------------------------------

CREATE_LATEST_VERSIONS_VIEW = """
CREATE VIEW IF NOT EXISTS ComponentLatestVersions AS
SELECT
    cv.id,
    cv.component_id,
    cv.version_number,
    cv.created_at,
    cv.code_snapshot_hash,
    cv.notes
FROM ComponentVersions cv
INNER JOIN (
    SELECT component_id, MAX(version_number) AS max_version
    FROM ComponentVersions
    GROUP BY component_id
) latest ON cv.component_id = latest.component_id
         AND cv.version_number = latest.max_version;
"""

# ---------------------------------------------------------------------------
# Ordered list of all DDL statements executed during initialisation
# ---------------------------------------------------------------------------

ALL_DDL: list[str] = [
    CREATE_COMPONENTS,
    CREATE_COMPONENT_VERSIONS,
    CREATE_HYPERPARAM_CONFIGS,
    CREATE_EXPERIMENTS,
    CREATE_RUNS,
    CREATE_EXECUTIONS,
    CREATE_EXECUTION_RUNS,
    *CREATE_INDEXES,
    CREATE_LATEST_VERSIONS_VIEW,
]
