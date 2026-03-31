"""Medium integration tests for the research-analysis SQLite loader."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import numpy as np
import polars as pl
import pytest
from research_analysis.learning_curve import step_weighted_returns_from_dataframe
from research_analysis.loader import load_sqlite_query


@pytest.fixture
def sqlite_database_path(tmp_path: Path) -> Path:
    database_path = tmp_path / "analysis.sqlite"

    with sqlite3.connect(database_path) as connection:
        connection.execute(
            "CREATE TABLE metrics (run_id TEXT NOT NULL, step INTEGER NOT NULL, value REAL NOT NULL)"
        )
        connection.executemany(
            "INSERT INTO metrics (run_id, step, value) VALUES (?, ?, ?)",
            [
                ("seed-0", 0, 1.0),
                ("seed-0", 1, 1.5),
                ("seed-1", 0, 0.8),
            ],
        )
        connection.commit()

    return database_path


@pytest.fixture
def sqlite_learning_curve_database_path(tmp_path: Path) -> Path:
    database_path = tmp_path / "learning_curve.sqlite"

    with sqlite3.connect(database_path) as connection:
        connection.execute(
            "CREATE TABLE episodic_returns (run_id TEXT NOT NULL, cumulative_steps INTEGER NOT NULL, episodic_return REAL NOT NULL)"
        )
        connection.executemany(
            "INSERT INTO episodic_returns (run_id, cumulative_steps, episodic_return) VALUES (?, ?, ?)",
            [
                ("seed-0", 2, 1.0),
                ("seed-0", 5, 3.0),
            ],
        )
        connection.commit()

    return database_path


def test_load_sqlite_query_returns_polars_dataframe(sqlite_database_path: Path) -> None:
    frame = load_sqlite_query(
        sqlite_database_path,
        "SELECT run_id, step, value FROM metrics ORDER BY run_id, step",
    )

    assert isinstance(frame, pl.DataFrame)
    assert frame.schema == {
        "run_id": pl.String,
        "step": pl.Int64,
        "value": pl.Float64,
    }
    assert frame.to_dict(as_series=False) == {
        "run_id": ["seed-0", "seed-0", "seed-1"],
        "step": [0, 1, 0],
        "value": [1.0, 1.5, 0.8],
    }


def test_load_sqlite_query_feeds_step_weighted_returns_from_dataframe(
    sqlite_learning_curve_database_path: Path,
) -> None:
    frame = load_sqlite_query(
        sqlite_learning_curve_database_path,
        "SELECT cumulative_steps, episodic_return FROM episodic_returns ORDER BY cumulative_steps",
    )

    result = step_weighted_returns_from_dataframe(
        frame,
        cumulative_steps_column="cumulative_steps",
        episodic_returns_column="episodic_return",
        end_step=7,
    )

    np.testing.assert_array_equal(result, np.array([1.0, 1.0, 3.0, 3.0, 3.0, 3.0, 3.0]))