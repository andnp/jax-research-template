"""Polars-first SQLite ingestion helpers for analysis workflows.

This module intentionally keeps the public surface area small: SQLite data
is loaded directly into a :class:`polars.DataFrame`, which matches ADR 009's
requirement that Polars be the first-class analysis data layer.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import polars as pl


def load_sqlite_query(database_path: str | Path, query: str) -> pl.DataFrame:
    """Load a SQLite query result into a Polars DataFrame.

    Args:
        database_path: Path to the SQLite database file.
        query: SQL query to execute against the database.

    Returns:
        A Polars DataFrame containing the query result.

    Raises:
        FileNotFoundError: If the SQLite database file does not exist.
    """
    path = Path(database_path)
    if not path.exists():
        raise FileNotFoundError(f"SQLite database does not exist: {path}")

    with sqlite3.connect(path) as connection:
        return pl.read_database(query=query, connection=connection)