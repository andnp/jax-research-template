from __future__ import annotations

import subprocess
from pathlib import Path

from experiment_definition.db import DatabaseManager

from .config import SlurmConfig
from .ownership import format_slurm_owner
from .script import build_job_script, write_job_script


def submit_experiment(
    config: SlurmConfig,
    execution_ids: list[int],
    db_path: Path,
    spec_file: Path,
    spec_name: str,
    *,
    planner_owner: str,
    working_dir: Path | None = None,
    script_path: Path | None = None,
    results_root: Path | None = None,
    dry_run: bool = False,
) -> str | None:
    script = build_job_script(
        config,
        execution_ids,
        spec_file,
        spec_name,
        working_dir=working_dir,
        results_root=results_root,
    )

    out_path = script_path or Path("slurm_submit.sh")
    write_job_script(script, out_path)

    if dry_run:
        return None

    try:
        result = subprocess.run(
            ["sbatch", "--parsable", str(out_path)],
            capture_output=True,
            text=True,
            check=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        with DatabaseManager(db_path) as database:
            database.initialize()
            database.fail_executions(execution_ids)
        raise

    job_id = result.stdout.strip().partition(";")[0]
    with DatabaseManager(db_path) as database:
        database.initialize()
        database.transfer_owners(
            {execution_id: planner_owner for execution_id in execution_ids},
            {execution_id: format_slurm_owner(job_id, index) for index, execution_id in enumerate(execution_ids)},
        )
    return job_id
