from __future__ import annotations

import subprocess
from pathlib import Path

from .config import SlurmConfig
from .script import build_job_script, write_job_script


def submit_experiment(
    config: SlurmConfig,
    execution_ids: list[int],
    db_path: Path,
    spec_file: Path,
    spec_name: str,
    *,
    working_dir: Path | None = None,
    script_path: Path | None = None,
    dry_run: bool = False,
):
    script = build_job_script(
        config,
        execution_ids,
        db_path,
        spec_file,
        spec_name,
        working_dir=working_dir,
    )

    out_path = script_path or Path("slurm_submit.sh")
    write_job_script(script, out_path)

    if dry_run:
        return None

    result = subprocess.run(
        ["sbatch", str(out_path)],
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout.strip()
