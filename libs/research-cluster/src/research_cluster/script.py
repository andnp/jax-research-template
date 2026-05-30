from __future__ import annotations

from pathlib import Path

from .config import SlurmConfig


def build_sbatch_flags(config: SlurmConfig):
    flags: list[str] = []
    if config.account:
        flags.append(f"#SBATCH --account={config.account}")
    if config.partition:
        flags.append(f"#SBATCH --partition={config.partition}")
    flags.append(f"#SBATCH --time={config.time}")
    flags.append(f"#SBATCH --cpus-per-task={config.cpus_per_task}")
    flags.append(f"#SBATCH --mem-per-cpu={config.mem_per_cpu}")
    if config.gpus_per_task is not None:
        flags.append(f"#SBATCH --gpus-per-task={config.gpus_per_task}")
    flags.append(f"#SBATCH --output={config.log_path}")
    return "\n".join(flags)


def build_job_script(
    config: SlurmConfig,
    execution_ids: list[int],
    db_path: Path,
    spec_file: Path,
    spec_name: str,
    *,
    working_dir: Path | None = None,
):
    if not execution_ids:
        raise ValueError("execution_ids must not be empty.")

    id_list = " ".join(str(eid) for eid in execution_ids)

    lines = [
        "#!/bin/bash",
        build_sbatch_flags(config),
        f"#SBATCH --array=0-{len(execution_ids) - 1}",
        "",
        "set -euo pipefail",
    ]

    if working_dir:
        lines.append(f'cd "{working_dir}"')

    if config.modules:
        lines.append("")
        lines.append("# Load modules")
        lines.extend(f"module load {m}" for m in config.modules)

    if config.setup_commands:
        lines.append("")
        lines.append("# Setup")
        lines.extend(config.setup_commands)

    lines.extend([
        "",
        "# Map array index to execution ID",
        f"EXECUTION_IDS=({id_list})",
        "EXECUTION_ID=${EXECUTION_IDS[$SLURM_ARRAY_TASK_ID]}",
        "",
        'echo "Array task $SLURM_ARRAY_TASK_ID -> Execution $EXECUTION_ID"',
        "",
        "uv run research experiment execute-batch \\",
        f'    "{db_path}" \\',
        '    --execution-id "$EXECUTION_ID" \\',
        f'    --spec-file "{spec_file}" \\',
        f'    --spec "{spec_name}"',
        "",
    ])

    return "\n".join(lines)


def write_job_script(
    script: str,
    output_path: Path,
):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(script)
    return output_path
