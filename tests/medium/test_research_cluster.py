from __future__ import annotations

import json
from pathlib import Path

import pytest

from research_cluster.config import SlurmConfig
from research_cluster.script import build_job_script, build_sbatch_flags, write_job_script
from research_cluster.submit import submit_experiment


def test_slurm_config_defaults() -> None:
    config = SlurmConfig()
    assert config.account is None
    assert config.partition is None
    assert config.time == "2:59:00"
    assert config.cpus_per_task == 1
    assert config.mem_per_cpu == "4G"
    assert config.gpus_per_task is None
    assert config.modules == []
    assert config.setup_commands == []
    assert config.log_path == "slurm-%A_%a.out"


def test_slurm_config_from_dict() -> None:
    config = SlurmConfig.from_dict({
        "account": "test-acct",
        "partition": "gpu",
        "time": "4:00:00",
        "gpus_per_task": 1,
    })
    assert config.account == "test-acct"
    assert config.partition == "gpu"
    assert config.time == "4:00:00"
    assert config.gpus_per_task == 1
    assert config.cpus_per_task == 1
    assert config.mem_per_cpu == "4G"


def test_slurm_config_from_json(tmp_path: Path) -> None:
    json_file = tmp_path / "slurm.json"
    json_file.write_text(json.dumps({
        "account": "json-acct",
        "partition": "batch",
        "time": "8:00:00",
        "cpus_per_task": 4,
    }))
    config = SlurmConfig.from_json(str(json_file))
    assert config.account == "json-acct"
    assert config.partition == "batch"
    assert config.time == "8:00:00"
    assert config.cpus_per_task == 4


def test_build_sbatch_flags_minimal() -> None:
    config = SlurmConfig()
    flags = build_sbatch_flags(config)
    assert "--time=2:59:00" in flags
    assert "--cpus-per-task=1" in flags
    assert "--mem-per-cpu=4G" in flags
    assert "--account" not in flags
    assert "--gpus-per-task" not in flags


def test_build_sbatch_flags_with_gpu_and_account() -> None:
    config = SlurmConfig(account="my-acct", gpus_per_task=2)
    flags = build_sbatch_flags(config)
    assert "--account=my-acct" in flags
    assert "--gpus-per-task=2" in flags


def test_build_job_script_contains_essentials() -> None:
    config = SlurmConfig()
    script = build_job_script(
        config,
        execution_ids=[1, 2, 3],
        db_path=Path("/data/exp.sqlite"),
        spec_file=Path("/specs/train.py"),
        spec_name="my_spec",
    )
    assert script.startswith("#!/bin/bash")
    assert "#SBATCH --array=0-2" in script
    assert "EXECUTION_IDS=(1 2 3)" in script
    assert "execute-batch" in script
    assert "/specs/train.py" in script
    assert "my_spec" in script


def test_build_job_script_with_modules() -> None:
    config = SlurmConfig(modules=["python/3.13", "cuda/12"])
    script = build_job_script(
        config,
        execution_ids=[1],
        db_path=Path("/data/exp.sqlite"),
        spec_file=Path("/specs/train.py"),
        spec_name="s",
    )
    assert "module load python/3.13" in script
    assert "module load cuda/12" in script


def test_build_job_script_empty_ids_raises() -> None:
    config = SlurmConfig()
    with pytest.raises(ValueError, match="must not be empty"):
        build_job_script(
            config,
            execution_ids=[],
            db_path=Path("/data/exp.sqlite"),
            spec_file=Path("/specs/train.py"),
            spec_name="s",
        )


def test_write_job_script_creates_file(tmp_path: Path) -> None:
    script = "#!/bin/bash\necho hello"
    out = tmp_path / "sub" / "job.sh"
    write_job_script(script, out)
    assert out.exists()
    assert out.read_text().startswith("#!/bin/bash")


def test_submit_experiment_dry_run(tmp_path: Path) -> None:
    config = SlurmConfig()
    script_path = tmp_path / "slurm_submit.sh"
    result = submit_experiment(
        config,
        execution_ids=[10, 20],
        db_path=Path("/data/exp.sqlite"),
        spec_file=Path("/specs/train.py"),
        spec_name="my_spec",
        script_path=script_path,
        dry_run=True,
    )
    assert result is None
    assert script_path.exists()
    assert "#!/bin/bash" in script_path.read_text()
