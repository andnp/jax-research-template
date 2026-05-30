from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class SlurmConfig:
    account: str | None = None
    partition: str | None = None
    time: str = "2:59:00"
    cpus_per_task: int = 1
    mem_per_cpu: str = "4G"
    gpus_per_task: int | None = None
    modules: list[str] = field(default_factory=list)
    setup_commands: list[str] = field(default_factory=list)
    log_path: str = "slurm-%A_%a.out"

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "SlurmConfig":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})

    @classmethod
    def from_json(cls, path: str) -> "SlurmConfig":
        import json
        from pathlib import Path

        with Path(path).open() as f:
            return cls.from_dict(json.load(f))
