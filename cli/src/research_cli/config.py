"""Typed loading and validation for workspace ``research.yaml`` files."""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, cast

import yaml

StorageBackend = Literal["local", "s3"]
AcceleratorLabel = Literal["cpu", "gpu", "tpu"]

_ALLOWED_STORAGE_BACKENDS = frozenset({"local", "s3"})
_ALLOWED_ACCELERATORS = frozenset({"cpu", "gpu", "tpu"})
_REQUIRED_KEYS = ("core_path", "storage_backend")
_KNOWN_TOP_LEVEL_KEYS = frozenset({"core_path", "storage_backend", "default_github_org", "github_owner", "doctor"})


class ResearchConfigError(ValueError):
    """Raised when ``research.yaml`` is missing or invalid."""


@dataclass(frozen=True, slots=True)
class DoctorConfig:
    expected_accelerators: tuple[AcceleratorLabel, ...] | None = None


@dataclass(frozen=True, slots=True)
class ResearchConfig:
    core_path: Path
    storage_backend: StorageBackend
    default_github_org: str | None = None
    github_owner: str | None = None
    doctor: DoctorConfig | None = None
    extra_values: dict[str, object] = field(default_factory=dict)


def load_research_config(config_path: Path):
    if not config_path.is_file():
        raise ResearchConfigError(
            f"research.yaml not found at '{config_path}'. Run `research workspace init` or create the file before running this command.",
        )

    try:
        raw_content = config_path.read_text(encoding="utf-8")
    except FileNotFoundError as exc:
        raise ResearchConfigError(
            f"research.yaml not found at '{config_path}'. Run `research workspace init` or create the file before running this command.",
        ) from exc

    try:
        loaded = yaml.safe_load(raw_content)
    except yaml.YAMLError as exc:
        raise ResearchConfigError(f"Malformed research.yaml at '{config_path}': {exc}") from exc

    raw_config = _require_top_level_mapping({} if loaded is None else loaded, config_path)
    missing_keys = [key for key in _REQUIRED_KEYS if key not in raw_config]
    if missing_keys:
        missing_text = ", ".join(missing_keys)
        raise ResearchConfigError(f"Invalid research.yaml at '{config_path}': missing required key(s): {missing_text}.")

    doctor_config = _load_doctor_config(raw_config.get("doctor"), config_path)
    extra_values = {key: value for key, value in raw_config.items() if key not in _KNOWN_TOP_LEVEL_KEYS}

    return ResearchConfig(
        core_path=Path(_require_string(raw_config, "core_path", config_path)),
        storage_backend=_validate_storage_backend(_require_string(raw_config, "storage_backend", config_path), config_path),
        default_github_org=_optional_string(raw_config, "default_github_org", config_path),
        github_owner=_optional_string(raw_config, "github_owner", config_path),
        doctor=doctor_config,
        extra_values=extra_values,
    )


def _require_top_level_mapping(value: object, config_path: Path):
    if isinstance(value, Mapping):
        return value
    raise ResearchConfigError(f"Invalid research.yaml at '{config_path}': expected a top-level mapping of keys to values.")


def _require_string(config: Mapping[str, object], key: str, config_path: Path):
    value = config.get(key)
    if isinstance(value, str) and value.strip():
        return value
    raise ResearchConfigError(f"Invalid research.yaml at '{config_path}': '{key}' must be a non-empty string.")


def _optional_string(config: Mapping[str, object], key: str, config_path: Path):
    value = config.get(key)
    if value is None:
        return None
    if isinstance(value, str):
        return value
    raise ResearchConfigError(f"Invalid research.yaml at '{config_path}': '{key}' must be a string when provided.")


def _load_doctor_config(raw_doctor: object, config_path: Path):
    if raw_doctor is None:
        return None
    if not isinstance(raw_doctor, Mapping):
        raise ResearchConfigError(f"Invalid research.yaml at '{config_path}': 'doctor' must be a mapping when provided.")

    expected_accelerators = _load_expected_accelerators(raw_doctor.get("expected_accelerators"), config_path)
    return DoctorConfig(expected_accelerators=expected_accelerators)


def _load_expected_accelerators(raw_value: object, config_path: Path):
    if raw_value is None:
        return None
    if isinstance(raw_value, str) or not isinstance(raw_value, Sequence):
        raise ResearchConfigError(
            f"Invalid research.yaml at '{config_path}': 'doctor.expected_accelerators' must be a list containing any of: cpu, gpu, tpu.",
        )

    invalid_labels: list[str] = []
    validated_labels: list[AcceleratorLabel] = []
    for raw_label in raw_value:
        if not isinstance(raw_label, str) or raw_label not in _ALLOWED_ACCELERATORS:
            invalid_labels.append(repr(raw_label))
            continue
        validated_labels.append(cast(AcceleratorLabel, raw_label))

    if invalid_labels:
        invalid_text = ", ".join(invalid_labels)
        raise ResearchConfigError(
            f"Invalid research.yaml at '{config_path}': invalid accelerator label(s) in 'doctor.expected_accelerators': {invalid_text}. Allowed values: cpu, gpu, tpu.",
        )

    return tuple(validated_labels)


def _validate_storage_backend(raw_backend: str, config_path: Path):
    if raw_backend in _ALLOWED_STORAGE_BACKENDS:
        return cast(StorageBackend, raw_backend)
    allowed_backends = ", ".join(sorted(_ALLOWED_STORAGE_BACKENDS))
    raise ResearchConfigError(
        f"Invalid research.yaml at '{config_path}': invalid storage_backend '{raw_backend}'. Allowed values: {allowed_backends}.",
    )