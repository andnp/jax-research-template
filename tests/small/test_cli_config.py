from pathlib import Path

import pytest
from research_cli.config import ResearchConfigError, load_research_config


def _write_config(tmp_path: Path, content: str):
    config_path = tmp_path / "research.yaml"
    config_path.write_text(content, encoding="utf-8")
    return config_path


def test_load_research_config_reads_required_and_optional_fields(tmp_path: Path) -> None:
    config_path = _write_config(
        tmp_path,
        "\n".join(
            [
                "core_path: core",
                "storage_backend: local",
                "default_github_org: rlcore",
                "github_owner: andy",
                "doctor:",
                "  expected_accelerators:",
                "    - cpu",
                "    - gpu",
                "extra_flag: true",
            ],
        ),
    )

    config = load_research_config(config_path)

    assert config.core_path == Path("core")
    assert config.storage_backend == "local"
    assert config.default_github_org == "rlcore"
    assert config.github_owner == "andy"
    assert config.doctor is not None
    assert config.doctor.expected_accelerators == ("cpu", "gpu")
    assert config.extra_values == {"extra_flag": True}


def test_load_research_config_allows_missing_doctor_section(tmp_path: Path) -> None:
    config_path = _write_config(tmp_path, "core_path: core\nstorage_backend: s3\n")

    config = load_research_config(config_path)

    assert config.core_path == Path("core")
    assert config.storage_backend == "s3"
    assert config.doctor is None


def test_load_research_config_raises_for_missing_file(tmp_path: Path) -> None:
    with pytest.raises(ResearchConfigError, match="research.yaml not found"):
        load_research_config(tmp_path / "research.yaml")


def test_load_research_config_raises_for_directory_path(tmp_path: Path) -> None:
    config_path = tmp_path / "research.yaml"
    config_path.mkdir()

    with pytest.raises(ResearchConfigError, match="is not a file"):
        load_research_config(config_path)


def test_load_research_config_raises_for_malformed_yaml(tmp_path: Path) -> None:
    config_path = _write_config(tmp_path, "core_path: [core\nstorage_backend: local\n")

    with pytest.raises(ResearchConfigError, match="Malformed research.yaml"):
        load_research_config(config_path)


@pytest.mark.parametrize(
    ("content", "missing_key"),
    [
        ("storage_backend: local\n", "core_path"),
        ("core_path: core\n", "storage_backend"),
    ],
)
def test_load_research_config_raises_for_missing_required_keys(tmp_path: Path, content: str, missing_key: str) -> None:
    config_path = _write_config(tmp_path, content)

    with pytest.raises(ResearchConfigError, match=rf"missing required key\(s\): {missing_key}"):
        load_research_config(config_path)


def test_load_research_config_raises_for_invalid_storage_backend(tmp_path: Path) -> None:
    config_path = _write_config(tmp_path, "core_path: core\nstorage_backend: filesystem\n")

    with pytest.raises(ResearchConfigError, match="invalid storage_backend 'filesystem'"):
        load_research_config(config_path)


def test_load_research_config_raises_for_invalid_accelerator_labels(tmp_path: Path) -> None:
    config_path = _write_config(
        tmp_path,
        "\n".join(
            [
                "core_path: core",
                "storage_backend: local",
                "doctor:",
                "  expected_accelerators:",
                "    - gpu",
                "    - cuda",
            ],
        ),
    )

    with pytest.raises(ResearchConfigError, match=r"invalid accelerator label\(s\)"):
        load_research_config(config_path)