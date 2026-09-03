import pytest
from process_control.benchmarks.registry import get_benchmark_entry


def test_get_benchmark_entry_returns_known_metadata() -> None:
    entry = get_benchmark_entry("chlorine")

    assert entry.name == "chlorine"
    assert entry.module == "process_control.benchmarks.chlorine"
    assert entry.obs_dim == 4
    assert entry.action_dim == 1


def test_get_benchmark_entry_reports_available_names() -> None:
    with pytest.raises(ValueError, match="Unknown benchmark 'missing'") as error:
        get_benchmark_entry("missing")

    assert "chlorine" in str(error.value)
    assert "bsm1" in str(error.value)
