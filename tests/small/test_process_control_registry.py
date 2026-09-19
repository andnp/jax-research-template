from types import SimpleNamespace

import pytest
from process_control.actuators.dosing_system import DIRECT, SUPERVISORY
from process_control.benchmarks.chlorine import ChlorineBenchmarkConfig
from process_control.benchmarks.h2s_scrubber import H2SScrubberConfig
from process_control.benchmarks.registry import get_action_bounds, get_benchmark_entry


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


def test_get_action_bounds_reads_scalar_configured_values() -> None:
    """Return configured scalar action bounds for a direct chlorine benchmark."""
    entry = get_benchmark_entry("chlorine")
    config = ChlorineBenchmarkConfig(pump_min_dose=1.0, pump_max_dose=10.0)

    assert get_action_bounds(entry, config) == (1.0, 10.0)


def test_get_action_bounds_reads_vector_configured_values() -> None:
    """Return configured vector action bounds for the H2S benchmark."""
    entry = get_benchmark_entry("h2s_scrubber")
    config = H2SScrubberConfig(
        ph_sp_min=8.5,
        ph_sp_max=10.5,
        orp_sp_min=450.0,
        orp_sp_max=750.0,
        makeup_flow_sp_min=12.0,
        makeup_flow_sp_max=28.0,
    )

    assert get_action_bounds(entry, config) == ((8.5, 450.0, 12.0), (10.5, 750.0, 28.0))


def test_get_action_bounds_omits_non_direct_chlorine_bounds() -> None:
    """Omit chlorine bounds when its configured control mode is not direct."""
    entry = get_benchmark_entry("chlorine")
    config = ChlorineBenchmarkConfig(control_mode=SUPERVISORY)

    assert get_action_bounds(entry, config) == (None, None)


@pytest.mark.parametrize(
    ("field", "value"),
    [("pump_min_dose", True), ("pump_max_dose", "invalid")],
)
def test_get_action_bounds_rejects_invalid_numeric_values(field: str, value: object) -> None:
    """Reject boolean and non-numeric configured action bounds with TypeError."""
    entry = get_benchmark_entry("chlorine")
    config_values: dict[str, object] = {"control_mode": DIRECT, "pump_min_dose": 0.0, "pump_max_dose": 8.0}
    config_values[field] = value
    config = SimpleNamespace(**config_values)

    with pytest.raises(TypeError, match=rf"configured action bound '{field}' must be numeric"):
        get_action_bounds(entry, config)
