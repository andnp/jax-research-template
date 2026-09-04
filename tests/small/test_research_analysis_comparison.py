"""Contract tests for comparison specifications."""

import pytest
from research_analysis.comparison import ComparisonSpec


def test_defaults_describe_a_reproducible_comparison() -> None:
    """Defaults provide a mean comparison with explicit uncertainty settings."""
    spec = ComparisonSpec()

    assert spec.estimand == "mean"
    assert spec.pairing == "auto"
    assert spec.alpha == 0.05
    assert spec.confidence_level == 0.95
    assert spec.pairing_keys == ("seed",)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"estimand": "iqm"}, "estimand"),
        ({"pairing": "unknown"}, "pairing"),
        ({"alpha": 0.0}, "alpha"),
        ({"n_resamples": 0}, "n_resamples"),
        ({"min_samples": 2}, "min_samples"),
        ({"missingness": "drop"}, "missingness"),
        ({"pairing_keys": ()}, "pairing_keys"),
    ],
)
def test_rejects_invalid_design_settings(kwargs: dict[str, object], message: str) -> None:
    """Invalid statistical design choices fail before data is analyzed."""
    with pytest.raises(ValueError, match=message):
        ComparisonSpec(**kwargs)  # type: ignore[arg-type]


def test_supports_explicit_paired_median_design() -> None:
    """Callers can declare median inference and a multi-key pairing unit."""
    spec = ComparisonSpec(
        estimand="median",
        pairing="paired",
        alpha=0.10,
        min_samples=5,
        pairing_keys=("benchmark", "seed"),
    )

    assert spec.estimand == "median"
    assert spec.pairing == "paired"
    assert spec.confidence_level == 0.90
    assert spec.min_samples == 5
