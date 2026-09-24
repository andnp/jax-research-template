"""Contract tests for bakeoff specifications."""

import pytest
from research_analysis.comparison import BakeoffSpec


def test_defaults_describe_a_reproducible_bakeoff() -> None:
    """Defaults provide a mean bakeoff with explicit uncertainty settings."""
    spec = BakeoffSpec()

    assert spec.estimand == "mean"
    assert spec.alpha == 0.05
    assert spec.confidence_level == 0.95
    assert spec.interval == "percentile"


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"estimand": "iqm"}, "estimand"),
        ({"alpha": 0.0}, "alpha"),
        ({"n_resamples": 0}, "n_resamples"),
        ({"min_samples": 2}, "min_samples"),
        ({"missingness": "drop"}, "missingness"),
        ({"interval": "bca"}, "interval"),
    ],
)
def test_rejects_invalid_design_settings(kwargs: dict[str, object], message: str) -> None:
    """Invalid statistical design choices fail before data is analyzed."""
    with pytest.raises(ValueError, match=message):
        BakeoffSpec(**kwargs)  # type: ignore[arg-type]


def test_supports_explicit_median_design() -> None:
    """Callers can declare median inference and stricter sample requirements."""
    spec = BakeoffSpec(
        estimand="median",
        alpha=0.10,
        min_samples=5,
    )

    assert spec.estimand == "median"
    assert spec.confidence_level == 0.90
    assert spec.min_samples == 5
