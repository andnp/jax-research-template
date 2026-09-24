"""Small tests for generic research trajectory plotting."""

from pathlib import Path

import pytest
from research_plot.plots import (
    EventWindow,
    TrajectoryPanel,
    TrajectorySeries,
    plot_distributions,
    plot_trajectory_panels,
)


def _panel() -> dict[str, TrajectoryPanel]:
    """Build a deterministic two-panel trajectory view."""
    x = [0.0, 1.0, 2.0]
    return {
        "state": TrajectoryPanel(
            ylabel="State",
            series={
                "median": TrajectorySeries(x=x, y=[1.0, 2.0, 3.0], lower=[0.5, 1.5, 2.5], upper=[1.5, 2.5, 3.5]),
                "worst decile": TrajectorySeries(x=x, y=[0.5, 1.0, 1.5]),
            },
        ),
        "action": TrajectoryPanel(
            ylabel="Action",
            series={"requested": TrajectorySeries(x=x, y=[0.0, 0.5, 1.0])},
        ),
    }


class TestTrajectoryPanels:
    def test_renders_bands_and_event_windows(self, tmp_path: Path) -> None:
        """Render a multi-panel view with uncertainty and event annotations."""
        output_path = tmp_path / "trajectory.png"

        resolved_path = plot_trajectory_panels(
            _panel(),
            output_path,
            title="Qualification",
            event_windows=(EventWindow(1.0, 2.0, "disturbance"),),
        )

        assert resolved_path == output_path
        assert output_path.exists()
        assert output_path.stat().st_size > 0

    def test_rejects_misaligned_series(self, tmp_path: Path) -> None:
        """Reject panel series that cannot share one time axis."""
        panel = {
            "state": TrajectoryPanel(
                ylabel="State",
                series={
                    "a": TrajectorySeries(x=[0.0, 1.0], y=[1.0, 2.0]),
                    "b": TrajectorySeries(x=[0.0, 2.0], y=[1.0, 2.0]),
                },
            )
        }

        with pytest.raises(ValueError, match="must share the same x values"):
            plot_trajectory_panels(panel, tmp_path / "invalid.png")

    def test_rejects_nonfinite_band(self, tmp_path: Path) -> None:
        """Reject uncertainty bands containing non-finite values."""
        panel = {
            "state": TrajectoryPanel(
                ylabel="State",
                series={
                    "median": TrajectorySeries(
                        x=[0.0, 1.0],
                        y=[1.0, 2.0],
                        lower=[0.0, float("nan")],
                        upper=[2.0, 3.0],
                    )
                },
            )
        }

        with pytest.raises(ValueError, match="bands must contain only finite"):
            plot_trajectory_panels(panel, tmp_path / "invalid.png")

    def test_rejects_inverted_event_window(self, tmp_path: Path) -> None:
        """Reject event intervals whose end precedes their start."""
        with pytest.raises(ValueError, match="start <= end"):
            plot_trajectory_panels(
                {"state": TrajectoryPanel(ylabel="State", series={"x": TrajectorySeries(x=[0.0], y=[1.0])})},
                tmp_path / "invalid.png",
                event_windows=(EventWindow(2.0, 1.0),),
            )


class TestRunDistributions:
    def test_renders_any_number_of_conditions_including_degenerate_ones(self, tmp_path: Path) -> None:
        """Conditions with one run or identical runs show their points instead of failing the density."""
        output_path = tmp_path / "runs" / "distribution.png"

        resolved_path = plot_distributions(
            {"a": [1.0, 2.0, 5.0], "b": [3.0, 3.0], "c": [4.0], "d": [0.0, -10.0, 2.0, 1.0]},
            output_path,
        )

        assert resolved_path == output_path
        assert output_path.exists()

    @pytest.mark.parametrize("samples", [{}, {"a": []}, {"a": [1.0, float("nan")]}])
    def test_rejects_empty_or_non_finite_samples(self, tmp_path: Path, samples: dict[str, list[float]]) -> None:
        """A missing or diverged run must not be drawn as if it were an ordinary outcome."""
        with pytest.raises(ValueError, match="sample|condition"):
            plot_distributions(samples, tmp_path / "distribution.png")
