"""Small tests for generic research trajectory plotting."""

from pathlib import Path

import pytest
from research_plot.plots import EventWindow, TrajectoryPanel, TrajectorySeries, plot_trajectory_panels


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
