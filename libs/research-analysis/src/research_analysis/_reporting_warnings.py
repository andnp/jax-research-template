"""Private statistical warning helpers for reporting."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from scipy import stats as scipy_stats

from research_analysis._reporting_models import HyperparameterConfigReport, HyperparameterSliceReport
from research_analysis.diagnostics.bakeoff import BakeoffWarning, BakeoffWarningCode


def boundary_warning(
    report: HyperparameterSliceReport,
    *,
    condition: str | None = None,
    cell: tuple[tuple[str, object], ...] = (),
    higher_is_better: bool = True,
) -> BakeoffWarning | None:
    """Flag a champion hyperparameter sitting at the edge of its swept grid while
    performance is still trending toward that edge over the grid's outer half, not
    merely plateaued there. A champion at the edge with an interior peak elsewhere
    (or a plateau right at the edge) is not flagged: only a still-climbing trend
    means the true optimum may lie outside the swept range.
    """
    values = report.values
    performance = report.best_performance
    if len(values) < 2:
        return None

    champion_idx = int(np.argmax(performance)) if higher_is_better else int(np.argmin(performance))
    if champion_idx == 0:
        edge_label, edge_value = "lower", values[0]
    elif champion_idx == len(values) - 1:
        edge_label, edge_value = "upper", values[-1]
    else:
        return None

    half = max(1, len(values) // 2)
    outer = performance[-half - 1 :] if edge_label == "upper" else performance[: half + 1]
    trend = np.diff(outer)
    if not higher_is_better:
        trend = -trend
    if edge_label == "lower":
        trend = -trend
    if float(np.mean(trend)) <= 0:
        return None

    return BakeoffWarning(
        code=BakeoffWarningCode.HYPERPARAMETER_AT_BOUNDARY,
        message=(
            f"{report.vary}={edge_value!r} sits at the {edge_label} edge of the swept grid "
            "and performance is still trending toward it; the true optimum may lie outside "
            "the swept range."
        ),
        condition=condition,
        cell=cell,
    )


def non_normality_warning(
    values: Sequence[float] | np.ndarray,
    *,
    condition: str | None = None,
    cell: tuple[tuple[str, object], ...] = (),
    alpha: float = 0.05,
) -> BakeoffWarning | None:
    """Shapiro-Wilk normality check against one cell's raw values, informational only.

    Per ADR-010, no normality pretest may select which statistical test runs; this
    never gates anything, it only surfaces a caveat for the reader to weigh (e.g.
    whether a median estimand would be more appropriate than a mean).
    """
    array = np.asarray(values, dtype=np.float64)
    if array.size < 3:
        return None
    _statistic, p_value = scipy_stats.shapiro(array)
    if p_value >= alpha:
        return None
    return BakeoffWarning(
        code=BakeoffWarningCode.NON_NORMAL_DISTRIBUTION,
        message=f"Shapiro-Wilk rejects normality (p={p_value:.4f} < alpha={alpha}); consider a robust estimand.",
        condition=condition,
        cell=cell,
    )


def insufficient_seeds_warning(
    report: HyperparameterConfigReport,
    n_seeds: int,
    *,
    condition: str | None = None,
    cell: tuple[tuple[str, object], ...] = (),
    min_seeds: int = 5,
    max_normalized_entropy: float = 0.5,
) -> BakeoffWarning | None:
    """Flag a champion pick backed by too few seeds, or unstable under resampling.

    Fires on either signal: fewer than ``min_seeds`` raw seeds, or a
    ``normalized_selection_entropy`` above ``max_normalized_entropy`` -- a low seed
    count can still bootstrap a stable pick, and a high entropy can occur even with
    many seeds when several configurations perform near-identically.
    """
    if n_seeds < min_seeds:
        return BakeoffWarning(
            code=BakeoffWarningCode.INSUFFICIENT_SEEDS,
            message=f"only {n_seeds} seed(s) backing this champion, fewer than the recommended {min_seeds}.",
            condition=condition,
            cell=cell,
        )
    if report.normalized_selection_entropy > max_normalized_entropy:
        return BakeoffWarning(
            code=BakeoffWarningCode.INSUFFICIENT_SEEDS,
            message=(
                f"champion selection is unstable under resampling (normalized entropy "
                f"{report.normalized_selection_entropy:.2f} > {max_normalized_entropy}); "
                "several configurations perform near-identically."
            ),
            condition=condition,
            cell=cell,
        )
    return None


def variance_warning(
    values: Sequence[float] | np.ndarray,
    *,
    condition: str | None = None,
    cell: tuple[tuple[str, object], ...] = (),
    reference_range: float | None = None,
    extreme_relative_std: float = 1.0,
) -> BakeoffWarning | None:
    """Flag a cell with zero cross-seed variance, or variance extreme relative to the
    metric's own cross-condition range (``reference_range``, typically max minus min
    across every condition and cell in the same report).
    """
    array = np.asarray(values, dtype=np.float64)
    std = float(np.std(array, ddof=1)) if array.size > 1 else 0.0
    if std == 0.0:
        return BakeoffWarning(
            code=BakeoffWarningCode.ZERO_VARIANCE,
            message="every seed produced an identical value; check for a fixed seed or a saturated metric.",
            condition=condition,
            cell=cell,
        )
    if reference_range is not None and reference_range > 0 and std > extreme_relative_std * reference_range:
        return BakeoffWarning(
            code=BakeoffWarningCode.EXTREME_VARIANCE,
            message=(
                f"cross-seed std ({std:.4g}) exceeds {extreme_relative_std:.0%} of the metric's own "
                f"cross-condition range ({reference_range:.4g})."
            ),
            condition=condition,
            cell=cell,
        )
    return None

