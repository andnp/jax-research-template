"""Bridge between experiment-definition and research-instrument.

Provides helpers to extract metric whitelists from experiment definitions
and configure the metrics collector accordingly.
"""

from __future__ import annotations

from experiment_definition.experiment import Experiment


def metric_whitelist(experiment: Experiment) -> frozenset[str]:
    """Extract the metric whitelist from an experiment definition.

    Returns a frozenset suitable for passing to
    ``research_instrument.collector.configure()``.

    Args:
        experiment: An experiment with metrics declared via ``add_metric()``.

    Returns:
        A frozenset of metric name strings.
    """
    return frozenset(m.name for m in experiment._state.metrics)
