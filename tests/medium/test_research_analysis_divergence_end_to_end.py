"""A diverged seed and a crashed seed travel from a real sweep into the bakeoff report without vanishing."""

from __future__ import annotations

from pathlib import Path

import pytest
from experiment_definition import Component, ComponentType, Experiment
from research_analysis.comparison import BakeoffSpec
from research_analysis.dataset.loading import load_experiment_dataset
from research_analysis.dataset.types import ExperimentDataset
from research_analysis.diagnostics.types import DiagnosticCode
from research_analysis.non_finite import NonFiniteValuesError, substitute_non_finite
from research_analysis.reporting import bakeoff_report
from research_runner import ExecutionContext, ExecutionResult, run_experiment

SEEDS = 6
DIVERGED_SEED = 2
CRASHED_SEED = 5
SPEC = BakeoffSpec(missingness="complete_case", n_resamples=999)


def train(ctx: ExecutionContext) -> ExecutionResult:
    curves = []
    for point in ctx.points:
        seed, condition = point.get("seed", int), point.get("condition_name", str)
        if condition == "variant" and seed == CRASHED_SEED:
            raise RuntimeError("simulated crash")
        final = float("nan") if condition == "variant" and seed == DIVERGED_SEED else seed + (condition == "variant")
        curves.append([0.0, final])
    return ExecutionResult(metric_curves={"return": curves})


@pytest.fixture
def dataset(tmp_path: Path) -> ExperimentDataset:
    experiment = Experiment("divergence", description="one diverged and one crashed seed")
    experiment.add_parameter("seed", list(range(SEEDS)))
    with experiment.for_component(Component(name="agent", path=Path("/nonexistent/agent.py"), type=ComponentType.ALGO)):
        experiment.add_parameter("condition_name", ["base", "variant"])

    db_path = tmp_path / "experiments.sqlite"
    # The crashing run is planned last, so the sweep has run every other seed when the crash propagates.
    with pytest.raises(RuntimeError, match="simulated crash"):
        run_experiment(
            db_path, experiment, train, executions_root=tmp_path / "executions", capture_git=False, max_runs_per_batch=1
        )
    return load_experiment_dataset(db_path, "divergence")


def test_accounting_counts_the_diverged_and_crashed_runs_in_their_cell(dataset: ExperimentDataset) -> None:
    """The variant cell reports the crash as failed and the NaN run as a completed, diverged outcome."""
    cells = {dict(item.cell)["condition_name"]: item for item in dataset.accounting.cells}

    variant = cells["variant"]
    assert (variant.planned, variant.completed, variant.failed, variant.pending) == (SEEDS, SEEDS - 1, 1, 0)
    assert variant.diverged == {"return": 1}
    assert (cells["base"].completed, cells["base"].failed, cells["base"].diverged) == (SEEDS, 0, {})


def test_diagnostics_name_the_diverged_and_crashed_runs(dataset: ExperimentDataset) -> None:
    """The diverged run stays in the runs table flagged non-finite; the crashed run is a RUN_FAILED diagnostic."""
    diverged = dataset.runs.filter(dataset.runs["non_finite_metrics"].list.len() > 0)
    by_code = {code: [d.run_id for d in dataset.diagnostics if d.code == code] for code in DiagnosticCode}

    assert diverged.select("condition_name", "seed").rows() == [("variant", DIVERGED_SEED)]
    assert by_code[DiagnosticCode.NON_FINITE_METRIC] == diverged["run_id"].to_list()
    assert len(by_code[DiagnosticCode.RUN_FAILED]) == 1
    assert by_code[DiagnosticCode.RUN_FAILED][0] not in dataset.runs["run_id"].to_list()


def test_bakeoff_refuses_diverged_values_by_default(dataset: ExperimentDataset) -> None:
    """Without a declared policy the report fails loudly rather than dropping or averaging the NaN run."""
    with pytest.raises(NonFiniteValuesError):
        bakeoff_report(dataset, "return", ["base", "variant"], spec=SPEC)


def test_bakeoff_counts_the_diverged_run_after_a_declared_substitution(dataset: ExperimentDataset) -> None:
    """A declared worst case keeps the diverged run in the variant cell and in the paired comparison."""
    substituted = substitute_non_finite(dataset, "return", worst_case=-100.0)

    report = bakeoff_report(substituted, "return", ["base", "variant"], spec=SPEC)

    assert report.cells["variant"][""].n == SEEDS - 1
    assert report.cells["base"][""].n == SEEDS
    assert [item.n_paired for item in report.differences] == [SEEDS - 1]
    assert substituted.accounting.cells == dataset.accounting.cells
