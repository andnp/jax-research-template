"""A diverged seed and a crashed seed travel from a real sweep into the bakeoff report without vanishing."""

from __future__ import annotations

import json
import textwrap
from dataclasses import dataclass
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from experiment_definition.component import Component, ComponentType
from experiment_definition.experiment import Experiment
from research_analysis.comparison import BakeoffSpec
from research_analysis.dataset.loading import load_experiment_dataset
from research_analysis.dataset.types import ExperimentDataset
from research_analysis.diagnostics.types import DiagnosticCode
from research_analysis.non_finite import NonFiniteValuesError, substitute_non_finite
from research_analysis.reporting import bakeoff_report
from research_cli.main import app
from research_runner.runner import run_experiment
from research_runner.types import ExperimentSpec
from typer.testing import CliRunner

SEEDS = 6
DIVERGED_SEED = 2
CRASHED_SEED = 5
SPEC = BakeoffSpec(missingness="complete_case", n_resamples=999)


@dataclass(frozen=True)
class _Arm:
    condition_name: str = "base"


def _crash_on(seed: np.ndarray) -> np.float32:
    if int(seed) == CRASHED_SEED:
        raise RuntimeError("simulated crash")
    return np.float32(0)


def train(config: _Arm, key: jax.Array) -> tuple[object, dict[str, jax.Array]]:
    """A variant arm diverges on one seed and crashes on another; `key[1]` is the run's seed."""
    seed = key[1]
    final = seed.astype(jnp.float32)
    if config.condition_name == "variant":
        final = jnp.where(seed == DIVERGED_SEED, jnp.nan, final + 1)
        final += jax.pure_callback(_crash_on, jax.ShapeDtypeStruct((), jnp.float32), seed, vmap_method="sequential")
    return None, {"return": jnp.stack([jnp.float32(0), final])}


@pytest.fixture
def dataset(tmp_path: Path) -> ExperimentDataset:
    experiment = Experiment("divergence", description="one diverged and one crashed seed")
    experiment.add_parameter("seed", list(range(SEEDS)))
    with experiment.for_component(Component(name="agent", path=Path("/nonexistent/agent.py"), type=ComponentType.ALGO)):
        experiment.add_parameter("condition_name", ["base", "variant"])
    experiment.add_metric("return")

    db_path = tmp_path / "experiments.sqlite"
    # The crashing run is planned last, so the sweep has run every other seed when the crash propagates.
    with pytest.raises(RuntimeError, match="simulated crash"):
        run_experiment(
            ExperimentSpec(experiment, config=_Arm(), train=train, results_root=tmp_path, max_runs_per_batch=1, capture_git=False)
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


# ── Through `research experiment report` ─────────────────────────────────────


def swept_spec_file(tmp_path: Path, worst_case: float | None) -> Path:
    spec_file = tmp_path / "divergence_spec.py"
    spec_file.write_text(
        textwrap.dedent(f"""\
            from __future__ import annotations
            from dataclasses import dataclass
            from pathlib import Path
            import jax
            import jax.numpy as jnp
            import numpy as np
            from experiment_definition.component import Component, ComponentType
            from experiment_definition.experiment import Experiment
            from research_analysis.comparison import BakeoffSpec
            from research_analysis.dataset.types import ExperimentDimensions
            from research_analysis.plan import AnalysisPlan, BakeoffAnalysis, analysis_plan
            from research_runner.registry import spec
            from research_runner.types import ExperimentSpec

            @dataclass(frozen=True)
            class Arm:
                condition_name: str = "base"

            def crash_on(seed):
                if int(seed) == {CRASHED_SEED}:
                    raise RuntimeError("simulated crash")
                return np.float32(0)

            def train(config, key):
                seed = key[1]
                final = seed.astype(jnp.float32)
                if config.condition_name == "variant":
                    final = jnp.where(seed == {DIVERGED_SEED}, jnp.nan, final + 1)
                    final += jax.pure_callback(crash_on, jax.ShapeDtypeStruct((), jnp.float32), seed, vmap_method="sequential")
                return None, {{"return": jnp.stack([jnp.float32(0), final])}}

            @spec
            def divergence() -> ExperimentSpec:
                experiment = Experiment("divergence", description="one diverged and one crashed seed")
                experiment.add_parameter("seed", list(range({SEEDS})))
                with experiment.for_component(Component(name="agent", path=Path("/nonexistent/agent.py"), type=ComponentType.ALGO)):
                    experiment.add_parameter("condition_name", ["base", "variant"])
                experiment.add_metric("return")
                return ExperimentSpec(
                    experiment, config=Arm(), train=train, results_root=Path("{tmp_path / "results"}"),
                    max_runs_per_batch=1, capture_git=False,
                )

            @analysis_plan(divergence)
            def plan() -> AnalysisPlan:
                return AnalysisPlan(
                    metric="return",
                    analyses=(BakeoffAnalysis("headline", ("base", "variant")),),
                    dimensions=ExperimentDimensions(),
                    stats=BakeoffSpec(missingness="complete_case", n_resamples=999),
                    non_finite_worst_case={worst_case},
                )
        """)
    )
    swept = CliRunner().invoke(app, ["experiment", "run", str(spec_file)])
    assert isinstance(swept.exception, RuntimeError), swept.output
    return spec_file


def report_json(spec_file: Path):
    result = CliRunner().invoke(app, ["experiment", "report", str(spec_file), "--json"])
    document = json.loads(result.stdout)
    variant = next(cell for cell in document["accounting"]["cells"] if cell["cell"]["condition_name"] == "variant")
    return result.exit_code, document, variant


def test_report_cli_fails_the_analysis_on_an_undeclared_divergence(tmp_path: Path) -> None:
    """Without a declared worst case the report names the diverged run and still accounts for both outcomes."""
    exit_code, document, variant = report_json(swept_spec_file(tmp_path, worst_case=None))

    assert (exit_code, document["status"], document["analyses"]) == (3, "analysis_failed", [])
    assert [(run["condition_name"], run["seed"]) for run in document["failure"]["non_finite_runs"]] == [
        ("variant", DIVERGED_SEED)
    ]
    assert (variant["failed"], variant["diverged"]) == (1, {"return": 1})


def test_report_cli_counts_the_diverged_run_under_a_declared_worst_case(tmp_path: Path) -> None:
    """A declared worst case yields results in which the diverged run is scored and the crash still diagnosed."""
    exit_code, document, variant = report_json(swept_spec_file(tmp_path, worst_case=-100.0))

    assert (exit_code, document["status"]) == (0, "ok")
    assert document["analyses"][0]["report"]["cells"]["variant"][""]["n"] == SEEDS - 1
    assert {item["code"] for item in document["non_fatal_diagnostics"]} >= {"run_failed", "non_finite_metric"}
    assert (variant["failed"], variant["diverged"]) == (1, {"return": 1})
