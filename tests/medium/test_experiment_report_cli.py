"""End-to-end tests for `research experiment report`: declare a plan, run the spec, report it."""

from __future__ import annotations

import json
import textwrap
from pathlib import Path

import pytest
from research_cli.main import app
from typer.testing import CliRunner

runner = CliRunner()


def _write_spec(tmp_path: Path, *, diverge: bool = False, declare_plan: bool = True) -> Path:
    results_root = tmp_path / "results"
    plan = textwrap.dedent("""\
        @analysis_plan(two_arms)
        def two_arms_plan() -> AnalysisPlan:
            return AnalysisPlan(
                "reward",
                (BakeoffAnalysis("headline", ("a", "b")),),
                ExperimentDimensions(),
                stats=BakeoffSpec(n_resamples=100),
            )
    """)
    spec_file = tmp_path / "two_arms.py"
    spec_file.write_text(
        textwrap.dedent(f"""\
            from __future__ import annotations
            import math
            from pathlib import Path
            from experiment_definition import Component, ComponentType, Experiment
            from research_analysis.comparison import BakeoffSpec
            from research_analysis.dataset.types import ExperimentDimensions
            from research_analysis.plan import AnalysisPlan, BakeoffAnalysis, analysis_plan
            from research_runner import ExecutionContext, ExecutionResult, ExperimentSpec, spec

            DIVERGE = {diverge}


            def _curve(condition: str, seed: int) -> list[float]:
                if DIVERGE and condition == "b" and seed == 0:
                    return [0.0, math.nan]
                return [0.0, seed + (2.0 if condition == "b" else 0.0)]


            @spec
            def two_arms() -> ExperimentSpec:
                experiment = Experiment("two-arms")
                with experiment.for_component(Component(name="algo", path=Path("algo.py"), type=ComponentType.ALGO)):
                    experiment.add_parameter("condition_name", ["a", "b"])
                with experiment.for_component(Component(name="env", path=Path("env.py"), type=ComponentType.ENV)):
                    pass
                experiment.add_parameter("seed", list(range(4)))
                experiment.add_metric("reward", frequency="per_update")

                def train(ctx: ExecutionContext) -> ExecutionResult:
                    curves = [
                        _curve(point.get("condition_name", str), point.get("seed", int)) for point in ctx.points
                    ]
                    return ExecutionResult(metric_curves={{"reward": curves}})

                return ExperimentSpec(
                    experiment=experiment, train_fn=train, results_root=Path("{results_root}"), capture_git=False
                )
        """)
        + ("\n\n" + plan if declare_plan else "")
    )
    return spec_file


def _run(spec_file: Path) -> None:
    result = runner.invoke(app, ["experiment", "run", str(spec_file)])
    assert result.exit_code == 0, result.output


def test_a_declared_plan_reports_json_and_markdown_under_the_results_root(tmp_path: Path) -> None:
    """Run then report: the plan is declared, and the files land under <results_root>/reports/."""
    spec_file = _write_spec(tmp_path)
    _run(spec_file)

    result = runner.invoke(app, ["experiment", "report", str(spec_file)])

    assert result.exit_code == 0, result.output
    report_dirs = list((tmp_path / "results" / "reports" / "two-arms").iterdir())
    assert len(report_dirs) == 1
    document = json.loads((report_dirs[0] / "report.json").read_text())
    assert (document["status"], document["plan_status"]) == ("ok", "declared")
    assert (report_dirs[0] / "report.md").read_text().startswith("# Report: two-arms")
    assert all((report_dirs[0] / path).exists() for path in document["plots"].values())
    assert "a − b" in result.output


def test_json_flag_prints_only_the_document(tmp_path: Path) -> None:
    """With --json, stdout is the report document itself, for an agent to parse."""
    spec_file = _write_spec(tmp_path)
    _run(spec_file)

    result = runner.invoke(app, ["experiment", "report", str(spec_file), "--json", "--output-dir", str(tmp_path / "results" / "out")])

    assert result.exit_code == 0, result.output
    document = json.loads(result.stdout)
    assert document["schema_version"] == 1
    assert (tmp_path / "results" / "out" / "report.json").exists()


def test_a_report_without_a_launch_snapshot_is_exploratory(tmp_path: Path) -> None:
    """A plan added after the runs is reported, but labelled exploratory with the reason."""
    _run(_write_spec(tmp_path, declare_plan=False))
    spec_file = _write_spec(tmp_path)

    result = runner.invoke(app, ["experiment", "report", str(spec_file), "--json", "--output-dir", str(tmp_path / "results" / "out")])

    document = json.loads(result.stdout)
    assert document["plan_status"] == "exploratory"
    assert document["exploratory_reasons"][0].startswith("no plan snapshot before execution")


def test_a_diverged_seed_exits_non_zero_with_the_run_named(tmp_path: Path) -> None:
    """A NaN score fails the analysis loudly: exit 3, no results, the diverged run listed."""
    spec_file = _write_spec(tmp_path, diverge=True)
    _run(spec_file)

    result = runner.invoke(app, ["experiment", "report", str(spec_file), "--json", "--output-dir", str(tmp_path / "results" / "out")])

    assert result.exit_code == 3
    document = json.loads(result.stdout)
    assert (document["status"], document["analyses"]) == ("analysis_failed", [])
    assert [run["seed"] for run in document["failure"]["non_finite_runs"]] == [0]


def test_a_spec_without_a_plan_is_refused(tmp_path: Path) -> None:
    """Nothing is reported for a spec with no declared plan: there is nothing declared to run."""
    spec_file = _write_spec(tmp_path, declare_plan=False)
    _run(spec_file)

    result = runner.invoke(app, ["experiment", "report", str(spec_file)])

    assert result.exit_code == 1
    assert "no declared analysis plan" in result.output


@pytest.mark.parametrize(
    "outside",
    ["scratch", ".", "results/../escaped"],
    ids=["sibling-scratch-dir", "parent-working-dir", "dot-dot-escape"],
)
def test_an_output_dir_outside_the_results_root_is_refused(tmp_path: Path, outside: str) -> None:
    """A report written outside the results root would not be durable with its results, so nothing is written."""
    spec_file = _write_spec(tmp_path)
    _run(spec_file)

    result = runner.invoke(app, ["experiment", "report", str(spec_file), "--output-dir", str(tmp_path / outside)])

    assert result.exit_code == 1
    assert "outside the results root" in result.output
    assert not (tmp_path / outside / "report.json").exists()
    assert not (tmp_path / "results" / "reports").exists()
