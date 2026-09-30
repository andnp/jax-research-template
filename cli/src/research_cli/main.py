# The CLI bootstrap must settle the JAX platform before runtime imports.
# ruff: noqa: E402

from __future__ import annotations

import os
import sys

# JAX reads JAX_PLATFORMS when it is first imported (transitively, via research_runner
# below), so this must be decided before any other import in this file.
if not ("experiment" in sys.argv and "run" in sys.argv and "--gpu" in sys.argv):
    os.environ["JAX_PLATFORMS"] = "cpu"

from research_cli.supervisor import die_with_supervisor

die_with_supervisor()

import typer

from research_cli.doctor import doctor_command
from research_cli.experiment import experiment_app
from research_cli.lifecycle import eject, harvest
from research_cli.project import project_app
from research_cli.workspace import workspace_app

app = typer.Typer(help="RL Research Monorepo Management CLI")
app.add_typer(project_app, name="project")
app.add_typer(workspace_app, name="workspace")
app.add_typer(experiment_app, name="experiment")
app.command(name="doctor", help="Run a read-only diagnostic sweep for this workspace.")(doctor_command)
app.command(help="Eject a shared library into a project-local components/ package.")(eject)
app.command(help="Harvest a project-local component into libs/.")(harvest)


@app.command()
def info() -> None:
    """Display information about the monorepo core."""
    typer.echo("RL Research Core v0.1.0")


if __name__ == "__main__":
    app()
