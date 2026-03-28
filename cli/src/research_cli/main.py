import typer

from research_cli.project import project_app
from research_cli.workspace import workspace_app

app = typer.Typer(help="RL Research Monorepo Management CLI")
app.add_typer(project_app, name="project")
app.add_typer(workspace_app, name="workspace")


@app.command()
def info() -> None:
    """Display information about the monorepo core."""
    typer.echo("RL Research Core v0.1.0")


if __name__ == "__main__":
    app()
