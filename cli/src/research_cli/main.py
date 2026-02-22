import typer

app = typer.Typer(help="RL Research Monorepo Management CLI")

@app.command()
def info():
    """Display information about the monorepo core."""
    typer.echo("RL Research Core v0.1.0")

if __name__ == "__main__":
    app()
