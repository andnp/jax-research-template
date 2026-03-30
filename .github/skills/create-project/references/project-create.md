# Project Creation Reference

## Command Surface
Current command:
- `research project create <name> [--dry-run] [--github-repo <owner/name>]`

Primary implementation:
- `cli/src/research_cli/project.py`
- `tests/medium/test_project_create_render.py`
- `docs/specs/research-cli.md`
- `templates/copier.yml`

## Preconditions
The current implementation resolves the enclosing workspace root upward from the current directory.

That means it can run from the shell root or from inside an existing child project repo, but the resolved workspace root must still contain `projects/`.

If `projects/` is missing at the resolved workspace root, it exits with an error instead of trying to infer or create a workspace.

## What the Command Does Today
`research project create` currently:
- resolves the workspace root upward from the current working directory,
- resolves the template root from this repository's `templates/` directory,
- resolves the target directory as `projects/<name>`,
- prints the resolved workspace root, template root, and target path,
- makes the Git ownership contract explicit: shared workspace files stay owned by the shell repo, while the new child project gets its own `git init`,
- supports `--dry-run` previews,
- renders the template into `projects/` with Copier,
- passes `project_name=<name>` and otherwise accepts template defaults,
- initializes a Git repository in the rendered project,
- optionally runs `gh repo create <owner/name> --private --source <project-root> --remote origin`.

## What It Does Not Do
The current implementation does not:
- prompt through all Copier answers from the CLI,
- update workspace configuration after project creation,
- create a GitHub repository unless `--github-repo` is supplied,
- overwrite an existing `projects/<name>` target.

## Expected Rendered Shape
The current medium test verifies that a successful non-dry-run create produces:
- `projects/<name>/README.md`
- `projects/<name>/pyproject.toml`
- `projects/<name>/train.py`
- `projects/<name>/.git/`

The test also verifies that the render lands directly in `projects/<name>` and does not create a nested `projects/<name>/<name>/` directory.

## Template Defaults
The current template definition includes default answers for fields such as:
- `description`
- `author`
- `python_version`
- `algorithm`
- `env_name`
- `num_seeds`

Because the CLI only passes `project_name`, these values currently come from `templates/copier.yml` defaults unless the implementation changes.
