# Technical Specification: Repo-Shared Agent Workflow Skills

## 1. Purpose
This repository exposes a small set of repo-shared workflow skills under `.github/skills/` so agent workflows can discover stable, reusable procedures without loading full repo documentation on every task.

This first slice defines the initial architecture and seeds two skills that are truthful against the current repository state:
- `bootstrap-shell`
- `create-project`

## 2. Scope
These skills cover repeatable workflow guidance for:
- bootstrapping a new research shell workspace with the current `research workspace init` command,
- creating a new project inside an existing workspace with the current `research project create` command.

They do not add new automation, wrapper scripts, or CLI behavior.

## 3. Design Constraints
- Skill folders live under `.github/skills/<stable-name>/`.
- Each folder name must match the `name` field in `SKILL.md`.
- Names must be product-oriented and stable rather than issue-specific.
- `SKILL.md` stays lean, keyword-rich, and procedural so it works as the discovery and execution surface.
- Repo-specific contracts, caveats, and command details live in `references/`.
- `assets/` is optional and should be added only when a reusable artifact is needed.
- Content must reflect the current repo truth from code, tests, and docs. Do not claim hidden automation or future behavior.

## 4. Current Source of Truth
The first two skills are grounded in the current implementation and docs:
- `cli/src/research_cli/workspace.py`
- `cli/src/research_cli/project.py`
- `tests/small/test_workspace_init.py`
- `tests/medium/test_project_create_render.py`
- `docs/specs/research-cli.md`
- `CONTRIBUTING.md`
- `templates/copier.yml`

## 5. Initial Skill Inventory
### 5.1 `bootstrap-shell`
Use for creating or previewing a shell workspace with `research workspace init`.

Current contract to preserve:
- creates a new workspace directory,
- initializes Git,
- creates `projects/`,
- optionally adds `core/` as a git submodule when `--core-url` is supplied,
- writes `pyproject.toml`, `research.yaml`, and `.gitignore`,
- supports `--dry-run`,
- does not run `uv sync` or install hooks automatically.

### 5.2 `create-project`
Use for creating or previewing a project with `research project create`.

Current contract to preserve:
- must run from a workspace root containing `projects/`,
- renders the template from `templates/` into `projects/<name>`,
- supports `--dry-run`,
- initializes a Git repository in the rendered project,
- optionally creates a private GitHub remote with `gh repo create` when `--github-repo` is provided,
- currently passes only `project_name` into the copier render and relies on template defaults for the remaining answers.

## 6. Directory Layout
```text
.github/skills/
├── README.md
├── bootstrap-shell/
│   ├── SKILL.md
│   └── references/
│       └── workspace-bootstrap.md
└── create-project/
    ├── SKILL.md
    └── references/
        └── project-create.md
```

## 7. Verification
This slice is verified by:
- manual frontmatter review,
- reading back the changed Markdown files,
- lightweight repo checks only when they provide real signal for Markdown or customization-file structure.

No dedicated skill-file checker currently exists in this repository.
