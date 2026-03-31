# Workspace Diagnosis Reference

## Command Surface
Current command:
- `research doctor`

Primary implementation:
- `cli/src/research_cli/doctor.py`
- `tests/medium/test_cli_doctor.py`
- `docs/specs/research-cli.md`

## Preconditions
The current implementation resolves the enclosing workspace root upward from `Path.cwd()` and loads `research.yaml` from that root.

That means it can be run from the shell root or from inside a child project repo, as long as the enclosing workspace contains `research.yaml`.

## What the Command Does Today
`research doctor` currently:
- resolves the enclosing workspace root upward from the current working directory,
- loads `research.yaml` from that workspace root,
- reports config validation for that file,
- resolves the configured `core_path` relative to the current working directory,
- resolves the configured `core_path` relative to the workspace root,
- checks whether the resolved Core path exists,
- checks whether that path is a Git working tree,
- checks whether `HEAD` is attached to a branch,
- checks whether Git status is clean, treating any porcelain output, including untracked files, as a failure,
- checks that `uv` responds to `--version`,
- attempts to import and probe JAX without mutating the environment,
- compares detected accelerators against `doctor.expected_accelerators` when configured,
- renders all three diagnostic groups plus an overall `PASS` or `FAIL` line,
- exits non-zero when any diagnostic fails.

## What It Does Not Do
The current implementation does not:
- edit `research.yaml`,
- install or upgrade packages,
- run `uv sync`,
- clean or reset the Core checkout,
- invoke `research workspace repair`,
- search parent directories for `research.yaml`.

## Failure Reporting
The current report always includes these groups:
- `Config validation`
- `Git health`
- `Environment health`

If config loading fails, downstream Git and accelerator checks are still rendered as failures with blocked explanations instead of being skipped silently.

## Output Scope
The command is diagnostic only. It reports current state for:
- workspace config shape,
- the configured Core checkout,
- local `uv` availability,
- local JAX import and accelerator detection.

It does not attempt remediation.
