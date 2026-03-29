---
name: diagnose-workspace
description: 'Run the current read-only workspace diagnostics. Use when running or interpreting `research doctor`, checking `research.yaml`, the configured Core checkout, or validating `uv` and JAX without mutating the workspace.'
---

# Diagnose Workspace

## When to Use
- Run `research doctor` against the current workspace root.
- Explain what the current diagnostic sweep checks today.
- Review config, Core Git health, and environment failures without mutating anything.
- Decide whether a later manual repair step is even warranted.

## Procedure
1. Read the current contract in [workspace diagnosis reference](./references/diagnose-workspace.md).
2. Confirm the command will run from the workspace root. Today, `research doctor` reads `research.yaml` only from the current working directory.
3. Run `research doctor`.
4. Review all reported groups before proposing follow-up: config validation, Git health, and environment health.
5. Keep the workflow strictly read-only:
   - do not edit `research.yaml`,
   - do not install or upgrade dependencies,
   - do not clean or reset the Core checkout,
   - do not invoke `research workspace repair` implicitly.
6. If the report suggests the narrow Core checkout repair flow, recommend `research workspace repair --dry-run` first. Otherwise, keep the guidance descriptive and grounded in the reported failures.
7. When summarizing results, stay inside the current command surface and messages.

## Notes
- Keep the guidance tied to `cli/src/research_cli/doctor.py`.
- Do not claim automatic remediation, dependency sync, or Git repair from this command.
