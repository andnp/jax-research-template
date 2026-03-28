#!/usr/bin/env python3
import argparse
import json
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

# --- Constants & Configuration ---

REPO_ROOT = Path(__file__).resolve().parent.parent

MUTATING_PREFIXES = ("edit/", "replace_", "write_", "create_", "delete_", "insert_", "git.", "but ", "multi_replace_")
MUTATING_SUBSTRINGS = ("write", "replace", "create", "delete", "insert", "append", "update", "apply", "patch", "save", "modify", "manage_todo")
SPECIFIC_MUTATORS = {"docs.insertText", "docs.appendText", "docs.replaceText", "docs.create", "docs.move", "merge_memories"}

# --- Data Structures ---

@dataclass
class HookContext:
    tool_name: str = ""
    session_id: str = ""
    timestamp: str = ""
    changed_files: list[str] = field(default_factory=list)
    is_mutating: bool = False

    @classmethod
    def from_stdin(cls) -> "HookContext":
        ctx = cls()
        try:
            raw = sys.stdin.read()
            if raw:
                data = json.loads(raw)
                ctx.tool_name = data.get("tool_name") or data.get("tool") or data.get("action", "")
                ctx.session_id = data.get("sessionId") or data.get("session_id", "")
                ctx.timestamp = data.get("timestamp", "")
        except Exception:
            pass

        ctx.is_mutating = ctx._check_mutation()
        if ctx.is_mutating:
            ctx.changed_files = ctx._get_git_changes()
        return ctx

    def _check_mutation(self) -> bool:
        if not self.tool_name:
            return True
        low = self.tool_name.lower()
        return (
            low.startswith(MUTATING_PREFIXES)
            or any(s in low for s in MUTATING_SUBSTRINGS)
            or self.tool_name in SPECIFIC_MUTATORS
        )

    def _get_git_changes(self) -> list[str]:
        res = subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True).stdout
        paths = []
        for line in res.splitlines():
            if len(line) <= 3:
                continue
            path = line[3:].strip().strip('"')
            paths.append(path.split(" -> ")[-1] if " -> " in path else path)
        return paths

# --- Utility Helpers ---

def sh(cmd: list[str], cwd: Path = REPO_ROOT) -> subprocess.CompletedProcess:
    """Convenience helper for running shell commands."""
    return subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)

def find_subrepo(path: str) -> str | None:
    """Finds the subrepo directory (containing pyproject.toml) for a path."""
    current = (REPO_ROOT / path).resolve()
    if current.is_file():
        current = current.parent
    candidates = [current] + list(current.parents)
    for p in candidates:
        if (p / "pyproject.toml").exists():
            return str(p.relative_to(REPO_ROOT))
        if p == REPO_ROOT:
            break
    return None

# --- Main Logic ---

def run_post_tool_hooks(ctx: HookContext) -> None:
    messages = []
    has_blocking_errors = False

    if ctx.is_mutating and ctx.changed_files:
        python_files = [f for f in ctx.changed_files if f.endswith(".py")]
        python_subrepos = {sub for f in python_files if (sub := find_subrepo(f))}

        for sub in sorted(python_subrepos):
            sub_path = REPO_ROOT / sub
            # Use 'uv run' to ensure we use the workspace's environment
            ruff = sh(["uv", "run", "ruff", "check", "."], sub_path)
            ty = sh(["uv", "run", "ty", "check", "."], sub_path)
            
            # Run pyright only on changed files in this subrepo
            sub_files = [str(REPO_ROOT / f) for f in python_files if find_subrepo(f) == sub]
            pyright = sh(["uv", "run", "pyright"] + sub_files, sub_path)

            failed = []
            if ruff.returncode != 0:
                failed.append("ruff")
            if ty.returncode != 0:
                failed.append("ty")
            if pyright.returncode != 0:
                failed.append("pyright")

            if failed:
                has_blocking_errors = True
                err = f"[Hook] Validation failed for subrepo: {sub}\n"
                if ruff.returncode != 0:
                    err += f"--- Ruff Errors ---\n{ruff.stdout}{ruff.stderr}\n"
                if ty.returncode != 0:
                    err += f"--- Ty Errors ---\n{ty.stdout}{ty.stderr}\n"
                if pyright.returncode != 0:
                    err += f"--- Pyright Errors (changed files) ---\n{pyright.stdout}{pyright.stderr}"
                messages.append(err)

    response = {}
    if messages:
        msg_text = "\n\n".join(messages)
        if has_blocking_errors:
            response["decision"] = "block"
            response["reason"] = msg_text
            response["systemMessage"] = msg_text
        else:
            response["systemMessage"] = msg_text
            response["additionalContext"] = msg_text

    print(json.dumps(response))

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("hook_type", choices=["session-start", "session-end", "post-tool-use"])
    args = parser.parse_args()

    if args.hook_type == "session-start":
        msg = (
            "💡 REMINDER: Use 'search_memories' to load recent context,\n"
            "   architectural goals, and active work journals."
        )
        print(json.dumps({
            "systemMessage": msg,
            "additionalContext": msg
        }))
    elif args.hook_type == "session-end":
        msg = (
            "💾 SESSION COMPLETE: Consider using 'create_memory' or\n"
            "   'append_memory' to log progress, decisions, or new facts."
        )
        print(json.dumps({"systemMessage": msg}))
    elif args.hook_type == "post-tool-use":
        run_post_tool_hooks(HookContext.from_stdin())

if __name__ == "__main__":
    main()
