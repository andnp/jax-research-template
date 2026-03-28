#!/bin/sh
set -eu

worktree_root="$(git rev-parse --show-toplevel)"
cd "$worktree_root"

git config --local core.hooksPath .githooks

git_common_dir="$(git rev-parse --git-common-dir)"
printf 'Configured shared Git hooks in %s/config with core.hooksPath=%s\n' "$git_common_dir" '.githooks'
