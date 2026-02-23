[project]
name = "{name}"
version = "0.1.0"
description = "Research shell workspace"
requires-python = ">=3.13"
dependencies = []

[build-system]
requires = ["setuptools>=61.0"]
build-backend = "setuptools.build_meta"

[tool.uv]
package = false

[tool.uv.workspace]
members = [
    "core/libs/*",
    "projects/*",
]
