"""Collector for structured dataset-loading diagnostics."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field

from research_analysis.diagnostics.types import DatasetDiagnostic, DiagnosticCode


@dataclass
class DiagnosticAccumulator:
    """Collects DatasetDiagnostic entries in place, avoiding a return-value threading convention."""

    _items: list[DatasetDiagnostic] = field(default_factory=list)

    def add(
        self,
        code: DiagnosticCode,
        message: str,
        run_id: int | None = None,
        execution_id: int | None = None,
    ) -> None:
        self._items.append(DatasetDiagnostic(code, message, run_id, execution_id))

    def extend(self, diagnostics: Iterable[DatasetDiagnostic]) -> None:
        self._items.extend(diagnostics)

    def fatal(self, fatal_codes: frozenset[DiagnosticCode]) -> tuple[DatasetDiagnostic, ...]:
        return tuple(item for item in self._items if item.code in fatal_codes)

    def as_tuple(self) -> tuple[DatasetDiagnostic, ...]:
        return tuple(self._items)
