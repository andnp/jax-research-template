from __future__ import annotations

from experiment_definition.db import ReclaimReport
from research_cli.reclaim import describe_reclaim


class TestDescribeReclaim:
    def test_clean_reclaim_is_one_short_line(self) -> None:
        """Nothing unresolved means no hint, just the count."""
        assert describe_reclaim(ReclaimReport([4, 5], [], [])) == "Reclaimed 2 dead execution(s)"

    def test_unresolved_executions_are_counted_with_the_invalidate_hint(self) -> None:
        """Unknown and ownerless live executions are counted separately and point at invalidate."""
        line = describe_reclaim(ReclaimReport([], [1, 2], [3]))

        assert line.startswith("Reclaimed 0 dead execution(s); 2 with an owner that cannot be checked and 1 without an owner")
        assert "research experiment invalidate" in line
