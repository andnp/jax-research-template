from experiment_definition.db import RunBatch
from research_cli.supervisor import PendingGroup, pending_groups


def _batch(group_key: str, runs: int) -> RunBatch:
    return RunBatch(None, None, None, {}, None, list(range(runs)), group_key)


class TestPendingGroups:
    def test_batches_of_one_group_are_counted_together(self) -> None:
        """A group split into several batches by a batch cap is still one group with all its runs."""
        assert pending_groups([_batch("a", 2), _batch("a", 2), _batch("a", 1)]) == [PendingGroup("a", 5)]

    def test_most_pending_group_starts_first(self) -> None:
        """Groups are ordered by descending pending runs, keeping listing order for ties."""
        batches = [_batch("small", 1), _batch("tie-1", 2), _batch("big", 3), _batch("tie-2", 2)]
        assert [group.key for group in pending_groups(batches)] == ["big", "tie-1", "tie-2", "small"]
