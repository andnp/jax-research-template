import dataclasses

from research_runner.config import traced


@dataclasses.dataclass(frozen=True)
class Opt:
    LR: float = traced(1e-3)
    STEPS: int = 10


class TestTraced:
    def test_field_keeps_its_declared_default(self):
        """A traced field behaves as an ordinary defaulted dataclass field."""
        assert Opt() == Opt(LR=1e-3, STEPS=10)

    def test_field_can_be_overridden_at_construction(self):
        """Marking a field traced does not change how it is set."""
        assert Opt(LR=0.5).LR == 0.5
