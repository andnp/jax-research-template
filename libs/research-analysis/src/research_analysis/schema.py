"""Source-key mappings for experiment analysis metadata."""

from collections.abc import Mapping
from dataclasses import dataclass


@dataclass(frozen=True)
class AnalysisSchema:
    """Map stored experiment metadata onto analysis dimensions."""

    condition_key: str = "condition_name"
    environment_key: str = "env_name"

    def condition_name(self, hyperparameters: Mapping[str, object]) -> str:
        """Return the mapped condition value or reject incomplete metadata."""
        return self._required_value(hyperparameters, self.condition_key, "condition")

    def environment_name(self, hyperparameters: Mapping[str, object]) -> str:
        """Return the mapped environment value or reject incomplete metadata."""
        return self._required_value(hyperparameters, self.environment_key, "environment")

    @staticmethod
    def _required_value(
        hyperparameters: Mapping[str, object], key: str, dimension: str
    ) -> str:
        value = hyperparameters.get(key)
        if value is None:
            raise ValueError(f"Missing {dimension} metadata key {key!r}.")
        return str(value)
