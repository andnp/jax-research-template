from typing import Protocol

import jax

from process_control.components import ComponentSchema


class KineticsModel[P](Protocol):
    """Reaction rates over a component vector, with parameters correctable to a temperature."""

    @property
    def schema(self) -> ComponentSchema: ...

    @property
    def oxygen(self) -> str:
        """Name of the dissolved-oxygen component aeration transfers into."""
        ...

    def at_temperature(self, params: P, temperature: jax.Array) -> P: ...

    def rates(self, components: jax.Array, params: P) -> jax.Array: ...
