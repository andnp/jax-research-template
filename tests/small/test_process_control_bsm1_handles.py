import dataclasses

import pytest
from process_control.benchmarks.bsm1 import (
    BSM1_NITRATE_CONTROL,
    BSM1_OXYGEN_CONTROL,
    HANDLE_LIMITS,
    Agent,
    AgentSetpoint,
    BSM1Handles,
    Fixed,
    action_space,
)

KLA_MAX = 360.0 / 24.0


class TestActionSpace:
    def test_default_actions_share_kla_3_and_4(self) -> None:
        """The default agent sets one kLa for reactors 3 and 4, kLa5 and Qw, within the Table 15 limits."""
        space = action_space(BSM1Handles())

        assert space.names == ("kla_34", "kla_5", "q_w")
        assert space.low == (0.0, 0.0, 0.0)
        assert space.high == (KLA_MAX, KLA_MAX, 1844.6 / 24.0)

    def test_handles_without_agent_control_add_no_actions(self) -> None:
        """Fixed and fixed-setpoint PI handles leave the agent nothing to choose."""
        handles = BSM1Handles(kla_3=Fixed(10.0), kla_4=Fixed(10.0), kla_5=BSM1_OXYGEN_CONTROL, q_w=Fixed(16.0))

        assert action_space(handles).names == ()

    def test_supervised_setpoints_follow_the_handle_commands(self) -> None:
        """An agent-chosen setpoint becomes an action after the direct handle commands, with its own bounds."""
        handles = BSM1Handles(q_a=dataclasses.replace(BSM1_NITRATE_CONTROL, setpoint=AgentSetpoint(0.0, 4.0)))
        space = action_space(handles)

        assert space.names == ("kla_34", "kla_5", "q_w", "q_a_setpoint")
        assert (space.low[-1], space.high[-1]) == (0.0, 4.0)

    def test_an_action_cannot_drive_handles_with_different_limits(self) -> None:
        """Sharing one action between a kLa and a flow would give it two incompatible ranges."""
        with pytest.raises(ValueError, match="different limits"):
            action_space(BSM1Handles(kla_5=Agent("shared"), q_w=Agent("shared")))

    def test_setpoint_actions_cannot_shadow_handle_commands(self) -> None:
        """A handle command named like a setpoint action would make the layout ambiguous."""
        handles = BSM1Handles(kla_5=Agent("q_a_setpoint"), q_a=dataclasses.replace(BSM1_NITRATE_CONTROL, setpoint=AgentSetpoint(0.0, 4.0)))

        with pytest.raises(ValueError, match="already a handle command"):
            action_space(handles)


class TestHandleValidation:
    @pytest.mark.parametrize(("handle", "value"), [("q_w", -1.0), ("kla_1", KLA_MAX + 1.0), ("q_r", HANDLE_LIMITS["q_r"][1] * 2.0)])
    def test_fixed_values_must_respect_the_limits(self, handle: str, value: float) -> None:
        """A handle fixed outside its Table 15 range is rejected."""
        with pytest.raises(ValueError, match=f"fixed {handle}"):
            BSM1Handles(**{handle: Fixed(value)})
