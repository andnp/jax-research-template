"""Configurable BSM1 benchmark on the coupled activated-sludge flowsheet.

Alex et al. (2008), Benchmark Simulation Model no. 1:
https://www2.iea.lth.se/publications/Reports/LTH-IEA-7229.pdf

Each control handle of Table 15 is driven by the agent, by a PI controller on a sensed plant quantity
(whose setpoint may itself be an agent action), or held fixed. Sensors, controllers, actuators, the
influent and the plant advance every control interval; the agent acts every `intervals_per_step`
intervals. Times are in hours, flows in m³/h and kLa in h⁻¹ throughout, actions included; the
performance indices in `info` keep the BSM1 per-day units.

The benchmark needs float64 (`jax_enable_x64`): in float32 the slow particulate states stop moving after
weeks of simulated time. Observations and rewards are returned as float32.
"""

from dataclasses import dataclass

from process_control.controllers.pi_controller import BSM1_NITRATE_CONTROLLER, BSM1_OXYGEN_CONTROLLER, PIControllerParams
from process_control.sensors.response_time import BSM1_NITRATE_SENSOR, BSM1_OXYGEN_SENSOR, SensorParams

_DAY = 24.0
Q0_STAB = 18446.0 / _DAY

HANDLES = ("kla_1", "kla_2", "kla_3", "kla_4", "kla_5", "q_a", "q_r", "q_w")
KLA_HANDLES = HANDLES[:5]
# Table 15, converted to hours.
HANDLE_LIMITS = {
    **dict.fromkeys(KLA_HANDLES, (0.0, 360.0 / _DAY)),
    "q_a": (0.0, 92230.0 / _DAY),
    "q_r": (0.0, 36892.0 / _DAY),
    "q_w": (0.0, 1844.6 / _DAY),
}


@dataclass(frozen=True)
class Fixed:
    """The handle is held at `value`."""

    value: float


@dataclass(frozen=True)
class Agent:
    """The handle follows the agent action `action`, which every handle naming it shares; by default the handle's own name."""

    action: str | None = None


@dataclass(frozen=True)
class AgentSetpoint:
    """A controller setpoint the agent chooses within [low, high]."""

    low: float
    high: float


@dataclass(frozen=True)
class PIControl:
    """The handle follows a PI controller acting on a sensed plant quantity toward a fixed or agent-chosen setpoint."""

    measurement: str
    setpoint: float | AgentSetpoint
    controller: PIControllerParams
    sensor: SensorParams


type HandleControl = Fixed | Agent | PIControl

# The BSM1 default controllers of Sec. 5: DO in reactor 5 by kLa5, nitrate in reactor 2 by Qa.
BSM1_OXYGEN_CONTROL = PIControl("s_o_5", 2.0, BSM1_OXYGEN_CONTROLLER, BSM1_OXYGEN_SENSOR)
BSM1_NITRATE_CONTROL = PIControl("s_no_2", 1.0, BSM1_NITRATE_CONTROLLER, BSM1_NITRATE_SENSOR)


@dataclass(frozen=True)
class BSM1Handles:
    """How each control handle of Table 15 is driven. Every kLa command passes through the BSM1 aeration actuator.

    The default lets the agent set a shared kLa for reactors 3 and 4, kLa5 and the wastage Qw, with Qa on
    the BSM1 nitrate controller, Qr at Q0,stab and the anoxic reactors unaerated.
    """

    kla_1: HandleControl = Fixed(0.0)
    kla_2: HandleControl = Fixed(0.0)
    kla_3: HandleControl = Agent("kla_34")
    kla_4: HandleControl = Agent("kla_34")
    kla_5: HandleControl = Agent()
    q_a: HandleControl = BSM1_NITRATE_CONTROL
    q_r: HandleControl = Fixed(Q0_STAB)
    q_w: HandleControl = Agent()

    def items(self) -> tuple[tuple[str, HandleControl], ...]:
        controls = (self.kla_1, self.kla_2, self.kla_3, self.kla_4, self.kla_5, self.q_a, self.q_r, self.q_w)
        return tuple(zip(HANDLES, controls, strict=True))

    def __post_init__(self) -> None:
        for name, control in self.items():
            low, high = HANDLE_LIMITS[name]
            if isinstance(control, Fixed) and not low <= control.value <= high:
                raise ValueError(f"fixed {name} = {control.value} lies outside its limits [{low}, {high}]")


@dataclass(frozen=True)
class ActionSpace:
    names: tuple[str, ...]
    low: tuple[float, ...]
    high: tuple[float, ...]


def action_space(handles: BSM1Handles) -> ActionSpace:
    """The agent's actions in order of first use: shared handle commands, then any supervised setpoints."""
    limits: dict[str, tuple[float, float]] = {}
    for name, control in handles.items():
        if not isinstance(control, Agent):
            continue
        action = control.action or name
        if limits.setdefault(action, HANDLE_LIMITS[name]) != HANDLE_LIMITS[name]:
            raise ValueError(f"action {action!r} drives handles with different limits")
    for name, control in handles.items():
        if not isinstance(control, PIControl) or not isinstance(control.setpoint, AgentSetpoint):
            continue
        action = f"{name}_setpoint"
        if action in limits:
            raise ValueError(f"action {action!r} is already a handle command")
        limits[action] = (control.setpoint.low, control.setpoint.high)
    return ActionSpace(
        names=tuple(limits), low=tuple(low for low, _ in limits.values()), high=tuple(high for _, high in limits.values())
    )
