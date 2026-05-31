from dataclasses import dataclass

import jax

from process_control.actuators.dosing_system import DosingSystemState
from process_control.disturbances.schedule import DisturbanceSchedule
from process_control.scenarios.diurnal_source import DiurnalSourceState
from process_control.sensors.flow_sensor import FlowSensorState
from process_control.units.contact_basin import ContactBasinState


@dataclass(frozen=True)
class PlantState:
    step_count: jax.Array
    source_state: DiurnalSourceState
    basin_state: ContactBasinState
    flow_sensor_state: FlowSensorState
    dosing_loop: DosingSystemState
    last_dose: jax.Array
    disturbance_schedule: DisturbanceSchedule


jax.tree_util.register_dataclass(
    PlantState,
    data_fields=[
        "step_count",
        "source_state",
        "basin_state",
        "flow_sensor_state",
        "dosing_loop",
        "last_dose",
        "disturbance_schedule",
    ],
    meta_fields=[],
)
