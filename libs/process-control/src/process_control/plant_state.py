from dataclasses import dataclass

import jax

from process_control.actuators.dose_pump import DosePumpState
from process_control.controllers.pi_controller import PIControllerState
from process_control.disturbances.schedule import DisturbanceSchedule
from process_control.scenarios.diurnal_source import DiurnalSourceState
from process_control.sensors.flow_sensor import FlowSensorState
from process_control.sensors.residual_analyzer import ResidualAnalyzerState
from process_control.units.contact_basin import ContactBasinState


@dataclass(frozen=True)
class PlantState:
    step_count: jax.Array
    source_state: DiurnalSourceState
    basin_state: ContactBasinState
    flow_sensor_state: FlowSensorState
    residual_sensor_state: ResidualAnalyzerState
    dose_pump_state: DosePumpState
    pi_state: PIControllerState
    last_dose: jax.Array
    disturbance_schedule: DisturbanceSchedule


jax.tree_util.register_dataclass(
    PlantState,
    data_fields=[
        "step_count",
        "source_state",
        "basin_state",
        "flow_sensor_state",
        "residual_sensor_state",
        "dose_pump_state",
        "pi_state",
        "last_dose",
        "disturbance_schedule",
    ],
    meta_fields=[],
)
