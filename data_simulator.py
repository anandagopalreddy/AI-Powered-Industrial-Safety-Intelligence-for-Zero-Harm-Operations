"""
Simulates the heterogeneous data sources a real plant would expose:
IoT gas sensors, SCADA-style readings, permit-to-work system, maintenance
CMMS activity, and worker location pings (e.g. from RFID/BLE badges).

In production these would be replaced by real connectors (OPC-UA, MQTT,
REST APIs into the permit/CMMS systems). The simulator keeps the same
data shapes so the risk engine and frontend need no changes to go live.
"""
import random
import uuid
from datetime import datetime, timedelta

from models import (
    GasReading, GasType, MaintenanceActivity, Permit, PermitType,
    WorkerLocation, Zone,
)

ZONES = [
    Zone(zone_id="Z1", name="Coke Oven Battery 3", x=20, y=30, hazard_class="COKE_OVEN"),
    Zone(zone_id="Z2", name="Gas Cleaning Plant", x=45, y=25, hazard_class="CONFINED_SPACE"),
    Zone(zone_id="Z3", name="Blast Furnace Bay", x=70, y=35, hazard_class="HIGH_TEMP"),
    Zone(zone_id="Z4", name="Tank Farm - Tar Storage", x=25, y=65, hazard_class="TANK_FARM"),
    Zone(zone_id="Z5", name="Sinter Plant", x=55, y=70, hazard_class="GENERAL"),
    Zone(zone_id="Z6", name="Rolling Mill", x=80, y=65, hazard_class="GENERAL"),
]

GAS_BASELINES = {
    GasType.H2S: dict(base=2.0, warn=10.0, crit=20.0),
    GasType.CO: dict(base=15.0, warn=50.0, crit=100.0),
    GasType.CH4: dict(base=0.5, warn=5.0, crit=10.0),
    GasType.O2: dict(base=20.9, warn=19.5, crit=18.0),  # low is bad
}

WORKER_NAMES = [
    "R. Kumar", "S. Patel", "A. Sharma", "V. Reddy", "M. Singh",
    "P. Nair", "K. Iyer", "D. Verma", "N. Joshi", "T. Rao",
]


class PlantState:
    """Holds the mutable in-memory state of the simulated plant."""

    def __init__(self):
        self.zones = ZONES
        self.gas_readings: dict[str, GasReading] = {}
        self.permits: dict[str, Permit] = {}
        self.maintenance: dict[str, MaintenanceActivity] = {}
        self.workers: dict[str, WorkerLocation] = {}
        self.incident_mode: dict[str, bool] = {z.zone_id: False for z in self.zones}
        self._seed()

    def _seed(self):
        now = datetime.utcnow()
        # seed baseline gas sensors, 1-2 per zone per gas type of interest
        for zone in self.zones:
            gas_types = [GasType.CO, GasType.H2S] if zone.hazard_class in (
                "COKE_OVEN", "CONFINED_SPACE", "TANK_FARM") else [GasType.CO]
            for gt in gas_types:
                sensor_id = f"{zone.zone_id}-{gt.value}-01"
                cfg = GAS_BASELINES[gt]
                self.gas_readings[sensor_id] = GasReading(
                    sensor_id=sensor_id, zone_id=zone.zone_id, gas_type=gt,
                    value_ppm=cfg["base"], threshold_warning=cfg["warn"],
                    threshold_critical=cfg["crit"], timestamp=now,
                )
        # seed a couple of active permits
        self._issue_permit("Z2", PermitType.CONFINED_SPACE, "Contractor Crew B")
        self._issue_permit("Z5", PermitType.GENERAL, "Shift Team A")
        # seed a maintenance job
        self._start_maintenance("Z1", "Scheduled gas pressure valve inspection")
        # seed worker locations
        for i, wid in enumerate([f"W{i:03d}" for i in range(1, 9)]):
            zone = random.choice(self.zones)
            self.workers[wid] = WorkerLocation(
                worker_id=wid, name=WORKER_NAMES[i % len(WORKER_NAMES)],
                zone_id=zone.zone_id,
            )

    def _issue_permit(self, zone_id, ptype, crew):
        pid = f"P-{uuid.uuid4().hex[:6].upper()}"
        now = datetime.utcnow()
        self.permits[pid] = Permit(
            permit_id=pid, permit_type=ptype, zone_id=zone_id,
            issued_to=crew, issued_at=now, valid_until=now + timedelta(hours=8),
        )
        return pid

    def _start_maintenance(self, zone_id, description):
        aid = f"M-{uuid.uuid4().hex[:6].upper()}"
        self.maintenance[aid] = MaintenanceActivity(
            activity_id=aid, zone_id=zone_id, description=description,
        )
        return aid

    def tick(self):
        """Advance the simulation by one step — random walk on gas readings,
        with a slow drift so the dashboard feels alive."""
        for reading in self.gas_readings.values():
            cfg = GAS_BASELINES[reading.gas_type]
            drift = random.uniform(-0.6, 0.6)
            if self.incident_mode.get(reading.zone_id) and reading.gas_type in (GasType.CO, GasType.H2S):
                drift += random.uniform(1.5, 4.0)  # escalating during incident
            new_val = max(0.0, reading.value_ppm + drift)
            # gentle pull back toward baseline when not in incident mode
            if not self.incident_mode.get(reading.zone_id):
                new_val += (cfg["base"] - new_val) * 0.1
            reading.value_ppm = round(new_val, 2)
            reading.timestamp = datetime.utcnow()

    def trigger_incident(self, zone_id: str):
        """Simulate the Vizag-style scenario: gas accumulation begins in a
        zone that also has active maintenance work — the compound condition
        the engine should catch."""
        self.incident_mode[zone_id] = True
        if not any(m.zone_id == zone_id and m.active for m in self.maintenance.values()):
            self._start_maintenance(zone_id, "Emergency valve maintenance (unplanned)")
        # also issue a hot work permit nearby to create the dangerous combination
        self._issue_permit(zone_id, PermitType.HOT_WORK, "Contractor Crew D")

    def clear_incident(self, zone_id: str):
        self.incident_mode[zone_id] = False
        for r in self.gas_readings.values():
            if r.zone_id == zone_id:
                cfg = GAS_BASELINES[r.gas_type]
                r.value_ppm = cfg["base"]


plant_state = PlantState()
