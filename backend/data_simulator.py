"""
Zero-Harm — Data Simulator
============================
Models a six-zone plant layout matching the hazard classes referenced in the
ET AI Hackathon 2026 brief. Generates realistic gas sensor drift, seeds active
permits and maintenance jobs, and exposes trigger_incident()/clear_incident()
so a judge can reproduce the compound failure pattern (Vizag, Jan 2025) on demand.

In production, this module is replaced by real OPC-UA/MQTT/REST connectors —
nothing else in the system needs to change, because this simulator produces
the exact data shapes a real integration would produce.
"""

import random
import time
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional


# ---------------------------------------------------------------------------
# Static plant layout — six zones matching common heavy-industrial hazard classes
# ---------------------------------------------------------------------------
@dataclass
class Zone:
    zone_id: str
    name: str
    x: float
    y: float
    hazard_class: str


ZONES: List[Zone] = [
    Zone("Z1", "Coke Oven Battery 3", 120, 90, "toxic_gas / fire"),
    Zone("Z2", "Gas Cleaning Plant", 320, 90, "toxic_gas / confined_space"),
    Zone("Z3", "Blast Furnace Bay", 520, 90, "high_temp / fire"),
    Zone("Z4", "Tar Tank Farm", 120, 260, "flammable / fire"),
    Zone("Z5", "Sinter Plant", 320, 260, "particulate / fire"),
    Zone("Z6", "Rolling Mill", 520, 260, "mechanical / fire"),
]

GAS_TYPE_BY_ZONE = {
    "Z1": "CO",
    "Z2": "H2S",
    "Z3": "CO",
    "Z4": "CH4",
    "Z5": "CO",
    "Z6": "O2_deficiency",
}

# Per-gas thresholds (ppm), except O2_deficiency which is inverted (percent) —
# kept simple/uniform here for the hackathon demo.
WARNING_THRESHOLD = 40.0
CRITICAL_THRESHOLD = 80.0


@dataclass
class GasReading:
    sensor_id: str
    zone_id: str
    gas_type: str
    value_ppm: float
    warning_threshold: float = WARNING_THRESHOLD
    critical_threshold: float = CRITICAL_THRESHOLD


@dataclass
class Permit:
    permit_id: str
    permit_type: str  # "hot_work" | "confined_space" | "general"
    zone_id: str
    issued_to: str
    status: str  # "active" | "closed"
    issued_at: float = field(default_factory=time.time)


@dataclass
class MaintenanceActivity:
    activity_id: str
    zone_id: str
    description: str
    active: bool = True


@dataclass
class WorkerLocation:
    worker_id: str
    name: str
    zone_id: str
    role: str = "Operator"
    timestamp: float = field(default_factory=time.time)


class PlantSimulator:
    """Holds all live plant state and advances it one tick at a time."""

    def __init__(self):
        self.zones: Dict[str, Zone] = {z.zone_id: z for z in ZONES}
        self.gas_readings: Dict[str, GasReading] = {}
        self.permits: List[Permit] = []
        self.maintenance: List[MaintenanceActivity] = []
        self.workers: List[WorkerLocation] = []
        self._incident_zones: set = set()
        self._permit_counter = 0
        self._activity_counter = 0
        self._worker_counter = 0

        self.previous_values: Dict[str, float] = {}
        self.event_log: List[dict] = []

        for z in ZONES:
            gas_type = GAS_TYPE_BY_ZONE[z.zone_id]
            initial_value = round(random.uniform(8, 18), 1)
            self.gas_readings[z.zone_id] = GasReading(
                sensor_id=f"GS-{z.zone_id}",
                zone_id=z.zone_id,
                gas_type=gas_type,
                value_ppm=initial_value,
            )
            self.previous_values[z.zone_id] = initial_value

        # Baseline worker roster — realistic mixed roles spread across zones,
        # not just placeholders. Movement between zones happens in tick().
        WORKER_ROSTER = [
            ("R. Kumar", "Engineer", "Z1"), ("S. Rao", "Supervisor", "Z3"),
            ("A. Sharma", "Operator", "Z2"), ("V. Reddy", "Safety Officer", "Z1"),
            ("K. Iyer", "Maintenance Crew", "Z3"), ("M. Singh", "Contractor", "Z4"),
            ("P. Nair", "Operator", "Z5"), ("D. Joshi", "Engineer", "Z6"),
            ("N. Verma", "Supervisor", "Z2"), ("T. Menon", "Contractor", "Z4"),
            ("J. Patel", "Maintenance Crew", "Z5"), ("L. Gupta", "Safety Officer", "Z6"),
        ]
        for name, role, zone_id in WORKER_ROSTER:
            self._add_worker(zone_id, name, role)

        # Baseline permits/maintenance so the dashboard doesn't look empty before
        # a judge triggers an incident. Deliberately "general" permit type (not
        # hot_work/confined_space) and placed where gas never reaches the warning
        # threshold at baseline (max ~35 ppm vs. 40 ppm threshold) — verified in
        # testing this adds visual richness without changing any risk score.
        self.permits.append(Permit(
            permit_id=self._next_permit_id(), permit_type="general",
            zone_id="Z6", issued_to="Facilities Team", status="active",
        ))
        self.maintenance.append(MaintenanceActivity(
            activity_id=self._next_activity_id(), zone_id="Z3",
            description="Boiler — scheduled inspection", active=True,
        ))
        self.maintenance.append(MaintenanceActivity(
            activity_id=self._next_activity_id(), zone_id="Z5",
            description="Conveyor belt — routine check", active=True,
        ))

    # -----------------------------------------------------------------
    def _add_worker(self, zone_id: str, name: str, role: str = "Field Crew"):
        self._worker_counter += 1
        self.workers.append(
            WorkerLocation(worker_id=f"W{self._worker_counter:03d}", name=name, zone_id=zone_id, role=role)
        )

    def log_event(self, zone_id: str, description: str, severity: str = "info"):
        """Appends a timestamped entry to the incident timeline (kept to the last 50)."""
        self.event_log.append({
            "timestamp": time.time(),
            "zone_id": zone_id,
            "zone_name": self.zones[zone_id].name if zone_id in self.zones else zone_id,
            "description": description,
            "severity": severity,  # "info" | "warning" | "critical"
        })
        self.event_log = self.event_log[-50:]

    def _next_permit_id(self) -> str:
        self._permit_counter += 1
        return f"P{self._permit_counter:04d}"

    def _next_activity_id(self) -> str:
        self._activity_counter += 1
        return f"M{self._activity_counter:04d}"

    # -----------------------------------------------------------------
    def tick(self):
        """Advance the simulation by one step (called every ~2 seconds)."""
        for zone_id, reading in self.gas_readings.items():
            self.previous_values[zone_id] = reading.value_ppm
            if zone_id in self._incident_zones:
                # Ramp gas upward during an active incident scenario
                reading.value_ppm = min(95.0, reading.value_ppm + random.uniform(4, 9))
            else:
                # Gentle random walk around nominal baseline
                drift = random.uniform(-2.5, 2.5)
                reading.value_ppm = max(3.0, min(35.0, reading.value_ppm + drift))

        # Workers move between zones over time (like badge/RFID location pings) —
        # each worker has a small chance per tick of relocating, so movement looks
        # organic rather than everyone shuffling at once. Workers currently in an
        # active incident zone stay put (they wouldn't wander into a live incident).
        for worker in self.workers:
            if worker.zone_id in self._incident_zones:
                continue
            if random.random() < 0.12:
                worker.zone_id = random.choice(list(self.zones.keys()))
                worker.timestamp = time.time()

    # -----------------------------------------------------------------
    def trigger_incident(self, zone_id: str) -> bool:
        """
        Reproduces the Vizag-pattern compound failure: a hot-work permit issued
        and a maintenance job started in a zone while gas levels are rising.
        """
        if zone_id not in self.zones:
            return False

        self._incident_zones.add(zone_id)
        self.log_event(zone_id, "Maintenance activity started", severity="info")

        self.permits.append(
            Permit(
                permit_id=self._next_permit_id(),
                permit_type="hot_work",
                zone_id=zone_id,
                issued_to="Contract Crew — Shift B",
                status="active",
            )
        )
        self.log_event(zone_id, "Hot-work permit issued", severity="warning")
        self.maintenance.append(
            MaintenanceActivity(
                activity_id=self._next_activity_id(),
                zone_id=zone_id,
                description="Emergency valve replacement",
                active=True,
            )
        )
        self._add_worker(zone_id, "Field Crew #2", role="Contractor")
        self._add_worker(zone_id, "Field Crew #3", role="Maintenance Crew")
        self.log_event(zone_id, "Gas levels beginning to rise", severity="warning")
        return True

    def clear_incident(self, zone_id: str) -> bool:
        """Resets a zone back to baseline nominal conditions."""
        if zone_id not in self.zones:
            return False

        self._incident_zones.discard(zone_id)
        self.gas_readings[zone_id].value_ppm = round(random.uniform(8, 18), 1)

        for permit in self.permits:
            if permit.zone_id == zone_id and permit.status == "active":
                permit.status = "closed"
        for activity in self.maintenance:
            if activity.zone_id == zone_id:
                activity.active = False
        # Only remove the temporary incident-response crew added by
        # trigger_incident() — the permanent roster stays put.
        self.workers = [
            w for w in self.workers
            if not (w.zone_id == zone_id and w.name.startswith("Field Crew #"))
        ]
        self.log_event(zone_id, "Zone cleared — evacuation stood down, permits closed", severity="info")
        return True

    # -----------------------------------------------------------------
    def snapshot(self) -> dict:
        """Serializable snapshot of the full plant state."""
        return {
            "zones": [asdict(z) for z in self.zones.values()],
            "gas_readings": [asdict(g) for g in self.gas_readings.values()],
            "permits": [asdict(p) for p in self.permits if p.status == "active"],
            "maintenance": [asdict(m) for m in self.maintenance if m.active],
            "workers": [asdict(w) for w in self.workers],
        }
