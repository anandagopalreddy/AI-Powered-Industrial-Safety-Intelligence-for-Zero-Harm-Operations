"""
Zero-Harm — Compound Risk Detection Engine
=============================================
Implements the fusion logic as four independent "agent" functions — one per
data source — whose outputs are combined by assess_zone(). Each agent can be
swapped for a trained ML model later without touching the fusion/scoring logic.

Weights match Section 6 of the technical documentation exactly, so the numbers
in the documentation and in this code always agree.
"""

from dataclasses import dataclass, field
from typing import List, Dict
from data_simulator import GasReading, Permit, MaintenanceActivity, WorkerLocation


# ---------------------------------------------------------------------------
# Weights — Section 6 of the documentation
# ---------------------------------------------------------------------------
WEIGHT_GAS_WARNING = 25
WEIGHT_GAS_CRITICAL = 45
WEIGHT_HOT_WORK_PLUS_GAS = 30
WEIGHT_CONFINED_SPACE_PLUS_GAS = 30
WEIGHT_MAINTENANCE_PLUS_GAS = 20
WEIGHT_OVERLAPPING_PERMITS = 10
WEIGHT_WORKER_EXPOSURE = 10
WEIGHT_AI_ANOMALY = 15  # NEW — Isolation Forest signal, separate from threshold rules

MAX_SCORE = 100

# Score bands -> action levels (Section 6)
def risk_band(score: int) -> str:
    if score >= 70:
        return "CRITICAL"
    if score >= 45:
        return "HIGH"
    if score >= 20:
        return "MODERATE"
    return "LOW"


RECOMMENDED_ACTION = {
    "LOW": "Routine monitoring.",
    "MODERATE": "Increase monitoring frequency; notify shift supervisor.",
    "HIGH": "Suspend relevant permits in this zone; notify safety officer immediately.",
    "CRITICAL": "Evacuate zone; suspend all active permits; dispatch emergency response.",
}


@dataclass
class ZoneAssessment:
    zone_id: str
    score: int
    risk_level: str
    triggers: List[str] = field(default_factory=list)
    recommended_action: str = ""


# ---------------------------------------------------------------------------
# Agent 1 — Gas Agent
# ---------------------------------------------------------------------------
def gas_agent(reading: GasReading) -> Dict:
    elevated = reading.value_ppm >= reading.warning_threshold
    critical = reading.value_ppm >= reading.critical_threshold
    return {"elevated": elevated, "critical": critical, "value": reading.value_ppm, "gas_type": reading.gas_type}


# ---------------------------------------------------------------------------
# Agent 2 — Permit Agent
# ---------------------------------------------------------------------------
def permit_agent(permits: List[Permit], zone_id: str) -> Dict:
    zone_permits = [p for p in permits if p.zone_id == zone_id and p.status == "active"]
    return {
        "hot_work_active": any(p.permit_type == "hot_work" for p in zone_permits),
        "confined_space_active": any(p.permit_type == "confined_space" for p in zone_permits),
        "overlapping_count": len(zone_permits),
    }


# ---------------------------------------------------------------------------
# Agent 3 — Maintenance Agent
# ---------------------------------------------------------------------------
def maintenance_agent(activities: List[MaintenanceActivity], zone_id: str) -> Dict:
    active = [a for a in activities if a.zone_id == zone_id and a.active]
    return {"maintenance_active": len(active) > 0, "activities": [a.description for a in active]}


# ---------------------------------------------------------------------------
# Agent 4 — Worker Density Agent
# ---------------------------------------------------------------------------
def worker_density_agent(workers: List[WorkerLocation], zone_id: str) -> Dict:
    present = [w for w in workers if w.zone_id == zone_id]
    return {"count": len(present), "names": [w.name for w in present]}


# ---------------------------------------------------------------------------
# Fusion & Compound Risk Scoring — Section 6
# ---------------------------------------------------------------------------
def assess_zone(
    zone_id: str,
    gas_reading: GasReading,
    permits: List[Permit],
    maintenance: List[MaintenanceActivity],
    workers: List[WorkerLocation],
    anomaly: "tuple[bool, float] | None" = None,
) -> ZoneAssessment:
    gas = gas_agent(gas_reading)
    permit_state = permit_agent(permits, zone_id)
    maint_state = maintenance_agent(maintenance, zone_id)
    worker_state = worker_density_agent(workers, zone_id)

    score = 0
    triggers: List[str] = []
    elevated = gas["elevated"]

    if elevated and not gas["critical"]:
        score += WEIGHT_GAS_WARNING
        triggers.append(f"Gas reading WARNING ({gas['value']:.1f} ppm {gas['gas_type']})")

    if gas["critical"]:
        score += WEIGHT_GAS_CRITICAL
        triggers.append(f"Gas reading CRITICAL ({gas['value']:.1f} ppm {gas['gas_type']})")

    if permit_state["hot_work_active"] and elevated:
        score += WEIGHT_HOT_WORK_PLUS_GAS
        triggers.append("Hot-work permit active during elevated gas — ignition risk (Vizag pattern)")

    if permit_state["confined_space_active"] and elevated:
        score += WEIGHT_CONFINED_SPACE_PLUS_GAS
        triggers.append("Confined-space entry active during elevated gas — asphyxiation risk")

    if maint_state["maintenance_active"] and elevated:
        score += WEIGHT_MAINTENANCE_PLUS_GAS
        triggers.append("Maintenance activity co-occurring with elevated gas")

    if permit_state["overlapping_count"] >= 2:
        score += WEIGHT_OVERLAPPING_PERMITS
        triggers.append(f"{permit_state['overlapping_count']} overlapping permits in zone — coordination risk")

    if worker_state["count"] >= 3 and elevated:
        score += WEIGHT_WORKER_EXPOSURE
        triggers.append(f"{worker_state['count']} workers exposed to elevated-risk zone")

    if anomaly is not None:
        is_anomaly, confidence = anomaly
        if is_anomaly:
            score += WEIGHT_AI_ANOMALY
            triggers.append(
                f"AI Anomaly Detector (Isolation Forest): gas pattern deviates from this "
                f"zone's historical norm — {confidence:.0f}% abnormality confidence"
            )

    score = min(score, MAX_SCORE)
    level = risk_band(score)

    return ZoneAssessment(
        zone_id=zone_id,
        score=score,
        risk_level=level,
        triggers=triggers,
        recommended_action=RECOMMENDED_ACTION[level],
    )


def assess_all_zones(simulator, anomaly_detector=None) -> List[ZoneAssessment]:
    """Runs assess_zone() for every zone in the plant, highest score first."""
    results = []
    for zone_id in simulator.zones:
        gas_reading = simulator.gas_readings[zone_id]

        anomaly = None
        if anomaly_detector is not None:
            previous = simulator.previous_values.get(zone_id, gas_reading.value_ppm)
            anomaly = anomaly_detector.score(zone_id, gas_reading.value_ppm, previous)

        results.append(
            assess_zone(
                zone_id, gas_reading, simulator.permits, simulator.maintenance,
                simulator.workers, anomaly=anomaly,
            )
        )
    results.sort(key=lambda a: a.score, reverse=True)
    return results
