"""
Compound Risk Detection Engine.

This is the core "intelligence layer" the problem statement asks for: it
does NOT just threshold a single sensor. It correlates gas readings,
active permits, maintenance activity, and worker presence to surface
dangerous *combinations* that no single system would flag on its own —
e.g. a hot-work permit issued in a zone where gas levels are climbing
and a confined-space job is already underway.

Architecturally this models a lightweight multi-agent pattern: each
`_check_*` method is an independent "agent" inspecting one data source,
and `assess_zone` is the fusion/orchestration step that combines their
findings into a single compound risk score. This keeps it easy to swap
any individual check for a real ML model later without touching the
fusion logic.
"""
from datetime import datetime

from models import GasType, PermitType, RiskAlert, RiskLevel, Zone
from data_simulator import PlantState

# Weights reflect how much each fused signal contributes to the compound
# score. Calibrated illustratively; in production these would be tuned
# against historical incident data (see Evaluation Focus in the brief).
WEIGHTS = {
    "gas_warning": 25,
    "gas_critical": 45,
    "hot_work_with_gas": 30,      # hot work permit + elevated gas = classic ignition risk
    "confined_space_with_gas": 30,  # confined space entry + elevated gas = asphyxiation risk
    "maintenance_with_gas": 20,   # maintenance activity co-occurring with gas rise
    "multiple_permits_overlap": 10,
    "worker_density": 10,
}


def _gas_agent(state: PlantState, zone_id: str):
    """Independent 'agent': inspects gas sensor status for a zone."""
    readings = [r for r in state.gas_readings.values() if r.zone_id == zone_id]
    worst = "NORMAL"
    detail = []
    for r in readings:
        if r.status == "CRITICAL":
            worst = "CRITICAL"
        elif r.status == "WARNING" and worst != "CRITICAL":
            worst = "WARNING"
        if r.status != "NORMAL":
            detail.append(f"{r.gas_type.value} at {r.value_ppm:.1f}ppm ({r.status})")
    return worst, detail


def _permit_agent(state: PlantState, zone_id: str):
    """Independent 'agent': inspects active permits for a zone."""
    return [p for p in state.permits.values() if p.zone_id == zone_id and p.status == "ACTIVE"]


def _maintenance_agent(state: PlantState, zone_id: str):
    """Independent 'agent': inspects active maintenance/CMMS jobs for a zone."""
    return [m for m in state.maintenance.values() if m.zone_id == zone_id and m.active]


def _worker_agent(state: PlantState, zone_id: str):
    """Independent 'agent': inspects worker headcount in a zone."""
    return [w for w in state.workers.values() if w.zone_id == zone_id]


def assess_zone(state: PlantState, zone: Zone) -> RiskAlert | None:
    """Fusion step: combine independent agent outputs into one compound
    risk score and, if material, produce a RiskAlert with the specific
    contributing triggers spelled out (this is what makes the alert
    actionable rather than just a number)."""
    gas_status, gas_detail = _gas_agent(state, zone.zone_id)
    permits = _permit_agent(state, zone.zone_id)
    maintenance = _maintenance_agent(state, zone.zone_id)
    workers = _worker_agent(state, zone.zone_id)

    score = 0.0
    triggers: list[str] = []

    if gas_status == "WARNING":
        score += WEIGHTS["gas_warning"]
        triggers.append(f"Elevated gas reading in {zone.name}: " + "; ".join(gas_detail))
    elif gas_status == "CRITICAL":
        score += WEIGHTS["gas_critical"]
        triggers.append(f"CRITICAL gas reading in {zone.name}: " + "; ".join(gas_detail))

    hot_work_active = any(p.permit_type == PermitType.HOT_WORK for p in permits)
    confined_space_active = any(p.permit_type == PermitType.CONFINED_SPACE for p in permits)

    if gas_status != "NORMAL" and hot_work_active:
        score += WEIGHTS["hot_work_with_gas"]
        triggers.append(
            "Hot work permit ACTIVE while gas levels are elevated — ignition risk "
            "(this combination preceded the Jan-2025 coke oven battery incident)."
        )

    if gas_status != "NORMAL" and confined_space_active:
        score += WEIGHTS["confined_space_with_gas"]
        triggers.append(
            "Confined space entry permit ACTIVE while gas levels are elevated — "
            "asphyxiation / toxic exposure risk."
        )

    if gas_status != "NORMAL" and maintenance:
        score += WEIGHTS["maintenance_with_gas"]
        job_desc = ", ".join(m.description for m in maintenance)
        triggers.append(f"Maintenance activity in progress during gas rise: {job_desc}")

    if len(permits) >= 2:
        score += WEIGHTS["multiple_permits_overlap"]
        triggers.append(f"{len(permits)} overlapping permits active in the same zone.")

    if len(workers) >= 3 and gas_status != "NORMAL":
        score += WEIGHTS["worker_density"]
        triggers.append(f"{len(workers)} workers present in a zone with elevated gas readings.")

    score = min(score, 100.0)

    if score == 0:
        return None

    if score >= 70:
        level = RiskLevel.CRITICAL
        action = ("Initiate evacuation protocol for this zone, suspend all active hot "
                   "work and confined space permits immediately, and dispatch emergency response team.")
    elif score >= 45:
        level = RiskLevel.HIGH
        action = ("Suspend hot work / confined space permits in this zone pending gas "
                   "re-verification; notify shift safety officer.")
    elif score >= 20:
        level = RiskLevel.MODERATE
        action = "Increase monitoring frequency and notify area supervisor."
    else:
        level = RiskLevel.LOW
        action = "Continue routine monitoring."

    return RiskAlert(
        alert_id=f"A-{zone.zone_id}-{int(datetime.utcnow().timestamp())}",
        zone_id=zone.zone_id,
        zone_name=zone.name,
        risk_level=level,
        risk_score=round(score, 1),
        triggers=triggers,
        recommended_action=action,
    )


def assess_all(state: PlantState) -> list[RiskAlert]:
    alerts = []
    for zone in state.zones:
        alert = assess_zone(state, zone)
        if alert:
            alerts.append(alert)
    # highest risk first
    alerts.sort(key=lambda a: a.risk_score, reverse=True)
    return alerts
