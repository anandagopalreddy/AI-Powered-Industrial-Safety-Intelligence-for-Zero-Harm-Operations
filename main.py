"""
Zero-Harm — API Layer
========================
FastAPI service exposing the fused plant state and live alerts as REST
endpoints (Section 9 of the technical documentation), plus a background task
that advances the simulation every two seconds so the dashboard feels live.
"""

import asyncio
import io
import time
from dataclasses import asdict
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from data_simulator import PlantSimulator
from risk_engine import assess_all_zones
from anomaly_detector import ZoneAnomalyDetector
from benchmark import run_benchmark
from history import HistoryTracker
from risk_prediction import predict_zone_risk
from report_generator import build_incident_report_pdf
from vision_agent import VisionAgent
from copilot import answer_question

app = FastAPI(title="Zero-Harm — Industrial Safety Intelligence", version="1.2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

simulator = PlantSimulator()
anomaly_detector = ZoneAnomalyDetector(zone_ids=list(simulator.zones.keys()))
history_tracker = HistoryTracker(zone_ids=list(simulator.zones.keys()))
vision_agent = VisionAgent()

FRONTEND_DIR = Path(__file__).resolve().parent.parent / "frontend"
_last_risk_levels: dict = {}


# ---------------------------------------------------------------------------
# Background simulation loop — also feeds the history tracker every tick
# ---------------------------------------------------------------------------
@app.on_event("startup")
async def start_background_loop():
    async def loop():
        while True:
            simulator.tick()
            alerts = assess_all_zones(simulator, anomaly_detector)
            for a in alerts:
                gas_reading = simulator.gas_readings[a.zone_id]
                previous = simulator.previous_values.get(a.zone_id, gas_reading.value_ppm)
                _, confidence = anomaly_detector.score(a.zone_id, gas_reading.value_ppm, previous)
                history_tracker.record(
                    zone_id=a.zone_id,
                    gas_ppm=gas_reading.value_ppm,
                    risk_score=a.score,
                    risk_level=a.risk_level,
                    ai_confidence=confidence,
                )
                # Log risk-level transitions to the incident timeline — this is
                # what makes the timeline panel show real state changes, not
                # just the permit/maintenance events logged at trigger time.
                previous_level = _last_risk_levels.get(a.zone_id)
                if previous_level is not None and previous_level != a.risk_level:
                    severity = "critical" if a.risk_level == "CRITICAL" else (
                        "warning" if a.risk_level in ("HIGH", "MODERATE") else "info"
                    )
                    simulator.log_event(
                        a.zone_id,
                        f"Risk level changed: {previous_level} → {a.risk_level} (score {a.score})",
                        severity=severity,
                    )
                _last_risk_levels[a.zone_id] = a.risk_level
            await asyncio.sleep(2)

    asyncio.create_task(loop())


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------
@app.get("/health")
def health():
    return {
        "status": "healthy",
        "service": "Zero-Harm AI",
        "version": app.version,
        "zones_tracked": len(simulator.zones),
    }


# ---------------------------------------------------------------------------
# Core API endpoints — Section 9 of the documentation
# ---------------------------------------------------------------------------
@app.get("/api/dashboard")
def get_dashboard():
    """Single fused payload: zones, sensors, permits, maintenance, workers, alerts."""
    snapshot = simulator.snapshot()
    alerts = assess_all_zones(simulator, anomaly_detector)
    return {
        "timestamp": time.time(),
        **snapshot,
        "alerts": [asdict(a) for a in alerts],
    }


@app.get("/api/zones")
def get_zones():
    return simulator.snapshot()["zones"]


@app.get("/api/sensors")
def get_sensors():
    return simulator.snapshot()["gas_readings"]


@app.get("/api/permits")
def get_permits():
    return simulator.snapshot()["permits"]


@app.get("/api/maintenance")
def get_maintenance():
    return simulator.snapshot()["maintenance"]


@app.get("/api/workers")
def get_workers():
    return simulator.snapshot()["workers"]


@app.get("/api/alerts")
def get_alerts():
    alerts = assess_all_zones(simulator, anomaly_detector)
    return [asdict(a) for a in alerts]


@app.get("/api/benchmark")
def get_benchmark():
    """
    Evaluation metrics computed live from the same scenario suite as
    Zero-Harm_Baseline_Comparison_Script.py — see Section 11 of the
    documentation. Includes a clearly-labelled business impact ESTIMATE
    (not measured data) — see benchmark.py for the stated assumptions.
    """
    return run_benchmark()


@app.get("/api/timeline")
def get_timeline(limit: int = 30):
    """
    Chronological incident timeline: permit/maintenance events logged at
    trigger/clear time, plus every real risk-level transition detected in the
    background loop (e.g. "LOW → MODERATE → HIGH → CRITICAL"). Most recent
    first. This is what the dashboard's Incident Timeline panel reads.
    """
    return list(reversed(simulator.event_log[-limit:]))


# ---------------------------------------------------------------------------
# Dashboard KPI summary — Plant Safety Score, Critical Zones, Active Alerts,
# Workers in Danger, Open Permits. This is the endpoint the top KPI bar reads.
# ---------------------------------------------------------------------------
@app.get("/api/kpis")
def get_kpis():
    alerts = assess_all_zones(simulator, anomaly_detector)
    band_counts = {"LOW": 0, "MODERATE": 0, "HIGH": 0, "CRITICAL": 0}
    for a in alerts:
        band_counts[a.risk_level] = band_counts.get(a.risk_level, 0) + 1

    scores = [a.score for a in alerts]
    # Plant Safety Score: 100 minus the average compound risk score across all
    # zones. A simple, explainable rollup — not a separately trained model.
    plant_safety_score = round(100 - (sum(scores) / len(scores)), 1) if scores else 100.0
    plant_safety_score = max(0.0, min(100.0, plant_safety_score))

    danger_zone_ids = {a.zone_id for a in alerts if a.risk_level in ("HIGH", "CRITICAL")}
    workers_in_danger = sum(1 for w in simulator.workers if w.zone_id in danger_zone_ids)
    open_permits = sum(1 for p in simulator.permits if p.status == "active")

    return {
        "timestamp": time.time(),
        "plant_safety_score": plant_safety_score,
        "critical_zones": band_counts["CRITICAL"],
        "high_zones": band_counts["HIGH"],
        "active_alerts": band_counts["HIGH"] + band_counts["CRITICAL"] + band_counts["MODERATE"],
        "workers_in_danger": workers_in_danger,
        "workers_total": len(simulator.workers),
        "open_permits": open_permits,
        "zone_risk_band_counts": band_counts,
    }


# ---------------------------------------------------------------------------
# Historical analytics — Section 15 of the documentation
# ---------------------------------------------------------------------------
@app.get("/api/history/{zone_id}")
def get_zone_history(zone_id: str):
    if zone_id not in simulator.zones:
        raise HTTPException(status_code=404, detail=f"Unknown zone_id: {zone_id}")
    return {
        "zone_id": zone_id,
        "zone_name": simulator.zones[zone_id].name,
        "points": history_tracker.get_zone_history(zone_id),
    }


@app.get("/api/history")
def get_all_history_summary():
    """Zone-wise safety score table + monthly-style risk distribution for the
    Historical Analytics tab. See history.py for what 'monthly' means here."""
    return {
        "zone_summaries": history_tracker.get_all_zone_summaries(),
        "risk_distribution": history_tracker.monthly_risk_distribution(),
        "zone_names": {zid: z.name for zid, z in simulator.zones.items()},
    }


# ---------------------------------------------------------------------------
# Risk prediction — heuristic linear trend extrapolation (see risk_prediction.py)
# ---------------------------------------------------------------------------
@app.get("/api/predict/{zone_id}")
def get_zone_prediction(zone_id: str, horizon_minutes: float = 15.0):
    if zone_id not in simulator.zones:
        raise HTTPException(status_code=404, detail=f"Unknown zone_id: {zone_id}")
    recent = history_tracker.get_recent_scores(zone_id, n=60)
    prediction = predict_zone_risk(recent, horizon_minutes=horizon_minutes)
    if prediction is None:
        return {
            "zone_id": zone_id,
            "available": False,
            "reason": "Not enough history yet — let the simulator run a bit longer "
                      "(needs at least ~10 seconds of ticks).",
        }
    return {"zone_id": zone_id, "available": True, **prediction}


# ---------------------------------------------------------------------------
# Incident report PDF
# ---------------------------------------------------------------------------
@app.get("/api/report/{zone_id}")
def get_incident_report(zone_id: str):
    if zone_id not in simulator.zones:
        raise HTTPException(status_code=404, detail=f"Unknown zone_id: {zone_id}")

    zone = simulator.zones[zone_id]
    gas_reading = simulator.gas_readings[zone_id]
    alerts = assess_all_zones(simulator, anomaly_detector)
    assessment = next((a for a in alerts if a.zone_id == zone_id), None)
    if assessment is None:
        raise HTTPException(status_code=500, detail="Could not compute assessment for this zone")

    pdf_bytes = build_incident_report_pdf(zone.name, zone.hazard_class, assessment, gas_reading)
    filename = f"zero-harm-incident-report-{zone_id}-{int(time.time())}.pdf"
    return StreamingResponse(
        io.BytesIO(pdf_bytes),
        media_type="application/pdf",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


# ---------------------------------------------------------------------------
# Vision agent — PPE / restricted-area detection (see vision_agent.py for the
# honest scope note: simulated by default, real YOLOv8 path is wired in but
# untrained/untested in this environment).
# ---------------------------------------------------------------------------
@app.get("/api/vision/status")
def get_vision_status():
    return vision_agent.status()


@app.get("/api/vision/{zone_id}")
def get_vision_check(zone_id: str):
    if zone_id not in simulator.zones:
        raise HTTPException(status_code=404, detail=f"Unknown zone_id: {zone_id}")
    return vision_agent.detect_ppe(zone_id)


@app.post("/api/vision/{zone_id}/analyze")
async def analyze_vision_frame(zone_id: str, file: UploadFile = File(...)):
    if zone_id not in simulator.zones:
        raise HTTPException(status_code=404, detail=f"Unknown zone_id: {zone_id}")
    image_bytes = await file.read()
    return vision_agent.detect_ppe(zone_id, image_bytes=image_bytes)


# ---------------------------------------------------------------------------
# AI Copilot — structured Q&A over live data (see copilot.py for scope note)
# ---------------------------------------------------------------------------
class CopilotQuestion(BaseModel):
    question: str


@app.post("/api/copilot/ask")
def copilot_ask(payload: CopilotQuestion):
    alerts = assess_all_zones(simulator, anomaly_detector)
    return answer_question(payload.question, simulator, alerts)


# ---------------------------------------------------------------------------
# Simulation controls
# ---------------------------------------------------------------------------
@app.post("/api/simulate/incident/{zone_id}")
def simulate_incident(zone_id: str):
    ok = simulator.trigger_incident(zone_id)
    if not ok:
        raise HTTPException(status_code=404, detail=f"Unknown zone_id: {zone_id}")
    return {"status": "incident triggered", "zone_id": zone_id}


@app.post("/api/simulate/clear/{zone_id}")
def simulate_clear(zone_id: str):
    ok = simulator.clear_incident(zone_id)
    if not ok:
        raise HTTPException(status_code=404, detail=f"Unknown zone_id: {zone_id}")
    return {"status": "zone cleared", "zone_id": zone_id}


# ---------------------------------------------------------------------------
# Serve the frontend dashboard
# ---------------------------------------------------------------------------
@app.get("/")
def serve_index():
    index_path = FRONTEND_DIR / "index.html"
    if index_path.exists():
        return FileResponse(index_path)
    raise HTTPException(status_code=404, detail="frontend/index.html not found")


if FRONTEND_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")
