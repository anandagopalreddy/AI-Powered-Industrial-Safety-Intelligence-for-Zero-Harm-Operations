"""
AI-Powered Industrial Safety Intelligence — API layer.

Exposes the fused plant state (sensors, permits, maintenance, workers)
and the Compound Risk Detection Engine's live alerts over REST, and
serves the dashboard frontend as static files.

Run:
    uvicorn main:app --reload --port 8000
"""
import asyncio
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from data_simulator import plant_state
from risk_engine import assess_all


@asynccontextmanager
async def lifespan(app: FastAPI):
    task = asyncio.create_task(_simulation_loop())
    yield
    task.cancel()


async def _simulation_loop():
    while True:
        plant_state.tick()
        await asyncio.sleep(2)


app = FastAPI(title="Industrial Safety Intelligence API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/zones")
def get_zones():
    return plant_state.zones


@app.get("/api/sensors")
def get_sensors():
    return list(plant_state.gas_readings.values())


@app.get("/api/permits")
def get_permits():
    return list(plant_state.permits.values())


@app.get("/api/maintenance")
def get_maintenance():
    return list(plant_state.maintenance.values())


@app.get("/api/workers")
def get_workers():
    return list(plant_state.workers.values())


@app.get("/api/alerts")
def get_alerts():
    return assess_all(plant_state)


@app.get("/api/dashboard")
def get_dashboard():
    """Single fused payload for the frontend — mirrors what a real
    'unified intelligence layer' would return to a safety officer's
    console in one call."""
    return {
        "zones": plant_state.zones,
        "sensors": list(plant_state.gas_readings.values()),
        "permits": list(plant_state.permits.values()),
        "maintenance": list(plant_state.maintenance.values()),
        "workers": list(plant_state.workers.values()),
        "alerts": assess_all(plant_state),
    }


@app.post("/api/simulate/incident/{zone_id}")
def simulate_incident(zone_id: str):
    """Trigger a Vizag-style compound risk scenario for demo purposes:
    gas accumulation + active maintenance + a new hot work permit, all
    in the same zone."""
    plant_state.trigger_incident(zone_id)
    return {"status": "incident triggered", "zone_id": zone_id}


@app.post("/api/simulate/clear/{zone_id}")
def clear_incident(zone_id: str):
    plant_state.clear_incident(zone_id)
    return {"status": "cleared", "zone_id": zone_id}


# Serve the dashboard frontend
app.mount("/", StaticFiles(directory=".", html=True), name="frontend")
