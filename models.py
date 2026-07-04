"""
Data models for the AI-Powered Industrial Safety Intelligence platform.
Covers gas sensors, work permits, maintenance activity, worker locations,
and the compound-risk alerts the engine produces.
"""
from datetime import datetime
from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field


class GasType(str, Enum):
    H2S = "H2S"          # Hydrogen Sulphide
    CO = "CO"            # Carbon Monoxide
    CH4 = "CH4"          # Methane
    O2 = "O2"            # Oxygen (low O2 = asphyxiation risk)


class PermitType(str, Enum):
    HOT_WORK = "HOT_WORK"
    CONFINED_SPACE = "CONFINED_SPACE"
    ELECTRICAL_ISOLATION = "ELECTRICAL_ISOLATION"
    HEIGHT_WORK = "HEIGHT_WORK"
    GENERAL = "GENERAL"


class RiskLevel(str, Enum):
    LOW = "LOW"
    MODERATE = "MODERATE"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


class Zone(BaseModel):
    zone_id: str
    name: str
    x: float  # normalized plant-layout coordinate (0-100)
    y: float
    hazard_class: str = "GENERAL"  # e.g. COKE_OVEN, TANK_FARM, CONFINED_SPACE


class GasReading(BaseModel):
    sensor_id: str
    zone_id: str
    gas_type: GasType
    value_ppm: float
    threshold_warning: float
    threshold_critical: float
    timestamp: datetime = Field(default_factory=datetime.utcnow)

    @property
    def status(self) -> str:
        if self.value_ppm >= self.threshold_critical:
            return "CRITICAL"
        if self.value_ppm >= self.threshold_warning:
            return "WARNING"
        return "NORMAL"


class Permit(BaseModel):
    permit_id: str
    permit_type: PermitType
    zone_id: str
    issued_to: str
    issued_at: datetime
    valid_until: datetime
    status: str = "ACTIVE"  # ACTIVE, CLOSED, SUSPENDED


class MaintenanceActivity(BaseModel):
    activity_id: str
    zone_id: str
    description: str
    active: bool = True
    started_at: datetime = Field(default_factory=datetime.utcnow)


class WorkerLocation(BaseModel):
    worker_id: str
    name: str
    zone_id: str
    updated_at: datetime = Field(default_factory=datetime.utcnow)


class RiskAlert(BaseModel):
    alert_id: str
    zone_id: str
    zone_name: str
    risk_level: RiskLevel
    risk_score: float  # 0-100
    triggers: list[str]
    recommended_action: str
    created_at: datetime = Field(default_factory=datetime.utcnow)
