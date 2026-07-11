"""
Zero-Harm — Historical Analytics Buffer
==========================================
Records a rolling window of per-zone state (gas ppm, compound risk score,
risk band, AI anomaly confidence) every simulation tick, so the dashboard can
show trends instead of only the current instant.

This is an in-memory ring buffer, not a database — it resets when the process
restarts. In production this would write to a time-series store (e.g.
TimescaleDB/InfluxDB) and the query functions below would become SQL queries
instead of list slicing; the public interface (get_zone_history /
get_all_zone_summaries) is written so that swap is a backend change only.
"""

import time
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Optional

MAX_POINTS_PER_ZONE = 300  # ~10 minutes of history at one point per 2-second tick


@dataclass
class HistoryPoint:
    timestamp: float
    gas_ppm: float
    risk_score: int
    risk_level: str
    ai_confidence: float = 0.0


class HistoryTracker:
    def __init__(self, zone_ids: List[str], max_points: int = MAX_POINTS_PER_ZONE):
        self.max_points = max_points
        self._history: Dict[str, Deque[HistoryPoint]] = {
            zid: deque(maxlen=max_points) for zid in zone_ids
        }

    def record(self, zone_id: str, gas_ppm: float, risk_score: int,
               risk_level: str, ai_confidence: float = 0.0):
        if zone_id not in self._history:
            self._history[zone_id] = deque(maxlen=self.max_points)
        self._history[zone_id].append(
            HistoryPoint(
                timestamp=time.time(),
                gas_ppm=gas_ppm,
                risk_score=risk_score,
                risk_level=risk_level,
                ai_confidence=ai_confidence,
            )
        )

    def get_zone_history(self, zone_id: str) -> List[dict]:
        points = self._history.get(zone_id, deque())
        return [
            {
                "timestamp": p.timestamp,
                "gas_ppm": p.gas_ppm,
                "risk_score": p.risk_score,
                "risk_level": p.risk_level,
                "ai_confidence": p.ai_confidence,
            }
            for p in points
        ]

    def get_recent_scores(self, zone_id: str, n: int) -> List[HistoryPoint]:
        points = list(self._history.get(zone_id, deque()))
        return points[-n:]

    def get_all_zone_summaries(self) -> Dict[str, dict]:
        """One summary row per zone: latest point + simple aggregates, for the
        Analytics tab's zone-wise safety score table."""
        summaries = {}
        for zone_id, points in self._history.items():
            if not points:
                summaries[zone_id] = {
                    "latest_score": None, "latest_level": None,
                    "avg_score_recent": None, "max_score_recent": None,
                    "points_recorded": 0,
                }
                continue
            recent = list(points)[-60:]  # last ~2 minutes
            scores = [p.risk_score for p in recent]
            summaries[zone_id] = {
                "latest_score": points[-1].risk_score,
                "latest_level": points[-1].risk_level,
                "avg_score_recent": round(sum(scores) / len(scores), 1),
                "max_score_recent": max(scores),
                "points_recorded": len(points),
            }
        return summaries

    def monthly_risk_distribution(self) -> Dict[str, int]:
        """
        Buckets every recorded point (across all zones, all time in the buffer)
        into its risk band. Named 'monthly' to match the KPI a production
        deployment would show once the buffer is backed by real historical
        storage spanning a month; in this in-memory demo it reflects whatever
        history has been recorded since the process started.
        """
        buckets = {"LOW": 0, "MODERATE": 0, "HIGH": 0, "CRITICAL": 0}
        for points in self._history.values():
            for p in points:
                buckets[p.risk_level] = buckets.get(p.risk_level, 0) + 1
        return buckets
