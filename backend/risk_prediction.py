"""
Zero-Harm — Short-Horizon Risk Prediction
============================================
WHAT THIS IS (be precise when explaining it to a judge)
---------------------------------------------------------
This is a linear trend extrapolation over a zone's own recent compound risk
score history — NOT a trained forecasting model. It fits a straight line
(scikit-learn's LinearRegression) through the last N recorded score points
and projects that line forward by the requested horizon (default 15 minutes).

Why this is still useful: a rule-based + AI compound score that has been
climbing steadily for the last few minutes is a genuinely informative signal
— "if this keeps going the way it's going, here's where it lands" — and it
requires no training data, which the hackathon prototype does not have.

Why this is NOT a "predictive AI model" in the deep-learning sense, and
should not be presented to judges as one: it cannot anticipate a new permit
being issued, a gas leak accelerating non-linearly, or any event outside the
recent trend. Say so if asked. The "confidence" value returned is the R²
(goodness-of-fit) of the linear fit over the recent window, not a calibrated
probability — a flat, noisy history will honestly report low confidence.

PRODUCTION UPGRADE PATH
-------------------------
Once real historical SCADA data is available (weeks/months, per Section 15 of
the documentation), this module is the natural place to swap in a trained
time-series model (e.g. gradient-boosted regressor on lagged features, or an
LSTM) without touching the API contract below — predict_zone_risk() would
keep the same return shape.
"""

from typing import Optional

import numpy as np
from sklearn.linear_model import LinearRegression

from risk_engine import risk_band

TICK_SECONDS = 2.0  # must match the background loop's asyncio.sleep() interval


def predict_zone_risk(recent_points: list, horizon_minutes: float = 15.0) -> Optional[dict]:
    """
    recent_points: list of history.HistoryPoint (or objects with .risk_score),
                   most-recent-last, ideally >= 5 points (10s of ticks) for a
                   meaningful fit.
    Returns None if there isn't enough history yet to fit a trend.
    """
    if len(recent_points) < 5:
        return None

    scores = np.array([p.risk_score for p in recent_points], dtype=float)
    t = np.arange(len(scores), dtype=float).reshape(-1, 1)

    model = LinearRegression()
    model.fit(t, scores)
    r_squared = model.score(t, scores)

    horizon_ticks = (horizon_minutes * 60.0) / TICK_SECONDS
    future_t = np.array([[len(scores) - 1 + horizon_ticks]])
    predicted_score = float(model.predict(future_t)[0])
    predicted_score = max(0.0, min(100.0, predicted_score))

    current_score = float(scores[-1])
    current_level = risk_band(int(round(current_score)))
    predicted_level = risk_band(int(round(predicted_score)))

    # R² can be negative for a flat/noisy series with a bad linear fit; clip
    # to [0, 1] and treat it as "how much to trust this straight-line guess",
    # not a true probability.
    confidence_pct = round(max(0.0, min(1.0, r_squared)) * 100, 1)

    return {
        "current_score": round(current_score, 1),
        "current_level": current_level,
        "predicted_score": round(predicted_score, 1),
        "predicted_level": predicted_level,
        "horizon_minutes": horizon_minutes,
        "confidence_percent": confidence_pct,
        "trend_slope_per_tick": round(float(model.coef_[0]), 3),
        "points_used": len(recent_points),
        "method": "linear_trend_extrapolation",
        "disclaimer": "Straight-line projection of this zone's own recent score history "
                       "(not a trained forecasting model). Confidence = R\u00b2 of the fit, "
                       "not a calibrated probability. See risk_prediction.py for details.",
    }
