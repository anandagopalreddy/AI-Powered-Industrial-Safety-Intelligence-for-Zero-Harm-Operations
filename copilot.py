"""
Zero-Harm — AI Copilot (Structured Q&A over live data)
==========================================================
HONEST SCOPE NOTE
--------------------
This is a structured intent-matcher over the system's own live data, NOT a
general-purpose LLM. It recognizes a small set of question shapes ("why is
<zone> critical/high?", "show active permits", "recommend corrective action
for <zone>", "show workers in <zone>") and answers them by reading directly
from the same risk_engine assessments and simulator state the dashboard
already uses — so every answer is grounded and reproducible, not generated
free-text.

PRODUCTION UPGRADE PATH
--------------------------
To turn this into genuine open-ended natural-language dialogue, the natural
next step is to route unmatched questions to a real LLM (e.g. the Anthropic
API) with the live dashboard snapshot passed in as context, so the model
answers from real data rather than guessing. That call would go where
`_fallback_answer()` is below — deliberately left as a stub with no network
call, since this environment has no API key configured and no outbound
network access to verify one working end-to-end.
"""

import re
from typing import List

ZONE_NAME_ALIASES = {
    "coke oven": "Z1", "z1": "Z1",
    "gas cleaning": "Z2", "z2": "Z2",
    "blast furnace": "Z3", "z3": "Z3",
    "tar tank": "Z4", "z4": "Z4",
    "sinter": "Z5", "z5": "Z5",
    "rolling mill": "Z6", "z6": "Z6",
}


def _resolve_zone_id(question: str, zones: dict) -> str | None:
    q = question.lower()
    for alias, zid in ZONE_NAME_ALIASES.items():
        if alias in q:
            return zid
    for zid, zone in zones.items():
        if zone.name.lower() in q:
            return zid
    return None


def answer_question(question: str, simulator, alerts: List) -> dict:
    q = question.strip().lower()
    zones = simulator.zones
    zone_id = _resolve_zone_id(question, zones)
    alert_by_zone = {a.zone_id: a for a in alerts}

    # --- "why is <zone> critical/high?" ---
    if re.search(r"\bwhy\b", q) and zone_id:
        alert = alert_by_zone.get(zone_id)
        zone_name = zones[zone_id].name
        if not alert or alert.risk_level == "LOW":
            answer = f"{zone_name} is currently LOW risk — no compound signals are active."
        else:
            triggers = "; ".join(alert.triggers) if alert.triggers else "no specific triggers logged"
            answer = (f"{zone_name} is {alert.risk_level} (score {alert.score}/100) because: {triggers}. "
                      f"Recommended action: {alert.recommended_action}")
        return {"intent": "explain_zone_risk", "zone_id": zone_id, "answer": answer}

    # --- "show active permits" ---
    if "permit" in q:
        active_permits = [p for p in simulator.permits if p.status == "active"]
        if zone_id:
            active_permits = [p for p in active_permits if p.zone_id == zone_id]
        if not active_permits:
            answer = "No active permits" + (f" in {zones[zone_id].name}." if zone_id else " plant-wide.")
        else:
            lines = [f"{p.permit_id} — {p.permit_type.replace('_', ' ')} in {zones[p.zone_id].name} (issued to {p.issued_to})"
                     for p in active_permits]
            answer = "Active permits:\n" + "\n".join(lines)
        return {"intent": "show_permits", "zone_id": zone_id, "answer": answer,
                "permits": [{"permit_id": p.permit_id, "permit_type": p.permit_type,
                             "zone_id": p.zone_id, "issued_to": p.issued_to} for p in active_permits]}

    # --- "recommend corrective action for <zone>" ---
    if "recommend" in q or "corrective" in q or "what should" in q:
        if not zone_id:
            return {"intent": "recommend_action", "zone_id": None,
                    "answer": "Which zone would you like a recommendation for? "
                              "Try naming it, e.g. 'recommend action for Blast Furnace Bay'."}
        alert = alert_by_zone.get(zone_id)
        zone_name = zones[zone_id].name
        if not alert or alert.risk_level == "LOW":
            answer = f"{zone_name} is nominal — routine monitoring is sufficient, no corrective action needed."
        else:
            answer = f"For {zone_name} ({alert.risk_level}, score {alert.score}): {alert.recommended_action}"
        return {"intent": "recommend_action", "zone_id": zone_id, "answer": answer}

    # --- "show workers in <zone>" ---
    if "worker" in q:
        workers = simulator.workers
        if zone_id:
            workers = [w for w in workers if w.zone_id == zone_id]
        if not workers:
            answer = "No workers currently logged" + (f" in {zones[zone_id].name}." if zone_id else ".")
        else:
            lines = [f"{w.name} — {zones[w.zone_id].name}" for w in workers]
            answer = "Workers:\n" + "\n".join(lines)
        return {"intent": "show_workers", "zone_id": zone_id, "answer": answer}

    return _fallback_answer(question)


def _fallback_answer(question: str) -> dict:
    """
    No matching intent. In production this is where an unmatched question
    would be routed to a real LLM with the live dashboard snapshot as
    context — intentionally left as a grounded stub here rather than a
    network call this environment can't make or verify.
    """
    return {
        "intent": "unrecognized",
        "zone_id": None,
        "answer": (
            "I can answer questions like: \u2018why is Blast Furnace Bay critical?\u2019, "
            "\u2018show active permits\u2019, \u2018recommend corrective action for Zone 3\u2019, "
            "or \u2018show workers in Coke Oven Battery\u2019. "
            "I'm a structured Q&A layer over the live plant data, not a general-purpose "
            "assistant — try rephrasing using a zone name or one of those question shapes."
        ),
    }
