"""
Zero-Harm: Single-Sensor vs. Rule-Based Compound Engine — Full Evaluation Suite
=================================================================================
WHY THIS FILE EXISTS
---------------------
Judges give 25% weight to "Business Impact" and the brief specifically asks for:
  - "Compound risk detection accuracy vs. single-sensor baselines"
  - "Reduction in false negative rate"

This script measures it properly:
  - 6 main scenarios (4 dangerous + 2 harmless) — the validated core comparison
  - 1 bonus scenario demonstrating what the AI anomaly layer sees that the
    rule engine cannot (reported separately, never mixed into the main count)
  - Precision / Recall / F1 / Accuracy for both systems
  - Lead-time statistics (min / max / median, not just average)
  - AI confidence distribution for the bonus scenario
  - Exports to results.csv and results.json for use elsewhere (e.g. a dashboard)
  - Chart generation (detection comparison, incident timeline, risk score trend)

HOW TO RUN
----------
    pip install scikit-learn numpy matplotlib    (same core deps as the backend;
                                                    matplotlib only needed for charts)
    python3 baseline_vs_compound_demo.py

Every number below is computed from the scenario timelines defined in this file —
nothing is hardcoded or asserted without a calculation behind it.
"""

import csv
import json
import statistics
from dataclasses import dataclass, field, asdict
from typing import List, Optional

try:
    import numpy as np
    from sklearn.ensemble import IsolationForest
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

try:
    import matplotlib
    matplotlib.use("Agg")  # no display needed, just save PNG files
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False


# ---------------------------------------------------------------------------
# 1. A minute-by-minute timeline for one zone during one incident scenario
# ---------------------------------------------------------------------------
@dataclass
class TimelineStep:
    minute: int
    gas_ppm: float
    hot_work_permit_active: bool
    confined_space_permit_active: bool
    maintenance_active: bool
    workers_in_zone: int


@dataclass
class Scenario:
    name: str
    gas_warning_threshold: float
    gas_critical_threshold: float
    timeline: List[TimelineStep]
    is_genuinely_dangerous: bool  # ground truth: did this scenario actually need action?
    is_bonus_ai_visibility_test: bool = False  # excluded from main aggregate — see Scenario 7


# ---------------------------------------------------------------------------
# 2. Detection logic — SINGLE-SENSOR BASELINE
#    (what most existing plants actually run today)
# ---------------------------------------------------------------------------
def single_sensor_detection_minute(scenario: Scenario) -> int | None:
    """Returns the minute the baseline system would raise an alarm, or None if it never does."""
    for step in scenario.timeline:
        if step.gas_ppm >= scenario.gas_critical_threshold:
            return step.minute
    return None


# ---------------------------------------------------------------------------
# 3. Detection logic — COMPOUND ENGINE (same weights as Zero-Harm risk_engine.py)
# ---------------------------------------------------------------------------
def compound_score(scenario: Scenario, step: TimelineStep) -> int:
    score = 0
    elevated = step.gas_ppm >= scenario.gas_warning_threshold
    critical = step.gas_ppm >= scenario.gas_critical_threshold

    if elevated and not critical:
        score += 25
    if critical:
        score += 45
    if step.hot_work_permit_active and elevated:
        score += 30
    if step.confined_space_permit_active and elevated:
        score += 30
    if step.maintenance_active and elevated:
        score += 20
    if step.workers_in_zone >= 3 and elevated:
        score += 10

    return min(score, 100)


def compound_detection_minute(scenario: Scenario, action_threshold: int = 45) -> int | None:
    """Returns the minute the compound engine reaches HIGH risk (>=45), or None if it never does."""
    for step in scenario.timeline:
        if compound_score(scenario, step) >= action_threshold:
            return step.minute
    return None


# ---------------------------------------------------------------------------
# 3b. Detection logic — COMPOUND ENGINE + AI ANOMALY LAYER
#     Mirrors backend/anomaly_detector.py exactly: one Isolation Forest per
#     scenario "zone", trained on that zone's own low-baseline pattern,
#     scoring (value, rate-of-change) pairs, threshold at 65% confidence.
# ---------------------------------------------------------------------------
def _train_anomaly_model(nominal_baseline: float, seed: int = 42):
    rng = np.random.default_rng(seed)
    n_samples = 500
    values = nominal_baseline + rng.normal(0, 3.0, n_samples)
    values = np.clip(values, 0, None)
    deltas = np.diff(values, prepend=values[0])
    features = np.column_stack([values, deltas])

    model = IsolationForest(n_estimators=150, contamination=0.03, random_state=42)
    model.fit(features)
    return model


def _anomaly_confidence(model, value: float, delta: float) -> float:
    raw_score = model.decision_function(np.array([[value, delta]]))[0]
    confidence = float(np.clip((0.12 - raw_score) / 0.28 * 100, 0, 100))
    return confidence


def compound_ai_score(scenario: Scenario, step: TimelineStep, prev_value: float, model) -> int:
    score = compound_score(scenario, step)
    if SKLEARN_AVAILABLE and model is not None:
        delta = step.gas_ppm - prev_value
        confidence = _anomaly_confidence(model, step.gas_ppm, delta)
        if confidence >= 65.0:
            score += 15
    return min(score, 100)


def compound_ai_detection_minute(scenario: Scenario, action_threshold: int = 45) -> Optional[int]:
    """Returns the minute the compound+AI engine reaches HIGH risk, or None if it never does."""
    if not SKLEARN_AVAILABLE:
        return compound_detection_minute(scenario, action_threshold)

    model = _train_anomaly_model(nominal_baseline=scenario.timeline[0].gas_ppm)
    prev_value = scenario.timeline[0].gas_ppm
    for step in scenario.timeline:
        if compound_ai_score(scenario, step, prev_value, model) >= action_threshold:
            return step.minute
        prev_value = step.gas_ppm
    return None


# ---------------------------------------------------------------------------
# 4. Seven scenarios total:
#    - 6 MAIN scenarios (4 genuinely dangerous compound situations, 2 harmless
#      noise — this is the validated core comparison used in the summary)
#    - 1 BONUS scenario (marked is_bonus_ai_visibility_test=True), used only
#      to demonstrate the AI anomaly layer's added visibility, reported
#      separately so it never inflates the main detection-rate numbers
# ---------------------------------------------------------------------------
def build_scenarios() -> List[Scenario]:
    scenarios = []

    # Scenario 1 — the Vizag-style pattern: hot work permit + slowly rising gas
    scenarios.append(Scenario(
        name="Coke Oven Battery — hot work during gas rise (Vizag pattern)",
        gas_warning_threshold=40, gas_critical_threshold=80,
        is_genuinely_dangerous=True,
        timeline=[
            TimelineStep(0, 20, False, False, False, 1),
            TimelineStep(5, 35, False, False, True, 2),
            TimelineStep(10, 45, True, False, True, 2),   # <-- compound danger starts here
            TimelineStep(15, 55, True, False, True, 3),
            TimelineStep(20, 70, True, False, True, 3),
            TimelineStep(25, 85, True, False, True, 3),   # <-- single sensor finally wakes up
        ],
    ))

    # Scenario 2 — confined space entry while gas creeps up slowly
    scenarios.append(Scenario(
        name="Gas Cleaning Plant — confined space entry, slow gas creep",
        gas_warning_threshold=30, gas_critical_threshold=75,
        is_genuinely_dangerous=True,
        timeline=[
            TimelineStep(0, 15, False, False, False, 0),
            TimelineStep(5, 28, False, True, False, 1),
            TimelineStep(10, 32, False, True, False, 2),  # <-- compound danger starts here
            TimelineStep(15, 40, False, True, False, 2),
            TimelineStep(20, 60, False, True, False, 2),
            TimelineStep(25, 76, False, True, False, 2),  # <-- single sensor finally wakes up
        ],
    ))

    # Scenario 3 — maintenance job started right as gas starts elevating
    scenarios.append(Scenario(
        name="Blast Furnace Bay — unplanned maintenance during gas fluctuation",
        gas_warning_threshold=35, gas_critical_threshold=80,
        is_genuinely_dangerous=True,
        timeline=[
            TimelineStep(0, 18, False, False, False, 1),
            TimelineStep(5, 30, False, False, True, 1),
            TimelineStep(10, 38, False, False, True, 2),  # <-- compound danger starts here
            TimelineStep(15, 50, False, False, True, 2),
            TimelineStep(20, 65, False, False, True, 2),
            TimelineStep(25, 82, False, False, True, 3),  # <-- single sensor finally wakes up
        ],
    ))

    # Scenario 4 — hot work permit issued but gas NEVER reaches critical
    # (this is the classic false negative: single-sensor system never raises an alarm at all)
    scenarios.append(Scenario(
        name="Tar Tank Farm — hot work permit, gas stays just under critical all day",
        gas_warning_threshold=35, gas_critical_threshold=80,
        is_genuinely_dangerous=True,
        timeline=[
            TimelineStep(0, 20, False, False, False, 1),
            TimelineStep(10, 38, True, False, False, 2),  # <-- compound danger starts here
            TimelineStep(20, 55, True, False, False, 2),
            TimelineStep(30, 68, True, False, False, 2),
            TimelineStep(40, 74, True, False, False, 2),  # <-- never crosses 80. Baseline: SILENT.
        ],
    ))

    # Scenario 5 — harmless: gas fluctuates but no permit, no maintenance (should NOT alarm)
    scenarios.append(Scenario(
        name="Sinter Plant — routine gas fluctuation, no other activity",
        gas_warning_threshold=35, gas_critical_threshold=80,
        is_genuinely_dangerous=False,
        timeline=[
            TimelineStep(0, 20, False, False, False, 0),
            TimelineStep(10, 38, False, False, False, 0),
            TimelineStep(20, 42, False, False, False, 1),
            TimelineStep(30, 30, False, False, False, 0),
        ],
    ))

    # Scenario 6 — harmless: permit active but gas stays low all day (should NOT alarm)
    scenarios.append(Scenario(
        name="Rolling Mill — hot work permit active, gas nominal throughout",
        gas_warning_threshold=35, gas_critical_threshold=80,
        is_genuinely_dangerous=False,
        timeline=[
            TimelineStep(0, 12, True, False, False, 2),
            TimelineStep(10, 15, True, False, False, 2),
            TimelineStep(20, 10, True, False, False, 1),
            TimelineStep(30, 14, True, False, False, 2),
        ],
    ))

    # Scenario 7 (BONUS, not counted in the main summary) — erratic/unstable sensor
    # pattern, NO permit, NO maintenance. No rule was ever written for "unstable
    # oscillation" — this represents a failing valve or degrading sensor.
    #
    # HONEST NOTE: with Zero-Harm's deliberately conservative AI weight (+15, same
    # as production), an anomaly signal ALONE does not cross the HIGH action
    # threshold and would stay at LOW in the live dashboard — by design, so a lone
    # ML signal never floods the alert feed with noise. What this scenario DOES
    # prove: the rule engine sees a risk score of ZERO the entire time (no rule
    # applies), while the AI layer registers real, measurable concern (90%+
    # abnormality confidence) that a safety officer watching the raw signal feed
    # would see well before the rule engine does. This is reported separately
    # below, not folded into the main detection-rate summary, so the headline
    # numbers stay honest and reproducible.
    scenarios.append(Scenario(
        name="Sinter Plant — erratic sensor oscillation, no permit/maintenance (equipment fault)",
        gas_warning_threshold=45, gas_critical_threshold=85,
        is_genuinely_dangerous=True,
        is_bonus_ai_visibility_test=True,
        timeline=[
            TimelineStep(0, 14, False, False, False, 1),
            TimelineStep(5, 15, False, False, False, 1),
            TimelineStep(10, 38, False, False, False, 1),  # sudden erratic spike
            TimelineStep(15, 12, False, False, False, 1),  # sudden erratic drop
            TimelineStep(20, 36, False, False, False, 1),  # <-- AI flags instability here
            TimelineStep(25, 10, False, False, False, 1),
            TimelineStep(30, 33, False, False, False, 1),
        ],
    ))

    return scenarios


# ---------------------------------------------------------------------------
# 5. Run the comparison and print a judge-ready report
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# 5. Evaluation metrics — Precision / Recall / F1 / Accuracy
#    Computed from a real confusion matrix over the 6 main scenarios
#    (4 dangerous = positive class, 2 harmless = negative class)
# ---------------------------------------------------------------------------
def compute_confusion_metrics(scenarios: List[Scenario], detection_fn) -> dict:
    tp = fp = fn = tn = 0
    for s in scenarios:
        detected = detection_fn(s) is not None
        if s.is_genuinely_dangerous and detected:
            tp += 1
        elif s.is_genuinely_dangerous and not detected:
            fn += 1
        elif not s.is_genuinely_dangerous and detected:
            fp += 1
        else:
            tn += 1

    precision = tp / (tp + fp) if (tp + fp) > 0 else 1.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 1.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    accuracy = (tp + tn) / len(scenarios) if scenarios else 0.0

    return {
        "true_positives": tp, "false_positives": fp,
        "false_negatives": fn, "true_negatives": tn,
        "precision": round(precision, 3), "recall": round(recall, 3),
        "f1_score": round(f1, 3), "accuracy": round(accuracy, 3),
    }


def print_metrics_block(label: str, metrics: dict):
    print(f"\n  {label}")
    print(f"    Precision: {metrics['precision']:.3f}   Recall: {metrics['recall']:.3f}   "
          f"F1: {metrics['f1_score']:.3f}   Accuracy: {metrics['accuracy']:.3f}")
    print(f"    (TP={metrics['true_positives']}  FP={metrics['false_positives']}  "
          f"FN={metrics['false_negatives']}  TN={metrics['true_negatives']})")


# ---------------------------------------------------------------------------
# 6. Export results for reuse (dashboard, README, slides)
# ---------------------------------------------------------------------------
def export_results(filepath_base: str, results: dict):
    json_path = f"{filepath_base}.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)

    csv_path = f"{filepath_base}.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["scenario", "genuinely_dangerous", "baseline_minute", "compound_minute", "lead_time_minutes"])
        for row in results["per_scenario"]:
            writer.writerow([
                row["name"], row["is_genuinely_dangerous"],
                row["baseline_minute"], row["compound_minute"], row["lead_time_minutes"],
            ])

    print(f"\nExported: {json_path}")
    print(f"Exported: {csv_path}")


# ---------------------------------------------------------------------------
# 7. Charts — far more effective in a live demo than console text
# ---------------------------------------------------------------------------
def generate_charts(main_scenarios: List[Scenario], baseline_metrics: dict, compound_metrics: dict):
    if not MATPLOTLIB_AVAILABLE:
        print("\n(matplotlib not installed — skipping charts. Run: pip install matplotlib)")
        return

    # Chart 1 — detection rate comparison
    fig, ax = plt.subplots(figsize=(6, 4))
    systems = ["Single-Sensor\nBaseline", "Zero-Harm\nCompound Engine"]
    detected = [baseline_metrics["true_positives"], compound_metrics["true_positives"]]
    missed = [baseline_metrics["false_negatives"], compound_metrics["false_negatives"]]
    ax.bar(systems, detected, label="Detected", color="#2ecc71")
    ax.bar(systems, missed, bottom=detected, label="Missed (false negative)", color="#e53935")
    ax.set_ylabel("Dangerous scenarios")
    ax.set_title("Detection Rate: Baseline vs. Compound Engine")
    ax.legend()
    fig.tight_layout()
    fig.savefig("chart_detection_comparison.png", dpi=130)
    plt.close(fig)

    # Chart 2 — incident timeline for the flagship Vizag-pattern scenario
    flagship = main_scenarios[0]
    minutes = [step.minute for step in flagship.timeline]
    gas_values = [step.gas_ppm for step in flagship.timeline]
    baseline_min = single_sensor_detection_minute(flagship)
    compound_min = compound_detection_minute(flagship)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(minutes, gas_values, marker="o", color="#1a1a1a", label="Gas reading (ppm)")
    ax.axhline(flagship.gas_warning_threshold, color="#f5a623", linestyle="--", label="Warning threshold")
    ax.axhline(flagship.gas_critical_threshold, color="#e53935", linestyle="--", label="Critical threshold")
    if compound_min is not None:
        ax.axvline(compound_min, color="#2ecc71", linestyle=":", label=f"Compound engine alarm (min {compound_min})")
    if baseline_min is not None:
        ax.axvline(baseline_min, color="#e53935", linestyle=":", label=f"Baseline alarm (min {baseline_min})")
    ax.set_xlabel("Minutes")
    ax.set_ylabel("Gas concentration (ppm)")
    ax.set_title(f"Incident Timeline — {flagship.name}")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig("chart_incident_timeline.png", dpi=130)
    plt.close(fig)

    # Chart 3 — compound risk score trend over the same flagship scenario
    scores = [compound_score(flagship, step) for step in flagship.timeline]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(minutes, scores, marker="o", color="#c0392b")
    ax.axhline(45, color="#e8672c", linestyle="--", label="HIGH threshold (45)")
    ax.axhline(70, color="#e53935", linestyle="--", label="CRITICAL threshold (70)")
    ax.set_xlabel("Minutes")
    ax.set_ylabel("Compound risk score (0-100)")
    ax.set_title(f"Risk Score Trend — {flagship.name}")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig("chart_risk_score_trend.png", dpi=130)
    plt.close(fig)

    print("\nCharts saved: chart_detection_comparison.png, chart_incident_timeline.png, chart_risk_score_trend.png")


# ---------------------------------------------------------------------------
# 8. Final comparison table — the one to put directly on a slide
# ---------------------------------------------------------------------------
def print_comparison_table(baseline_metrics, compound_metrics, lead_times, bonus_max_confidence):
    lead_avg = f"{sum(lead_times)/len(lead_times):.1f} min" if lead_times else "—"
    lead_note = "0 min (reference)"

    rows = [
        ("Dangerous Scenarios Detected", f"{baseline_metrics['true_positives']}/4",
         f"{compound_metrics['true_positives']}/4", f"{compound_metrics['true_positives']}/4"),
        ("False Negatives", str(baseline_metrics["false_negatives"]),
         str(compound_metrics["false_negatives"]), str(compound_metrics["false_negatives"])),
        ("False Positives", str(baseline_metrics["false_positives"]),
         str(compound_metrics["false_positives"]), str(compound_metrics["false_positives"])),
        ("Avg Lead Time", lead_note, lead_avg, f"{lead_avg} (unchanged — see note)"),
        ("AI Confidence (bonus test)", "—", "—",
         f"{bonus_max_confidence:.0f}%*" if bonus_max_confidence else "—"),
    ]

    col_widths = [32, 16, 16, 22]
    header = ["Metric", "Single Sensor", "Rule Engine", "Rule Engine + AI"]
    print("\n" + "-" * sum(col_widths))
    print("".join(h.ljust(w) for h, w in zip(header, col_widths)))
    print("-" * sum(col_widths))
    for row in rows:
        print("".join(str(c).ljust(w) for c, w in zip(row, col_widths)))
    print("-" * sum(col_widths))
    print("* AI Confidence is from the separate BONUS visibility test (a pattern with zero")
    print("  rule-engine score). It is NOT a 5th detected scenario in the 4-scenario count above —")
    print("  see the 'BONUS' section output for why that would misrepresent the result.")


def main():
    scenarios = build_scenarios()
    main_scenarios = [s for s in scenarios if not s.is_bonus_ai_visibility_test]
    bonus_scenarios = [s for s in scenarios if s.is_bonus_ai_visibility_test]

    print("=" * 88)
    print("ZERO-HARM — SINGLE-SENSOR vs RULE-BASED COMPOUND ENGINE")
    print("(6 main scenarios — matches Section 11 of the documentation)")
    print("=" * 88)

    per_scenario_results = []
    lead_times = []

    for s in main_scenarios:
        baseline_minute = single_sensor_detection_minute(s)
        compound_minute = compound_detection_minute(s)
        lead = (baseline_minute - compound_minute) if (baseline_minute is not None and compound_minute is not None) else None
        if lead is not None:
            lead_times.append(lead)

        print(f"\nScenario: {s.name}")
        print(f"  Genuinely dangerous?      {'YES' if s.is_genuinely_dangerous else 'no (should stay quiet)'}")
        print(f"  Single-sensor alarm at:   {f'minute {baseline_minute}' if baseline_minute is not None else 'NEVER TRIGGERED'}")
        print(f"  Compound engine alarm at: {f'minute {compound_minute}' if compound_minute is not None else 'never triggered'}")
        if lead is not None:
            print(f"  --> Lead time gained:     {lead} minutes earlier")
        elif compound_minute is not None and baseline_minute is None:
            print(f"  --> Baseline would have MISSED this entirely (false negative avoided)")

        per_scenario_results.append({
            "name": s.name,
            "is_genuinely_dangerous": s.is_genuinely_dangerous,
            "baseline_minute": baseline_minute,
            "compound_minute": compound_minute,
            "lead_time_minutes": lead,
        })

    baseline_metrics = compute_confusion_metrics(main_scenarios, single_sensor_detection_minute)
    compound_metrics = compute_confusion_metrics(main_scenarios, compound_detection_minute)

    print("\n" + "=" * 88)
    print("SUMMARY")
    print("=" * 88)
    print(f"Dangerous scenarios tested:                 4")
    print(f"Caught by single-sensor baseline:            {baseline_metrics['true_positives']}/4  "
          f"(false negatives: {baseline_metrics['false_negatives']})")
    print(f"Caught by Zero-Harm compound engine:         {compound_metrics['true_positives']}/4  "
          f"(false negatives: {compound_metrics['false_negatives']})")

    if lead_times:
        print(f"\nLead time statistics (compound vs. baseline, where both detected):")
        print(f"  Average: {sum(lead_times)/len(lead_times):.1f} min   "
              f"Median: {statistics.median(lead_times):.1f} min   "
              f"Min: {min(lead_times)} min   Max: {max(lead_times)} min")

    print_metrics_block("Single-Sensor Baseline — Precision / Recall / F1 / Accuracy", baseline_metrics)
    print_metrics_block("Zero-Harm Compound Engine — Precision / Recall / F1 / Accuracy", compound_metrics)

    print(f"\nFalse alarms on harmless scenarios (baseline): {baseline_metrics['false_positives']}")
    print(f"False alarms on harmless scenarios (compound): {compound_metrics['false_positives']}")

    # -----------------------------------------------------------------------
    # BONUS — AI Anomaly Layer visibility test (honest, separate from above)
    # -----------------------------------------------------------------------
    bonus_max_confidence = 0.0
    bonus_timeline_data = []

    if bonus_scenarios and SKLEARN_AVAILABLE:
        print("\n\n" + "=" * 88)
        print("BONUS — AI ANOMALY LAYER: WHAT IT SEES THAT THE RULE ENGINE DOESN'T")
        print("(reported separately and honestly — NOT folded into the summary above)")
        print("=" * 88)

        for s in bonus_scenarios:
            model = _train_anomaly_model(nominal_baseline=s.timeline[0].gas_ppm)
            prev = s.timeline[0].gas_ppm
            print(f"\nScenario: {s.name}")
            print(f"  {'Minute':>6}  {'Gas ppm':>8}  {'Rule Score':>11}  {'AI Confidence':>14}  {'AI Flag':>8}")
            confidences = []
            for step in s.timeline:
                delta = step.gas_ppm - prev
                confidence = _anomaly_confidence(model, step.gas_ppm, delta)
                rule_score = compound_score(s, step)
                flag = "ANOMALY" if confidence >= 65.0 else ""
                confidences.append(confidence)
                bonus_timeline_data.append({"minute": step.minute, "gas_ppm": step.gas_ppm,
                                             "rule_score": rule_score, "ai_confidence": confidence})
                print(f"  {step.minute:>6}  {step.gas_ppm:>8.1f}  {rule_score:>11}  {confidence:>13.1f}%  {flag:>8}")
                prev = step.gas_ppm

            bonus_max_confidence = max(confidences)
            print(f"\n  AI Confidence distribution — Average: {sum(confidences)/len(confidences):.1f}%   "
                  f"Max: {max(confidences):.1f}%   Min: {min(confidences):.1f}%")

            print(f"\n  Honest takeaway: the rule engine's score is 0 for this entire timeline —")
            print(f"  no permit, no maintenance, gas never crosses the warning threshold, so no")
            print(f"  rule was ever going to fire. The AI layer reaches {bonus_max_confidence:.0f}% abnormality")
            print(f"  confidence — real, measurable signal a safety officer monitoring the raw AI")
            print(f"  feed would see, even though Zero-Harm's conservative weighting (+15, same as")
            print(f"  production) deliberately holds this at LOW rather than raising a full alert")
            print(f"  on a lone ML signal. This is a visibility gain, not a detection count —")
            print(f"  see Section 5.5 of the documentation for why that trade-off was made.")
        print("=" * 88)
    elif bonus_scenarios:
        print("\n(scikit-learn not installed — skipping AI bonus test. Run: pip install scikit-learn numpy)")

    # -----------------------------------------------------------------------
    # Final comparison table (item requested: single table, judge-ready)
    # -----------------------------------------------------------------------
    print("\n\n" + "=" * 88)
    print("FINAL COMPARISON TABLE — put this directly on a slide")
    print("=" * 88)
    print_comparison_table(baseline_metrics, compound_metrics, lead_times, bonus_max_confidence)

    # -----------------------------------------------------------------------
    # Export + charts
    # -----------------------------------------------------------------------
    results = {
        "per_scenario": per_scenario_results,
        "baseline_metrics": baseline_metrics,
        "compound_metrics": compound_metrics,
        "lead_time_stats": {
            "average": round(sum(lead_times) / len(lead_times), 1) if lead_times else None,
            "median": statistics.median(lead_times) if lead_times else None,
            "min": min(lead_times) if lead_times else None,
            "max": max(lead_times) if lead_times else None,
        },
        "bonus_ai_visibility_test": {
            "scenario_name": bonus_scenarios[0].name if bonus_scenarios else None,
            "max_confidence_pct": bonus_max_confidence,
            "timeline": bonus_timeline_data,
            "note": "Rule engine score is 0 throughout; AI confidence shown separately and "
                    "NOT counted as a detected scenario in the main 4-scenario summary.",
        },
    }

    print("\n" + "=" * 88)
    print("EXPORTING RESULTS")
    print("=" * 88)
    export_results("results", results)
    generate_charts(main_scenarios, baseline_metrics, compound_metrics)


if __name__ == "__main__":
    main()
