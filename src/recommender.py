"""
Physics-Informed Maintenance Recommendation Engine.
Integrates mechanical thresholds, thermodynamic balances, and ML predictive risk
to generate actionable Standard Operating Procedures (SOPs) for plant engineers.
Tailored for heavy industrial operations (e.g., Solar Industries India Limited).
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List


def _safe_float(val: Any, default: float) -> float:
    """Safely extracts float from dictionary value, handling None, NaN, strings, and missing keys."""
    if val is None or val == "":
        return default
    try:
        f = float(val)
        return default if np.isnan(f) else f
    except (ValueError, TypeError):
        return default


def _safe_str(val: Any, default: str) -> str:
    """Safely extracts string uppercase code, handling None and non-string types."""
    if val is None or val == "":
        return default
    s = str(val).strip().upper()
    return s if s in ["L", "M", "H"] else default


def generate_maintenance_recommendation(
    sensor_data: Dict[str, Any],
    failure_probability: float,
    anomaly_severity_pct: float = 0.0,
) -> Dict[str, Any]:
    """
    Evaluates equipment state across 5 mechatronic domains and ML inference
    to produce a consolidated prescriptive maintenance recommendation.
    """
    air_temp = _safe_float(sensor_data.get("Air temperature [K]"), 300.0)
    process_temp = _safe_float(sensor_data.get("Process temperature [K]"), 310.0)
    speed_rpm = _safe_float(sensor_data.get("Rotational speed [rpm]"), 1500.0)
    torque_nm = _safe_float(sensor_data.get("Torque [Nm]"), 40.0)
    tool_wear = _safe_float(sensor_data.get("Tool wear [min]"), 0.0)
    prod_type = _safe_str(sensor_data.get("Type"), "M")

    # Derived physics variables
    delta_t = round(process_temp - air_temp, 2)
    omega = speed_rpm * (2 * 3.14159265 / 60.0)
    power_w = round(torque_nm * omega, 2)
    overstrain_val = round(tool_wear * torque_nm, 2)

    tool_wear_limit = 240.0
    tool_rul_estimate = max(0.0, round(tool_wear_limit - tool_wear, 1))

    subsystem_reports = []
    immediate_actions = []
    scheduled_actions = []
    risk_level = "NOMINAL"
    urgency = "LOW"

    # 1. Tooling Subsystem Evaluation (TWF)
    if tool_wear >= 200:
        risk_level = "CRITICAL"
        urgency = "IMMEDIATE"
        immediate_actions.append(
            f"REPLACE CUTTING TOOL IMMEDIATELY (Current wear: {tool_wear:.0f} min >= 200 min limit). High risk of Tool Wear Failure (TWF)."
        )
        subsystem_reports.append({
            "subsystem": "Tooling & Insert Assembly",
            "status": "CRITICAL",
            "diagnostic": f"Tool accumulated {tool_wear:.0f} mins of cutting contact. Tip wear threshold exceeded.",
            "sop": "Execute standard tool change procedure SOP-TL-04. Inspect toolholder taper for micro-fretting.",
        })
    elif tool_wear >= 160:
        if risk_level != "CRITICAL":
            risk_level = "WARNING"
            urgency = "MEDIUM"
        scheduled_actions.append(
            f"Stage replacement tool insert for next scheduled shift pause (Wear: {tool_wear:.0f} min)."
        )
        subsystem_reports.append({
            "subsystem": "Tooling & Insert Assembly",
            "status": "ELEVATED",
            "diagnostic": f"Tool approaching terminal wear envelope ({tool_rul_estimate:.0f} min estimated RUL remaining).",
            "sop": "Pre-calibrate spare tooling insert; inspect edge for crater wear or chipping.",
        })
    else:
        subsystem_reports.append({
            "subsystem": "Tooling & Insert Assembly",
            "status": "HEALTHY",
            "diagnostic": f"Tool wear within safe operating margins ({tool_rul_estimate:.0f} min estimated RUL remaining).",
            "sop": "Continue regular shift lubrication and tool offset tracking.",
        })

    # 2. Thermal Dissipation Evaluation (HDF)
    if delta_t < 8.6 and speed_rpm < 1380:
        risk_level = "CRITICAL"
        urgency = "IMMEDIATE"
        immediate_actions.append(
            f"HEAT DISSIPATION FAILURE HAZARD: Process ΔT is only {delta_t:.2f} K at low speed {speed_rpm:.0f} RPM."
        )
        subsystem_reports.append({
            "subsystem": "Thermal & Coolant Circulation",
            "status": "CRITICAL",
            "diagnostic": "Insufficient thermal dissipation gradient across heat sink / cooling jacket.",
            "sop": "Check coolant flow rate, filter cleanliness, heat exchanger circulation pump, and chiller valves.",
        })
    elif delta_t < 9.2 and speed_rpm < 1400:
        if risk_level != "CRITICAL":
            risk_level = "WARNING"
            if urgency == "LOW":
                urgency = "MEDIUM"
        scheduled_actions.append(
            f"PRE-TRIP THERMAL ADVISORY: Temperature gradient narrowing (ΔT: {delta_t:.2f} K near 8.6 K critical threshold)."
        )
        subsystem_reports.append({
            "subsystem": "Thermal & Coolant Circulation",
            "status": "ELEVATED",
            "diagnostic": f"Thermal dissipation margin degraded (ΔT = {delta_t:.2f} K approaching trip limit).",
            "sop": "Flush coolant supply nozzles and check chiller fluid temperature.",
        })
    elif process_temp > 313.15:  # > 40 °C
        if risk_level != "CRITICAL":
            risk_level = "WARNING"
        scheduled_actions.append(f"Inspect equipment cooling fans and ambient heat dissipation (Process Temp: {process_temp:.1f} K).")
        subsystem_reports.append({
            "subsystem": "Thermal & Coolant Circulation",
            "status": "ELEVATED",
            "diagnostic": f"High process surface temperature ({process_temp:.1f} K / {process_temp - 273.15:.1f} °C).",
            "sop": "Verify ventilation louvers and check coolant mixture concentration.",
        })
    else:
        subsystem_reports.append({
            "subsystem": "Thermal & Coolant Circulation",
            "status": "HEALTHY",
            "diagnostic": f"Normal thermal equilibrium (ΔT = {delta_t:.2f} K).",
            "sop": "Maintain routine coolant reservoir checks.",
        })

    # 3. Spindle Drive & Power Subsystem (PWF)
    if power_w < 3500 or power_w > 9000:
        risk_level = "CRITICAL"
        urgency = "IMMEDIATE"
        immediate_actions.append(
            f"POWER TRIP DANGER: Spindle mechanical power is {power_w:.1f} W (Normal envelope: 3,500 W - 9,000 W)."
        )
        subsystem_reports.append({
            "subsystem": "Spindle Drive & Inverter Motor",
            "status": "CRITICAL",
            "diagnostic": f"Spindle mechanical power anomaly ({power_w:.1f} W). Suspected motor drive stall, belt slip, or VFD fault.",
            "sop": "Perform electrical load audit on VFD inverter; measure motor 3-phase current balance and belt tension.",
        })
    elif (3500 <= power_w < 3800) or (8200 < power_w <= 9000):
        if risk_level != "CRITICAL":
            risk_level = "WARNING"
            if urgency == "LOW":
                urgency = "MEDIUM"
        scheduled_actions.append(
            f"POWER ENVELOPE WARNING: Spindle mechanical power is {power_w:.1f} W (near tripping boundaries 3.5 kW / 9.0 kW)."
        )
        subsystem_reports.append({
            "subsystem": "Spindle Drive & Inverter Motor",
            "status": "ELEVATED",
            "diagnostic": f"Spindle operating near power trip limits ({power_w:.1f} W).",
            "sop": "Review cut depth, feed rate, and check VFD current draw.",
        })
    else:
        subsystem_reports.append({
            "subsystem": "Spindle Drive & Inverter Motor",
            "status": "HEALTHY",
            "diagnostic": f"Spindle mechanical power within optimal operating band ({power_w:.1f} W).",
            "sop": "Maintain periodic vibration and motor bearing temperature baseline logging.",
        })

    # 4. Overstrain & Structural Load (OSF)
    strain_limit = 11000.0 if prod_type == "L" else (12000.0 if prod_type == "M" else 13000.0)
    if overstrain_val > strain_limit:
        risk_level = "CRITICAL"
        urgency = "IMMEDIATE"
        immediate_actions.append(
            f"OVERSTRAIN BREAKAGE HAZARD: Wear × Torque is {overstrain_val:.0f} (Exceeds {prod_type}-variant threshold of {strain_limit:.0f})."
        )
        subsystem_reports.append({
            "subsystem": "Mechanical Structure & Spindle Bearing",
            "status": "CRITICAL",
            "diagnostic": "Excessive continuous torque combined with high tool wear creating dangerous mechanical fatigue.",
            "sop": "Reduce spindle feed rate immediately by 25%; perform non-destructive inspection of chuck jaws and spindle arbor.",
        })
    elif overstrain_val >= 0.85 * strain_limit:
        if risk_level != "CRITICAL":
            risk_level = "WARNING"
            if urgency == "LOW":
                urgency = "MEDIUM"
        scheduled_actions.append(
            f"STRUCTURAL OVERSTRAIN ADVISORY: Wear × Torque is {overstrain_val:.0f} (At {overstrain_val/strain_limit*100:.0f}% of fatigue rupture limit {strain_limit:.0f})."
        )
        subsystem_reports.append({
            "subsystem": "Mechanical Structure & Spindle Bearing",
            "status": "ELEVATED",
            "diagnostic": f"Overstrain index {overstrain_val:.0f} approaching critical mechanical fatigue threshold.",
            "sop": "Inspect insert wear lands, reduce feed rate by 15%, and check for workpiece material hard-spots.",
        })
    else:
        subsystem_reports.append({
            "subsystem": "Mechanical Structure & Spindle Bearing",
            "status": "HEALTHY",
            "diagnostic": f"Overstrain index {overstrain_val:.0f} well below fatigue limit ({strain_limit:.0f}).",
            "sop": "Normal operation allowed.",
        })

    # 5. Machine Learning Failure Probability & Anomaly Integration
    if failure_probability >= 0.70:
        risk_level = "CRITICAL"
        urgency = "IMMEDIATE"
        immediate_actions.append(
            f"ML PREDICTIVE ALERT: Machine failure probability is {failure_probability * 100:.1f}%. Initiate Lockout-Tagout (LOTO) protocol."
        )
    elif failure_probability >= 0.35:
        if risk_level != "CRITICAL":
            risk_level = "WARNING"
            urgency = "MEDIUM"
        scheduled_actions.append(
            f"ML ELEVATED RISK: Machine failure probability is {failure_probability * 100:.1f}%. Dispatch maintenance inspection within 2 hours."
        )

    if anomaly_severity_pct >= 80.0:
        if risk_level != "CRITICAL":
            risk_level = "WARNING"
        scheduled_actions.append(
            f"ANOMALY WARNING: Sensor deviation score is {anomaly_severity_pct:.1f}%. Unusual multi-variable sensor drift detected."
        )

    # Compile Final Guidance
    if not immediate_actions and not scheduled_actions:
        scheduled_actions.append("All mechatronic telemetry channels within nominal bounds. Continue scheduled autonomous production.")

    status_badge_color = {
        "CRITICAL": "#FF4B4B",
        "WARNING": "#FFA500",
        "NOMINAL": "#00CC66",
    }.get(risk_level, "#00CC66")

    return {
        "status": risk_level,
        "status_color": status_badge_color,
        "urgency": urgency,
        "failure_probability_pct": round(failure_probability * 100.0, 1),
        "anomaly_severity_pct": round(anomaly_severity_pct, 1),
        "tool_rul_minutes": tool_rul_estimate,
        "immediate_actions": immediate_actions,
        "scheduled_actions": scheduled_actions,
        "subsystems": subsystem_reports,
        "metrics_summary": {
            "Power_W": power_w,
            "Temp_Difference_K": delta_t,
            "Overstrain_Index": overstrain_val,
            "Tool_Wear_Min": tool_wear,
            "Torque_Nm": torque_nm,
            "Speed_RPM": speed_rpm,
        },
    }


def generate_fleet_audit_report(
    df_scored: pd.DataFrame,
    critical_threshold: float = 0.50,
    warning_threshold: float = 0.25,
    top_n: int = 15,
) -> str:
    """
    Generates a formal, executive-ready plant inspection report formatted in Markdown.
    Includes fleet summary, operational risk distribution, high-priority work orders,
    and prescribed Standard Operating Procedures (SOPs).
    """
    from datetime import datetime

    total_units = len(df_scored)
    prob_col = "Failure_Probability" if "Failure_Probability" in df_scored.columns else None
    anomaly_col = "Is_Anomaly" if "Is_Anomaly" in df_scored.columns else None
    severity_col = "Anomaly_Severity_Pct" if "Anomaly_Severity_Pct" in df_scored.columns else None

    if prob_col:
        crit_count = int((df_scored[prob_col] >= critical_threshold).sum())
        warn_count = int(((df_scored[prob_col] >= warning_threshold) & (df_scored[prob_col] < critical_threshold)).sum())
        nom_count = total_units - crit_count - warn_count
        avg_risk = round(float(df_scored[prob_col].mean()) * 100.0, 1)
    else:
        crit_count, warn_count, nom_count, avg_risk = 0, 0, total_units, 0.0

    anomaly_count = int(df_scored[anomaly_col].sum()) if anomaly_col else 0

    timestamp_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    lines = [
        "# EXECUTIVE PLANT RELIABILITY & MAINTENANCE AUDIT REPORT",
        f"**Generated**: {timestamp_str} | **System**: PMAD Industrial Mechatronics Engine",
        f"**Target Site**: High-Reliability Energetic Manufacturing / Solar Industries India Limited",
        "",
        "---",
        "",
        "## 1. Fleet Operational Health Summary",
        f"- **Total Monitored Equipment Units**: {total_units:,}",
        f"- **Critical Failure Risk Units (Prob >= {critical_threshold*100:.0f}%)**: {crit_count} ({crit_count/max(1, total_units)*100:.1f}%)",
        f"- **Warning State Units (Prob >= {warning_threshold*100:.0f}%)**: {warn_count} ({warn_count/max(1, total_units)*100:.1f}%)",
        f"- **Nominal Operating Units**: {nom_count} ({nom_count/max(1, total_units)*100:.1f}%)",
        f"- **Unsupervised Anomalies Detected**: {anomaly_count} ({anomaly_count/max(1, total_units)*100:.1f}%)",
        f"- **Mean Fleet Breakdown Probability**: {avg_risk}%",
        "",
        "---",
        "",
        "## 2. Priority Maintenance Work Orders (Top Dispatch Queue)",
        "",
    ]

    # Sort top priority units
    if prob_col:
        df_priority = df_scored.sort_values(by=prob_col, ascending=False).head(top_n)
    elif severity_col:
        df_priority = df_scored.sort_values(by=severity_col, ascending=False).head(top_n)
    else:
        df_priority = df_scored.head(top_n)

    id_col = "Product ID" if "Product ID" in df_priority.columns else ("UDI" if "UDI" in df_priority.columns else None)

    lines.append("| Priority | Equipment ID | Failure Risk | Anomaly Severity | Tool Wear | Power (W) | Prescribed SOP |")
    lines.append("| :--- | :--- | :--- | :--- | :--- | :--- | :--- |")

    for rank, (idx, row) in enumerate(df_priority.iterrows(), 1):
        eq_id = str(row[id_col]) if id_col else f"Unit-{idx}"
        f_prob = f"{row[prob_col]*100:.1f}%" if prob_col else "N/A"
        a_sev = f"{row[severity_col]:.1f}%" if severity_col else "N/A"
        t_wear = f"{row.get('Tool wear [min]', 0):.0f} min"
        p_watt = f"{row.get('Power_W', 0):.0f} W"

        # Determine concise SOP tag
        rec = generate_maintenance_recommendation(
            row.to_dict(),
            float(row.get(prob_col, 0.0)) if prob_col else 0.0,
            float(row.get(severity_col, 0.0)) if severity_col else 0.0,
        )
        sop_summary = rec["immediate_actions"][0] if rec["immediate_actions"] else (rec["scheduled_actions"][0] if rec["scheduled_actions"] else "SOP-RT-01 Routine")
        sop_clean = sop_summary.replace("|", "/").strip()
        if len(sop_clean) > 65:
            sop_clean = sop_clean[:62] + "..."

        lines.append(f"| #{rank} | `{eq_id}` | **{f_prob}** | {a_sev} | {t_wear} | {p_watt} | {sop_clean} |")

    lines.extend([
        "",
        "---",
        "",
        "## 3. Prescriptive Standard Operating Procedures (SOP Reference)",
        "- **SOP-TL-04 (Tool Tip Replacement)**: Tool wear exceeded 200 min. Mandatory replacement of cutting insert before micro-fracturing.",
        "- **SOP-TH-02 (Cooling Jacket Descaling)**: Process delta-T collapsed below 8.6 K at low speeds. Inspect coolant pump flow rate and heat exchanger plates.",
        "- **SOP-DR-01 (Drive Motor Envelope Audit)**: Mechanical power exceeded 9,000 W or dropped below 3,500 W. Inspect inverter drive and mechanical torque transmission.",
        "- **SOP-OS-03 (Overstrain Load Redistribution)**: High torque on worn tool exceeding structural limits. Reduce cutting feed rate immediately.",
        "- **SOP-AN-05 (Sensor Calibration & Bearing Inspection)**: Unsupervised density anomaly without classified failure. Inspect multi-axis vibration and transducer drift.",
        "",
        "---",
        "*Report verified and generated autonomously by PMAD Industrial Analytics Suite.*",
    ])

    return "\n".join(lines)

