"""
Tests for physics-informed prescriptive maintenance recommender engine.
"""

import pytest
import pandas as pd
from src.recommender import (
    generate_maintenance_recommendation,
    generate_fleet_audit_report,
)


def test_nominal_operating_state():
    nominal_telemetry = {
        "Type": "M",
        "Air temperature [K]": 300.0,
        "Process temperature [K]": 310.0,
        "Rotational speed [rpm]": 1500,
        "Torque [Nm]": 40.0,
        "Tool wear [min]": 30,
    }
    rec = generate_maintenance_recommendation(
        sensor_data=nominal_telemetry,
        failure_probability=0.04,
        anomaly_severity_pct=15.0,
    )

    assert rec["status"] == "NOMINAL"
    assert rec["urgency"] == "LOW"
    assert len(rec["immediate_actions"]) == 0
    assert len(rec["scheduled_actions"]) > 0
    assert rec["tool_rul_minutes"] == 210.0


def test_critical_tool_wear_failure():
    worn_tool_telemetry = {
        "Type": "M",
        "Air temperature [K]": 300.0,
        "Process temperature [K]": 310.0,
        "Rotational speed [rpm]": 1500,
        "Torque [Nm]": 40.0,
        "Tool wear [min]": 215,  # Exceeds 200 min
    }
    rec = generate_maintenance_recommendation(
        sensor_data=worn_tool_telemetry,
        failure_probability=0.65,
        anomaly_severity_pct=40.0,
    )

    assert rec["status"] == "CRITICAL"
    assert rec["urgency"] == "IMMEDIATE"
    assert any("REPLACE CUTTING TOOL" in action for action in rec["immediate_actions"])
    assert rec["tool_rul_minutes"] == 25.0


def test_heat_dissipation_failure():
    # Low delta T (< 8.6) and low speed (< 1380 rpm)
    hdf_telemetry = {
        "Type": "L",
        "Air temperature [K]": 302.0,
        "Process temperature [K]": 305.0,  # dT = 3.0 K
        "Rotational speed [rpm]": 1250,     # < 1380
        "Torque [Nm]": 40.0,
        "Tool wear [min]": 50,
    }
    rec = generate_maintenance_recommendation(
        sensor_data=hdf_telemetry,
        failure_probability=0.55,
        anomaly_severity_pct=60.0,
    )

    assert rec["status"] == "CRITICAL"
    assert any("HEAT DISSIPATION" in action for action in rec["immediate_actions"])


def test_power_failure_envelope():
    # Extreme torque and speed leading to power > 9000 W
    # Power = Torque * (rpm * 2 * pi / 60). For 100 Nm and 1500 rpm: ~ 15,707 W
    high_power_telemetry = {
        "Type": "H",
        "Air temperature [K]": 300.0,
        "Process temperature [K]": 310.0,
        "Rotational speed [rpm]": 1500,
        "Torque [Nm]": 100.0,
        "Tool wear [min]": 50,
    }
    rec = generate_maintenance_recommendation(
        sensor_data=high_power_telemetry,
        failure_probability=0.88,
        anomaly_severity_pct=85.0,
    )

    assert rec["status"] == "CRITICAL"
    assert any("POWER TRIP DANGER" in action for action in rec["immediate_actions"])


def test_overstrain_breakage():
    # Wear * Torque > 11000 for Type L
    # Wear = 200, Torque = 60 -> 12,000 > 11000
    strain_telemetry = {
        "Type": "L",
        "Air temperature [K]": 300.0,
        "Process temperature [K]": 310.0,
        "Rotational speed [rpm]": 1500,
        "Torque [Nm]": 60.0,
        "Tool wear [min]": 200,
    }
    rec = generate_maintenance_recommendation(
        sensor_data=strain_telemetry,
        failure_probability=0.72,
        anomaly_severity_pct=70.0,
    )

    assert rec["status"] == "CRITICAL"
    assert any("OVERSTRAIN BREAKAGE" in action for action in rec["immediate_actions"])


def test_recommender_none_and_empty_values():
    # Robustness against None and empty strings in sensor dictionary
    corrupt_telemetry = {
        "Type": None,
        "Air temperature [K]": None,
        "Process temperature [K]": "",
        "Rotational speed [rpm]": None,
        "Torque [Nm]": None,
        "Tool wear [min]": None,
    }
    rec = generate_maintenance_recommendation(
        sensor_data=corrupt_telemetry,
        failure_probability=0.10,
        anomaly_severity_pct=10.0,
    )
    assert rec["status"] in ["NOMINAL", "WARNING", "CRITICAL"]
    assert rec["tool_rul_minutes"] >= 0.0


def test_recommender_early_warning_overstrain():
    # Type L limit is 11,000. 85% limit is 9,350. Wear=150, Torque=65 -> 9,750 (elevated advisory)
    pre_fatigue_telemetry = {
        "Type": "L",
        "Air temperature [K]": 300.0,
        "Process temperature [K]": 310.0,
        "Rotational speed [rpm]": 1200,  # 1200 RPM * 65 Nm = 8.17 kW (healthy power band)
        "Torque [Nm]": 65.0,
        "Tool wear [min]": 150,  # 150 * 65 = 9750 (88.6% of 11,000 limit)
    }
    rec = generate_maintenance_recommendation(
        sensor_data=pre_fatigue_telemetry,
        failure_probability=0.20,
        anomaly_severity_pct=25.0,
    )
    assert rec["status"] == "WARNING"
    assert any("STRUCTURAL OVERSTRAIN ADVISORY" in action for action in rec["scheduled_actions"])


def test_recommender_early_warning_power():
    # Power = Torque * (rpm * 2 * pi / 60) = 3600 W (near 3500 W lower limit)
    # 3600 / (1500 * 2 * pi / 60) = ~22.9 Nm
    low_power_telemetry = {
        "Type": "M",
        "Air temperature [K]": 300.0,
        "Process temperature [K]": 310.0,
        "Rotational speed [rpm]": 1500,
        "Torque [Nm]": 23.0,  # ~3612 W (near 3500 W limit)
        "Tool wear [min]": 50,
    }
    rec = generate_maintenance_recommendation(
        sensor_data=low_power_telemetry,
        failure_probability=0.25,
        anomaly_severity_pct=30.0,
    )
    assert rec["status"] == "WARNING"
    assert any("POWER ENVELOPE WARNING" in action for action in rec["scheduled_actions"])


def test_generate_fleet_audit_report():
    df_sample = pd.DataFrame([
        {
            "UDI": 1,
            "Product ID": "M14860",
            "Type": "M",
            "Air temperature [K]": 300.0,
            "Process temperature [K]": 310.0,
            "Rotational speed [rpm]": 1500,
            "Torque [Nm]": 40.0,
            "Tool wear [min]": 30,
            "Failure_Probability": 0.05,
            "Anomaly_Severity_Pct": 10.0,
            "Machine failure": 0,
        },
        {
            "UDI": 2,
            "Product ID": "L47181",
            "Type": "L",
            "Air temperature [K]": 302.0,
            "Process temperature [K]": 306.0,
            "Rotational speed [rpm]": 1280,
            "Torque [Nm]": 65.0,
            "Tool wear [min]": 220,
            "Failure_Probability": 0.85,
            "Anomaly_Severity_Pct": 85.0,
            "Machine failure": 1,
        },
    ])
    report = generate_fleet_audit_report(df_sample, critical_threshold=0.5, warning_threshold=0.25)
    assert isinstance(report, str)
    assert "EXECUTIVE PLANT RELIABILITY & MAINTENANCE AUDIT REPORT" in report
    assert "Priority Maintenance Work Orders" in report
    assert "L47181" in report

