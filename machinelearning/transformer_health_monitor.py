# -*- coding: utf-8 -*-

import os
import logging
from datetime import datetime
from typing import Dict, List, Any
import pandas as pd

from DataProcessing.programFiles import Database
from .forecast_engine import TransformerForecastEngine

logger = logging.getLogger(__name__)

# Health weight constants
WEIGHTS = {
    "Secondary Voltage-A-phase (V)": 0.6, "Secondary Voltage-B-phase (V)": 0.6, "Secondary Voltage-C-phase (V)": 0.6,
    "Secondary Current-A-phase(A)": 0.65, "Secondary Current-B-phase(A)": 0.65, "Secondary Current-C-phase(A)": 0.65,
    "PF%": 0.4,
    "VTHD-A-B": 0.3, "VTHD-B-C": 0.3, "VTHD-A-C": 0.3,
    "Winding-Temp-A(°C)": 0.9, "Winding-Temp-B(°C)": 0.9, "Winding-Temp-C(°C)": 0.9,
}

COLOR_SCORES = {"Green": 1.0, "Yellow": 0.5, "Red": 0.2}

SUBSYSTEM1_COLUMN_MAP = {
    "avg_secondary_voltage_a_phase": "Secondary Voltage-A-phase (V)",
    "avg_secondary_voltage_b_phase": "Secondary Voltage-B-phase (V)",
    "avg_secondary_voltage_c_phase": "Secondary Voltage-C-phase (V)",
    "avg_secondary_current_a_phase": "Secondary Current-A-phase(A)",
    "avg_secondary_current_b_phase": "Secondary Current-B-phase(A)",
    "avg_secondary_current_c_phase": "Secondary Current-C-phase(A)",
    "avg_power_factor": "PF%",
    "avg_vTHD_a_phase": "VTHD-A-B",
    "avg_vTHD_b_phase": "VTHD-B-C",
    "avg_vTHD_c_phase": "VTHD-A-C",
    "avg_winding_temp_a_phase": "Winding-Temp-A(°C)",
    "avg_winding_temp_b_phase": "Winding-Temp-B(°C)",
    "avg_winding_temp_c_phase": "Winding-Temp-C(°C)",
}


class TransformerHealthMonitor:
    """
    Main class for transformer health monitoring system.
    """

    def __init__(self, database: Database):

        self.db = database

        # CRITICAL FIX — forecasting engine MUST receive the DB instance
        self.forecast_engine = TransformerForecastEngine(database=self.db)

        logger.info("Transformer Health Monitor initialized")

    # ----------------------------------------------------------------------
    def test_connection(self):
        """Test database functionality."""
        print("=" * 60)
        print(" DATABASE CONNECTION TEST ")
        print("=" * 60)

        ok = self.db.test_connection()
        print("Database Connection:", "SUCCESS" if ok else "FAILED")

        self.db.print_connection_status()
        return ok

    # ----------------------------------------------------------------------
    def initialize_database_schema(self):
        print("\nInitializing HealthScores + ForecastData tables...")

        self.db.initialize_schema()

        print(" -> Database schema initialized.")
        print(" -> HealthScores and ForecastData ready.")

    # ----------------------------------------------------------------------
    def run_health_assessments(self, transformer_names: List[str]):
        print("\nRunning health assessments...")

        results = {}

        for tname in transformer_names:
            try:
                res = self.calculate_health_score(tname)
                if res:
                    results[tname] = res
            except Exception as e:
                logger.error(f"Error assessing {tname}: {e}")
                print(f"Error assessing {tname}: {e}")

        self._print_overall_summary(results)
        return results

    # ----------------------------------------------------------------------
    def calculate_health_score(self, transformer_name):
        logger.info(f"Calculating health score for {transformer_name}...")

        try:
            specs = self.db.get_rated_specs(transformer_name)
            if not specs:
                print(f"No rated specs for {transformer_name}")
                return None

            latest = self.db.get_latest_averages(transformer_name)
            if not latest:
                print(f"No subsystem averages found for {transformer_name}")
                return None

            results = {}
            weighted_sum = 0
            weight_total = 0

            critical = []
            warns = []

            for col, avg_value in latest.items():
                if col not in SUBSYSTEM1_COLUMN_MAP:
                    continue

                var_name = SUBSYSTEM1_COLUMN_MAP[col]

                if var_name not in specs:
                    continue

                rated = float(specs[var_name])
                if rated == 0:
                    continue

                diff_ratio = abs(avg_value - rated) / rated

                if diff_ratio <= 0.05:
                    status = "Green"
                elif diff_ratio <= 0.10:
                    status = "Yellow"
                    warns.append(f"{var_name}: {diff_ratio*100:.1f}% deviation")
                else:
                    status = "Red"
                    critical.append(f"{var_name}: {diff_ratio*100:.1f}% deviation")

                weight = WEIGHTS.get(var_name, 1)
                score = COLOR_SCORES[status]

                weighted_sum += score * weight
                weight_total += weight

                results[var_name] = {
                    "Average": avg_value,
                    "Rated": rated,
                    "Status": status,
                    "Deviation_Percent": diff_ratio * 100,
                    "Weight": weight,
                    "Score": score
                }

            if weight_total == 0:
                return None

            overall_score = weighted_sum / weight_total
            overall_color = "Green" if overall_score >= 0.79 else "Yellow" if overall_score >= 0.49 else "Red"

            self.db.save_health_results(transformer_name, results, overall_score, overall_color)

            self._print_health_summary(
                transformer_name, results, overall_score, overall_color, critical, warns
            )

            return {
                "transformer_name": transformer_name,
                "overall_score": overall_score,
                "overall_color": overall_color,
                "critical_issues": critical,
                "warnings": warns
            }

        except Exception as e:
            logger.error(f"Health score error for {transformer_name}: {e}")
            print(f"Health error {transformer_name}: {e}")
            return None

    # ----------------------------------------------------------------------
    def run_lifetime_forecasting(self, transformer_names, method="ensemble"):
        print("\nRunning lifetime forecasting...")
        print("=" * 60)

        results = {}

        for tname in transformer_names:
            try:
                life_df = self.db.get_transformer_lifetime_data(tname)

                if life_df.empty:
                    print(f"No lifetime data for {tname}")
                    continue

                score = self.db.get_latest_health_score(tname)

                # calls our fixed forecast_engine
                res = self.forecast_engine.forecast_transformer_lifetime(
                    tname, life_df, score, method
                )

                if res:
                    results[tname] = res

            except Exception as e:
                print(f"Forecasting error for {tname}: {e}")
                logger.error(f"Forecasting error {tname}: {e}")

        return results

    # ----------------------------------------------------------------------
    def run_health_monitoring(self):
        print("Transformer Health Monitoring System")
        print("=" * 60)

        if not self.test_connection():
            return False

        self.initialize_database_schema()

        names = self.db.get_transformer_names()
        if not names:
            print("No transformers found.")
            return False

        print(f"Found {len(names)} transformers: {names}")

        self.run_health_assessments(names)

        print("\nRunning forecasting...")
        f_results = self.run_lifetime_forecasting(names)

        if f_results:
            self.print_forecast_summary(f_results)

        print("\nHealth monitoring complete.")
        return True

    # ----------------------------------------------------------------------
    def _print_health_summary(self, name, results, score, color, critical, warns):
        print(f"\n{'='*60}")
        print(f"HEALTH: {name}")
        print(f"Score: {score:.2f}  Status: {color}")
        print(f"{'='*60}")

    # ----------------------------------------------------------------------
    def _print_overall_summary(self, results):
        if not results:
            print("No health results.")
            return

        scores = [v["overall_score"] for v in results.values()]
        colors = [v["overall_color"] for v in results.values()]

        print("\nOverall Assessment Summary")
        print("=" * 60)
        print(f"Average score: {sum(scores)/len(scores):.2f}")
        print(f"Green: {colors.count('Green')}")
        print(f"Yellow: {colors.count('Yellow')}")
        print(f"Red: {colors.count('Red')}")

    # ----------------------------------------------------------------------
    def print_forecast_summary(self, f_results):
        print("\nForecast Summary")
        print("=" * 60)

        for name, res in f_results.items():
            years = res.get("remaining_life_years", None)
            print(f"\n{name}:")
            print(f"  Model: {res['model_name']}")
            print(f"  R²: {res['r2_score']:.3f}")

            if years:
                print(f"  20% Remaining in: {years:.1f} years")
            else:
                print("  20% Remaining: > 50 years")

    # ----------------------------------------------------------------------
    def close(self):
        self.db.close()
