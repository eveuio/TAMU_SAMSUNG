import os
import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime

# --- Configuration Constants ---
DB_PATH = "C:/Users/bigal/Capstone-alex/data/my_database.db"

# Health Monitoring Weights and Scores
WEIGHTS = {
    "Secondary Voltage-A-phase (V)": 0.6, "Secondary Voltage-B-phase (V)": 0.6, "Secondary Voltage-C-phase (V)": 0.6,
    "Secondary Current-A-phase(A)": 0.65, "Secondary Current-B-phase(A)": 0.65, "Secondary Current-C-phase(A)": 0.65,
    "PF%": 0.4,
    "VTHD-A-B": 0.3, "VTHD-B-C": 0.3, "VTHD-A-C": 0.3,
    "Winding-Temp-A(°C)": 0.9, "Winding-Temp-B(°C)": 0.9, "Winding-Temp-C(°C)": 0.9,
}
COLOR_SCORES = {"Green": 1.0, "Yellow": 0.5, "Red": 0.2}

# Mapping from Subsystem 1's column names to your subsystem's variable names
SUBSYSTEM1_COLUMN_MAP = {
    "avg_secondary_voltage_a_phase": "Secondary Voltage-A-phase (V)",
    "avg_secondary_voltage_b_phase": "Secondary Voltage-B-phase (V)",
    "avg_secondary_voltage_c_phase": "Secondary Voltage-C-phase (V)",
    "avg_secondary_current_a_phase": "Secondary Current-A-phase(A)",
    "avg_secondary_current_b_phase": "Secondary Current-B-phase(A)",
    "avg_secondary_current_c_phase": "Secondary Current-C-phase(A)",
    "avg_power_factor": "PF%",
    "avg_vTHD_a_phase": "VTHD-A-B", # Note: Assuming a mapping, this might need adjustment
    "avg_vTHD_b_phase": "VTHD-B-C", # Note: Assuming a mapping, this might need adjustment
    "avg_vTHD_c_phase": "VTHD-A-C", # Note: Assuming a mapping, this might need adjustment
    "avg_winding_temp_a_phase": "Winding-Temp-A(°C)",
    "avg_winding_temp_b_phase": "Winding-Temp-B(°C)",
    "avg_winding_temp_c_phase": "Winding-Temp-C(°C)",
}

class Database:
    """
    A unified class to manage database interactions for Subsystem 2 of the transformer monitoring project.
    This class reads pre-processed data from Subsystem 1 and generates health scores and forecasts.
    """
    def __init__(self, db_path=DB_PATH):
        self.db_path = db_path
        self.conn = sqlite3.connect(self.db_path)
        self.conn.row_factory = sqlite3.Row # Allows accessing columns by name
        self.cursor = self.conn.cursor()
        print(f"Database Manager (Subsystem 2) initialized. Connected to {self.db_path}")

    def close(self):
        self.conn.close()
        print("Database connection closed.")

    def initialize_schema(self):
        """Creates the tables required by Subsystem 2 if they don't exist."""
        # Create TransformerSpecs table for rated values
        self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS TransformerSpecs (
                transformer_name TEXT,
                variable_name TEXT,
                rated_value REAL,
                PRIMARY KEY (transformer_name, variable_name)
            )
        """)
        # Create HealthScores table for this subsystem's output
        self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS HealthScores (
                transformer_name TEXT,
                date TEXT,
                variable_name TEXT,
                average_value REAL,
                rated_value REAL,
                status TEXT,
                overall_score REAL,
                overall_color TEXT
            )
        """)
        # Create ForecastData table for the forecasting model's output
        self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS ForecastData (
                transformer_name TEXT,
                forecast_date TEXT,
                predicted_lifetime REAL,
                PRIMARY KEY (transformer_name, forecast_date)
            )
        """)
        self.conn.commit()
        print("Initialized Subsystem 2 schema: 'TransformerSpecs', 'HealthScores', and 'ForecastData' tables are ready.")

    def seed_transformer_specs(self):
        """Populates the TransformerSpecs table with predefined rated values."""
        # This data would come from your hardcoded specs
        transformer_specs_data = {
            "LTR_A01_Data": {"Secondary Voltage-A-phase (V)": 480, "Secondary Current-A-phase(A)": 601, "PF%": 93, "VTHD-A-B": 2.5, "Winding-Temp-A(°C)": 63},
            "LTR_22B01_Data": {"Secondary Voltage-A-phase (V)": 480, "Secondary Current-A-phase(A)": 1002, "PF%": 93, "VTHD-A-B": 2.5, "Winding-Temp-A(°C)": 63},
        }
        insert_query = "INSERT OR REPLACE INTO TransformerSpecs (transformer_name, variable_name, rated_value) VALUES (?, ?, ?)"
        
        count = 0
        for transformer, specs in transformer_specs_data.items():
            for variable, value in specs.items():
                 # This logic ensures all phases/pairs get a spec if only one is defined
                if "(V)" in variable or "(A)" in variable or "(°C)" in variable:
                    for phase in ["-A-", "-B-", "-C-"]:
                        self.cursor.execute(insert_query, (transformer, variable.replace("-A-", phase), value))
                        count += 1
                elif "VTHD" in variable:
                     for pair in ["-A-B", "-B-C", "-A-C"]:
                        self.cursor.execute(insert_query, (transformer, variable.replace("-A-B", pair), value))
                        count += 1
                else:
                    self.cursor.execute(insert_query, (transformer, variable, value))
                    count += 1
        self.conn.commit()
        print(f"Seeded TransformerSpecs table with {count} records.")

    def get_rated_specs(self, transformer_name):
        """Fetches the rated specifications for a given transformer."""
        query = "SELECT variable_name, rated_value FROM TransformerSpecs WHERE transformer_name = ?"
        specs_df = pd.read_sql_query(query, self.conn, params=(transformer_name,))
        if specs_df.empty:
            return None
        return dict(zip(specs_df["variable_name"], specs_df["rated_value"]))

    def get_latest_averages(self, transformer_name):
        """Fetches the latest row of pre-calculated averages from Subsystem 1's table."""
        averages_table_name = f"{transformer_name}_average_metrics_day"
        try:
            query = f'SELECT * FROM "{averages_table_name}" ORDER BY timestamp DESC LIMIT 1'
            latest_averages = self.cursor.execute(query).fetchone()
            return latest_averages
        except sqlite3.OperationalError:
            print(f"[Error] Could not find Subsystem 1 averages table: '{averages_table_name}'.")
            return None

    def save_health_results(self, transformer_name, results, overall_score, overall_color):
        """Saves the calculated health scores and statuses to the HealthScores table."""
        today_str = datetime.now().strftime("%Y-%m-%d")
        insert_query = "INSERT INTO HealthScores (transformer_name, date, variable_name, average_value, rated_value, status, overall_score, overall_color) VALUES (?, ?, ?, ?, ?, ?, ?, ?)"
        
        for var, vals in results.items():
            self.cursor.execute(insert_query, (
                transformer_name, today_str, var, vals["Average"], vals["Rated"], vals["Status"], overall_score, overall_color
            ))
        
        self.conn.commit()
        print(f"'{transformer_name}' -> Health results saved successfully.")

    def save_forecast_results(self, transformer_name, forecast_df):
        """
        Clears old forecast data and saves the new forecast results to the ForecastData table.
        """
        # Clear any previous forecasts for this transformer
        self.cursor.execute("DELETE FROM ForecastData WHERE transformer_name = ?", (transformer_name,))
        
        # Save the new forecast data
        forecast_df.to_sql('ForecastData', self.conn, if_exists='append', index=False)
        
        self.conn.commit()
        print(f"'{transformer_name}' -> Forecast results saved successfully.")

    def get_transformer_lifetime_data(self, transformer_name):
        """Fetches lifetime data from the table created by Subsystem 1."""
        lifetime_table = f"{transformer_name}_lifetime_continuous_loading"
        try:
            # Assuming 'total_phase_lifetime' is the column to use for forecasting
            query = f'SELECT timestamp as DATETIME, total_phase_lifetime as Lifetime_Percentage FROM "{lifetime_table}"'
            df = pd.read_sql_query(query, self.conn)
            df["DATETIME"] = pd.to_datetime(df["DATETIME"], errors="coerce")
            return df
        except Exception as e:
            print(f"[Error] Could not find or read lifetime data from '{lifetime_table}': {e}")
            return pd.DataFrame()

    def get_latest_health_score(self, transformer_name):
        """Gets the most recent overall_score for a transformer from the HealthScores table."""
        query = "SELECT overall_score FROM HealthScores WHERE transformer_name = ? ORDER BY date DESC LIMIT 1"
        result = self.cursor.execute(query, (transformer_name,)).fetchone()
        return result[0] if result else 0.5 # Default score if none found

    def get_transformer_names(self):
        """
        Finds transformer names by looking for the average metrics tables created by Subsystem 1.
        """
        self.cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name LIKE '%_average_metrics_day'")
        tables = self.cursor.fetchall()
        # Extracts 'LTR_A01_Data' from 'LTR_A01_Data_average_metrics_day'
        return [table['name'].replace('_average_metrics_day', '') for table in tables]
    