# -*- coding: utf-8 -*-
"""
Database wrapper for Subsystem 2
This allows us to use the shared database.py without importing transformerFunctions
"""
import sqlite3
import pandas as pd
from datetime import datetime
import os

class Database:
    """
    Subsystem 2 database interface - uses the shared transformerDB.db
    """
    
    def __init__(self, db_path):
        self.db_path = db_path
        self.conn = sqlite3.connect(self.db_path)
        self.cursor = self.conn.cursor()
        self.conn.row_factory = sqlite3.Row
        print(f"Database Manager (Subsystem 2) initialized. Connected to {self.db_path}")
    
    def close(self):
        self.conn.close()
        print("Database connection closed.")
    
    def initialize_schema(self):
        """Creates the HealthScores table required by Subsystem 2 if it doesn't exist."""
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
        self.conn.commit()
        print("Initialized HealthScores table.")
    
    def seed_transformer_specs(self):
        """Placeholder - specs are in transformers table."""
        pass
    
    def test_connection(self):
        """Test database connection."""
        try:
            self.cursor.execute("SELECT 1")
            return True
        except Exception:
            return False
    
    def print_connection_status(self):
        """Print connection status."""
        status = self.test_connection()
        print("Database Connection: SUCCESS" if status else "Database Connection: FAILED")
        
        file_exists = os.path.exists(self.db_path)
        print(f"Database File: {self.db_path}")
        print(f"File exists: {file_exists}")
        
        # Check tables
        self.cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = [row[0] for row in self.cursor.fetchall()]
        print(f"Database Tables: {len(tables)} total")
        
        return status
    
    def get_transformer_names(self):
        """Get transformer names from transformers table."""
        try:
            self.cursor.execute("SELECT transformer_name FROM transformers")
            transformer_names = [row[0] for row in self.cursor.fetchall()]
            
            if transformer_names:
                # Check for averaged tables
                self.cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
                all_tables = [row[0] for row in self.cursor.fetchall()]
                
                averaged_tables = []
                for name in transformer_names:
                    avg_table = f"{name}_average_metrics_day"
                    if avg_table in all_tables:
                        averaged_tables.append(name)
                
                if averaged_tables:
                    print(f"Production mode: Found {len(averaged_tables)} transformers with averaged data")
                return averaged_tables
            else:
                print("No transformers found in the 'transformers' table")
                return []
        except Exception as e:
            print(f"Error getting transformer names: {e}")
            return []
    
    def get_rated_specs(self, transformer_name):
        """Get rated specs from transformers table."""
        try:
            query = "SELECT rated_voltage_LV, rated_current_LV, rated_avg_winding_temp_rise FROM transformers WHERE transformer_name = ?"
            specs_df = pd.read_sql_query(query, self.conn, params=(transformer_name,))
            
            if specs_df.empty:
                return None
            
            # Create rated specs dictionary
            rated_specs = {}
            if not specs_df.empty:
                rated_specs["Secondary Voltage-A-phase (V)"] = specs_df.iloc[0]['rated_voltage_LV']
                rated_specs["Secondary Voltage-B-phase (V)"] = specs_df.iloc[0]['rated_voltage_LV']
                rated_specs["Secondary Voltage-C-phase (V)"] = specs_df.iloc[0]['rated_voltage_LV']
                rated_specs["Secondary Current-A-phase(A)"] = specs_df.iloc[0]['rated_current_LV']
                rated_specs["Secondary Current-B-phase(A)"] = specs_df.iloc[0]['rated_current_LV']
                rated_specs["Secondary Current-C-phase(A)"] = specs_df.iloc[0]['rated_current_LV']
                rated_specs["Winding-Temp-A(°C)"] = specs_df.iloc[0]['rated_avg_winding_temp_rise']
                rated_specs["Winding-Temp-B(°C)"] = specs_df.iloc[0]['rated_avg_winding_temp_rise']
                rated_specs["Winding-Temp-C(°C)"] = specs_df.iloc[0]['rated_avg_winding_temp_rise']
                rated_specs["PF%"] = 93.0
                rated_specs["VTHD-A-B"] = 2.5
                rated_specs["VTHD-B-C"] = 2.5
                rated_specs["VTHD-A-C"] = 2.5
            
            return rated_specs
        except Exception as e:
            print(f"Error getting rated specs: {e}")
            return None
    
    def get_latest_averages(self, transformer_name):
        """Get latest averaged data."""
        try:
            averaged_table = f"{transformer_name}_average_metrics_day"
            query = f'SELECT * FROM "{averaged_table}" ORDER BY DATETIME DESC LIMIT 1'
            avg_data = self.cursor.execute(query).fetchone()
            
            if avg_data:
                return dict(avg_data)
            return None
        except Exception as e:
            print(f"Error getting latest averages: {e}")
            return None
    
    def get_transformer_lifetime_data(self, transformer_name):
        """Get lifetime data."""
        try:
            lifetime_table = f"{transformer_name}_lifetime_continuous_loading"
            query = f'SELECT timestamp as DATETIME, total_phase_lifetime as Lifetime_Percentage FROM "{lifetime_table}"'
            df = pd.read_sql_query(query, self.conn)
            if not df.empty:
                df["DATETIME"] = pd.to_datetime(df["DATETIME"], errors="coerce")
                return df
        except:
            pass
        return pd.DataFrame()
    
    def get_latest_health_score(self, transformer_name):
        """Get latest health score."""
        try:
            query = "SELECT overall_score FROM HealthScores WHERE transformer_name = ? ORDER BY date DESC LIMIT 1"
            result = self.cursor.execute(query, (transformer_name,)).fetchone()
            return result[0] if result else 0.5
        except:
            return 0.5
    
    def save_health_results(self, transformer_name, results, overall_score, overall_color):
        """Save health results."""
        today_str = datetime.now().strftime("%Y-%m-%d")
        insert_query = "INSERT INTO HealthScores (transformer_name, date, variable_name, average_value, rated_value, status, overall_score, overall_color) VALUES (?, ?, ?, ?, ?, ?, ?, ?)"
        
        for var, vals in results.items():
            self.cursor.execute(insert_query, (
                transformer_name, today_str, var, vals["Average"], vals["Rated"], vals["Status"], overall_score, overall_color
            ))
        
        self.conn.commit()
        print(f"'{transformer_name}' -> Health results saved successfully.")
    
    def save_forecast_results(self, transformer_name, forecast_df):
        """Save forecast results."""
        self.cursor.execute("DELETE FROM ForecastData WHERE transformer_name = ?", (transformer_name,))
        forecast_df.to_sql('ForecastData', self.conn, if_exists='append', index=False)
        self.conn.commit()
        print(f"'{transformer_name}' -> Forecast results saved successfully.")

