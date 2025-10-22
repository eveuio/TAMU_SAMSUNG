import os
import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime

# --- Configuration Constants ---
DB_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', 'my_database.db'))

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
    
    def test_connection(self):
        """
        Test database connection and return detailed status information.
        Returns a dictionary with connection status and diagnostic information.
        """
        try:
            # Test basic connection
            self.cursor.execute("SELECT 1")
            basic_connection = True
            connection_error = None
        except Exception as e:
            basic_connection = False
            connection_error = str(e)
        
        # Test database file existence and permissions
        file_exists = os.path.exists(self.db_path)
        file_readable = os.access(self.db_path, os.R_OK) if file_exists else False
        file_writable = os.access(self.db_path, os.W_OK) if file_exists else False
        
        # Test table existence
        try:
            self.cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
            tables = [row[0] for row in self.cursor.fetchall()]
            table_count = len(tables)
        except Exception as e:
            tables = []
            table_count = 0
            table_error = str(e)
        
        # Test Subsystem 1 data availability
        subsystem1_tables = [table for table in tables if table.startswith('LTR_') and '_test' not in table]
        transformer_names = subsystem1_tables
        
        # Test Subsystem 2 tables
        subsystem2_tables = ['TransformerSpecs', 'HealthScores', 'ForecastData']
        missing_subsystem2_tables = [table for table in subsystem2_tables if table not in tables]
        
        # Test data availability
        data_available = False
        if transformer_names:
            try:
                # Test if we can read from a Subsystem 1 table (direct table name)
                test_table = transformer_names[0]
                self.cursor.execute(f'SELECT COUNT(*) FROM "{test_table}"')
                row_count = self.cursor.fetchone()[0]
                data_available = row_count > 0
            except Exception:
                data_available = False
        
        # Compile status information
        status = {
            'connection_status': 'SUCCESS' if basic_connection else 'FAILED',
            'connection_error': connection_error,
            'database_path': self.db_path,
            'file_exists': file_exists,
            'file_readable': file_readable,
            'file_writable': file_writable,
            'total_tables': table_count,
            'subsystem1_tables': len(subsystem1_tables),
            'subsystem2_tables': len([t for t in subsystem2_tables if t in tables]),
            'missing_subsystem2_tables': missing_subsystem2_tables,
            'transformer_names': transformer_names,
            'data_available': data_available,
            'overall_status': 'HEALTHY' if (basic_connection and file_exists and data_available) else 'NEEDS_ATTENTION'
        }
        
        return status
    
    def print_connection_status(self):
        """
        Print a detailed connection status report to the console.
        """
        print("\n" + "="*60)
        print("DATABASE CONNECTION TEST")
        print("="*60)
        
        status = self.test_connection()
        
        # Basic connection status
        if status['connection_status'] == 'SUCCESS':
            print("✅ Database Connection: SUCCESS")
        else:
            print("❌ Database Connection: FAILED")
            print(f"   Error: {status['connection_error']}")
            return False
        
        # File status
        print(f"\n📁 Database File: {status['database_path']}")
        if status['file_exists']:
            print("✅ File exists")
        else:
            print("❌ File does not exist")
            return False
        
        if status['file_readable']:
            print("✅ File is readable")
        else:
            print("❌ File is not readable")
        
        if status['file_writable']:
            print("✅ File is writable")
        else:
            print("❌ File is not writable")
        
        # Table status
        print(f"\n📊 Database Tables: {status['total_tables']} total")
        print(f"   Subsystem 1 tables: {status['subsystem1_tables']}")
        print(f"   Subsystem 2 tables: {status['subsystem2_tables']}")
        
        if status['missing_subsystem2_tables']:
            print(f"   Missing Subsystem 2 tables: {', '.join(status['missing_subsystem2_tables'])}")
        
        # Transformer data status
        if status['transformer_names']:
            print(f"\n🔌 Available Transformers: {len(status['transformer_names'])}")
            for name in status['transformer_names']:
                print(f"   - {name}")
        else:
            print("\n⚠️  No transformer data found from Subsystem 1")
        
        # Data availability
        if status['data_available']:
            print("✅ Data is available for processing")
        else:
            print("❌ No data available for processing")
        
        # Overall status
        print(f"\n🎯 Overall Status: {status['overall_status']}")
        
        if status['overall_status'] == 'HEALTHY':
            print("✅ Database is ready for health monitoring")
            return True
        else:
            print("⚠️  Database needs attention before running health monitoring")
            return False

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
        """Fetches the latest averaged data from Subsystem 1's tables."""
        try:
            # First try production mode: look for averaged table
            averaged_table = f"{transformer_name}_average_metrics_day"
            try:
                query = f'SELECT * FROM "{averaged_table}" ORDER BY timestamp DESC LIMIT 1'
                avg_data = self.cursor.execute(query).fetchone()
                
                if avg_data:
                    # Production mode: return pre-calculated averages
                    return dict(avg_data)
            except sqlite3.OperationalError:
                # Averaged table doesn't exist, continue to development mode
                pass
            
            # Development mode: calculate from raw data
            query = f'SELECT * FROM "{transformer_name}" ORDER BY DATETIME DESC LIMIT 1'
            raw_data = self.cursor.execute(query).fetchone()
            
            if not raw_data:
                print(f"[Error] No data found in table: '{transformer_name}'.")
                return None
            
            # Convert to dict for easier access
            raw_dict = dict(raw_data)
            
            # Helper function to safely convert to float
            def safe_float(value):
                try:
                    return float(value) if value is not None else 0.0
                except (ValueError, TypeError):
                    return 0.0
            
            # Calculate averages from raw data (development mode)
            avg_data = {
                'timestamp': raw_dict['DATETIME'],
                'avg_secondary_voltage_a_phase': safe_float(raw_dict.get('Secondary Voltage-A-phase (V)')),
                'avg_secondary_voltage_b_phase': safe_float(raw_dict.get('Secondary Voltage-B-phase (V)')),
                'avg_secondary_voltage_c_phase': safe_float(raw_dict.get('Secondary Voltage-C-phase (V)')),
                'avg_secondary_voltage_total_phase': (
                    safe_float(raw_dict.get('Secondary Voltage-A-phase (V)')) +
                    safe_float(raw_dict.get('Secondary Voltage-B-phase (V)')) +
                    safe_float(raw_dict.get('Secondary Voltage-C-phase (V)'))
                ) / 3,
                'avg_secondary_current_a_phase': safe_float(raw_dict.get('Secondary Current-A-phase(A)')),
                'avg_secondary_current_b_phase': safe_float(raw_dict.get('Secondary Current-B-phase(A)')),
                'avg_secondary_current_c_phase': safe_float(raw_dict.get('Secondary Current-C-phase(A)')),
                'avg_secondary_current_total_phase': (
                    safe_float(raw_dict.get('Secondary Current-A-phase(A)')) +
                    safe_float(raw_dict.get('Secondary Current-B-phase(A)')) +
                    safe_float(raw_dict.get('Secondary Current-C-phase(A)'))
                ) / 3,
                'avg_vTHD_a_phase': safe_float(raw_dict.get('VTHD-A-B')),
                'avg_vTHD_b_phase': safe_float(raw_dict.get('VTHD-B-C')),
                'avg_vTHD_c_phase': safe_float(raw_dict.get('VTHD-A-C')),
                'avg_vTHD_total_phase': (
                    safe_float(raw_dict.get('VTHD-A-B')) +
                    safe_float(raw_dict.get('VTHD-B-C')) +
                    safe_float(raw_dict.get('VTHD-A-C'))
                ) / 3,
                'avg_power_factor': safe_float(raw_dict.get('PF%')),
                'avg_winding_temp_a_phase': safe_float(raw_dict.get('Winding-Temp-A(°C)')),
                'avg_winding_temp_b_phase': safe_float(raw_dict.get('Winding-Temp-B(°C)')),
                'avg_winding_temp_c_phase': safe_float(raw_dict.get('Winding-Temp-C(°C)')),
                'avg_winding_temp_total_phase': (
                    safe_float(raw_dict.get('Winding-Temp-A(°C)')) +
                    safe_float(raw_dict.get('Winding-Temp-B(°C)')) +
                    safe_float(raw_dict.get('Winding-Temp-C(°C)'))
                ) / 3,
                'lifetime_percentage': safe_float(raw_dict.get('Lifetime_Percentage'))
            }
            return avg_data
            
        except Exception as e:
            print(f"[Error] Unexpected error processing data for '{transformer_name}': {e}")
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
        """Fetches lifetime data from the main transformer table."""
        try:
            # First try to get from separate lifetime table (production mode)
            lifetime_table = f"{transformer_name}_lifetime_continuous_loading"
            try:
                query = f'SELECT timestamp as DATETIME, total_phase_lifetime as Lifetime_Percentage FROM "{lifetime_table}"'
                df = pd.read_sql_query(query, self.conn)
                if not df.empty:
                    df["DATETIME"] = pd.to_datetime(df["DATETIME"], errors="coerce")
                    return df
            except:
                pass  # Fall through to main table
            
            # Fallback: get from main transformer table (development mode)
            query = f'SELECT DATETIME, Lifetime_Percentage FROM "{transformer_name}" WHERE Lifetime_Percentage IS NOT NULL'
            df = pd.read_sql_query(query, self.conn)
            df["DATETIME"] = pd.to_datetime(df["DATETIME"], errors="coerce")
            return df
        except Exception as e:
            print(f"[Error] Could not find or read lifetime data from '{transformer_name}': {e}")
            return pd.DataFrame()

    def get_latest_health_score(self, transformer_name):
        """Gets the most recent overall_score for a transformer from the HealthScores table."""
        query = "SELECT overall_score FROM HealthScores WHERE transformer_name = ? ORDER BY date DESC LIMIT 1"
        result = self.cursor.execute(query, (transformer_name,)).fetchone()
        return result[0] if result else 0.5 # Default score if none found

    def get_transformer_names(self):
        """
        Finds transformer names by looking for data tables created by Subsystem 1.
        Handles mixed scenarios where some transformers have averaged tables and others don't.
        """
        # Get all LTR transformer tables (both raw and averaged)
        self.cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name LIKE 'LTR_%' AND name NOT LIKE '%_test%'")
        all_tables = self.cursor.fetchall()
        all_table_names = [table['name'] for table in all_tables]
        
        # Separate averaged and raw tables
        averaged_tables = [name for name in all_table_names if name.endswith('_average_metrics_day')]
        raw_tables = [name for name in all_table_names if not name.endswith('_average_metrics_day') and not name.endswith('_lifetime_continuous_loading')]
        
        # Get transformer names from averaged tables
        averaged_transformers = [name.replace('_average_metrics_day', '') for name in averaged_tables]
        
        # Get transformer names from raw tables
        raw_transformers = raw_tables
        
        # Combine and deduplicate transformer names
        all_transformers = list(set(averaged_transformers + raw_transformers))
        
        if averaged_tables and raw_tables:
            print(f"🔄 Mixed mode: {len(averaged_tables)} averaged tables, {len(raw_tables)} raw tables")
        elif averaged_tables:
            print(f"📊 Production mode: Using pre-calculated averaged data from Subsystem 1")
        else:
            print(f"🔧 Development mode: Using raw data and calculating averages on-the-fly")
        
        return all_transformers
    