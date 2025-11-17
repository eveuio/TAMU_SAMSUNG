# -*- coding: utf-8 -*-

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
        self.conn.row_factory = sqlite3.Row  # MUST BE BEFORE CURSOR
        self.cursor = self.conn.cursor()
        print(f"Database Manager (Subsystem 2) initialized. Connected to {self.db_path}")
    
    def close(self):
        self.conn.close()
        print("Database connection closed.")
    
    def initialize_schema(self):
        """Creates the HealthScores and ForecastData tables required by Subsystem 2 if they don't exist."""
        # Drop and recreate the HealthScores table to ensure proper column types
        self.cursor.execute("DROP TABLE IF EXISTS HealthScores")
        self.cursor.execute("""
            CREATE TABLE HealthScores (
                transformer_name TEXT,
                date TEXT,
                variable_name TEXT,
                average_value TEXT,
                rated_value TEXT,
                status TEXT,
                overall_score TEXT,
                overall_color TEXT
            )
        """)
        
        # Create ForecastData table if it doesn't exist
        self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS ForecastData (
                transformer_name TEXT,
                forecast_date TEXT,
                predicted_lifetime REAL,
                PRIMARY KEY (transformer_name, forecast_date)
            )
        """)
        
        self.conn.commit()
        print("Initialized HealthScores table with TEXT columns for numeric values.")
        print("ForecastData table is ready.")
    
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
            query = "SELECT * FROM transformers WHERE transformer_name = ?"
            specs_df = pd.read_sql_query(query, self.conn, params=(transformer_name,))
            
            if specs_df.empty:
                return None
            
            # Create rated specs dictionary with hexadecimal to decimal conversion
            rated_specs = {}
            if not specs_df.empty:
                
                # Helper function to convert hex to decimal if needed
                def convert_value(value):
                    if value is None:
                        return 0.0
                    
                    # Handle bytes objects (binary data from database)
                    if isinstance(value, bytes):
                        try:
                            import struct
                            # Try to interpret as little-endian double (8 bytes) - SQLite REAL format
                            if len(value) == 8:
                                return float(struct.unpack('<d', value)[0])
                            # Try to interpret as little-endian 32-bit integer
                            elif len(value) == 4:
                                return float(struct.unpack('<i', value)[0])
                            # If that fails, try to decode as string
                            else:
                                value = value.decode('utf-8', errors='ignore')
                        except Exception:
                            return 0.0
                    
                    if isinstance(value, str):
                        # Handle various hex formats
                        if value.startswith('0x') or value.startswith('0X'):
                            try:
                                return float(int(value, 16))
                            except ValueError:
                                pass
                        # Try regular float conversion first (most common case)
                        try:
                            return float(value)
                        except ValueError:
                            pass
                        # Handle hex without 0x prefix (only as last resort)
                        try:
                            if all(c in '0123456789ABCDEFabcdef' for c in value) and len(value) > 2:
                                return float(int(value, 16))
                        except ValueError:
                            pass
                        return 0.0
                    
                    # For numeric types (int, float), convert directly
                    try:
                        return float(value)
                    except (ValueError, TypeError):
                        return 0.0
                
                rated_specs["Secondary Voltage-A-phase (V)"] = convert_value(specs_df.iloc[0]['rated_voltage_LV'])
                rated_specs["Secondary Voltage-B-phase (V)"] = convert_value(specs_df.iloc[0]['rated_voltage_LV'])
                rated_specs["Secondary Voltage-C-phase (V)"] = convert_value(specs_df.iloc[0]['rated_voltage_LV'])
                rated_specs["Secondary Current-A-phase(A)"] = convert_value(specs_df.iloc[0]['rated_current_LV'])
                rated_specs["Secondary Current-B-phase(A)"] = convert_value(specs_df.iloc[0]['rated_current_LV'])
                rated_specs["Secondary Current-C-phase(A)"] = convert_value(specs_df.iloc[0]['rated_current_LV'])
                rated_specs["Winding-Temp-A(°C)"] = convert_value(specs_df.iloc[0]['rated_avg_winding_temp_rise'])
                rated_specs["Winding-Temp-B(°C)"] = convert_value(specs_df.iloc[0]['rated_avg_winding_temp_rise'])
                rated_specs["Winding-Temp-C(°C)"] = convert_value(specs_df.iloc[0]['rated_avg_winding_temp_rise'])
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
            
            # Use pandas to read the data, which handles the conversion properly
            df = pd.read_sql_query(query, self.conn)
            
            if not df.empty:
                # Convert to dictionary, excluding the DATETIME index
                result_dict = df.iloc[0].to_dict()
                # Remove DATETIME from the result
                if 'DATETIME' in result_dict:
                    del result_dict['DATETIME']
                return result_dict
            return None
        except Exception as e:
            print(f"Error getting latest averages: {e}")
            import traceback
            print(f"Traceback: {traceback.format_exc()}")
            return None
    
    def get_manufacture_date(self, transformer_name):
        """Get manufacture_date from transformers table."""
        try:
            query = "SELECT manufacture_date FROM transformers WHERE transformer_name = ?"
            result = self.cursor.execute(query, (transformer_name,)).fetchone()
            if result and result[0]:
                return result[0]
        except Exception as e:
            print(f"Error getting manufacture_date for {transformer_name}: {e}")
        return None
    
    def get_transformer_lifetime_data(self, transformer_name):
        """Get lifetime data from lifetime_transient_loading table with remainingLifetime_percent."""
        try:
            # Get manufacture_date from transformers table
            manufacture_date = self.get_manufacture_date(transformer_name)
            
            # Get lifetime data from transient_loading table
            lifetime_table = f"{transformer_name}_lifetime_transient_loading"
            query = f'SELECT timestamp as DATETIME, remainingLifetime_percent as Lifetime_Percentage FROM "{lifetime_table}"'
            df = pd.read_sql_query(query, self.conn)
            
            if not df.empty:
                # Convert DATETIME to proper datetime format
                df["DATETIME"] = pd.to_datetime(df["DATETIME"], errors="coerce")
                
                # If manufacture_date is available and DATETIME is missing or invalid, use manufacture_date as reference
                if manufacture_date:
                    try:
                        manufacture_date_dt = pd.to_datetime(manufacture_date, errors="coerce")
                        if manufacture_date_dt and df["DATETIME"].isna().any():
                            # Fill missing DATETIME values with manufacture_date + offset based on row index
                            for idx, row in df.iterrows():
                                if pd.isna(row["DATETIME"]):
                                    # Use manufacture_date + days based on row index
                                    df.at[idx, "DATETIME"] = manufacture_date_dt + pd.Timedelta(days=idx)
                    except Exception as e:
                        print(f"Warning: Could not process manufacture_date {manufacture_date}: {e}")
                
                return df
        except Exception as e:
            print(f"Error getting lifetime data for {transformer_name}: {e}")
            import traceback
            print(f"Traceback: {traceback.format_exc()}")
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
            
            # Ensure rated_value is stored as a proper numeric value
            # Convert to float and ensure it's not stored as binary data
            rated_val = vals["Rated"]
            if rated_val is None:
                rated_value = 0.0
            else:
                # Force conversion to float to avoid binary representation
                try:
                    rated_value = float(rated_val)
                except (ValueError, TypeError):
                    print(f"Warning: Could not convert rated value '{rated_val}' to float for {var}")
                    rated_value = 0.0
            
            # Ensure average_value is also properly converted
            avg_val = vals["Average"]
            if avg_val is None:
                average_value = 0.0
            else:
                try:
                    average_value = float(avg_val)
                except (ValueError, TypeError):
                    print(f"Warning: Could not convert average value '{avg_val}' to float for {var}")
                    average_value = 0.0
            
            # Ensure values are stored as proper numeric types
            # Convert to string first to avoid binary storage issues
            rated_value_str = str(rated_value)
            average_value_str = str(average_value)
            overall_score_str = str(overall_score)
            
            
            self.cursor.execute(insert_query, (
                transformer_name, today_str, var, average_value_str, rated_value_str, vals["Status"], overall_score_str, overall_color
            ))
        
        self.conn.commit()
        print(f"'{transformer_name}' -> Health results saved successfully.")
    
    def save_forecast_results(self, transformer_name, forecast_df):
        """Save forecast results to ForecastData table."""
        try:
            # Check if DataFrame is empty
            if forecast_df.empty:
                print(f"Warning: Empty forecast DataFrame for {transformer_name}. Nothing to save.")
                return
            
            # Check required columns
            required_columns = ['transformer_name', 'forecast_date', 'predicted_lifetime']
            missing_columns = [col for col in required_columns if col not in forecast_df.columns]
            if missing_columns:
                print(f"Error: Missing required columns in forecast DataFrame: {missing_columns}")
                return
            
            # Delete existing forecast data for this transformer
            self.cursor.execute("DELETE FROM ForecastData WHERE transformer_name = ?", (transformer_name,))
            
            # Ensure transformer_name is in the DataFrame (it should already be there from create_forecast_dataframe)
            if 'transformer_name' not in forecast_df.columns:
                forecast_df['transformer_name'] = transformer_name
            
            # Print debug info
            print(f"Debug: Saving {len(forecast_df)} forecast records for {transformer_name}")
            print(f"Debug: DataFrame columns: {list(forecast_df.columns)}")
            print(f"Debug: First few rows:\n{forecast_df.head()}")
            
            # Save to database
            forecast_df.to_sql('ForecastData', self.conn, if_exists='append', index=False)
            self.conn.commit()
            
            # Verify the save
            self.cursor.execute("SELECT COUNT(*) FROM ForecastData WHERE transformer_name = ?", (transformer_name,))
            count = self.cursor.fetchone()[0]
            print(f"'{transformer_name}' -> Forecast results saved successfully. {count} records in database.")
        except Exception as e:
            print(f"Error saving forecast results for {transformer_name}: {e}")
            import traceback
            print(f"Traceback: {traceback.format_exc()}")
            self.conn.rollback()
