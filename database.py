
import numpy
from parameterCalculations import avgAmbientTemp
from datetime import datetime
from pandas import DataFrame
import pandas
import time
import os
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import sqlite3
# from hotSpotPrediction import createDataSets
from transformerFunctions import Transformer



class Database:
    def __init__(self, dbpath):
        self.conn = sqlite3.connect(dbpath)
        self.cursor = self.conn.cursor()
        self.dbpath = dbpath
        self.conn.row_factory = sqlite3.Row # Allows accessing columns by name

#?-------------------------CORE-DATABASE-FUNCTIONS-------------------------------#
   
    #!Populate format for transformer rated values, creating empty storage structure for Transformer Data and filling in transformer rated values. Import all known data
    def addTransformer(self):
        #TODO: Retrive all "new" status transformers from Database and create objects for them
        
        #TODO: Push all new transformer objects to transformer object list manager
        
        #TODO: populate raw data table for transformer:
        self.populateRawDataTable(transformer:Transformer)
        
        #TODO: Rated Values for Transformer
        # Create empty table to fill with rated values
        self.cursor.execute('''
                    CREATE TABLE IF NOT EXISTS transformers (
                       id INTEGER PRIMARY KEY AUTOINCREMENT,
                       transformer_name TEXT UNIQUE,
                       rated_voltage_HV NUMERIC,
                       rated_current_HV NUMERIC,
                       rated_voltage_LV NUMERIC,
                       rated_current_LV NUMERIC,
                       rated_thermal_class NUMERIC,
                       rated_avg_winding_temp_rise NUMERIC,
                       winding_material TEXT,
                       weight_CoreAndCoil_kg NUMERIC,
                       weight_Total_kg NUMERIC,
                       rated_impedance NUMERIC,
                       status TEXT UNIQUE)
                    ''')
        self.conn.commit()
        
        # Input Rated values from nameplate to recently created table
        self.cursor.execute('''
                    INSERT OR IGNORE INTO transformers (
                       transformer_name,
                       rated_voltage_HV,
                       rated_current_HV,
                       rated_voltage_LV,
                       rated_current_LV,
                       rated_thermal_class,
                       rated_avg_winding_temp_rise,
                       winding_material,
                       weight_CoreAndCoil_kg,
                       weight_Total_kg,
                       rated_impedance)
                       VALUES (?,?,?,?,?,?,?,?,?,?,?)
                       ''',
                       (transformer.name,
                        transformer.HV,
                        transformer.ratedCurrentHV,
                        transformer.LV,
                        transformer.RatedCurrentLV,
                        transformer.thermalClass_rated,
                        transformer.avgWindingTempRise_rated,
                        transformer.windingMaterial,
                        transformer.weight_CoreAndCoil,
                        transformer.weightTotal,
                        transformer.impedance
                        )
                       )
        self.conn.commit()
        
        #TODO: Lifetime Table (Continuous)
        #Set up table format
        self.cursor.execute(f'''
                        CREATE TABLE IF NOT EXISTS "{transformer.name}_lifetime_continuous_loading" (
                        timestamp TEXT UNIQUE,
                        a_phase_load_current NUMERIC,
                        b_phase_load_current NUMERIC,
                        c_phase_load_current NUMERIC,
                        total_phase_load_current NUMERIC,
                        a_phase_winding_temp NUMERIC,
                        b_phase_winding_temp NUMERIC,
                        c_phase_winding_temp NUMERIC,
                        total_phase_winding_temp NUMERIC,
                        a_phase_thermoD_hot_spot NUMERIC,
                        b_phase_thermoD_hot_spot NUMERIC,
                        c_phase_thermoD_hot_spot NUMERIC,
                        total_phase_thermoD_hot_spot NUMERIC,
                        a_phase_lifetime NUMERIC,
                        b_phase_lifetime NUMERIC,
                        c_phase_lifetime NUMERIC,
                        total_phase_lifetime NUMERIC
                        )
                        ''')

        #TODO: Lifetime Table (Transient)
        #Set up table format
        self.cursor.execute(f'''
                            CREATE TABLE IF NOT EXISTS "{transformer.name}_lifetime_transient_loading" (
                            timestamp TEXT UNIQUE,
                            a_phase_load_current NUMERIC,
                            b_phase_load_current NUMERIC,
                            c_phase_load_current NUMERIC,
                            total_phase_load_current NUMERIC,
                            a_phase_winding_temp NUMERIC,
                            b_phase_winding_temp NUMERIC,
                            c_phase_winding_temp NUMERIC,
                            total_phase_winding_temp NUMERIC,
                            a_phase_thermoD_hot_spot NUMERIC,
                            b_phase_thermoD_hot_spot NUMERIC,
                            c_phase_thermoD_hot_spot NUMERIC,
                            total_phase_thermoD_hot_spot NUMERIC,
                            a_phase_lifetime_consumption NUMERIC,
                            b_phase_lifetime_consumption NUMERIC,
                            c_phase_lifetime_consumption NUMERIC,
                            total_phase_lifetime_consumption NUMERIC
                            )
                            ''')
        

        #TODO: Full Range of Average Values (Day)
        #Set up table format
        self.cursor.execute(f'''
                            CREATE TABLE IF NOT EXISTS "{transformer.name}_average_metrics_day" (
                            timestamp TEXT UNIQUE,
                            avg_secondary_voltage_a_phase NUMERIC,
                            avg_secondary_voltage_b_phase NUMERIC,
                            avg_secondary_voltage_c_phase NUMERIC,
                            avg_secondary_voltage_total_phase NUMERIC,
                            avg_secondary_current_a_phase NUMERIC,
                            avg_secondary_current_b_phase NUMERIC,
                            avg_secondary_current_c_phase NUMERIC,
                            avg_secondary_current_total_phase NUMERIC,
                            avg_vTHD_a_phase NUMERIC,
                            avg_vTHD_b_phase NUMERIC,
                            avg_vTHD_c_phase NUMERIC,
                            avg_vTHD_total_phase NUMERIC,
                            avg_power_factor NUMERIC,
                            avg_winding_temp_a_phase NUMERIC,
                            avg_winding_temp_b_phase NUMERIC,
                            avg_winding_temp_c_phase NUMERIC,
                            avg_winding_temp_total_phase NUMERIC
                            )
                            ''')

        #TODO: Full Range of Average Values (Hour)
        #Set up table format
        self.cursor.execute(f'''
                        CREATE TABLE IF NOT EXISTS "{transformer.name}_average_metrics_hour" (
                        timestamp TEXT UNIQUE,
                        avg_secondary_voltage_a_phase NUMERIC,
                        avg_secondary_voltage_b_phase NUMERIC,
                        avg_secondary_voltage_c_phase NUMERIC,
                        avg_secondary_voltage_total_phase NUMERIC,
                        avg_secondary_current_a_phase NUMERIC,
                        avg_secondary_current_b_phase NUMERIC,
                        avg_secondary_current_c_phase NUMERIC,
                        avg_secondary_current_total_phase NUMERIC,
                        avg_vTHD_a_phase NUMERIC,
                        avg_vTHD_b_phase NUMERIC,
                        avg_vTHD_c_phase NUMERIC,
                        avg_vTHD_total_phase NUMERIC,
                        avg_power_factor NUMERIC,
                        avg_winding_temp_a_phase NUMERIC,
                        avg_winding_temp_b_phase NUMERIC,
                        avg_winding_temp_c_phase NUMERIC,
                        avg_winding_temp_total_phase NUMERIC
                        )
                        ''')
        #TODO: Health Prediction Metrics Table:
        self.cursor.execute(f'''
                        CREATE TABLE IF NOT EXISTS "{transformer.name}_HealthScores" (
                        transformer_name TEXT,
                        date TEXT,
                        variable_name TEXT,
                        average_value REAL,
                        rated_value REAL,
                        status TEXT,
                        overall_score REAL,
                        overall_color TEXT
                        )
                    ''')
        self.conn.commit()

        self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS ForecastData (
                transformer_name TEXT,
                forecast_date TEXT,
                predicted_lifetime REAL,
                PRIMARY KEY (transformer_name, forecast_date)
            )
            """)
        
        self.conn.commit()
    
    #! Remove transformer rated values from master table and all associated metrics tables   
    def removeTransformer(self, transformer):
        #TODO: Remove instance from main transformers table
        self.cursor.execute("DELETE FROM transformers WHERE transformer_name = ?", (transformer.name,))

        #TODO: Remove charts with name of transformer
        self.cursor.execute(f'''DROP TABLE IF EXISTS '{transformer.name}_average_metrics_hour'
                            ''')
        self.cursor.execute(f'''DROP TABLE IF EXISTS '{transformer.name}_average_metrics_day'
                            ''')
        self.cursor.execute(f'''DROP TABLE IF EXISTS '{transformer.name}_lifetime_continuous_loading'
                            ''')
        self.cursor.execute(f'''DROP TABLE IF EXISTS '{transformer.name}_lifetime_transient_loading'
                            ''')
        self.cursor.execute(f'''DROP TABLE IF EXISTS '{transformer.name}fullRange'
                            ''')
        #TODO: Add ML tables:-------------------------------------------------------------------
        self.cursor.execute(f'''DROP TABLE IF EXISTS '{transformer.name}_HealthScores'
                            ''')
        self.cursor.execute(f'''DROP TABLE IF EXISTS '{transformer.name}_trainingData'
                            ''')
        self.cursor.execute(f'''DROP TABLE IF EXISTS '{transformer.name}_testingData'
                            ''')
        #TODO:------------------------------------------------------------------------------------
        self.conn.commit()

    #! Populate Initial Average Tables per Transformer
    def createAverageReport(self,transformer):
        table_name= transformer.name+"fullRange"
        transformerData = pandas.read_sql_query(f'''SELECT * FROM "{table_name}"''',self.db.conn)
        transformerData['DATETIME'] = pandas.to_datetime(transformerData['DATETIME'])
        transformerData.index = transformerData['DATETIME']
        # fullDateRange = transformerData['DATETIME'].iloc[1:-1].tolist()

        #TODO: Precalculate RMS current, RMS voltage and ambient temp for all Timestamps, add to transformerData Dataframe:
        hsTempA = transformerData.columns[1] 
        hsTempB = transformerData.columns[2]
        hsTempC = transformerData.columns[3]

        voltageA = transformerData.columns[4]
        voltageB = transformerData.columns[5]
        voltageC = transformerData.columns[6]

        currentA = transformerData.columns[7]
        currentB = transformerData.columns[8]
        currentC = transformerData.columns[9]

        transformerData['HS_AVG'] = numpy.sqrt((transformerData[hsTempA]**2+transformerData[hsTempB]**2+transformerData[hsTempC]**2)/3)
        transformerData['I_RMS']= numpy.sqrt((transformerData[currentA]**2+transformerData[currentB]**2+transformerData[currentC]**2)/3)
        transformerData['V_RMS']= numpy.sqrt((transformerData[voltageA]**2+transformerData[voltageB]**2+transformerData[voltageC]**2)/3)
        transformerData['T_ambient'] = avgAmbientTemp(transformerData['I_RMS']/self.RatedCurrentLV)

        #TODO: Rename and shuffle columns to match desired order

        #Current Column names:
        name_dict = {
            transformerData.columns[0]: 'DATETIME',
            transformerData.columns[1]: f'''LTR_"{transformer.name}"_TEMP_WL_AVG''',
            transformerData.columns[2]: f'''LTR_"{transformer.name}"_TEMP_WC_AVG''',
            transformerData.columns[3]: f'''LTR_"{transformer.name}"_TEMP_WR_AVG''',
            transformerData.columns[4]: 'avg_secondary_voltage_total_phase',
            transformerData.columns[5]: 'avg_secondary_current_a_phase',
            transformerData.columns[6]: 'avg_secondary_current_b_phase',
            transformerData.columns[7]: 'avg_secondary_current_c_phase',
            transformerData.columns[8]: 'avg_secondary_current_total_phase',
            transformerData.columns[9]: 'avg_vTHD_a_phase',
            transformerData.columns[10]: 'avg_vTHD_b_phase',
            transformerData.columns[11]: 'avg_vTHD_c_phase',
            transformerData.columns[12]: 'avg_vTHD_total_phase',
            transformerData.columns[13]: 'avg_power_factor'
        }

        #Desired column names
        rename_dict = {
            transformerData.columns[0]: 'datetime',
            transformerData.columns[1]: 'avg_secondary_voltage_a_phase',
            transformerData.columns[2]: 'avg_secondary_voltage_b_phase',
            transformerData.columns[3]: 'avg_secondary_voltage_c_phase',
            transformerData.columns[4]: 'avg_secondary_voltage_total_phase',
            transformerData.columns[5]: 'avg_secondary_current_a_phase',
            transformerData.columns[6]: 'avg_secondary_current_b_phase',
            transformerData.columns[7]: 'avg_secondary_current_c_phase',
            transformerData.columns[8]: 'avg_secondary_current_total_phase',
            transformerData.columns[9]: 'avg_vTHD_a_phase',
            transformerData.columns[10]: 'avg_vTHD_b_phase',
            transformerData.columns[11]: 'avg_vTHD_c_phase',
            transformerData.columns[12]: 'avg_vTHD_total_phase',
            transformerData.columns[13]: 'avg_power_factor',
            transformerData.columns[14]: 'avg_winding_temp_a_phase',
            transformerData.columns[15]: 'avg_winding_temp_b_phase',
            transformerData.columns[16]: 'avg_winding_temp_c_phase',
            transformerData.columns[17]: 'avg_winding_temp_total_phase',
        }

        hourly_avg = transformerData.resample('H').mean(numeric_only=True)
        daily_avg = transformerData.resample('D').mean(numeric_only=True)

        hourly_avg.to_sql(name= f'''{self.name}_average_metrics_hour''',con=self.db.conn,if_exists = "replace",chunksize=5000,method ="multi", index=True, index_label = "DATETIME")
        daily_avg.to_sql(name= f'''{self.name}_average_metrics_day''',con=self.db.conn,if_exists = "replace",chunksize=5000,method ="multi",index=True, index_label = "DATETIME")
        
        return 
    #------------------------eFCMS-interaction-with-database--------------------#
    #! Collect all availble and relevant data stored for a specific transformer 
    def populateRawDataTable(self,transformer):
        #TODO: Load Previous Data for transformer into database and create table. Local import for now, will be eFCMS specific later
        previousData = pandas.read_excel("/home/eveuio/DataProcessing/IncompleteTransformerData_historical/22A03_noLastWeek.xlsx")
        previousData.to_sql(name=transformer.name+"fullRange",con=self.conn,if_exists="replace", index=False)
        self.conn.commit()
        return

    #!Collect relevant data points and append timestamp + data to appropritate day/hour data table
    def update_transformer_average_data(self):
        #TODO: Real-time calculations needed after inital inserts, need to check for most recent timestamp, do averaging and insert into averaging tables, Needs to update every averging table at once, use master transformer list
        #TODO: need unique database connection for threading
        conn = sqlite3.connect(self.dbpath, check_same_thread=True)  

        #TODO: Get transformer names and rated current LV from master table
        master_table= "transformers"
        transformerNamesAndCurrents = pandas.read_sql_query(f'SELECT transformer_name, rated_current_LV FROM "{master_table}"', conn)

        transformer_names = transformerNamesAndCurrents['transformer_name'].tolist()
        transformer_currentLV_list = transformerNamesAndCurrents['rated_current_LV'].tolist()

        #TODO: update all transformer averages for all listed in master database. need to process name, rated_lv in parallel; use zip() to accomplish this
        time.sleep(2)
        while True:
            for name, rated_lv in zip(transformer_names, transformer_currentLV_list):
                raw_table = f"{name}fullRange"
                hourly_table = f"{name}_hourlyTest"
                daily_table = f"{name}_dailyTest"

                #TODO: Get last processed timestamps from hourly table
                last_timestamp_hour_df = pandas.read_sql_query(f'SELECT MAX("DATETIME") AS last_ts FROM "{hourly_table}"', conn)
                last_timestamp_list = pandas.to_datetime(last_timestamp_hour_df["last_ts"].iloc[0]) if not last_timestamp_hour_df.empty else None

                last_daily_df = pandas.read_sql_query(f'SELECT MAX("DATETIME") as last_ts FROM "{daily_table}"', conn)
                last_daily_ts = pandas.to_datetime(last_daily_df['last_ts'].iloc[0]) if last_daily_df['last_ts'].iloc[0] else None
                
                #TODO: Retrieve all data since last processed timestamp from raw table
                if last_timestamp_list is None:
                    query = f'SELECT * FROM "{raw_table}"'
                    continue
                else:
                    query = f'SELECT * FROM "{raw_table}" WHERE DATETIME > "{last_timestamp_list}"'

                transformerData = pandas.read_sql_query(query, conn, parse_dates=["DATETIME"])
                transformerData.index = transformerData['DATETIME']

                # #TODO: Precalculate RMS current, RMS voltage and ambient temp for all Timestamps, add to transformerData Dataframe:
                hsTempA = transformerData.columns[1]
                hsTempB = transformerData.columns[2]
                hsTempC = transformerData.columns[3]

                voltageA = transformerData.columns[4]
                voltageB = transformerData.columns[5]
                voltageC = transformerData.columns[6]

                currentA = transformerData.columns[7]
                currentB = transformerData.columns[8]
                currentC = transformerData.columns[9]

                transformerData['HS_AVG'] = (transformerData[hsTempA]+transformerData[hsTempB]+transformerData[hsTempC])/3
                transformerData['I_RMS']= numpy.sqrt((transformerData[currentA]**2+transformerData[currentB]**2+transformerData[currentC]**2)/3)
                transformerData['V_RMS']= numpy.sqrt((transformerData[voltageA]**2+transformerData[voltageB]**2+transformerData[voltageC]**2)/3)
                transformerData['T_ambient'] = avgAmbientTemp((transformerData['I_RMS']/rated_lv))

                hourly_avg = transformerData.resample('H').mean(numeric_only=True)
                daily_avg = transformerData.resample('D').mean(numeric_only=True)
                
                #TODO: need to ensure that no partial day averages are written:
                daily_avg=pandas.DataFrame(daily_avg[daily_avg.index > last_daily_ts])

                #TODO: append dataframe to end of existing hour/daily tables if not empty:
                if not hourly_avg.empty:
                    hourly_avg.index.name = 'DATETIME'
                    hourly_avg.to_sql(hourly_table,con=conn,if_exists="append",index=True,chunksize=5000,method="multi")
                    
                    #TODO: fix print statement to make hour_timestamp the datetime of the row inserted
                    print(f"[{time.strftime('%H:%M:%S')}] Inserted hourly rows up to datetime: {hourly_avg.index[-1]}")


                if not daily_avg.empty:
                    daily_avg.index.name = 'DATETIME'
                    daily_avg.to_sql(daily_table,con=conn,if_exists="append",index=True,chunksize=5000,method="multi")

                    #TODO: fix print statement to make hour_timestamp the datetime of the row inserted
                    print(f"[{time.strftime('%H:%M:%S')}] Inserted daily rows up to datetime: {daily_avg.index[-1]}")
            
            #TODO: Wait until raw table had enough data to complete an averaging section
            time.sleep(12)
        return 

#?==============================----functions-needed-by-health-monitoring--------==========================---#

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
 
    #TODO: Fix to represent database structure
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

    def get_transformer_lifetime_data(self, transformer_name):

        """Fetches lifetime data from the main transformer table."""
        try:
            # First try to get from separate lifetime table (production mode)
            lifetime_table = f"{transformer_name}_lifetime_continuous_loading"
            try:
                query = f'SELECT timestamp as DATETIME, total_phase_lifetime as Lifetime_Percentage FROM "{lifetime_table}"'
                df = pandas.read_sql_query(query, self.conn)
                if not df.empty:
                    df["DATETIME"] = pandas.to_datetime(df["DATETIME"], errors="coerce")
                    return df
            except:
                pass  # Fall through to main table
            
            # Fallback: get from main transformer table (development mode)
            query = f'SELECT DATETIME, Lifetime_Percentage FROM "{transformer_name}" WHERE Lifetime_Percentage IS NOT NULL'
            df = pandas.read_sql_query(query, self.conn)
            df["DATETIME"] = pandas.to_datetime(df["DATETIME"], errors="coerce")
            return df
        
        except Exception as e:
            print(f"[Error] Could not find or read lifetime data from '{transformer_name}': {e}")
            return pandas.DataFrame()

    def get_rated_specs(self, transformer_name):

        """Fetches the rated specifications for a given transformer."""
        query = "SELECT variable_name, rated_value FROM TransformerSpecs WHERE transformer_name = ?"
        specs_df = pandas.read_sql_query(query, self.conn, params=(transformer_name,))
        if specs_df.empty:
            return None
        return dict(zip(specs_df["variable_name"], specs_df["rated_value"]))

    def get_latest_health_score(self, transformer_name):
        """Gets the most recent overall_score for a transformer from the HealthScores table."""
        query = "SELECT overall_score FROM HealthScores WHERE transformer_name = ? ORDER BY date DESC LIMIT 1"
        result = self.cursor.execute(query, (transformer_name,)).fetchone()
        return result[0] if result else 0.5 # Default score if none found
    
    def get_latest_averages(self, transformer_name):
        """Fetches the latest averaged data from Subsystem 1's tables."""
        try:
            # Look for averaged table from Subsystem 1
            averaged_table = f"{transformer_name}_average_metrics_day"
            query = f'SELECT * FROM "{averaged_table}" ORDER BY timestamp DESC LIMIT 1'
            avg_data = self.cursor.execute(query).fetchone()
            
            if avg_data:
                # Return pre-calculated averages from Subsystem 1
                return dict(avg_data)
            else:
                print(f"[Error] No averaged data found in table: '{averaged_table}'.")
                return None
            
        except sqlite3.OperationalError as e:
            print(f"[Error] Averaged table '{averaged_table}' does not exist: {e}")
            return None
        except Exception as e:
            print(f"[Error] Unexpected error processing data for '{transformer_name}': {e}")
            return None
