
import numpy
from datetime import datetime
from pandas import DataFrame
import pandas
import time
import os
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import sqlite3
import sqlalchemy
import openpyxl
import json

import sqlalchemy
from sqlalchemy import text
from sqlalchemy import inspect
from sqlalchemy.orm import sessionmaker
from sqlalchemy import inspect
from fastapi import FastAPI, HTTPException
from sqlalchemy.ext.declarative import DeclarativeMeta

from sqlalchemy.engine import Engine
from .transformerFunctions import Transformer


class Database:
    def __init__(self, db_path, session_factory:sessionmaker, orm_transformers:DeclarativeMeta, engine:Engine):
        self.conn = sqlite3.connect(db_path)
        self.cursor = self.conn.cursor()
        self.db_path = db_path

        self.SessionLocal = session_factory
        self.orm_transformers = orm_transformers              # so fastAPI can access correctly
        self.engine = engine
        self.conn.row_factory = sqlite3.Row                   # Allows accessing columns by name

        #Create initial transformer master table
        self.cursor.execute('''
                    CREATE TABLE IF NOT EXISTS transformers (
                       id INTEGER PRIMARY KEY AUTOINCREMENT,
                       transformer_name TEXT UNIQUE,
                       kva NUMERIC,
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
                       manufacture_date TEXT,
                            
                       status TEXT)
                    ''')
        self.conn.commit()

        #Create initial forecast data master table
        self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS ForecastData (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                transformer_name TEXT,
                forecast_date TEXT,
                predicted_lifetime REAL
                
            )
            """)
        self.conn.commit()

        self.cursor.execute('''
                    CREATE TABLE IF NOT EXISTS HealthScores (
                       id INTEGER PRIMARY KEY AUTOINCREMENT,
                       transformer_name TEXT,
                       date TEXT,
                       variable_name TEXT,
                       average_value REAL,
                       rated_value REAL,
                       status TEXT,
                       overall_score REAL,
                       overall_color TEXT)
                    ''')
        self.conn.commit()



#?=======================-------CORE-DATABASE-FUNCTIONS---------===========================================================================================================--#
   
    #!Populate format for transformer rated values, creating empty storage structure for Transformer Data and filling in transformer rated values. Import all known data
    def addTransformer(self):
        # Step 1: Retrieve first 'new' transformer from ORM
        with self.SessionLocal() as db:
            new_transformer_row = db.query(self.orm_transformers).filter_by(status="new").first()
            if not new_transformer_row:
                raise HTTPException(status_code=404, detail="No new transformer found")

            transformer_name = new_transformer_row.transformer_name

        # Step 2: Create all associated tables using engine
        #TODO: change to reflect current structure for lifetime tables
        tables_to_create = { 
            
            f"{transformer_name}_lifetime_transient_loading": """ 
                DATETIME TEXT UNIQUE,
                LifetimeConsumption_day_percent NUMERIC,
                remainingLifetime_percent NUMERIC
                
            """,
            f"{transformer_name}_average_metrics_day": """
                DATETIME TEXT UNIQUE,
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
            """,
            f"{transformer_name}_average_metrics_hour": """
                DATETIME TEXT UNIQUE,
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
            """,
        }

        with self.engine.begin() as conn:
            for table_name, columns_sql in tables_to_create.items():
                conn.execute(text(f'CREATE TABLE IF NOT EXISTS "{table_name}" ({columns_sql})'))


        # Step 3: Update transformer status to active
        with self.SessionLocal() as db:
            db.query(self.orm_transformers)\
            .filter_by(transformer_name=transformer_name)\
            .update({"status": "active"})
            db.commit()

        # Step 4: Populate raw data and averages
        self.populateRawDataTable(transformer_name)
        self.createAverageReport(transformer_name)

        # Step 5: populate transient lifetime consumption
        self.write_lifetime_transient_df(transformer_name)

        return
        
    #! Remove transformer rated values from master table and all associated metrics tables   
    def removeTransformer(self, xfmr_name: str):
        with self.SessionLocal() as db:  
            if not xfmr_name:
                raise HTTPException(status_code=404, detail="Transformer not found")

            
            # related tables to drop 
            tables_to_drop = [
                f"{xfmr_name}_average_metrics_hour",
                f"{xfmr_name}_average_metrics_day",
                f"{xfmr_name}_lifetime_continuous_loading",
                f"{xfmr_name}_lifetime_transient_loading",
                f"{xfmr_name}fullRange"
                ]
            #Connection for table deletion
            with self.engine.begin() as conn:

                for table in tables_to_drop:
                    conn.execute(text(f'DROP TABLE IF EXISTS "{table}"'))

            
            with self.engine.connect() as conn:
                conn.execute(
                    text("DELETE FROM ForecastData WHERE transformer_name = :name"),
                    {"name": xfmr_name}
                )
                conn.execute(
                    text("DELETE FROM HealthScores WHERE transformer_name = :name"),
                    {"name": xfmr_name}
                )
                conn.commit()  # Explicitly commit

            return xfmr_name
        
    #! Populate Initial Average Tables per Transformer
    def createAverageReport(self,transformer_name, update=False):
       
        table_name = transformer_name + "fullRange"
        
        with self.engine.connect() as conn:
            
            transformerData = pandas.read_sql_query(
                    f'SELECT * FROM "{table_name}"',
                    con=conn
                )
           
        # Detect datetime column
        datetime_col = next((col for col in transformerData.columns if "date" in col.lower()), None)
        if datetime_col is None:
            raise ValueError(f"No datetime-like column found in {table_name}. Expected a column containing 'date'.")

        # Rename and clean
        transformerData = transformerData.rename(columns={datetime_col: 'DATETIME'})
        transformerData['DATETIME'] = transformerData['DATETIME'].astype(str).str.strip()

        # Parse using exact format
        transformerData['DATETIME'] = pandas.to_datetime(
            transformerData['DATETIME'],
            format='%Y-%m-%d %H:%M:%S',
            errors='coerce'
        )

        # Drop invalid rows
        transformerData = transformerData.dropna(subset=['DATETIME'])

        # Set index
        transformerData = transformerData.set_index('DATETIME', drop=False)

        if update:
            # print("TransformerData tail before timestamp filtering: ",transformerData.tail())
            
            with self.engine.connect() as conn:
                #TODO: fetch last timestamp from {transformer_name}_average_metrics_hour
                
                avg_table_name = transformer_name + "_average_metrics_hour"
                last_avg_ts_query = f'SELECT MAX(DATETIME) as max_ts FROM "{avg_table_name}"'
                last_avg_ts = pandas.read_sql_query(last_avg_ts_query, con=conn)['max_ts'].iloc[0]
                
                # print(f"Last timestamp in fullRange: {last_full_range_ts}")
                # print(f"Last timestamp in average_metrics_hour: {last_avg_ts}")
            
           
            #TODO: create dataframe 'newTransformerData'that contains the last hour mentioned in _average_metrics_hour to the last timestamp in {transformer_name}fullRange
            newTransformerData = transformerData[transformerData['DATETIME'] >= last_avg_ts]

            #TODO: set transformerData[] to be 'newTransformerData'
            transformerData = newTransformerData

            #TODO: need to initialize first timestamps for later data insertion
            firstHour_transformerData = transformerData["DATETIME"].min().hour
            firstDay_transformerData = transformerData["DATETIME"].min().day

            # print("TransformerData tail after timestamp filtering: ",transformerData.tail())


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

        # Optional columns (handle missing ones)
        vthd_cols = transformerData.columns[10:13] if len(transformerData.columns) > 12 else []
        pf_col = transformerData.columns[13] if len(transformerData.columns) >= 13 else None

        transformerData['HS_AVG'] = numpy.sqrt((transformerData[hsTempA]**2+transformerData[hsTempB]**2+transformerData[hsTempC]**2)/3)
        transformerData['I_RMS']= numpy.sqrt((transformerData[currentA]**2+transformerData[currentB]**2+transformerData[currentC]**2)/3)
        transformerData['V_RMS']= numpy.sqrt((transformerData[voltageA]**2+transformerData[voltageB]**2+transformerData[voltageC]**2)/3)
        
        if len(vthd_cols) == 3:
            transformerData['VTHD_RMS'] = numpy.sqrt((transformerData[vthd_cols[0]]**2 + transformerData[vthd_cols[1]]**2 + transformerData[vthd_cols[2]]**2) / 3)
        else:
            transformerData['VTHD_RMS'] = numpy.nan
        # transformerData['T_ambient'] = avgAmbientTemp(transformerData['I_RMS']/transformer.RatedCurrentLV)

        if pf_col is not None:
            transformerData[pf_col] = pandas.to_numeric(transformerData[pf_col], errors='coerce')
            # print("power factor column present:", pf_col)

        #TODO: Rename and shuffle columns to match desired order
        # Column rename mapping
        rename_map = {
            'DATETIME': 'DATETIME',
            transformerData.columns[4]: 'avg_secondary_voltage_a_phase',
            transformerData.columns[5]: 'avg_secondary_voltage_b_phase',
            transformerData.columns[6]: 'avg_secondary_voltage_c_phase',
            'V_RMS': 'avg_secondary_voltage_total_phase',
            transformerData.columns[7]: 'avg_secondary_current_a_phase',
            transformerData.columns[8]: 'avg_secondary_current_b_phase',
            transformerData.columns[9]: 'avg_secondary_current_c_phase',
            'I_RMS': 'avg_secondary_current_total_phase',
            transformerData.columns[1]: 'avg_winding_temp_a_phase',
            transformerData.columns[2]: 'avg_winding_temp_b_phase',
            transformerData.columns[3]: 'avg_winding_temp_c_phase',
            'HS_AVG': 'avg_winding_temp_total_phase'
        }

        # Add VTHD and PF columns only if they exist
        if len(vthd_cols) == 3:
            rename_map.update({
                vthd_cols[0]: 'avg_vTHD_a_phase',
                vthd_cols[1]: 'avg_vTHD_b_phase',
                vthd_cols[2]: 'avg_vTHD_c_phase',
                'VTHD_RMS': 'avg_vTHD_total_phase'
            })
        if pf_col:
            rename_map.update(
                {pf_col: 'avg_power_factor'}
            )



        # Rename existing columns
        existing_cols = [c for c in rename_map if c in transformerData.columns]
        transformerData.rename(columns={c: rename_map[c] for c in existing_cols}, inplace=True)

        # Now add missing “placeholder” columns:
        desired_order = [
            'DATETIME',
            'avg_secondary_voltage_a_phase',
            'avg_secondary_voltage_b_phase',
            'avg_secondary_voltage_c_phase',
            'avg_secondary_voltage_total_phase',
            'avg_secondary_current_a_phase',
            'avg_secondary_current_b_phase',
            'avg_secondary_current_c_phase',
            'avg_secondary_current_total_phase',
            'avg_vTHD_a_phase',
            'avg_vTHD_b_phase',
            'avg_vTHD_c_phase',
            'avg_vTHD_total_phase',
            'avg_power_factor',
            'avg_winding_temp_a_phase',
            'avg_winding_temp_b_phase',
            'avg_winding_temp_c_phase',
            'avg_winding_temp_total_phase'
        ]

        # Fill missing ones with NaN
        for col in desired_order:
            if col not in transformerData.columns:
                transformerData[col] = numpy.nan

        # Reorder neatly
        transformerData = transformerData[desired_order]
    
        hourly_avg = transformerData.resample('h').mean(numeric_only=True)
        daily_avg = transformerData.resample('d').mean(numeric_only=True)

        hourly_avg = hourly_avg.reset_index()
        daily_avg = daily_avg.reset_index()

        
        # After resampling and resetting index
        hourly_avg['DATETIME'] = hourly_avg['DATETIME'].dt.strftime('%Y-%m-%d %H:%M:%S')
        daily_avg['DATETIME']  = daily_avg['DATETIME'].dt.strftime('%Y-%m-%d %H:%M:%S')

        if update:
            #TODO: need to identify the timestamp seen in both the average and daily frames if that timestamp hour/day appears in the fullRange set (ie original set had an incomplete day/hour)
            
            # Fetch first timestamp from existing hourly and daily tables
            with self.engine.connect() as conn:
                # Hourly
                result_hour = conn.execute(
                    text(f'SELECT MAX(DATETIME) FROM "{transformer_name}_average_metrics_hour"')
                ).scalar()
                last_hour_avg_ts = pandas.to_datetime(result_hour) if result_hour else None

                # Daily
                result_day = conn.execute(
                    text(f'SELECT MAX(DATETIME) FROM "{transformer_name}_average_metrics_day"')
                ).scalar()
                last_day_avg_ts = pandas.to_datetime(result_day) if result_day else None

                # print("last timestamp in hourly table",last_hour_avg_ts)
                # print("last timestamp in daily table",last_day_avg_ts)

            #TODO: if first timestamp hour of transformerData and first timestamp hour of average metrics hour are the same, delete the last row in average_metrics_hour
            
            if last_hour_avg_ts.hour == firstHour_transformerData:
                    with self.engine.begin() as conn:
                        conn.execute(
                            text(f'DELETE FROM "{transformer_name}_average_metrics_hour" WHERE DATETIME = :ts'),
                            {"ts": last_hour_avg_ts.strftime("%Y-%m-%d %H:%M:%S")}
                        )

            #TODO: if first timestamp day of transformerData and first timestamp day of average metrics day are the same, delete the last row in average_metrics_day
            
            if  last_day_avg_ts.day == firstDay_transformerData:
                    with self.engine.begin() as conn:
                        conn.execute(
                            text(f'DELETE FROM "{transformer_name}_average_metrics_day" WHERE DATETIME = :ts'),
                            {"ts": last_day_avg_ts.strftime("%Y-%m-%d %H:%M:%S")}
                        )

            #TODO: Append only the new hourly/daily averages
            hourly_avg.to_sql(
                name=f"{transformer_name}_average_metrics_hour",
                con=self.engine,
                if_exists="append",   # append instead of replace
                chunksize=5000,
                method="multi",
                index=False
            )
            daily_avg.to_sql(
                name=f"{transformer_name}_average_metrics_day",
                con=self.engine,
                if_exists="append",   # append instead of replace
                chunksize=5000,
                method="multi",
                index=False
            )
        else:
            # Replace the tables with full data
            hourly_avg.to_sql(
                name=f"{transformer_name}_average_metrics_hour",
                con=self.engine,
                if_exists="replace",
                chunksize=5000,
                method="multi",
                index=False
            )
            daily_avg.to_sql(
                name=f"{transformer_name}_average_metrics_day",
                con=self.engine,
                if_exists="replace",
                chunksize=5000,
                method="multi",
                index=False
            )

        return 
    
    def createDataSet(self,transformer_name):
        table_name = transformer_name + "fullRange"

        # Pull the entire table (or consider chunks for very large tables)
        query = f'''SELECT * FROM "{table_name}" ORDER BY "DATETIME" ASC'''
        transformerData = pandas.read_sql_query(query, self.conn)
        transformerData['DATETIME'] = pandas.to_datetime(transformerData['DATETIME'])

        # Identify relevant columns
        hsA, hsB, hsC = transformerData.columns[1:4]
        voltageA, voltageB, voltageC = transformerData.columns[4:7]
        currentA, currentB, currentC = transformerData.columns[7:10]

        # Calculate max hotspot, RMS current, RMS voltage, ambient temp
        transformerData['hotspot_temp_max'] = numpy.max(transformerData[[hsA, hsB, hsC]].values, axis=1)
        transformerData['V_RMS'] = numpy.sqrt((transformerData[voltageA]**2 + transformerData[voltageB]**2 + transformerData[voltageC]**2)/3)
        transformerData['I_RMS'] = numpy.sqrt((transformerData[currentA]**2 + transformerData[currentB]**2 + transformerData[currentC]**2)/3)
        transformerData['T_ambient'] = 26.67 + (43.3333-26.67)*(transformerData['I_RMS']/transformer.RatedCurrentLV)
        transformerData['phaseCurrentMax'] = numpy.max(transformerData[[currentA, currentB, currentB]].values, axis=1)
        transformerData['phaseVoltageMax'] = numpy.max(transformerData[[voltageA, voltageB, voltageC]].values, axis=1)
        
        num_lags = 20

        for lag in range(1, num_lags + 1):
            transformerData[f'phaseCurrentLag{lag}'] = transformerData['phaseCurrentMax'].shift(lag)
            transformerData[f'phaseVoltageLag{lag}'] = transformerData['phaseVoltageMax'].shift(lag)
            # transformerData[f'T_ambient_lag{lag}'] = transformerData['T_ambient'].shift(lag)
            transformerData[f'hotspot_lag{lag}'] = transformerData['hotspot_temp_max'].shift(lag)

        # Drop rows with NaN values introduced by lagging
        transformerData.dropna(inplace=True)

        # Set datetime index
        transformerData.set_index('DATETIME', inplace=True)

        # Rolling window size for 2 months
        # Assuming 10-min intervals: 2 months ≈ 60*24*6 = 8640 rows
        window_size_training = 8640
        window_size_testing = 2000 
        window_size_validation = 2000

        # Compute rolling std of hotspot temperature
        rolling_std_training = transformerData['hotspot_temp_max'].rolling(window=window_size_training).std()

        # Find the window with the highest std
        max_std_idx = rolling_std_training.idxmax()
        start_window = max_std_idx - pandas.Timedelta(minutes=window_size_training*10)
        end_window_training = max_std_idx
        training_window = transformerData.loc[start_window:end_window_training]

        print("Selected window for training:")
        print("Start:", start_window, "End:", end_window_training, "Std:", rolling_std_training.max())
        print('\n')

        #TODO: Add another table for validation data, identify next largest std window outside of training set and sample 2k points from that
        hotspot_series_copy = transformerData['hotspot_temp_max'].copy()

        # Mask training window rows (set them to NaN)
        validation_start_pos = transformerData.index.get_loc(training_window.index[0])
        validation_end_pos = transformerData.index.get_loc(training_window.index[-1])
        hotspot_series_copy.iloc[validation_start_pos:validation_end_pos+1] = numpy.nan

        # Compute rolling std on remaining data
        rolling_std_validation = hotspot_series_copy.rolling(window=window_size_validation).std()

        # mark start and end dates of this section
        max_std_idx_validation = rolling_std_validation.idxmax()
        start_window_validation = max_std_idx_validation - pandas.Timedelta(minutes = window_size_validation*10)
        end_window_validation = max_std_idx_validation
        validation_window = transformerData.loc[start_window_validation:end_window_validation]

        print("Selected window for validation:")
        print("Start:", start_window_validation, "End:", end_window_validation, "Std:", rolling_std_validation.max())
        print('\n')


        #TODO: Add another table for testing data, identify next largest std window outside of training and validation set and sample ~2k points from that
        # Create a copy of the series
        hotspot_series_copy = transformerData['hotspot_temp_max'].copy()

        # Mask training window rows (set them to NaN)
        train_start_pos = transformerData.index.get_loc(training_window.index[0])
        train_end_pos = transformerData.index.get_loc(training_window.index[-1])
        hotspot_series_copy.iloc[train_start_pos:train_end_pos+1] = numpy.nan

        #Mask Validation Window rows
        validation_start_pos = transformerData.index.get_loc(validation_window.index[0])
        validation_end_pos = transformerData.index.get_loc(validation_window.index[-1])
        hotspot_series_copy.iloc[validation_start_pos:validation_end_pos+1] = numpy.nan

        # Compute rolling std on remaining data
        rolling_std_testing = hotspot_series_copy.rolling(window=window_size_testing).std()

        # mark start and end dates of this section
        max_std_idx_testing = rolling_std_testing.idxmax()
        start_window_testing = max_std_idx_testing - pandas.Timedelta(minutes = window_size_testing*10)
        end_window_testing = max_std_idx_testing
        testing_window = transformerData.loc[start_window_testing:end_window_testing]

        print("Selected window for testing:")
        print("Start:", start_window_testing, "End:", end_window_testing, "Std:", rolling_std_testing.max())
        print('\n')

        # --- SAVE TO DATABASE ---
        training_window.to_sql(
            name=f"{transformer_name}_trainingData",
            con=self.conn,
            if_exists="replace",
            chunksize=5000,
            method="multi"
        )

        validation_window.to_sql(
            name=f"{transformer_name}_validationData",
            con=self.conn,
            if_exists="replace",
            chunksize=5000,
            method="multi"
        )

        testing_window.to_sql(
            name=f"{transformer_name}_testingData",
            con=self.conn,
            if_exists="replace",
            chunksize=5000,
            method="multi"
        )

    #! Calcuate and write dataFrame containing transient lifetime calculations:
    def write_lifetime_transient_df(self, transformer_name: str):
        #TODO: Fetch transformer metadata from master table
        query = "SELECT * FROM transformers WHERE transformer_name = ?"
        rated_specs = pandas.read_sql_query(query, con=self.engine, params=(transformer_name,))
        if rated_specs.empty:
            raise ValueError(f"No transformer found with name {transformer_name}")

        #TODO: Fetch average metrics
        avg_table = f"{transformer_name}_average_metrics_hour"
        avg_metrics = pandas.read_sql_table(avg_table, con=self.engine)

        #TODO: Create Transformer instance
        xfmr = Transformer(rated_specs=rated_specs)

        # TODO: Compute lifetime
        lifetime_df = xfmr.lifetime_TransientLoading(avg_metrics_hour=avg_metrics)

        #TODO: Write lifetime table
        lifetime_table = f"{transformer_name}_lifetime_transient_loading"
        
        lifetime_df.to_sql(
            lifetime_table,
            con=self.engine,
            if_exists="replace",
            index=False,
            chunksize=5000,
            method="multi"
        )
        return
        
    #------------------------eFCMS-interaction-or-data-table-population-with-database--------------------#
    
    #! Collect all availible and relevant data stored for a specific transformer. Update a json file to store the last datetime of the datafile, and the time modified. 
    def populateRawDataTable(self, transformer_name, update = False):
        """
        Populate the raw data table from Excel file and update the timestamp cache.
        """
        excel_dir = os.path.join(
            os.path.abspath(os.path.join(os.path.dirname(__file__), '..')),
            'CompleteTransformerData'
        )
        file_path = os.path.join(excel_dir, f'{transformer_name}.xlsx')
        
        previousData = pandas.read_excel(file_path)
        
        # TODO: need to only populate raw data table with non-zero values for the first 9 columns. Winding temp, current, voltage are the most important and table must start with a complete non-zero valued row)
        numeric_part = previousData.iloc[:, 0:10]  # first 10 columns (datetime plus winding temp (x3), current (x3) and voltage (x3))
    
        numeric_part = numeric_part.fillna(0) # Replace NaN with 0 just to be safe
        
        mask = (numeric_part != 0).all(axis=1) # Boolean mask: True if all first 9 columns are non-zero
        
        # Get the index of the first row where all 10 columns are non-zero, ensures a complete inital dataset to start with
        if mask.any():
            start_index = mask.idxmax()  # gives first True index
            previousData = previousData.loc[start_index:].reset_index(drop=True)
        else:
            return
    
        
        if (update == False):
            # Write to SQL
             # Format datetime
            previousData["DATETIME"] = pandas.to_datetime(previousData["DATETIME"]).dt.strftime('%Y-%m-%d %H:%M:%S')
           
            previousData.to_sql(
                name=f'{transformer_name}fullRange',
                con=self.engine,
                if_exists="replace",
                index=False
            )
        else:
            #TODO: Collect last_timestamp from file_timestamps.json under {transformer_name}.xlsx
            cache_file = os.path.join(excel_dir, 'file_timestamps.json')
            with open(cache_file, 'r') as f:
                timestamp_cache = json.load(f)

            last_timestamp_str = timestamp_cache.get(f'{transformer_name}.xlsx', {}).get("last_timestamp")

            #TODO: not including last timestamp, create dataframe containing last timestamp + 1 to max timestamp seen in DataProcessing/CompleteTransformerData/{transformer_name}.xlsx
            
            last_timestamp = pandas.to_datetime(last_timestamp_str)
            
            previousData["DATETIME"] = pandas.to_datetime(previousData["DATETIME"])

            new_data = previousData[previousData["DATETIME"] > last_timestamp]

            if new_data.empty:
                return
            

            #TODO: write data frame to sql in append mode, since we are adding new data to the end of the file
             # Format datetime
            new_data["DATETIME"] = pandas.to_datetime(previousData["DATETIME"]).dt.strftime('%Y-%m-%d %H:%M:%S')

            with self.engine.begin() as conn:
                new_data.to_sql(
                    name=f'{transformer_name}fullRange',
                    con=conn,
                    if_exists="append",
                    index=False
                )

            #TODO: change/update timestamp cache
            
            timestamp_cache[f'{transformer_name}.xlsx'] = {
                "last_timestamp": last_timestamp_str,
                "last_uploaded": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }

            with open(cache_file, 'w') as f:
                json.dump(timestamp_cache, f, indent=2)

        return

    # #! Update all data tables, recalculate averages, recalculate lifetime consumption
    # def updateTransformerDataTables(self, transformer_name):
    #     #TODO: 
   
    
   






#?================----functions-needed-by-health-monitoring--------=======================================================================================================---#

    def close(self):
        self.conn.close()
        print("Database connection closed.")

    def test_connection(self, session=None):
        """
        Test database connection and return detailed status information.
        Returns a dictionary with connection status and diagnostic information.
        Accepts optional SQLAlchemy session for thread-safe usage.
        """
        import os

        basic_connection = False
        connection_error = None
        table_names = []

        try:
            # Use provided session or create a temporary one
            if session is None:
                with self.SessionLocal() as session_local:
                    result = session_local.execute(text("SELECT name FROM sqlite_master WHERE type='table'")).fetchall()
                    table_names = [r[0] for r in result]
                    basic_connection = True
            else:
                result = session.execute(text("SELECT name FROM sqlite_master WHERE type='table'")).fetchall()
                table_names = [r[0] for r in result]
                basic_connection = True
        except Exception as e:
            connection_error = str(e)
            table_names = []

        # Database file checks
        file_exists = os.path.exists(self.db_path)
        file_readable = os.access(self.db_path, os.R_OK) if file_exists else False
        file_writable = os.access(self.db_path, os.W_OK) if file_exists else False

        # Step 1: Fetch transformer names from transformers table
        transformer_names = []
        try:
            if session is None:
                with self.SessionLocal() as s:
                    result = s.execute("SELECT transformer_name FROM transformers").fetchall()
            else:
                result = session.execute("SELECT transformer_name FROM transformers").fetchall()
            transformer_names = [r[0] for r in result]
        except Exception:
            transformer_names = []

        # Step 2: Filter subsystem1 tables (averaged metrics)
        subsystem1_tables = [
            table for table in table_names
            if any(name in table for name in transformer_names) and '_test' not in table
        ]

        # Subsystem2 tables
        subsystem2_tables = ['HealthScores', 'ForecastData']
        found_subsystem2_tables = [t for t in subsystem2_tables if t in table_names]

        # Test data availability
        data_available = False
        if subsystem1_tables:
            test_table = subsystem1_tables[0]
            try:
                if session is None:
                    with self.SessionLocal() as s:
                        row_count = s.execute(f'SELECT COUNT(*) FROM "{test_table}"').scalar()
                else:
                    row_count = session.execute(f'SELECT COUNT(*) FROM "{test_table}"').scalar()
                data_available = row_count > 0
            except Exception:
                data_available = False

        # Compile status info
        status = {
            'connection_status': 'SUCCESS' if basic_connection else 'FAILED',
            'connection_error': connection_error,
            'database_path': self.db_path,
            'file_exists': file_exists,
            'file_readable': file_readable,
            'file_writable': file_writable,
            'total_tables': len(table_names),
            'subsystem1_tables': len(subsystem1_tables),
            'subsystem2_tables': len(found_subsystem2_tables),
            'missing_subsystem2_tables': [t for t in subsystem2_tables if t not in table_names],
            'transformer_names': subsystem1_tables,
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
            print(" Database Connection: SUCCESS")
        else:
            print(" Database Connection: FAILED")
            print(f"   Error: {status['connection_error']}")
            return False
        
        # File status
        print(f"\n Database File: {status['database_path']}")
        if status['file_exists']:
            print(" File exists")
        else:
            print(" File does not exist")
            return False
        
        if status['file_readable']:
            print(" File is readable")
        else:
            print(" File is not readable")
        
        if status['file_writable']:
            print(" File is writable")
        else:
            print(" File is not writable")
        
        # Table status
        print(f"\n Database Tables: {status['total_tables']} total")
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
            print("\n  No transformer data found from Subsystem 1")
        
        # Data availability
        if status['data_available']:
            print("Data is available for processing")
        else:
            print("No data available for processing")
        
        # Overall status
        print(f"\nOverall Status: {status['overall_status']}")
        
        if status['overall_status'] == 'HEALTHY':
            print("Database is ready for health monitoring")
            return True
        else:
            print("Database needs attention before running health monitoring")
            return False
        
    def save_health_results(self, transformer_name, results, overall_score, overall_color, session=None):
        """
        Saves the calculated health scores and statuses to the HealthScores table.
        Accepts optional SQLAlchemy session for thread safety.
        """
        from sqlalchemy import text
        from datetime import datetime

        today_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        try:
            if session is not None:
                bind = session.bind
            else:
                # Create a temporary thread-safe session
                with self.SessionLocal() as local_session:
                    bind = local_session.bind

            insert_query = text("""
                INSERT OR REPLACE INTO HealthScores 
                (transformer_name, date, variable_name, average_value, rated_value, status, overall_score, overall_color)
                VALUES (:transformer_name, :date, :variable_name, :average_value, :rated_value, :status, :overall_score, :overall_color)
            """)

            delete_query = text("""
                DELETE FROM HealthScores
                WHERE transformer_name = :transformer_name
                AND variable_name = :variable_name
            """)

            with bind.begin() as conn:
                for var, vals in results.items():
                    conn.execute(delete_query, {
                        "transformer_name": transformer_name,
                        "variable_name": var
                    })

                    conn.execute(insert_query, {
                        "transformer_name": transformer_name,
                        "date": today_str,
                        "variable_name": var,
                        "average_value": float(vals["Average"]),  # cast to float
                        "rated_value": float(vals["Rated"]),      # cast to float
                        "status": vals["Status"],                 # keep as TEXT
                        "overall_score": float(overall_score),   # cast to float
                        "overall_color": overall_color
                    })

            print(f"'{transformer_name}' -> Health results saved successfully.")

        except Exception as e:
            print(f"[Error] Could not save health results for '{transformer_name}': {e}")
            import traceback
            print(traceback.format_exc())


    def save_forecast_results(self, transformer_name, forecast_df, session=None):
        """
        Clears old forecast data and saves the new forecast results to the ForecastData table.
        Accepts optional SQLAlchemy session for thread safety.
        """
        try:
            if session is not None:
                bind = session.bind
            else:
                # Use thread-safe temporary session
                with self.SessionLocal() as local_session:
                    bind = local_session.bind

            # Clear old forecasts
            from sqlalchemy import text
            with bind.begin() as conn:
                conn.execute(
                    text("DELETE FROM ForecastData WHERE transformer_name = :name"),
                    {"name": transformer_name}
                )

            # Add transformer_name to the forecast_df
            forecast_df['transformer_name'] = transformer_name

            # Save the new forecast data
            forecast_df.to_sql('ForecastData', bind, if_exists='append', index=False)

            print(f"'{transformer_name}' -> Forecast results saved successfully.")

        except Exception as e:
            print(f"[Error] Could not save forecast results for '{transformer_name}': {e}")
            import traceback
        print(traceback.format_exc())

    def get_transformer_names(self, session=None):
        """
        Finds transformer names by looking for data tables created by Subsystem 1.
        Gets transformer names from the 'transformers' table, then looks for matching averaged metrics tables.
        """
        import pandas as pd

        try:
            if session is not None:
                bind = session.bind
            else:
                # Use a thread-safe temporary session
                with self.SessionLocal() as local_session:
                    bind = local_session.bind

            # Step 1: Fetch transformer names from the transformers table
            df = pd.read_sql_query("SELECT transformer_name FROM transformers", bind)
            transformer_names = df['transformer_name'].tolist()
            
            if not transformer_names:
                print("No transformers found in the 'transformers' table")
                return []

            # Step 2: Get all tables from the database
            all_tables_df = pd.read_sql_query(
                "SELECT name FROM sqlite_master WHERE type='table'", bind
            )
            all_tables = all_tables_df['name'].tolist()

            # Step 3: Find tables that match transformer averaged metrics
            averaged_tables = [
                name for name in transformer_names
                if f"{name}_average_metrics_day" in all_tables
            ]

            if averaged_tables:
                print(f"Production mode: Found {len(averaged_tables)} transformers with averaged data")
            else:
                print(f"Warning: No averaged tables found for any transformers")

            return averaged_tables

        except Exception as e:
            print(f"[Error] Could not fetch transformer names: {e}")
            import traceback
            print(f"[Error] Traceback: {traceback.format_exc()}")
            return []

    def get_transformer_lifetime_data(self, transformer_name, session=None):
        """Fetches lifetime data from the main transformer table."""
        import pandas as pd

        try:
            if session is not None:
                bind = session.bind
            else:
                # Use a new thread-safe session
                with self.SessionLocal() as local_session:
                    bind = local_session.bind

            # Try separate lifetime table (production mode)
            lifetime_table = f"{transformer_name}_lifetime_transient_loading"
            try:
                query = f'SELECT timestamp as DATETIME, LifetimePercent_remaining as Lifetime_Percentage FROM "{lifetime_table}"'
                df = pd.read_sql_query(query, bind)
                if not df.empty:
                    df["DATETIME"] = pd.to_datetime(df["DATETIME"], errors="coerce")
                    return df
            except Exception:
                pass  # Fall through to main table

            # Fallback: main transformer table (development mode)
            query = f'SELECT DATETIME, Lifetime_Percentage FROM "{transformer_name}" WHERE Lifetime_Percentage IS NOT NULL'
            df = pd.read_sql_query(query, bind)
            df["DATETIME"] = pd.to_datetime(df["DATETIME"], errors="coerce")
            return df

        except Exception as e:
            print(f"[Error] Could not find or read lifetime data from '{transformer_name}': {e}")
            import traceback
            print(f"[Error] Traceback: {traceback.format_exc()}")
            return pd.DataFrame()

    def get_rated_specs(self, transformer_name, session=None):
        """Fetches the rated specifications for a given transformer."""
        import pandas as pd
        
        query = "SELECT transformer_name, rated_voltage_LV, rated_current_LV, rated_avg_winding_temp_rise FROM transformers WHERE transformer_name = ?"
        
        if session is not None:
            # Use SQLAlchemy session and pandas read_sql_query
            specs_df = pd.read_sql_query(query, session.bind, params=(transformer_name,))
        else:
            # Fallback: create a new session to ensure thread safety
            with self.SessionLocal() as local_session:
                specs_df = pd.read_sql_query(query, local_session.bind, params=(transformer_name,))
        
        if specs_df.empty:
            return None
        
        # Create rated specs dictionary mapping variable names to rated values
        rated_specs = {
            "Secondary Voltage-A-phase (V)": float(specs_df.iloc[0]['rated_voltage_LV']),
            "Secondary Voltage-B-phase (V)": float(specs_df.iloc[0]['rated_voltage_LV']),
            "Secondary Voltage-C-phase (V)": float(specs_df.iloc[0]['rated_voltage_LV']),
            "Secondary Current-A-phase(A)": float(specs_df.iloc[0]['rated_current_LV']),
            "Secondary Current-B-phase(A)": float(specs_df.iloc[0]['rated_current_LV']),
            "Secondary Current-C-phase(A)": float(specs_df.iloc[0]['rated_current_LV']),
            "Winding-Temp-A(°C)": float(specs_df.iloc[0]['rated_avg_winding_temp_rise']),
            "Winding-Temp-B(°C)": float(specs_df.iloc[0]['rated_avg_winding_temp_rise']),
            "Winding-Temp-C(°C)": float(specs_df.iloc[0]['rated_avg_winding_temp_rise']),
            "PF%": 93.0,
            "VTHD-A-B": 2.5,
            "VTHD-B-C": 2.5,
            "VTHD-A-C": 2.5
        }
        
        return rated_specs
    
    def get_latest_health_score(self, transformer_name, session=None):
        """Gets the most recent overall_score for a transformer from the HealthScores table."""
        import pandas as pd

        query = "SELECT overall_score FROM HealthScores WHERE transformer_name = ? ORDER BY date DESC LIMIT 1"

        if session is not None:
            # Use the provided SQLAlchemy session
            df = pd.read_sql_query(query, session.bind, params=(transformer_name,))
        else:
            # Create a new session to ensure thread safety
            with self.SessionLocal() as local_session:
                df = pd.read_sql_query(query, local_session.bind, params=(transformer_name,))

        if not df.empty:
            return df.iloc[0]['overall_score']
        else:
            return 0.5  # Default score if none found
        
    def initialize_schema(self):
        pass
    
    def seed_transformer_specs(self):
        """This method is not needed as specs are already in the transformers table."""
        # The rated specs are already in the transformers table
        # This is kept for compatibility with transformer_health_monitor.py
        pass

    def get_latest_averages(self, transformer_name, session=None):
        """Fetches the latest averaged data from Subsystem 1's tables."""
        import pandas as pd

        try:
            averaged_table = f"{transformer_name}_average_metrics_day"
            query = f'SELECT * FROM "{averaged_table}" ORDER BY DATETIME DESC LIMIT 1'

            if session is not None:
                # Use provided SQLAlchemy session
                df = pd.read_sql_query(query, session.bind)
            else:
                # Use a new thread-safe session
                with self.SessionLocal() as local_session:
                    df = pd.read_sql_query(query, local_session.bind)

            if not df.empty:
                result_dict = df.iloc[0].to_dict()
                result_dict.pop('DATETIME', None)  # Remove DATETIME if present
                return result_dict
            else:
                print(f"[Error] No averaged data found in table: '{averaged_table}'.")
                return None

        except Exception as e:
            print(f"[Error] Unexpected error processing data for '{transformer_name}': {e}")
            import traceback
            print(f"[Error] Traceback: {traceback.format_exc()}")
            return None
