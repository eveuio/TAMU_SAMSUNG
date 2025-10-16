
import numpy
from parameterCalculations import avgAmbientTemp
from datetime import datetime
from pandas import DataFrame
import pandas
import time
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import sqlite3
# from hotSpotPrediction import createDataSets
# from transformerFunctions import Transformer

class Database:
    def __init__(self, dbpath):
        self.dbconnect = sqlite3.connect(dbpath)
        self.cursor = self.dbconnect.cursor()
        self.dbpath = dbpath

    #-------------------------DATABASE-FUNCTIONS-------------------------------#
    #!Populate Table with given type of information, insert a single row of data. Use for both lifetime and averages, with the option of an iterator for the averages section/tables
    def insertData(self, transformerName, dataType, dataSet:DataFrame, iterator = ""):
        #TODO: dataframe->list in a dataframe object from transformer averaging. Single Row of Data
        dataSet_list = dataSet.iloc[0]

        #TODO: Determine which table data will go to (avgValues, Continuous lifetime. Transient Lifetime)
        #Average Value Table will have dataType = avgValues_{type}
        if dataType[0:9] == "avgValues":
            dataString = f"{dataType[10:]}"
           
            if(dataString != "power_factor"):
                self.cursor.execute(f'''
                                    INSERT OR REPLACE INTO "{transformerName}_average_metrics_{iterator}"(
                                    timestamp,
                                    avg_{dataString}_a_phase,
                                    avg_{dataString}_b_phase,
                                    avg_{dataString}_c_phase,
                                    avg_{dataString}_total_phase
                                    ) 
                                    VALUES(?,?,?,?,?)
                                    ON CONFLICT(timestamp) DO UPDATE SET
                                    avg_{dataString}_a_phase = excluded.avg_{dataString}_a_phase,
                                    avg_{dataString}_b_phase = excluded.avg_{dataString}_b_phase,
                                    avg_{dataString}_c_phase = excluded.avg_{dataString}_c_phase,
                                    avg_{dataString}_total_phase = excluded.avg_{dataString}_total_phase
                                    ''',
                                    dataSet_list
                                    )
            else:
                self.cursor.execute(f'''
                                    INSERT OR REPLACE INTO "{transformerName}_average_metrics_{iterator}"(
                                    timestamp,
                                    avg_{dataString}
                                    )
                                    VALUES(?,?)
                                    ON CONFLICT(timestamp) DO UPDATE SET
                                    avg_{dataString} = excluded.avg_{dataString}
                                    ''',
                                    dataSet_list
                                    )
            self.dbconnect.commit()
        
        #Lifetime(Continuous) Table:
        elif dataType == "lifetime_continuous":
            self.cursor.execute(f'''
                            INSERT OR REPLACE INTO "{transformerName}_lifetime_continuous_loading"(
                            timestamp,
                            a_phase_load_current,
                            b_phase_load_current,
                            c_phase_load_current,
                            total_phase_load_current,
                            a_phase_winding_temp,
                            b_phase_winding_temp,
                            c_phase_winding_temp,
                            total_phase_winding_temp,
                            a_phase_thermoD_hot_spot,
                            b_phase_thermoD_hot_spot,
                            c_phase_thermoD_hot_spot,
                            total_phase_thermoD_hot_spot,
                            a_phase_lifetime,
                            b_phase_lifetime,
                            c_phase_lifetime,
                            total_phase_lifetime
                            ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)''',
                            dataSet_list
                            )
            self.dbconnect.commit()

        #Lifetime(Transient) Table:
        elif dataType == "lifetime_transient":
            self.cursor.execute(f'''
                        INSERT OR REPLACE INTO "{transformerName}_lifetime_transient_loading"(
                        timestamp,
                        a_phase_load_current,
                        b_phase_load_current,
                        c_phase_load_current,
                        total_phase_load_current,
                        a_phase_thermoD_hot_spot,
                        b_phase_thermoD_hot_spot,
                        c_phase_thermoD_hot_spot,
                        total_phase_thermoD_hot_spot,
                        a_phase_lifetime_consumption,
                        b_phase_lifetime_consumption,
                        c_phase_lifetime_consumption,
                        total_phase_lifetime_consumption
                        ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)''',
                        dataSet_list
                        )
            self.dbconnect.commit()
        
        # Health Monitoring Table
        elif dataType =="health":
            self.cursor.execute(f'''
                    INSERT OR REPLACE INTO "{transformerName}_health"(
                    transformer_name,
                    date,
                    variable_name,
                    average_value,
                    rated_value,
                    status,
                    overall_score,
                    overall_color
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)''',
                    dataSet_list
                    )
            self.dbconnect.commit()

        else:
            print("Invalid Data type")
            return

    #!Populate format for transformer rated values, creating empty storage structure for Transformer Data and filling in transformer rated values. Import all known data
    def addTransformer(self, transformer):
        #TODO: populate raw data table for transformer:
        self.populateRawDataTable(transformer)
        
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
                       rated_impedance NUMERIC)
                    ''')
        self.dbconnect.commit()
        
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
        self.dbconnect.commit()
        
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
        
        self.dbconnect.commit()

        #TODO: Import all previous data from transformer, mirror copy file pulled from eFCMS
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
        #TODO:------------------------------------------------------------------------------------
        self.dbconnect.commit()

    #------------------------eFCMS-interaction-with-database--------------------#
    #! Collect all availble and relevant data stored for a specific transformer 
    def populateRawDataTable(self,transformer):
        #TODO: Load Previous Data for transformer into database and create table. Local import for now, will be eFCMS specific later
        previousData = pandas.read_excel("/home/eveuio/DataProcessing/IncompleteTransformerData_historical/22A03_noLastWeek.xlsx")
        previousData.to_sql(name=transformer.name+"fullRange",con=self.dbconnect,if_exists="replace", index=False)
        self.dbconnect.commit()
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


            
        