
import math
import numpy
from datetime import date
import pandas
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import sqlite3
import os


import math
import numpy
import pandas as pd
from datetime import date
from sqlalchemy.engine import Engine


class Transformer:
    def __init__(self, rated_specs, engine: Engine):
        """
        Initialize transformer instance based on data stored in the 'transformers' table.
        Fetches all rated parameters automatically from the database.
        :param name: Transformer name (transformer_name in the DB)
        :param engine: SQLAlchemy engine (FastAPI-compatible)
        """
        # self.name = name
        self.engine = engine

        # # Load transformer metadata from DB
        # query = f"""
        #     SELECT * FROM transformers
        #     WHERE transformer_name = :xfmr_name
        #     LIMIT 1
        # """
        # df = pd.read_sql_query(query, self.engine, params={"xfmr_name": name})

        # if df.empty:
        #     raise ValueError(f"Transformer '{name}' not found in the database.")

        # Assign rated values from the transformers table
        row = rated_specs.iloc[0]
        self.HV = row["rated_voltage_HV"]
        self.LV = row["rated_voltage_LV"]
        self.KVA = row["kva"]
        self.ratedCurrentHV = row["rated_current_HV"]
        self.RatedCurrentLV = row["rated_current_LV"]
        self.thermalClass_rated = row["rated_thermal_class"]
        self.avgWindingTempRise_rated = row["rated_avg_winding_temp_rise"]
        self.windingMaterial = row["winding_material"]
        self.weight_CoreAndCoil = row["weight_CoreAndCoil_kg"]
        self.weightTotal = row["weight_Total_kg"]
        self.impedance = row["rated_impedance"] * 0.1
        self.manufactureDate = int(row["manufacture_date"])
        self.status = row["status"]

        # Derived values
        self.hotSpotWindingTemp_rated = self.thermalClass_rated - 10
        self.age = date.today().year - self.manufactureDate
        self.XR_Ratio = 6
        self.MaterialConstant = 225 if self.windingMaterial == "Aluminum" else 235

        # Time constant calculation
        self._calculate_time_constant()


    # --------------------------------------------------------------------
    # 🔹 Helper: precompute thermal time constant
    # --------------------------------------------------------------------
    def _calculate_time_constant(self):
        # impedance_shortCircuit = self.impedance * self.LV / self.RatedCurrentLV
        # R = impedance_shortCircuit / numpy.sqrt(1 + self.XR_Ratio**2)
        # W_r = R * (self.RatedCurrentLV) ** 2
        # d_vhsR = self.avgWindingTempRise_rated + 20

        # if self.windingMaterial == "Aluminum":
        #     C_h = 0.044 * self.weight_CoreAndCoil
        # else:
        #     C_h = 0.033 * self.weight_CoreAndCoil

        # self.ratedTimeConstant = C_h * d_vhsR / W_r

        #TODO: X r ratio not known for all transformers, hard code for now
        self.ratedTimeConstant = 3
        


    # --------------------------------------------------------------------
    # 🔹 Rated steady-state lifetime
    # --------------------------------------------------------------------
    def ratedLifeTime(self):
        ambientTemp = 30
        T = 273 + ambientTemp + 1.2 * self.avgWindingTempRise_rated
        b = math.log(2) / (
            1 / (self.hotSpotWindingTemp_rated + 273)
            - 1 / (self.hotSpotWindingTemp_rated + 273 + 6)
        )
        a = math.e ** (math.log(180000) - b / (self.hotSpotWindingTemp_rated + 273))
        L = a * math.e ** (b / T)
        return L / 8766  # years


    # --------------------------------------------------------------------
    # 🔹 Continuous loading lifetime model
    # --------------------------------------------------------------------
    def lifetime_ContinuousLoading(self, avg_metrics) -> pd.DataFrame:
        # Arrhenius constants
        b = math.log(2) / (1 / (self.hotSpotWindingTemp_rated + 273)- 1/(self.hotSpotWindingTemp_rated + 273 + 6))
        a = math.exp(math.log(180000) - b / (self.hotSpotWindingTemp_rated + 273))

        transformerData = avg_metrics.copy()
        transformerData["DATETIME"] = pd.to_datetime(transformerData["DATETIME"])

        # Assuming hotspot temp columns are already present in avg_metrics
        hsA, hsB, hsC = transformerData.columns[15:18]  # adjust indices as needed
        transformerData["hotspot_temp_max"] = numpy.max(transformerData[[hsA, hsB, hsC]].values, axis=1)

        lifetimeInHours = a * numpy.exp(b / (transformerData["hotspot_temp_max"] + 273.15))
        transformerData["Lifetime_Years"] = lifetimeInHours / 8766

        return transformerData


    # --------------------------------------------------------------------
    # 🔹 Transient (non-constant) loading lifetime model, calculates consumption per hour and returns a total amount per day
    # --------------------------------------------------------------------
    def lifetime_TransientLoading(self, avg_metrics_hour:pandas.DataFrame):

        # Constants and parameters
        b = math.log(2) / (1 / (self.hotSpotWindingTemp_rated + 273)- 1 / (self.hotSpotWindingTemp_rated + 273 + 6))
        a = math.e ** (math.log(180000) - b / (self.hotSpotWindingTemp_rated + 273))
        m = 0.8

        transformerData = avg_metrics_hour

        # Approximate remaining life percentage at start of datasheet
        start_ts = pd.to_datetime(transformerData['DATETIME'].iloc[0])
        end_ts   = pd.to_datetime(transformerData['DATETIME'].iloc[-1])

        elapsed_years = (end_ts - start_ts).total_seconds() / (365.25 * 24 * 3600)

        age_at_start = self.age - elapsed_years

        currentLifetime_percent = 100 - age_at_start

        # Set up additional data needed for calculations below; final and initial hotspots per period
        transformerData["DATETIME"] = pd.to_datetime(transformerData["DATETIME"])
        transformerData["T_ambient"] = 26.67 + (43.3333 - 26.67) * (transformerData["avg_secondary_current_total_phase"] / self.RatedCurrentLV)
        transformerData["d_vhs_initial"] = (transformerData["avg_winding_temp_total_phase"] - transformerData["T_ambient"])
        transformerData["d_vhs_final"] = (transformerData["avg_winding_temp_total_phase"].shift(1)- transformerData["T_ambient"].shift(1))

        d_vhs_rated = 30 + self.avgWindingTempRise_rated

        # Correct formula: protect against division-by-zero and NaNs
        transformerData["tau_total_hour"] = (self.ratedTimeConstant * ((transformerData["d_vhs_final"] / d_vhs_rated)- (transformerData["d_vhs_initial"] / d_vhs_rated))/ ((transformerData["d_vhs_final"] / d_vhs_rated) ** (1 / m)- (transformerData["d_vhs_initial"] / d_vhs_rated) ** (1 / m)))

        # Ultimate hot spot rise per hour
        transformerData["ultimateHotSpotRise"] = ((transformerData["d_vhs_final"] - transformerData["d_vhs_initial"])/ (1 - numpy.exp(-1 / transformerData["tau_total_hour"]))+ transformerData["d_vhs_initial"])

        # Hot spot temp (K)
        transformerData["thermoDynamicHS_kelvin"] = (273.15 + transformerData["T_ambient"] + transformerData["ultimateHotSpotRise"])

        # Hourly lifetime consumption (% of total life)
        transformerData["LifetimeConsumption_hour_percent"] = (180000 * (1 / a) * numpy.exp(-b / transformerData["thermoDynamicHS_kelvin"]))

       # Aggregate hourly consumption into daily totals
        transformerData_daily = (
            transformerData.resample("D", on="DATETIME")
            .sum(numeric_only=True)[["LifetimeConsumption_hour_percent"]]
            .rename(columns={"LifetimeConsumption_hour_percent": "LifetimeConsumption_day_percent"})
            .reset_index()  # <-- keeps DATETIME as a column
        )

        # Initialize remaining lifetime starting from currentLifetime_percent
        transformerData_daily["remainingLifetime_percent"] = (
            currentLifetime_percent - transformerData_daily["LifetimeConsumption_day_percent"].cumsum()
        )

        # Prevent going below zero
        transformerData_daily["remainingLifetime_percent"] = transformerData_daily["remainingLifetime_percent"].clip(lower=0)
        
        return transformerData_daily











































































# DB_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'transformerDB.db')) # database file in TAMU_SAMSUNG/transformerDB.db

# # TODO: Transformer class; initialize rated values and define lifetime functions:
# class Transformer:
#     def __init__(self,
#                  name,
#                  ratedCurrent_H,
#                  ratedVoltage_H, 
#                  ratedCurrent_L, 
#                  ratedVoltage_L,
#                  impedance, 
#                  windingMaterial, 
#                  thermalClass_rated, 
#                  avgWindingTempRise_rated, 
#                  weight_CoreAndCoil,
#                  weight_total,
#                  manufactureDate,
#                  XR_Ratio=6,
#                  status="new",
#                  ratedKVA=0
#                  ):
#         self.name = name
#         self.thermalClass_rated = thermalClass_rated
#         self.avgWindingTempRise_rated = avgWindingTempRise_rated
#         self.hotSpotWindingTemp_rated = self.thermalClass_rated-10
#         self.windingMaterial = windingMaterial
#         self.weight_CoreAndCoil = weight_CoreAndCoil      #must be in kg
#         self.weightTotal = weight_total                   #must be in kg
#         self.HV = ratedVoltage_H
#         self.LV = ratedVoltage_L
#         self.ratedCurrentHV = ratedCurrent_H
#         self.RatedCurrentLV = ratedCurrent_L
#         self.impedance = impedance*0.1
#         self.age = manufactureDate - date.today().year
#         self.XR_Ratio = 6

#         self.db_path = DB_PATH
#         self.conn = None
#         if DB_PATH is not None:
#             self.conn = sqlite3.connect(DB_PATH)

#         if(windingMaterial == "Aluminum"):
#             self.MaterialConstant = 225
#         else:
#             self.MaterialConstant = 235
        
#         # From Power Dry Transformer Data Bulletin
#         # if (ratedKVA == 1000):
#         #     self.totalLosses =10700 
#         # elif(ratedKVA == 1500):
#         #     self.totalLosses = 13900
#         # elif(ratedKVA == 2000):
#         #     self.totalLosses = 16300
#         # elif(ratedKVA == 2500):
#         #     self.totalLosses = 20650

#         #TODO: ---------Add Pre-Calculated Time Constant---------
        
#         #TODO: Short circuit impedance of winding
#         impedance_shortCircuit = self.impedance*ratedVoltage_L/ratedCurrent_L
        
#         #TODO: Real Component of winding impedance
#         R = (impedance_shortCircuit)/numpy.sqrt(1+XR_Ratio**2)
        
#         #TODO: Winding losses based off real portion of impedance
#         W_r= (R)*(self.RatedCurrentLV)**2
#         # print(self.name +" Winding Losses in W: ", W_r)

#         #TODO: Rated Hot-Spot Temp, based off 20C ambient temp
#         d_vhsR = self.avgWindingTempRise_rated+20
        
#         # IEEE loading guide, based off epoxy windings 
#         #TODO: Thermal Capacity of transformer in C
#         if(self.windingMaterial == "Aluminum"):
#             C_h = 0.044*(self.weight_CoreAndCoil) #unit: watt-hour/C
#         else: #copper windings
#             C_h = 0.033*(self.weight_CoreAndCoil) #unit: watt-hour/C

#         #TODO: Calculate time constant
#         self.ratedTimeConstant = C_h*(d_vhsR)/(W_r)
    

#         #TODO: --------------------------------------------------

#     #--------------------------------LIFETIME-MODELS-------------------------------------------#
#     # ! Rated model (w/Transformer Rated Values and ambient temp):
#     def ratedLifeTime(self):
#         ambientTemp = 30
#         Z = 1.2
#         # Lifetime Calculations:
#         T = 273 + ambientTemp + 1.2*self.avgWindingTempRise_rated*(1)**1.6

#         b = math.log(2)/(1/(self.hotSpotWindingTemp_rated +273)- 1/(self.hotSpotWindingTemp_rated +273+6))
#         a = math.e**(math.log(180000)-b/(self.hotSpotWindingTemp_rated+273))

#         L = a*math.e**(b/T)

#         # print("a: ",a)
#         # print("b: ",b)
#         # print("T: ",T)
#         # print("Rated Lifetime: ", L/8766)
        
#         return (L/8766)
    
#     #! model given real current loading, constant load
#     def lifetime_ContinuousLoading(self):
#         #TODO: define what a and b to use given rated winding temp rated value
#         b = math.log(2)/(1/(self.hotSpotWindingTemp_rated +273)- 1/(self.hotSpotWindingTemp_rated +273+6))
#         a = math.e**(math.log(180000)-b/(self.hotSpotWindingTemp_rated+273))

#         #TODO: Collect max winding temp data from {transformer.name}_average_metrics_hour
#         table_name = f'''{self.name}_average_metrics_hour'''
#         query = f'''SELECT * FROM "{table_name}" ORDER BY "DATETIME" ASC'''
#         transformerData = pandas.read_sql_query(query, self.conn)
#         transformerData['DATETIME'] = pandas.to_datetime(transformerData['DATETIME'])

#         # Identify relevant columns: Hot spot data
#         hsA, hsB, hsC = transformerData.columns[15:18]

#         # Identify max hotspot
#         transformerData['hotspot_temp_max'] = numpy.max(transformerData[[hsA, hsB, hsC]].values, axis=1)

#         #TODO: Compute lifetime given data from phaseMax at given timestamp (ie, phase with largest recorded winding temp)
#         lifetimeInHours = a*numpy.exp(b/(transformerData['hotspot_temp_max'] + 273.15))
#         transformerData['Lifetime_Years'] = lifetimeInHours/8766

#         #TODO: Push to SQLite DB
#         transformerData.to_sql(
#             name=f'''{self.name}_trialLifetimeData_Continuous1''',
#             con=self.conn,
#             if_exists="replace",
#             chunksize=5000,
#             method="multi",
#             index=False
#         )
        
#         return

#     #! Transient Loading (Load not constant):
#     def lifetime_TransientLoading(self):
#         b = math.log(2)/(1/(self.hotSpotWindingTemp_rated +273)- 1/(self.hotSpotWindingTemp_rated +273+6))
#         a = math.e**(math.log(180000)-b/(self.hotSpotWindingTemp_rated+273))
#         m = 0.8
#         #TODO: Collect max winding temp data from {transformer.name}_average_metrics_hour
#         table_name = f'''{self.name}_average_metrics_day'''
#         query = f'''SELECT * FROM "{table_name}" ORDER BY "DATETIME" ASC'''
#         transformerData = pandas.read_sql_query(query, self.conn)
#         transformerData['DATETIME'] = pandas.to_datetime(transformerData['DATETIME'])
        
#         #TODO: Initial hotspot temp for loading considerations
#         transformerData['T_ambient'] = 26.67 + (43.3333-26.67)*(transformerData['avg_secondary_current_total_phase']/self.RatedCurrentLV)
#         transformerData['d_vhs_initial'] = transformerData['avg_winding_temp_total_phase'] - transformerData['T_ambient']

#         #TODO: Final hotspot temps for loading consideration (1 hour after d_vhs_initial)
#         transformerData['d_vhs_final'] = transformerData['avg_winding_temp_total_phase'].shift(1) - transformerData['T_ambient'].shift(1)

#         #TODO: Instantaneous time constant for given loading conditions:
#         d_vhs_rated = 30 + self.avgWindingTempRise_rated
#         transformerData['tau_total_hour'] = self.ratedTimeConstant*(((transformerData['d_vhs_final']/d_vhs_rated)-(transformerData['d_vhs_initial']/d_vhs_rated))/((transformerData['d_vhs_final']/d_vhs_rated)**(1/m)-(transformerData['d_vhs_initial']/d_vhs_rated)**(1/m)))
        
#         #TODO: ultimate hot spot rise using a time period t of 24 hours (1 day). This is now the d_vhs, since t > 5 tau (per IEC 60076-12)
#         transformerData['ultimateHotSpotRise'] = (transformerData['d_vhs_final']-transformerData['d_vhs_initial'])/(1-math.e**(-24/transformerData['tau_total_hour']))+transformerData['d_vhs_initial']
        
#         #TODO: Lifetime consumption in hours
#         transformerData['thermoDynamicHS_kelvin'] = 273.15 + transformerData['T_ambient'] + transformerData['ultimateHotSpotRise']
#         transformerData['LifetimeConsumption_day'] = 180000*24*(1/a)*math.e**(-b/transformerData['thermoDynamicHS_kelvin'])

#         #TODO: Push transformerData to sql table
#         transformerData.to_sql(
#             name=f'''{self.name}_trialLifetimeData_Transient2''',
#             con=self.conn,
#             if_exists="replace",
#             chunksize=5000,
#             method="multi",
#             index=False
#         )

#         return


#     #--------------------------------FRONT-END----------------------------------#
    
    
