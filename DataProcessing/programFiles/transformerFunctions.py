
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
    def __init__(self, rated_specs):
        """
        Calculate transient lifetime consumption given rated specs from 'transformers' table in transformerDB.db
        """
        
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
        self.hotSpotWindingTemp_rated = self.thermalClass_rated-10
        self.age = date.today().year - self.manufactureDate
        self.MaterialConstant = 225 if self.windingMaterial == "Aluminum" else 235

        self._determineXR_ratio()
        self._calculate_time_constant()


    #! Compute time constant based on rated values and estimated X/R ratio
    def _calculate_time_constant(self):
       
        impedance_shortCircuit = self.impedance * self.LV / self.RatedCurrentLV
        R = impedance_shortCircuit / numpy.sqrt(1 + self.XR_Ratio**2)
        W_r = R * (self.RatedCurrentLV) ** 2
        d_vhsR = self.avgWindingTempRise_rated + 20

        if self.windingMaterial == "Aluminum":
            C_h = 0.044 * self.weight_CoreAndCoil
        else:
            C_h = 0.033 * self.weight_CoreAndCoil

        # self.ratedTimeConstant = C_h * d_vhsR / W_r
        self.ratedTimeConstant = 0.5


    #! Determine X/R ratio based on data from power dry 2 transformer bulletin
    def _determineXR_ratio(self):
        #TODO: X r ratio not known for all transformers, hard code for now based on power dry X/R ratios from data bulletin:
        if self.KVA < 300:
            # no data in bulletin for KVA < 225
            self.XR_Ratio = 2.12
        
        elif self.KVA >= 300 and self.KVA < 500:
            self.XR_rRatio= 2.38
        
        elif self.KVA >= 500 and self.KVA < 750:
            self.XR_Ratio= 3.36
        
        elif self.KVA >= 750 and self.KVA < 1000:
            self.XR_Ratio= 4.05
        
        elif self.KVA >= 1000 and self.KVA < 1500:
            self.XR_Ratio= 4.28
        
        elif self.KVA >= 1500 and self.KVA < 2000:
            self.XR_Ratio= 4.89
        
        elif self.KVA >= 2000 and self.KVA < 2500:
            self.XR_Ratio= 5.40
        
        elif self.KVA >= 2500 and self.KVA < 3000:
            self.XR_Ratio= 5.60
        
        elif self.KVA >= 3000 and self.KVA < 3750:
            self.XR_Ratio= 5.28
        
        elif self.KVA >= 3750 and self.KVA < 5000:
            self.XR_Ratio= 5.72
        
        else:
            #KVA number larger than 5000, no data in bulletin beyond 5000
            self.XR_Ratio = 6.8

    
    #! Calculates lifetime consumption per hour and returns a dataframe with consumption per hour, overall remaining lifetime percentage and 
    def lifetime_TransientLoading(self, avg_metrics_hour: pd.DataFrame):
        #TODO: arrhenius constants (a,b) and IEC parameters (m)
        b = math.log(2) / (1 / (self.hotSpotWindingTemp_rated + 273.15) - 1 / (self.hotSpotWindingTemp_rated + 273.15 + 6))
        a = math.e ** (math.log(180000) - b / (self.hotSpotWindingTemp_rated + 273.15))
        m = 0.8

        # Work on a copy to avoid mutating the original DataFrame accidentally
        transformerData = avg_metrics_hour.copy()

        #TODO: Calculate initial lifetime percentage accounting for IEEE reccomended ~1% per year
        transformerData["DATETIME"] = pd.to_datetime(transformerData["DATETIME"], errors="coerce")
        transformerData = transformerData.sort_values("DATETIME").reset_index(drop=True)

        start_ts = transformerData["DATETIME"].iloc[0]
        end_ts   = transformerData["DATETIME"].iloc[-1]
        elapsed_years = (end_ts - start_ts).total_seconds() / (365 * 24 * 3600)
        age_at_start = self.age - elapsed_years
        
        currentLifetime_percent = 100 - age_at_start  

        #TODO: Calculate ambient temperature, initial and final HS temp rised per period
        # transformerData["T_ambient"] = 26.67 + (43.3333 - 26.67) * (transformerData["avg_secondary_current_total_phase"] / self.RatedCurrentLV)
        transformerData["T_ambient"]= 30

        transformerData["d_vhs_initial"] = transformerData["avg_winding_temp_total_phase"] - transformerData["T_ambient"]
        transformerData["d_vhs_final"]   = transformerData["avg_winding_temp_total_phase"].shift(1) - transformerData["T_ambient"].shift(1)

        # Drop the first row created by shift (NaNs)
        transformerData = transformerData.dropna(subset=["d_vhs_initial", "d_vhs_final", "T_ambient"]).reset_index(drop=True)

        #TODO: Need to account for when d_vhs_inital and d_vhs final are equal to each other when calculating tau to avoid a 0/0 situation, ie steady state or continuous loading. 
        # Rated reference hot spot temperature
        d_vhs_rated = 20 + self.avgWindingTempRise_rated

        # ----- Ratios and masks -----
        ratio_final   = transformerData["d_vhs_final"]   / d_vhs_rated
        ratio_initial = transformerData["d_vhs_initial"] / d_vhs_rated

        # Detect near-equality (0/0 case)
        eps = 1e-12
        equal_mask = numpy.isclose(ratio_final, ratio_initial, rtol=1e-9, atol=1e-12)


        # Fractional powers: guard negative bases for non-integer exponent
        neg_mask = (ratio_final < 0) | (ratio_initial < 0)
        pow_final   = numpy.where(neg_mask, numpy.nan, numpy.power(ratio_final, 1.0/m))
        pow_initial = numpy.where(neg_mask, numpy.nan, numpy.power(ratio_initial, 1.0/m))

        # Standard tau (may be invalid when numerator & denominator ~ 0)
        num_tau = (ratio_final - ratio_initial)
        den_tau = (pow_final   - pow_initial)
        near_zero_den = numpy.abs(den_tau) < eps
        tau_standard = self.ratedTimeConstant * (num_tau / numpy.where(near_zero_den, numpy.nan, den_tau))

        # L’Hôpital limit for equal-deltas: tau = RT * m * r^(1 - 1/m), r = ratio_initial (== ratio_final)
        r = ratio_initial
        tau_limit = self.ratedTimeConstant * m * numpy.power(r, 1.0 - (1.0/m))

        # Combine: use limit when equal OR when denominator is near-zero
        transformerData["tau_total_hour"] = numpy.where(equal_mask | near_zero_den, tau_limit, tau_standard)
        
        #TODO: Calculate ultimate HS rise per hour
        tau = transformerData["tau_total_hour"]
        
        safe_tau = numpy.where(numpy.isfinite(tau) & (numpy.abs(tau) > eps), tau, numpy.nan)

        decay = numpy.exp(-1.0 / safe_tau)                     # NaN-safe
        denominator_ultimate = 1.0 - decay
        safe_den_ultimate = numpy.where(numpy.abs(denominator_ultimate) < eps, numpy.nan, denominator_ultimate)

        transformerData["ultimateHotSpotRise"] = (
            (transformerData["d_vhs_final"] - transformerData["d_vhs_initial"]) / safe_den_ultimate
            + transformerData["d_vhs_initial"]
        )

        # In equal-delta rows, numerator==0; physically ultimate rise should stay at the initial value.
        fix_mask = equal_mask & ~numpy.isfinite(transformerData["ultimateHotSpotRise"])
        transformerData.loc[fix_mask, "ultimateHotSpotRise"] = transformerData.loc[fix_mask, "d_vhs_initial"]

        #TODO: Calculate thermodynamic HS rise in Kelvin; T = 273.15 + ambientTemp + ultimateHS_rise
        transformerData["thermoDynamicHS_kelvin"] = 273.15 + transformerData["T_ambient"] + transformerData["ultimateHotSpotRise"]

        
        tempK = transformerData["thermoDynamicHS_kelvin"]
        safe_tempK = numpy.where(~numpy.isfinite(tempK) | (tempK <= 0), numpy.nan, tempK)

        #TODO: Calculate lifetime consumption per tiem period (hour) as a percentage of 180,000
        transformerData["LifetimeHourPerHourConsumption"] = 180000.0 * (1/ a) * numpy.exp(-b / safe_tempK)

        transformerData["LifetimeConsumption_hour_percent"] = 100 * 1 * (1/ a) * numpy.exp(-b / safe_tempK)
        
        #TODO: Calculate Aggregate totals of lifetime consumption per day
        daily_group = transformerData.resample("D", on="DATETIME")
        
        transformerData_daily = (
            daily_group.sum(numeric_only=True)[["LifetimeConsumption_hour_percent", "LifetimeHourPerHourConsumption"]]
            .rename(columns={
                "LifetimeConsumption_hour_percent": "LifetimeConsumption_day_percent",
                "LifetimeHourPerHourConsumption": "LifetimeHoursConsumed_day"
            })
            .reset_index()
        )


        #TODO: Calculate Remaining lifetime based on aggregate total consumed per day
        # Remaining lifetime from current starting point
        transformerData_daily["remainingLifetime_percent"] = (
            currentLifetime_percent - transformerData_daily["LifetimeConsumption_day_percent"].cumsum()).clip(lower=0)  # don’t let it go below zero

        # Formatting DATETIME as string
        transformerData_daily["DATETIME"] = transformerData_daily["DATETIME"].dt.strftime("%Y-%m-%d %H:%M:%S")

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
    
    
