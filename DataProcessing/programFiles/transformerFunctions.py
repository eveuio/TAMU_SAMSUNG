
import math
import numpy
from datetime import date
import pandas
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import sqlite3
import os
from scipy.integrate import simpson


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
        b = math.log(2) / (1 / (self.hotSpotWindingTemp_rated + 273) - 1 / (self.hotSpotWindingTemp_rated + 273 + 6))
        a = math.e ** (math.log(180000) - b / (self.hotSpotWindingTemp_rated + 273))
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
        d_vhs_rated = 1.25*self.avgWindingTempRise_rated

        # ----- Ratios and masks -----
        ratio_final   = transformerData["d_vhs_final"]   / d_vhs_rated
        ratio_initial = transformerData["d_vhs_initial"] / d_vhs_rated

        # Detect near-equality (0/0 case)
        eps = 1e-6
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

        decay = numpy.exp(-60/ safe_tau)                     # NaN-safe
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
        transformerData["thermoDynamicHS_kelvin"] = 273 + transformerData["T_ambient"] + transformerData["ultimateHotSpotRise"]

        
        tempK = transformerData["thermoDynamicHS_kelvin"]
        safe_tempK = numpy.where(~numpy.isfinite(tempK) | (tempK <= 0), numpy.nan, tempK)

        #TODO: Calculate lifetime consumption per tiem period (hour) as a percentage of 180,000
        transformerData["LifetimeHourPerHourConsumption"] = 180000.0 * (1/ a) * numpy.exp(-b / safe_tempK)

        transformerData["LifetimeConsumption_hour_percent"] = 100 * 1 * (1/ a) * numpy.exp(-b / safe_tempK)
        
        
        #TODO: Calculate Aggregate totals of lifetime consumption per day via simpsons rule
       
        daily_data = []

        transformerData_sorted = transformerData.sort_values("DATETIME").reset_index(drop=True)
        transformerData_sorted["_date"] = pd.to_datetime(transformerData_sorted["DATETIME"]).dt.date

        unique_dates = transformerData_sorted["_date"].unique()

        for i, date in enumerate(unique_dates):
            # Get current day's data (hours 0-23)
            day_mask = transformerData_sorted["_date"] == date
            day_data = transformerData_sorted[day_mask].copy()
            
            n = len(day_data)
            
            if n < 3:
                # Not enough points, use sum
                lifetime_day = day_data["LifetimeConsumption_hour_percent"].sum()
                hours_day = day_data["LifetimeHourPerHourConsumption"].sum()
            else:
                # Try to get hour 0 of next day (which is hour 24 of current day)
                if i + 1 < len(unique_dates):
                    next_date = unique_dates[i + 1]
                    next_day_first = transformerData_sorted[transformerData_sorted["_date"] == next_date].head(1)
                    
                    if not next_day_first.empty:
                        # We have 25 points (hours 0-24), use Simpson's rule
                        

                        extended_data = pd.concat([day_data, next_day_first], ignore_index=True)

                        # if i == 0:
                        #     print(f"=== DEBUG: First day ({date}) ===")
                        #     print(f"Number of points: {len(extended_data)}")
                        #     print(f"\nLifetimeHourPerHourConsumption values (h/h):")
                        #     for idx, val in enumerate(extended_data["LifetimeHourPerHourConsumption"].values):
                        #         print(f"Hour {idx}: {val:.6f}")
                        #     print(f"\nSum before Simpson's: {extended_data['LifetimeHourPerHourConsumption'].sum():.6f}")

                        lifetime_day = simpson(extended_data["LifetimeConsumption_hour_percent"].values, dx=1.0)
                        hours_day = simpson(extended_data["LifetimeHourPerHourConsumption"].values, dx=1.0)
                    else:
                        # No next day, use Simpson's on what we have (24 points)
                        lifetime_day = simpson(day_data["LifetimeConsumption_hour_percent"].values, dx=1.0)
                        hours_day = simpson(day_data["LifetimeHourPerHourConsumption"].values, dx=1.0)
                else:
                    # Last day in dataset, no next day available
                    # Use Simpson's on 24 points
                    lifetime_day = simpson(day_data["LifetimeConsumption_hour_percent"].values, dx=1.0)
                    hours_day = simpson(day_data["LifetimeHourPerHourConsumption"].values, dx=1.0)
            
            daily_data.append({
                "DATETIME": pd.Timestamp(date),
                "LifetimeConsumption_day_percent": lifetime_day,
                "LifetimeHourPerHourConsumption": hours_day
            })

        transformerData_daily = pd.DataFrame(daily_data)

        # Clean up temporary column
        if "_date" in transformerData.columns:
            transformerData = transformerData.drop(columns=["_date"], errors="ignore")


        #TODO: Calculate Remaining lifetime based on aggregate total consumed per day
        # Remaining lifetime from current starting point
        transformerData_daily["remainingLifetime_percent"] = (
            currentLifetime_percent - transformerData_daily["LifetimeConsumption_day_percent"].cumsum()).clip(lower=0)  # don’t let it go below zero

        # Formatting DATETIME as string
        transformerData_daily["DATETIME"] = transformerData_daily["DATETIME"].dt.strftime("%Y-%m-%d %H:%M:%S")

        return transformerData_daily
