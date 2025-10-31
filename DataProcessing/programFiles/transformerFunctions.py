
import math
import numpy
from datetime import date
import pandas
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import sqlite3
import os

DB_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'transformerDB.db')) # database file in TAMU_SAMSUNG/transformerDB.db

# TODO: Transformer class; initialize rated values and define lifetime functions:
class Transformer:
    def __init__(self,
                 name,
                 ratedCurrent_H,
                 ratedVoltage_H, 
                 ratedCurrent_L, 
                 ratedVoltage_L,
                 impedance, 
                 windingMaterial, 
                 thermalClass_rated, 
                 avgWindingTempRise_rated, 
                 weight_CoreAndCoil,
                 weight_total,
                 manufactureDate,
                 XR_Ratio=6,
                 status="new",
                 ratedKVA=0
                 ):
        self.name = name
        self.thermalClass_rated = thermalClass_rated
        self.avgWindingTempRise_rated = avgWindingTempRise_rated
        self.hotSpotWindingTemp_rated = self.thermalClass_rated-10
        self.windingMaterial = windingMaterial
        self.weight_CoreAndCoil = weight_CoreAndCoil      #must be in kg
        self.weightTotal = weight_total                   #must be in kg
        self.HV = ratedVoltage_H
        self.LV = ratedVoltage_L
        self.ratedCurrentHV = ratedCurrent_H
        self.RatedCurrentLV = ratedCurrent_L
        self.impedance = impedance*0.1
        self.age = manufactureDate - date.today().year
        self.XR_Ratio = 6

        self.db_path = DB_PATH
        self.conn = None
        if DB_PATH is not None:
            self.conn = sqlite3.connect(DB_PATH)

        if(windingMaterial == "Aluminum"):
            self.MaterialConstant = 225
        else:
            self.MaterialConstant = 235
        
        # From Power Dry Transformer Data Bulletin
        # if (ratedKVA == 1000):
        #     self.totalLosses =10700 
        # elif(ratedKVA == 1500):
        #     self.totalLosses = 13900
        # elif(ratedKVA == 2000):
        #     self.totalLosses = 16300
        # elif(ratedKVA == 2500):
        #     self.totalLosses = 20650

        #TODO: ---------Add Pre-Calculated Time Constant---------
        
        #TODO: Short circuit impedance of winding
        impedance_shortCircuit = self.impedance*ratedVoltage_L/ratedCurrent_L
        
        #TODO: Real Component of winding impedance
        R = (impedance_shortCircuit)/numpy.sqrt(1+XR_Ratio**2)
        
        #TODO: Winding losses based off real portion of impedance
        W_r= (R)*(self.RatedCurrentLV)**2
        # print(self.name +" Winding Losses in W: ", W_r)

        #TODO: Rated Hot-Spot Temp, based off 20C ambient temp
        d_vhsR = self.avgWindingTempRise_rated+20
        
        # IEEE loading guide, based off epoxy windings 
        #TODO: Thermal Capacity of transformer in C
        if(self.windingMaterial == "Aluminum"):
            C_h = 0.044*(self.weight_CoreAndCoil) #unit: watt-hour/C
        else: #copper windings
            C_h = 0.033*(self.weight_CoreAndCoil) #unit: watt-hour/C

        #TODO: Calculate time constant
        self.ratedTimeConstant = C_h*(d_vhsR)/(W_r)
    

        #TODO: --------------------------------------------------

    #--------------------------------LIFETIME-MODELS-------------------------------------------#
    # ! Rated model (w/Transformer Rated Values and ambient temp):
    def ratedLifeTime(self):
        ambientTemp = 30
        Z = 1.2
        # Lifetime Calculations:
        T = 273 + ambientTemp + 1.2*self.avgWindingTempRise_rated*(1)**1.6

        b = math.log(2)/(1/(self.hotSpotWindingTemp_rated +273)- 1/(self.hotSpotWindingTemp_rated +273+6))
        a = math.e**(math.log(180000)-b/(self.hotSpotWindingTemp_rated+273))

        L = a*math.e**(b/T)

        # print("a: ",a)
        # print("b: ",b)
        # print("T: ",T)
        # print("Rated Lifetime: ", L/8766)
        
        return (L/8766)
    
    #! model given real current loading, constant load
    def lifetime_ContinuousLoading(self):
        #TODO: define what a and b to use given rated winding temp rated value
        b = math.log(2)/(1/(self.hotSpotWindingTemp_rated +273)- 1/(self.hotSpotWindingTemp_rated +273+6))
        a = math.e**(math.log(180000)-b/(self.hotSpotWindingTemp_rated+273))

        #TODO: Collect max winding temp data from {transformer.name}_average_metrics_hour
        table_name = f'''{self.name}_average_metrics_hour'''
        query = f'''SELECT * FROM "{table_name}" ORDER BY "DATETIME" ASC'''
        transformerData = pandas.read_sql_query(query, self.conn)
        transformerData['DATETIME'] = pandas.to_datetime(transformerData['DATETIME'])

        # Identify relevant columns: Hot spot data
        hsA, hsB, hsC = transformerData.columns[15:18]

        # Identify max hotspot
        transformerData['hotspot_temp_max'] = numpy.max(transformerData[[hsA, hsB, hsC]].values, axis=1)

        #TODO: Compute lifetime given data from phaseMax at given timestamp (ie, phase with largest recorded winding temp)
        lifetimeInHours = a*numpy.exp(b/(transformerData['hotspot_temp_max'] + 273.15))
        transformerData['Lifetime_Years'] = lifetimeInHours/8766

        #TODO: Push to SQLite DB
        transformerData.to_sql(
            name=f'''{self.name}_trialLifetimeData_Continuous1''',
            con=self.conn,
            if_exists="replace",
            chunksize=5000,
            method="multi",
            index=False
        )
        
        return

    #! Transient Loading (Load not constant):
    def lifetime_TransientLoading(self):
        #TODO: 1) Given Values/Data and associated splicing for dataframe object to list
        ambientTemp = 31.67 # taken manually from temperature gun, treated as average until more data is availible 
        q=1.6
        m=0.8
        
        #TODO: 4) Calculate Time Constant for Specific Load (per phase)
        
           
        #TODO: Initial hotspot temps for loading considerations for each phase
        # Z_test_aPhase = (avgWindingTemp_aPhase[rowNum]-ambientTemp)/(150*(avgLoadCurrent_aPhase[rowNum]/self.RatedCurrentLV)**(q))
        # Z_test_bPhase = (avgWindingTemp_bPhase[rowNum]-ambientTemp)/(150*(avgLoadCurrent_bPhase[rowNum]/self.RatedCurrentLV)**(q))
        # Z_test_cPhase = (avgWindingTemp_cPhase[rowNum]-ambientTemp)/(150*(avgLoadCurrent_cPhase[rowNum]/self.RatedCurrentLV)**(q))
        # Z_test_totalPhase = (avgWindingTemp_totalPhase[rowNum]-ambientTemp)/(150*(avgLoadCurrent_totalPhase[rowNum]/self.RatedCurrentLV)**(q))
        
        Z_test_aPhase = 1.2
        Z_test_bPhase = 1.2
        Z_test_cPhase = 1.2
        Z_test_totalPhase = 1.2

        d_vhs_a_phase_initial = Z_test_aPhase*self.avgWindingTempRise_rated*(avgLoadCurrent_aPhase[rowNum]/self.RatedCurrentLV)**(q)
        d_vhs_b_phase_initial = Z_test_bPhase*self.avgWindingTempRise_rated*(avgLoadCurrent_bPhase[rowNum]/self.RatedCurrentLV)**(q)
        d_vhs_c_phase_initial = Z_test_cPhase*self.avgWindingTempRise_rated*(avgLoadCurrent_cPhase[rowNum]/self.RatedCurrentLV)**(q)
        d_vhs_total_phase_initial = Z_test_totalPhase*self.avgWindingTempRise_rated*(avgLoadCurrent_totalPhase[rowNum]/self.RatedCurrentLV)**(q)

        #TODO: Final hotspot temps for loading considerations for each phase
        d_vhs_a_phase_final = Z_test_aPhase*self.avgWindingTempRise_rated*(avgLoadCurrent_aPhase[rowNum+1]/self.RatedCurrentLV)**(q)
        d_vhs_b_phase_final = Z_test_bPhase*self.avgWindingTempRise_rated*(avgLoadCurrent_bPhase[rowNum+1]/self.RatedCurrentLV)**(q)
        d_vhs_c_phase_final = Z_test_cPhase*self.avgWindingTempRise_rated*(avgLoadCurrent_cPhase[rowNum+1]/self.RatedCurrentLV)**(q)
        d_vhs_total_phase_final = Z_test_totalPhase*self.avgWindingTempRise_rated*(avgLoadCurrent_totalPhase[rowNum+1]/self.RatedCurrentLV)**(q)
        
        #TODO: Time Constant (per phase)
        d_vhsR = self.avgWindingTempRise_rated + 30
        tau_a_phase = self.ratedTimeConstant*((d_vhs_a_phase_final/d_vhsR)-(d_vhs_a_phase_initial/d_vhsR))/((d_vhs_a_phase_final/d_vhsR)**(1/m)-(d_vhs_a_phase_initial/d_vhsR)**(1/m))
        tau_b_phase = self.ratedTimeConstant*(((d_vhs_b_phase_final/d_vhsR)-(d_vhs_b_phase_initial/d_vhsR))/((d_vhs_b_phase_final/d_vhsR)**(1/m)-(d_vhs_b_phase_initial/d_vhsR)**(1/m)))
        tau_c_phase = self.ratedTimeConstant*(((d_vhs_c_phase_final/d_vhsR)-(d_vhs_c_phase_initial/d_vhsR))/((d_vhs_c_phase_final/d_vhsR)**(1/m)-(d_vhs_c_phase_initial/d_vhsR)**(1/m)))
        tau_total_phase = self.ratedTimeConstant*(((d_vhs_total_phase_final/d_vhsR)-(d_vhs_total_phase_initial/d_vhsR))/((d_vhs_total_phase_final/d_vhsR)**(1/m)-(d_vhs_total_phase_initial/d_vhsR)**(1/m)))

        #TODO: ultimate hot spot rise per phase, using a time period t of 24 hours (1 day). THis is now the d_vhs, since t > 5 tau (per IEC 60076-12)
        ultimateHotSpotRise_a_phase = (d_vhs_a_phase_final-d_vhs_a_phase_initial)/(1-math.e**(-24*dayNumber/tau_a_phase))+d_vhs_a_phase_initial
        ultimateHotSpotRise_b_phase = (d_vhs_b_phase_final-d_vhs_b_phase_initial)/(1-math.e**(-24*dayNumber/tau_b_phase))+d_vhs_b_phase_initial
        ultimateHotSpotRise_c_phase = (d_vhs_c_phase_final-d_vhs_c_phase_initial)/(1-math.e**(-24/tau_c_phase))+d_vhs_c_phase_initial
        ultimateHotSpotRise_total_phase = (d_vhs_total_phase_final-d_vhs_total_phase_initial)/(1-math.e**(-24/tau_total_phase))+d_vhs_total_phase_initial

        #TODO: Lifetime consumption in hours
        T_a_phase = 273 + ambientTemp + ultimateHotSpotRise_a_phase
        T_b_phase = 273 + ambientTemp + ultimateHotSpotRise_b_phase
        T_c_phase = 273 + ambientTemp + ultimateHotSpotRise_c_phase
        T_total_phase = 273 + ambientTemp + ultimateHotSpotRise_total_phase

        L_consumption_a_phase = 180000*dayNumber*24*(1/a)*math.e**(-b/T_a_phase)
        L_consumption_b_phase = 180000*dayNumber*24*(1/a)*math.e**(-b/T_b_phase)
        L_consumption_c_phase = 180000*dayNumber*24*(1/a)*math.e**(-b/T_c_phase)
        L_consumption_total_phase = 180000*dayNumber*24*(1/a)*math.e**(-b/T_total_phase)

        #TODO:(per phase) Append Lifetime Consumption, Thermodynamic Hot Spot Value to their respective lists




        #TODO: Go through each lifetime consumption list, convert to percentage of 180k hours and subtract from total. Append Result to lifetime percent list
    
        
        #TODO: Create excel file and graph from above approach
        
        return


    #--------------------------------FRONT-END----------------------------------#
    
    
