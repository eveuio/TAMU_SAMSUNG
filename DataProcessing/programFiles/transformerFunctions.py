
import math
import numpy
from datetime import datetime
import pandas
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import sqlite3
import os


DB_PATH = os.path.abspath('transformerDB.db') # in lieu of preventing "circular definitions" or something like that

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
        self.age = manufactureDate - datetime.year()
        self.XR_Ratio = 6

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

        #TODO: Collect max winding temp data from 

        #TODO: Compute lifetime given data from phaseMax at given timestamp (ie, phase with largest recorded winding temp)
        
        return

    #! Transient Loading (Load not constant):
    def lifetime_TransientLoading(self):
        #TODO: 1) Given Values/Data and associated splicing for dataframe object to list
        ambientTemp = 31.67 # taken manually from temperature gun, treated as average until more data is availible 
        q=1.6
        m=0.8

        #Determine number of days to average, chosen based on 4-5x time constant
        dayNumber = 1

        avgLoadCurrent = self.avgLoadCurrent(dayNumber)
        avgWindingTemp =self.avgWindingTemp(dayNumber)
        b = math.log(2)/(1/(self.hotSpotWindingTemp_rated +273)- 1/(self.hotSpotWindingTemp_rated +273+6))
        a = math.e**(math.log(180000)-b/(self.hotSpotWindingTemp_rated+273))
        
        dates = avgLoadCurrent.iloc[:, 0]                      
        avgLoadCurrent_aPhase= avgLoadCurrent.iloc[:, 1]          
        avgLoadCurrent_bPhase = avgLoadCurrent.iloc[:, 2]         
        avgLoadCurrent_cPhase = avgLoadCurrent.iloc[:, 3]         
        avgLoadCurrent_totalPhase = avgLoadCurrent.iloc[:, 4]  

        avgWindingTemp_aPhase = avgWindingTemp.iloc[:, 1]   
        avgWindingTemp_bPhase = avgWindingTemp.iloc[:, 2]         
        avgWindingTemp_cPhase = avgWindingTemp.iloc[:, 3]         
        avgWindingTemp_totalPhase = avgWindingTemp.iloc[:, 4]  
        
        #TODO: 4) Calculate Time Constant for Specific Load (per phase)
        T_a_phase = 0
        T_b_phase = 0
        T_c_phase = 0
        T_total_phase = 0

        L_consumption_a_phase = 0.0
        L_consumption_b_phase = 0.0
        L_consumption_c_phase = 0.0
        L_consumption_total_phase = 0.0

        T_aPhase_L = []
        T_bPhase_L = []
        T_cPhase_L = []
        T_totalPhase_L = []

        lifetimeConsumption_aPhase = []
        lifetimeConsumption_bPhase = []
        lifetimeConsumption_cPhase = []
        lifetimeConsumption_totalPhase = []
        
        rowNum = 0
        for item in range(len(dates)-1):

            if (not any([
                    avgLoadCurrent_aPhase[rowNum],
                    avgLoadCurrent_bPhase[rowNum],
                    avgLoadCurrent_cPhase[rowNum],
                    avgLoadCurrent_totalPhase[rowNum]]
                    ) or ( 
                    any(value == 0 for value in [
                    avgWindingTemp_aPhase[rowNum],
                    avgWindingTemp_bPhase[rowNum],
                    avgWindingTemp_cPhase[rowNum],
                    avgWindingTemp_totalPhase[rowNum],
                    avgLoadCurrent_aPhase[rowNum],
                    avgLoadCurrent_bPhase[rowNum],
                    avgLoadCurrent_cPhase[rowNum],
                    avgLoadCurrent_totalPhase[rowNum]]
                    )
                    )):
                
                lifetimeConsumption_aPhase.append(0)
                lifetimeConsumption_bPhase.append(0)
                lifetimeConsumption_cPhase.append(0)
                lifetimeConsumption_totalPhase.append(0)

                T_aPhase_L.append(273+ambientTemp)
                T_bPhase_L.append(273+ambientTemp)
                T_cPhase_L.append(273+ambientTemp)
                T_totalPhase_L.append(273+ambientTemp)

            else:
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
                T_aPhase_L.append(T_a_phase)
                T_bPhase_L.append(T_b_phase)
                T_cPhase_L.append(T_c_phase)
                T_totalPhase_L.append(T_total_phase) 

                lifetimeConsumption_aPhase.append(L_consumption_a_phase)
                lifetimeConsumption_bPhase.append(L_consumption_b_phase)
                lifetimeConsumption_cPhase.append(L_consumption_c_phase)
                lifetimeConsumption_totalPhase.append(L_consumption_total_phase)


            databaseInsert = pandas.DataFrame([[dates[rowNum],
                                            avgLoadCurrent_aPhase[rowNum],
                                            avgLoadCurrent_bPhase[rowNum],
                                            avgLoadCurrent_cPhase[rowNum],
                                            avgLoadCurrent_totalPhase[rowNum],
                                            T_a_phase,
                                            T_b_phase,
                                            T_c_phase,
                                            T_total_phase,
                                            L_consumption_a_phase,
                                            L_consumption_b_phase,
                                            L_consumption_c_phase,
                                            L_consumption_total_phase]])
            
            self.db.insertData(transformerName=self.name,dataType="lifetime_transient",dataSet=databaseInsert,iterator="")
            rowNum+=1
        
        #TODO: Create Excel Sheet for all data, testing purposes
        headers_lifetimeMetrics_all = ['Date/Time','A Phase Load Current', 'B Phase Load Current', 'C Phase Load Current', 'Total Phase Load Current','A Phase ThermoD_HotSpot', 'B Phase ThermoD_HotSpot', 'C Phase ThermoD_HotSpot','Total Phase ThermoD_HotSpot', 'A Phase Lifetime Consumption', 'B Phase Lifetime Consumption', 'C Phase Lifetime Consumption','Total Phase Lifetime Consumption']
        headers_lifetimeMetrics_lifetimeConsumption =['Date/Time','A Phase Lifetime Consumption', 'B Phase Lifetime Consumption', 'C Phase Lifetime Consumption', 'Total Phase Lifetime Consumption']
        headers_lifetimeMetrics_thermoD_hotSpot = ['Date/Time','A Phase Ultimate Thermodynamic Hot Spot', 'B Phase Ultimate Thermodynamic Hot Spot', 'C Phase Ultimate Thermodynamic Hot Spot', 'Total Phase Ultimate Thermodynamic Hot Spot']
        
        lifetimeMetrics_all = pandas.DataFrame([dates.to_list(),avgLoadCurrent_aPhase.to_list(),avgLoadCurrent_bPhase.to_list(), avgLoadCurrent_cPhase.to_list(), avgLoadCurrent_totalPhase.to_list(),T_aPhase_L, T_bPhase_L, T_cPhase_L, T_totalPhase_L,lifetimeConsumption_aPhase, lifetimeConsumption_bPhase, lifetimeConsumption_cPhase, lifetimeConsumption_totalPhase]).T
        lifetimeMetrics_thermoD_hotSpot = pandas.DataFrame([dates.to_list(), T_aPhase_L, T_bPhase_L, T_cPhase_L, T_totalPhase_L]).T
        lifetimeMetrics_lifetimeConsumption = pandas.DataFrame([dates.to_list(), lifetimeConsumption_aPhase, lifetimeConsumption_bPhase, lifetimeConsumption_cPhase, lifetimeConsumption_totalPhase]).T
        
        lifetimeMetrics_all.columns = headers_lifetimeMetrics_all
        lifetimeMetrics_thermoD_hotSpot.columns = headers_lifetimeMetrics_thermoD_hotSpot
        lifetimeMetrics_lifetimeConsumption.columns = headers_lifetimeMetrics_lifetimeConsumption
        
        filename_allMetrics = "/home/eveuio/DataProcessing/lifeTimeData/Transient/"+self.name+"/lifeTimeMetrics_all.xlsx"
        filename_thermoD_hotSpot = "/home/eveuio/DataProcessing/lifeTimeData/Transient/"+self.name+"/thermoD_hotSpot.xlsx"
        filename_lifetimeConsumptionOnly = "/home/eveuio/DataProcessing/lifeTimeData/Transient/"+self.name+"/lifetimeMetrics_lifetimeConsumption.xlsx"
        
        
        #TODO: Create individual excel sheets for graphs
        lifetimeMetrics_all.to_excel(filename_allMetrics,index=False)
        lifetimeMetrics_thermoD_hotSpot.to_excel(filename_thermoD_hotSpot,index=False)
        lifetimeMetrics_lifetimeConsumption.to_excel(filename_lifetimeConsumptionOnly,index=False)

        #TODO: Graphs for Visual Validation:
        output_path2 = filename_lifetimeConsumptionOnly[:-5]+"_graph.png"
        output_path3 = filename_thermoD_hotSpot[:-5]+"_graph.png"

        self.createAverageGraph(filename_thermoD_hotSpot, "Thermodynamic Hot Spot Temp", " Temperature (Kelvin)", output_path3)
        self.createAverageGraph(filename_lifetimeConsumptionOnly, "Lifetime Consumption", "h/h", output_path2)


        #TODO: Create Graph and Sheet for Lifetime Consumption as a remaining percentage of 180,000 h
        # IEEE recommends accounting for 1% lifeloss per year to potential overloading, starting percent = 100 - self.age at beginning of data
        startingPercent = self.age - (2025-int(dates[0][0:4]))

        overallLifetimeConsumption_a_phase = 100.0 - startingPercent
        overallLifetimeConsumption_b_phase = 100.0 - startingPercent
        overallLifetimeConsumption_c_phase = 100.0 - startingPercent
        overallLifetimeConsumption_total_phase = 100.0 - startingPercent

        currentYear = int(dates[0][0:4])

        lifetimePercent_a_L =[]
        lifetimePercent_b_L =[]
        lifetimePercent_c_L =[]
        lifetimePercent_total_L =[]

        # Append starting percent to each respective list:
        lifetimePercent_a_L.append(overallLifetimeConsumption_a_phase)
        lifetimePercent_b_L.append(overallLifetimeConsumption_b_phase)
        lifetimePercent_c_L.append(overallLifetimeConsumption_c_phase)
        lifetimePercent_total_L.append(overallLifetimeConsumption_total_phase)


        #TODO: Go through each lifetime consumption list, convert to percentage of 180k hours and subtract from total. Append Result to lifetime percent list
        for row in range(len(dates)-1):

            #1 percent = 1800h/180,000h, divide each consumption by 1800
            if (int(dates[row][0:4])!= currentYear):
                overallLifetimeConsumption_a_phase = overallLifetimeConsumption_a_phase - lifetimeConsumption_aPhase[row]/1800 - 1
                overallLifetimeConsumption_b_phase = overallLifetimeConsumption_b_phase - lifetimeConsumption_bPhase[row]/1800 - 1 
                overallLifetimeConsumption_c_phase = overallLifetimeConsumption_c_phase - lifetimeConsumption_cPhase[row]/1800 - 1
                overallLifetimeConsumption_total_phase = overallLifetimeConsumption_total_phase - lifetimeConsumption_totalPhase[row]/1800 - 1
                currentYear += 1
            else:
                overallLifetimeConsumption_a_phase = overallLifetimeConsumption_a_phase - lifetimeConsumption_aPhase[row]/1800
                overallLifetimeConsumption_b_phase = overallLifetimeConsumption_b_phase - lifetimeConsumption_bPhase[row]/1800
                overallLifetimeConsumption_c_phase = overallLifetimeConsumption_c_phase - lifetimeConsumption_cPhase[row]/1800
                overallLifetimeConsumption_total_phase = overallLifetimeConsumption_total_phase - lifetimeConsumption_totalPhase[row]/1800

            lifetimePercent_a_L.append(overallLifetimeConsumption_a_phase)
            lifetimePercent_b_L.append(overallLifetimeConsumption_b_phase)
            lifetimePercent_c_L.append(overallLifetimeConsumption_c_phase)
            lifetimePercent_total_L.append(overallLifetimeConsumption_total_phase)
        
        #TODO: Create excel file and graph from above approach
        filename_totalPercentConsumption = "/home/eveuio/DataProcessing/lifeTimeData/Transient/"+self.name+"/lifetimePercentConsumption.xlsx"

        lifetimePercentConsumption = pandas.DataFrame([dates.to_list(), lifetimePercent_a_L, lifetimePercent_b_L, lifetimePercent_c_L, lifetimePercent_total_L]).T
        headers_lifetimeConsumption =['Date/Time','A Phase Lifetime Consumption', 'B Phase Lifetime Consumption', 'C Phase Lifetime Consumption', 'Total Phase Lifetime Consumption']
        lifetimePercentConsumption.columns = headers_lifetimeConsumption

        lifetimePercentConsumption.to_excel(filename_totalPercentConsumption,index=False)

        output_path4 = filename_totalPercentConsumption[:-5]+"_graph.png"

        self.createAverageGraph(filename_totalPercentConsumption, "Lifetime Consumption as a Percentage of Total", "Percent", output_path4)
    


        
    #----------------------------AVERAGING FUNCTIONS FOR DATA PROCESSING-----------------------#
    
    #! Average Load Current:
    def avgLoadCurrent(self, dayNumber):
        transformerData = self.transformerData.active
        dates = [cell.value for cell in transformerData['A'][4:-1]]
        
        newDates = []
        a_phase_averages = []
        b_phase_averages = []
        c_phase_averages = []
        total_phase_averages = []

        a_phase_total = 0.0
        b_phase_total = 0.0
        c_phase_total = 0.0

        a_phase_average = 0.0
        b_phase_average = 0.0
        c_phase_average = 0.0
        total_phase_average = 0.0

        a_Phase_current = 0.0
        b_Phase_current = 0.0
        c_Phase_current = 0.0
        validRowCount = 0

        rowNum = 0
        maxRow = transformerData.max_row - 1 
        dayCounter = 1

        for row in transformerData.iter_rows(min_row = 5, max_row = maxRow, min_col=5, max_col= 7):
            
            a_Phase_current = row[0].value
            b_Phase_current = row[1].value
            c_Phase_current = row[2].value

            #need to skip over rows that have no value or null
            if ((a_Phase_current != None) and (b_Phase_current != None) and (c_Phase_current!= None)):
                    if((a_Phase_current != 0) and (b_Phase_current != 0) and (c_Phase_current!= 0)):
                        a_phase_total += a_Phase_current
                        b_phase_total += b_Phase_current
                        c_phase_total += c_Phase_current
                        validRowCount+=1
           
            
            if ((row[0].row == maxRow) or (dates[rowNum].hour == 23 and dates[rowNum].minute == 50)):
                #calculate average per phase, total phase and then append to respective list
                
                if((dayCounter == dayNumber) or (row[0].row == maxRow)):
                    # print("Check: ",row[0].row == maxRow)
                    if(a_phase_total != 0 and b_phase_total and c_phase_total != 0):
                        a_phase_average =a_phase_total/validRowCount
                        b_phase_average =b_phase_total/validRowCount
                        c_phase_average =c_phase_total/validRowCount
                        total_phase_average =(a_phase_total+b_phase_total+c_phase_total)/(3*validRowCount)
                    
                    a_phase_averages.append(a_phase_average)
                    b_phase_averages.append(b_phase_average)
                    c_phase_averages.append(c_phase_average)
                    total_phase_averages.append(total_phase_average)

                    newDates.append(dates[rowNum])
                
                    #set totals and counters back to zero
                    a_Phase_current = 0.0
                    b_Phase_current = 0.0
                    c_Phase_current = 0.0

                    dayCounter = 1
                    validRowCount = 0
                    a_phase_total = 0.0
                    b_phase_total = 0.0
                    c_phase_total = 0.0
                else:
                    dayCounter+=1
            rowNum+= 1
        
        dateStrings =[dt.strftime("%Y-%m-%d %H:%M:%S") for dt in newDates]
        avgLoadCurrentList = pandas.DataFrame([dateStrings,a_phase_averages,b_phase_averages,c_phase_averages,total_phase_averages]).T
        
        headers = ['Date/Time', 'A Phase', 'B Phase', 'C Phase','Total Phase']
    
        avgLoadCurrentList.columns = headers
        filename = "/home/eveuio/DataProcessing/lifeTimeData/Continuous/"+self.name+"/loadCurrent.xlsx"
        avgLoadCurrentList.to_excel(filename,index=False)
        
        self.createAverageGraph(filename,"Load Current (Daily)","Load Current(A)", output_path = filename[:-5]+"_graph.png")
        return avgLoadCurrentList

    #! Average Winding Temp
    def avgWindingTemp(self,dayNumber):
        # transformerData = self.transformerData.active
        ambientTemp = 31.67
        table_name= self.name+"fullRange"
        transformerData = pandas.read_sql_query(f'''SELECT * FROM {table_name}''',self.db.conn)
        # dates = [cell.value for cell in transformerData['A'][4:-1]]
        dates = transformerData['A'].iloc[4:-1].tolist()
        newDates = []
        a_phase_averages = []
        b_phase_averages = []
        c_phase_averages = []
        total_phase_averages = []

        a_phase_total = 0.0
        b_phase_total = 0.0
        c_phase_total = 0.0

        a_Phase_wTemp = 0.0
        b_Phase_wTemp = 0.0
        c_Phase_wTemp = 0.0

        a_phase_average = 0.0
        b_phase_average = 0.0
        c_phase_average = 0.0
        total_phase_average =0.0
        validRowCount = 0

        rowNum = 0
        maxRow = transformerData.max_row -1
        dayCounter = 1

        for row in transformerData.iter_rows(min_row = 5, max_row = maxRow, min_col=12, max_col= 14):

            a_Phase_wTemp = row[0].value
            b_Phase_wTemp = row[1].value
            c_Phase_wTemp = row[2].value

            # validRowCount += 1
            if ((a_Phase_wTemp != None) and (b_Phase_wTemp != None) and (c_Phase_wTemp!= None)):
                    if((a_Phase_wTemp != 0) and (b_Phase_wTemp != 0) and (c_Phase_wTemp!= 0)):
                        a_phase_total += a_Phase_wTemp
                        b_phase_total += b_Phase_wTemp
                        c_phase_total += c_Phase_wTemp
                        validRowCount+=1
            
                

            if ((row[0].row == maxRow) or (dates[rowNum].hour == 23 and dates[rowNum].minute == 50)):
                #calculate average per phase, total phase and then append to respective list
                
                if((dayCounter == dayNumber)):
                    if(a_phase_total != 0 and b_phase_total != 0 and c_phase_total != 0):
                        a_phase_average=a_phase_total/validRowCount
                        b_phase_average=b_phase_total/validRowCount
                        c_phase_average=c_phase_total/validRowCount
                        total_phase_average=(a_phase_total+b_phase_total+c_phase_total)/(3*validRowCount)
                    
                    a_phase_averages.append(a_phase_average)
                    b_phase_averages.append(b_phase_average)
                    c_phase_averages.append(c_phase_average)
                    total_phase_averages.append(total_phase_average)
                    newDates.append(dates[rowNum])
                
                    #set totals and counters back to zero
                    a_Phase_wTemp = 0.0
                    b_Phase_wTemp = 0.0
                    c_Phase_wTemp = 0.0

                    dayCounter = 1
                    validRowCount = 0
                    a_phase_total = 0.0
                    b_phase_total = 0.0
                    c_phase_total = 0.0
    
                else:
                    dayCounter += 1
                

            # increase row number on current date, set previous date to current date
            rowNum+= 1

        dateStrings =[dt.strftime("%Y-%m-%d %H:%M:%S") for dt in newDates]
        avgLoadCurrentList = pandas.DataFrame([dateStrings,a_phase_averages,b_phase_averages,c_phase_averages,total_phase_averages]).T

        headers = ['Date/Time', 'A Phase', 'B Phase', 'C Phase','Total Phase']
        
        avgLoadCurrentList.columns = headers
        
        filename = "/home/eveuio/DataProcessing/lifeTimeData/Continuous/"+self.name+"/avgWindingTemp.xlsx"
        avgLoadCurrentList.to_excel(filename,index=False)

        self.createAverageGraph(filename, "Average Winding Temperature (C)", "Temperature (C)", output_path = filename[:-5]+"_graph.png")

        return avgLoadCurrentList


    #--------------------------------FRONT-END----------------------------------#
    
    #! Used to create average graphs of given filename; used by createAvgReport
    def createAverageGraph(self, filename, title, y_axis, output_path = ""):
        sheet = pandas.read_excel(filename)
        
        sheet.iloc[:,0] = pandas.to_datetime(sheet.iloc[:,0])
        x_val = sheet.iloc[:,0]

        #treat NAN in dataset as a hole and not plot them
        for col in sheet.columns[1:]:
            sheet[col] = pandas.to_numeric(sheet[col], errors='coerce')

        plt.figure(figsize=(10,5))
        
        count = 1  # Assuming 'count' is initialized outside this loop, to track columns
        num_columns = len(sheet.columns)
        
        #plot values
        for column in sheet.columns[1:]:
            # Check if there are more than 2 columns and if it's not the last column
            sheet[column] = pandas.to_numeric(sheet[column], errors='coerce')

            # Filter out rows where either x_val or y is missing
            valid_mask = x_val.notna() & sheet[column].notna()

            if num_columns > 2 and column != sheet.columns[-1]:
                plt.plot(x_val[valid_mask], sheet[column][valid_mask],
                        label=column, linestyle='--', alpha=0.5)
            else:
                plt.plot(x_val[valid_mask], sheet[column][valid_mask],
                        label=column, linestyle='-')
            count += 1
        
        plt.xlabel(sheet.columns[0])
        plt.ylabel(y_axis)
        plt.title(title)
        plt.legend()

        plt.ticklabel_format(useOffset=False, style='plain', axis='y')
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d %H:%M"))  # Customize format if needed
        plt.gca().xaxis.set_major_locator(mdates.AutoDateLocator())  # Automatically adjust the ticks
        plt.xticks(rotation=45)

        plt.grid(True)
        plt.savefig(output_path, dpi = 300, bbox_inches='tight')
        plt.close()



