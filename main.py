from transformerFunctions import Transformer
from transformerFunctions import avgAmbientTemp
from hotSpotPrediction import createDataSets
from database import Database
from pseudoServer import pseudo_server

from datetime import datetime
import pandas
import numpy
import threading

#--------------------------------MAIN------------------------------------------------------------------#
database = Database(dbpath="/home/eveuio/DataProcessing/transformerDB")

serverThread = threading.Thread(target= pseudo_server, args = (database,), daemon=False)
updateMetricsThread = threading.Thread(target=database.update_transformer_average_data, args = (), daemon = False)

transformer22A03 = Transformer(name = "22A03",
                               ratedCurrent_H=116,
                               ratedVoltage_H=12470,
                               ratedCurrent_L=3007,
                               ratedVoltage_L=480,
                               impedance=5.8,
                               windingMaterial="Aluminum",
                               thermalClass_rated=220,
                               avgWindingTempRise_rated=115,
                               weight_CoreAndCoil=5670,
                               weight_total=6804,
                               database=database,
                               age=19,
                               ratedKVA=2500,
                               XR_Ratio=3.5)
database.addTransformer(transformer22A03)
transformer22A03.createAverageReport()

serverThread.start()
updateMetricsThread.start()






# transformerEX02A3 = Transformer(name = "EX02A3",
#                               ratedCurrent_H=1203,
#                               ratedVoltage_H=208,
#                               ratedCurrent_L=2776,
#                               ratedVoltage_L=208,
#                               impedance=6.73,
#                               windingMaterial="Aluminum",
#                               thermalClass_rated=220,
#                               avgWindingTempRise_rated=115,
#                               weight_CoreAndCoil=4423,
#                               weight_total=5398,
#                               database=database,
#                               age=8,
#                               ratedKVA=1000,
#                               XR_Ratio=4.3)

# transformer22B01 = Transformer(name = "22B01",
#                               ratedCurrent_H=116,
#                               ratedVoltage_H=12470,
#                               ratedCurrent_L=3007,
#                               ratedVoltage_L=480,
#                               impedance=5.76,
#                               windingMaterial="Aluminum",
#                               thermalClass_rated=220,
#                               avgWindingTempRise_rated=115,
#                               weight_CoreAndCoil=5670,
#                               weight_total=6804,
#                               database=database,
#                               age=19,
#                               ratedKVA=2500,
#                               XR_Ratio=3.4
#                               )


# transformer21A05 = Transformer(name = "21A05",
#                               ratedCurrent_H=69.4,
#                               ratedVoltage_H=12470,
#                               ratedCurrent_L=1804,
#                               ratedVoltage_L=480,
#                               impedance=5.82,
#                               windingMaterial="Aluminum",
#                               thermalClass_rated=220,
#                               avgWindingTempRise_rated=115,
#                               weight_CoreAndCoil=3538,
#                               weight_total=4536,
#                               database=database,
#                               age=19,
#                               ratedKVA=1500,
#                               XR_Ratio=3.5)
#?---------------------------------------------------------------------------------------------------------

# database.addTransformer(transformer22B01)
# database.addTransformer(transformerEX02A3)
# database.addTransformer(transformer22A03)
# database.addTransformer(transformer21A05)

# createDataSets(transformerEX02A3, database)
# createDataSets(transformer22B01, database)
# createDataSets(transformer22A03, database)
# createDataSets(transformer21A05, database)

#?-------------------------------POPULATE HISTORICAL AVERAGE TABLES---------------------------------------------------------------------
# transformer22B01.createAverageReport()
# transformerEX02A3.createAverageReport()
transformer22A03.createAverageReport()
# transformer21A05.createAverageReport()



#? -----------------------------TEST-FUNCTIONS-----------------------------------#

#! Check rated values of lifetime given ambient temp
# checkRatedLifetime(transformerB01)

#! Check start values of Secondary voltage, vTHD and winding temp
# checkAveragesEndCases(dataType="secondary_voltage",database=database)
# checkAveragesEndCases(dataType="vTHD",database=database)
# checkAveragesEndCases(dataType="winding_temp",database=database)

#! Check 
#?---------------------------Delete-Transformer----------------------------------#
# database.removeTransformer(transformerEX02A3)










