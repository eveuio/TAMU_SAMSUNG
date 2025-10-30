# from transformerFunctions import Transformer
# from database import Database
# from workingCopyHSPredict import particle_filter_for_SVR, train_and_evaluate_svr, createDataSet
# # from pseudoServer import pseudo_server
# from sklearn.metrics import mean_squared_error, mean_absolute_error


# from datetime import datetime
# import threading
# import matplotlib.pyplot as plt
# import os
# import sys
# import logging

# #--------------------------------MAIN------------------------------------------------------------------#
# DB_PATH = os.path.abspath('transformerDB.db')
# database = Database(db_path=DB_PATH)

# # serverThread = threading.Thread(target= pseudo_server, args = (database,), daemon=False)
# # updateMetricsThread = threading.Thread(target=database.update_transformer_average_data, args = (), daemon = False)

# transformer22A03 = Transformer(name = "22A03",
#                                ratedCurrent_H=116,
#                                ratedVoltage_H=12470,
#                                ratedCurrent_L=3007,
#                                ratedVoltage_L=480,
#                                impedance=5.8,
#                                windingMaterial="Aluminum",
#                                thermalClass_rated=220,
#                                avgWindingTempRise_rated=115,
#                                weight_CoreAndCoil=5670,
#                                weight_total=6804,
#                                age=19,
#                                ratedKVA=2500,
#                                XR_Ratio=3.5,
#                                status = "new")

# database.addTransformer(transformer22A03)
# database.createAverageReport(transformer22A03)

# #?-------------------------------POPULATE HISTORICAL AVERAGE TABLES---------------------------------------------------------------------
# createDataSet(transformer22A03,database)

# #?-------------------------------------------------------------------------------------------------------------------------------------------



















































# #=========================GRAVEYARD=========================================================#
# # serverThread.start()
# # updateMetricsThread.start()


# # transformerEX02A3 = Transformer(name = "EX02A3",
# #                               ratedCurrent_H=1203,
# #                               ratedVoltage_H=208,
# #                               ratedCurrent_L=2776,
# #                               ratedVoltage_L=208,
# #                               impedance=6.73,
# #                               windingMaterial="Aluminum",
# #                               thermalClass_rated=220,
# #                               avgWindingTempRise_rated=115,
# #                               weight_CoreAndCoil=4423,
# #                               weight_total=5398,
# #                               database=database,
# #                               age=8,
# #                               ratedKVA=1000,
# #                               XR_Ratio=4.3)

# # transformer22B01 = Transformer(name = "22B01",
# #                               ratedCurrent_H=116,
# #                               ratedVoltage_H=12470,
# #                               ratedCurrent_L=3007,
# #                               ratedVoltage_L=480,
# #                               impedance=5.76,
# #                               windingMaterial="Aluminum",
# #                               thermalClass_rated=220,
# #                               avgWindingTempRise_rated=115,
# #                               weight_CoreAndCoil=5670,
# #                               weight_total=6804,
# #                               database=database,
# #                               age=19,
# #                               ratedKVA=2500,
# #                               XR_Ratio=3.4
# #                               )


# # transformer21A05 = Transformer(name = "21A05",
# #                               ratedCurrent_H=69.4,
# #                               ratedVoltage_H=12470,
# #                               ratedCurrent_L=1804,
# #                               ratedVoltage_L=480,
# #                               impedance=5.82,
# #                               windingMaterial="Aluminum",
# #                               thermalClass_rated=220,
# #                               avgWindingTempRise_rated=115,
# #                               weight_CoreAndCoil=3538,
# #                               weight_total=4536,
# #                               database=database,
# #                               age=19,
# #                               ratedKVA=1500,
# #                               XR_Ratio=3.5)
# #?---------------------------------------------------------------------------------------------------------

# # database.addTransformer(transformer22B01)
# # database.addTransformer(transformerEX02A3)
# # database.addTransformer(transformer22A03)
# # database.addTransformer(transformer21A05)

# # createDataSets(transformerEX02A3, database)
# # createDataSets(transformer22B01, database)
# # createDataSets(transformer22A03, database)
# # createDataSets(transformer21A05, database)

# # database.addTransformer(transformer22B01)
# # database.addTransformer(transformerEX02A3)
# # database.addTransformer(transformer22A03)
# # database.addTransformer(transformer21A05)

# # createDataSets(transformerEX02A3, database)
# # createDataSets(transformer22B01, database)
# # createDataSets(transformer22A03, database)
# # createDataSets(transformer21A05, database)

# # transformer22B01.createAverageReport()
# # transformerEX02A3.createAverageReport()
# # transformer22A03.createAverageReport()
# # transformer21A05.createAverageReport()

# # database.removeTransformer(transformerEX02A3)


