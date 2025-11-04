# # from .transformerFunctions import Transformer
# from database import Database
# # # from workingCopyHSPredict import particle_filter_for_SVR, train_and_evaluate_svr, createDataSet
# # # from pseudoServer import pseudo_server

# # from TAMU_SAMSUNG.machinelearning.transformer_health_monitor import TransformerHealthMonitor
# # from sklearn.metrics import mean_squared_error, mean_absolute_error
# from transformerFunctions import Transformer

# from datetime import datetime
# # import threading
# import matplotlib.pyplot as plt
# import os
# import sys
# import logging
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# # #--------------------------------MAIN------------------------------------------------------------------#
# #TODO: get transformerDB path from 2 folders up (ie TAMU_SAMSUNG/transformerDB.db)
# DB_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'transformerDB.db'))
# print(DB_PATH)
# #TODO: Add connection to DB
# database = Database(db_path=DB_PATH)


# transformer22A03 = Transformer(name = "22A03",
                                # ratedVoltage_H=12470,
#                                ratedCurrent_H=116,
#                         
#                                ratedCurrent_L=3007,
#                                ratedVoltage_L=480,
#                                impedance=5.8,
#                                windingMaterial="Aluminum",
#                                thermalClass_rated=180,
#                                avgWindingTempRise_rated=115,
#                                weight_CoreAndCoil=5670,
#                                weight_total=6804,
#                                ratedKVA=2500,
#                                manufactureDate=2006,
#                                XR_Ratio=3.5,
#                                status = "new")

# database.addTransformer(transformer=transformer22A03,testingMode=True)
# database.createAverageReport(transformer22A03)
# transformer22A03.lifetime_ContinuousLoading()
# transformer22A03.lifetime_TransientLoading()




# # #?-------------------------------POPULATE HISTORICAL AVERAGE TABLES---------------------------------------------------------------------
# # # createDataSet(transformer22A03,database)

# # #?-------------------------------------------------------------------------------------------------------------------------------------------



















































# # #=========================GRAVEYARD=========================================================#
# # # serverThread.start()
# # # updateMetricsThread.start()


# # # transformerEX02A3 = Transformer(name = "EX02A3",
# # #                               ratedCurrent_H=1203,
# # #                               ratedVoltage_H=208,
# # #                               ratedCurrent_L=2776,
# # #                               ratedVoltage_L=208,
# # #                               impedance=6.73,
# # #                               windingMaterial="Aluminum",
# # #                               thermalClass_rated=220,
# # #                               avgWindingTempRise_rated=115,
# # #                               weight_CoreAndCoil=4423,
# # #                               weight_total=5398,
# # #                               database=database,
# # #                               age=8,
# # #                               ratedKVA=1000,
# # #                               XR_Ratio=4.3)

# # # transformer22B01 = Transformer(name = "22B01",
# # #                               ratedCurrent_H=116,
# # #                               ratedVoltage_H=12470,
# # #                               ratedCurrent_L=3007,
# # #                               ratedVoltage_L=480,
# # #                               impedance=5.76,
# # #                               windingMaterial="Aluminum",
# # #                               thermalClass_rated=220,
# # #                               avgWindingTempRise_rated=115,
# # #                               weight_CoreAndCoil=5670,
# # #                               weight_total=6804,
# # #                               database=database,
# # #                               age=19,
# # #                               ratedKVA=2500,
# # #                               XR_Ratio=3.4
# # #                               )


# # # transformer21A05 = Transformer(name = "21A05",
# # #                               ratedCurrent_H=69.4,
# # #                               ratedVoltage_H=12470,
# # #                               ratedCurrent_L=1804,
# # #                               ratedVoltage_L=480,
# # #                               impedance=5.82,
# # #                               windingMaterial="Aluminum",
# # #                               thermalClass_rated=220,
# # #                               avgWindingTempRise_rated=115,
# # #                               weight_CoreAndCoil=3538,
# # #                               weight_total=4536,
# # #                               database=database,
# # #                               age=19,
# # #                               ratedKVA=1500,
# # #                               XR_Ratio=3.5)
# # #?---------------------------------------------------------------------------------------------------------

# # # database.addTransformer(transformer22B01)
# # # database.addTransformer(transformerEX02A3)
# # # database.addTransformer(transformer22A03)
# # # database.addTransformer(transformer21A05)

# # # createDataSets(transformerEX02A3, database)
# # # createDataSets(transformer22B01, database)
# # # createDataSets(transformer22A03, database)
# # # createDataSets(transformer21A05, database)

# # # database.addTransformer(transformer22B01)
# # # database.addTransformer(transformerEX02A3)
# # # database.addTransformer(transformer22A03)
# # # database.addTransformer(transformer21A05)

# # # createDataSets(transformerEX02A3, database)
# # # createDataSets(transformer22B01, database)
# # # createDataSets(transformer22A03, database)
# # # createDataSets(transformer21A05, database)

# # # transformer22B01.createAverageReport()
# # # transformerEX02A3.createAverageReport()
# # # transformer22A03.createAverageReport()
# # # transformer21A05.createAverageReport()

# # # database.removeTransformer(transformerEX02A3)


