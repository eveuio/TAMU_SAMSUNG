import math
import numpy
from datetime import date
import pandas
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import sqlite3
import os

DB_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'transformerDB.db')) # database file in TAMU_SAMSUNG/transformerDB.db




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
        