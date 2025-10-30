import pandas
import sqlite3
import time
from database import Database

def pseudo_server(database:Database):
    #TODO: Identify all transformers in program
    
    #TODO: import all last weeks data from all transformers
    lastWeeksData = pandas.read_excel("/home/eveuio/DataProcessing/IncompleteTransformerData_lastWeek/22A03_LastWeek.xlsx")

    #TODO: convert to pandas object from excel; for each index/timestamp in last week, insert single row to fullRange, print "inserted data, timestamp: " wait 10 min
    lastWeeksData['DATETIME'] = pandas.to_datetime(lastWeeksData['DATETIME'])
    lastWeeksData.index = lastWeeksData['DATETIME']

    #TODO: open connection to DB seperate from main to avoid corruption potential
    conn = sqlite3.connect(database.db_path, check_same_thread=True)  

    #TOD0:
    # --- Insert rows one by one ---
    for i in range(len(lastWeeksData)):
        #TODO: identify/slice row from dataFrame, append to fullRange table
        single_row = lastWeeksData.iloc[[i]]  # keep as DataFrame
        single_row.to_sql("22A03fullRange", conn, if_exists="append", index=False)

        #TODO: Print inserted row timestamp
        dt_value = single_row.index[0]  
        print(f"[{time.strftime('%H:%M:%S')}] Inserted row, datetime: {dt_value}")

        #TODO: wait 5 sec before inserting next datapoint
        if i < len(lastWeeksData) - 1:
            time.sleep(5)  
    return
   
    

    

    