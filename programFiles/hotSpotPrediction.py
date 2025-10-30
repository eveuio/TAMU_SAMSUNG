from src.transformerFunctions import Transformer
from src.database import Database
import pandas
import numpy
#! Return temperature output from predictive model plus error margin; a list
def predictHotSpot(transformer, loadCurrent):
    #TODO: Create variables
    estimatedHSTemp = 0.0
    errorMargin = 0.0

    #TODO: Call input to prediction, store in variables


    return [estimatedHSTemp,errorMargin]

#! Create datasets for model training. Needs to be at least 100 points (16 hours), with enough variations in load and winding temperature. 
def createDataSets(transformer:Transformer, database:Database):
    #TODO: Identify location of transformer data table and store locally in dataframe, table should be labelled "{transformer.name}fullDataRange"
    table_name= transformer.name+"fullRange"
    transformerData = pandas.read_sql_query(f'''SELECT * FROM "{table_name}"''',database.dbconnect)
    transformerData['DATETIME'] = pandas.to_datetime(transformerData['DATETIME'])
    # fullDateRange = transformerData['DATETIME'].iloc[1:-1].tolist()

    #TODO: Precalculate RMS current, RMS voltage and ambient temp for all Timestamps, add to transformerData Dataframe:
    voltageA = transformerData.columns[4]
    voltageB = transformerData.columns[5]
    voltageC = transformerData.columns[6]

    currentA = transformerData.columns[7]
    currentB = transformerData.columns[8]
    currentC = transformerData.columns[9]

    transformerData['I_RMS']= numpy.sqrt((transformerData[currentA]**2+transformerData[currentB]**2+transformerData[currentC]**2)/3)
    transformerData['V_RMS']= numpy.sqrt((transformerData[voltageA]**2+transformerData[voltageB]**2+transformerData[voltageC]**2)/3)
    transformerData['T_ambient'] = avgAmbientTemp(transformerData['I_RMS']/transformer.RatedCurrentLV)

    #TODO: Define iteration interval, needs to be at least 2-5x time constant and a whole number (ex: 2h tau gives 10 hour interval, which is 60 datapoint)
    # iterationInterval = 2*round(transformer.ratedTimeConstant)*6
    #TODO: Evaluate which ones for 70/15/15, return partitioned list with dataset number. For now, choronological. Also skip 5x time constants between sets to ensure no leakage
    
    transformerData.to_sql(name= f'''{transformer.name}_trainingData''',con=database.dbconnect,if_exists = "replace",chunksize=5000,method ="multi")
    
    return 

#! Calculate Ambient Temp given load percentage:
def avgAmbientTemp(loadCurrentPercent):
    #TODO: lightly loaded transformers 10 degrees above ambient room temp, heavily loaded transformers 30 degrees above ambient room temp
    ambientTemp = 23.8889 + 10 + loadCurrentPercent*(40.556-23.8889)
    return ambientTemp


    logC, logGamma, epsilon = params
    C = 10**logC
    gamma = 10**logGamma
    model = SVR(kernel='rbf', C=C, gamma=gamma, epsilon=epsilon)
    model.fit(X_train, y_train)
    preds = model.predict(X_train)
    mse = mean_squared_error(y_train, preds)
    likelihood = np.exp(-mse)
    return mse, likelihood


