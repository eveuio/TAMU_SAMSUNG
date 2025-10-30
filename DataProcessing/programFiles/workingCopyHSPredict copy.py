import numpy
import pandas
import sqlite3
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import GridSearchCV, KFold

from transformerFunctions import Transformer
from database import Database

import matplotlib.pyplot as plt
import time
#TODO:==========================================CALCULATION-FUNCTIONS==================================================================#
#! Return avgAmbientTemp given load current percent
def avgAmbientTemp(loadCurrentPercent):
    #TODO: lightly loaded transformers 10 degrees above ambient room temp, heavily loaded transformers 30 degrees above ambient room temp
    ambientTemp = 23.8889 + loadCurrentPercent*(40.556-23.8889)
    return ambientTemp

#! Create 3 datasets, one for training, one for validation and one for testing. easy to parse for ML and doesnt average out any characteristics 
def createDataSet(transformer:Transformer, database:Database):
    table_name = transformer.name + "fullRange"

    # Pull the entire table (or consider chunks for very large tables)
    query = f'''SELECT * FROM "{table_name}" ORDER BY "DATETIME" ASC'''
    transformerData = pandas.read_sql_query(query, database.conn)
    transformerData['DATETIME'] = pandas.to_datetime(transformerData['DATETIME'])

    # Identify relevant columns
    hsA, hsB, hsC = transformerData.columns[1:4]
    voltageA, voltageB, voltageC = transformerData.columns[4:7]
    currentA, currentB, currentC = transformerData.columns[7:10]

    # Calculate max hotspot, RMS current, RMS voltage, ambient temp
    transformerData['hotspot_temp_max'] = numpy.max(transformerData[[hsA, hsB, hsC]].values, axis=1)
    transformerData['V_RMS'] = numpy.sqrt((transformerData[voltageA]**2 + transformerData[voltageB]**2 + transformerData[voltageC]**2)/3)
    transformerData['I_RMS'] = numpy.sqrt((transformerData[currentA]**2 + transformerData[currentB]**2 + transformerData[currentC]**2)/3)
    transformerData['T_ambient'] = avgAmbientTemp(transformerData['I_RMS']/transformer.RatedCurrentLV)
    transformerData['phaseCurrentMax'] = numpy.max(transformerData[[currentA, currentB, currentB]].values, axis=1)
    transformerData['phaseVoltageMax'] = numpy.max(transformerData[[voltageA, voltageB, voltageC]].values, axis=1)
    
    # Set datetime index
    transformerData.set_index('DATETIME', inplace=True)

    # Rolling window size for 2 months
    # Assuming 10-min intervals: 2 months ≈ 60*24*6 = 8640 rows
    window_size_training = 8640
    window_size_testing = 2000 
    window_size_validation = 2000

    # Compute rolling std of hotspot temperature
    rolling_std_training = transformerData['hotspot_temp_max'].rolling(window=window_size_training).std()

    # Find the window with the highest std
    max_std_idx = rolling_std_training.idxmax()
    start_window = max_std_idx - pandas.Timedelta(minutes=window_size_training*10)
    end_window_training = max_std_idx
    training_window = transformerData.loc[start_window:end_window_training]

    print("Selected 2-month window for training:")
    print("Start:", start_window, "End:", end_window_training, "Std:", rolling_std_training.max())
    print('\n')

    #TODO: Add another table for validation data, identify next largest std window outside of training set and sample 2k points from that
    hotspot_series_copy = transformerData['hotspot_temp_max'].copy()

    # Mask training window rows (set them to NaN)
    validation_start_pos = transformerData.index.get_loc(training_window.index[0])
    validation_end_pos = transformerData.index.get_loc(training_window.index[-1])
    hotspot_series_copy.iloc[validation_start_pos:validation_end_pos+1] = numpy.nan

    # Compute rolling std on remaining data
    rolling_std_validation = hotspot_series_copy.rolling(window=window_size_validation).std()

    # mark start and end dates of this section
    max_std_idx_validation = rolling_std_validation.idxmax()
    start_window_validation = max_std_idx_validation - pandas.Timedelta(minutes = window_size_validation*10)
    end_window_validation = max_std_idx_validation
    validation_window = transformerData.loc[start_window_validation:end_window_validation]

    print("Selected window for validation:")
    print("Start:", start_window_validation, "End:", end_window_validation, "Std:", rolling_std_validation.max())
    print('\n')


    #TODO: Add another table for testing data, identify next largest std window outside of training and validation set and sample ~2k points from that
    # Create a copy of the series
    hotspot_series_copy = transformerData['hotspot_temp_max'].copy()

    # Mask training window rows (set them to NaN)
    train_start_pos = transformerData.index.get_loc(training_window.index[0])
    train_end_pos = transformerData.index.get_loc(training_window.index[-1])
    hotspot_series_copy.iloc[train_start_pos:train_end_pos+1] = numpy.nan

    #Mask Validation Window rows
    validation_start_pos = transformerData.index.get_loc(validation_window.index[0])
    validation_end_pos = transformerData.index.get_loc(validation_window.index[-1])
    hotspot_series_copy.iloc[validation_start_pos:validation_end_pos+1] = numpy.nan

    # Compute rolling std on remaining data
    rolling_std_testing = hotspot_series_copy.rolling(window=window_size_testing).std()

    # mark start and end dates of this section
    max_std_idx_testing = rolling_std_testing.idxmax()
    start_window_testing = max_std_idx_testing - pandas.Timedelta(minutes = window_size_testing*10)
    end_window_testing = max_std_idx_testing
    testing_window = transformerData.loc[start_window_testing:end_window_testing]

    print("Selected window for testing:")
    print("Start:", start_window_testing, "End:", end_window_testing, "Std:", rolling_std_testing.max())
    print('\n')

    # --- SAVE TO DATABASE ---
    training_window.to_sql(
        name=f"{transformer.name}_trainingData",
        con=database.conn,
        if_exists="replace",
        chunksize=5000,
        method="multi"
    )

    validation_window.to_sql(
        name=f"{transformer.name}_validationData",
        con=database.conn,
        if_exists="replace",
        chunksize=5000,
        method="multi"
    )

    testing_window.to_sql(
        name=f"{transformer.name}_testingData",
        con=database.conn,
        if_exists="replace",
        chunksize=5000,
        method="multi"
    )

    

    
#TODO:==========================================PARTICLE-FILTER-FUNCTIONS===============================================================# 

#! Evaluate particle using calculated C, gamma epsilon parameters. Return a Mean Square Error and Probability Weight
def evaluate_particle(params, Xw_train, yw_train, Xw_validate,yw_validate):
    logC, logGamma, epsilon = params
    C = 10**logC
    gamma = 10**logGamma
    model = SVR(kernel='rbf', C=C, gamma=gamma, epsilon=epsilon)
    model.fit(Xw_train, yw_train)
    preds = model.predict(Xw_validate)
    mse = mean_squared_error(yw_validate, preds)
    likelihood = numpy.exp(-mse/0.1)
    return mse, likelihood
    
#! Calculate final SVR parameters based on historical data (C, gamma, epsilon)
def particle_filter_for_SVR(transformer:Transformer, database:Database):
    
    n_particles=400
    n_steps=30

    # process_noise_std = [0.9, 0.003, 0.2]  # C, gamma, epsilon
    process_noise_std = [0.1, 0.01, 0.005]

    print("\nProcess noise: ", process_noise_std)
    print("\n")
    random_state=42
    
    rng = numpy.random.default_rng(random_state)
    
    # --- Load data from training/testing sets---
    db_path=database.dbpath
    
    table_name_training= f'''{transformer.name}_trainingData'''
    table_name_validation = f'''{transformer.name}_validationData'''
    table_name_testing = f'''{transformer.name}_testingData'''
    
    conn = sqlite3.connect(db_path)

    query_training = f'''SELECT DATETIME, 
                hotspot_temp_max,
                phaseCurrentMax,
                phaseVoltageMax
                FROM "{table_name_training}"'''
    
    query_validation = f'''SELECT DATETIME, 
                hotspot_temp_max,
                phaseCurrentMax,
                phaseVoltageMax
                FROM "{table_name_validation}"'''
    
    query_testing = f'''SELECT DATETIME, 
                phaseCurrentMax,
                phaseVoltageMax,
                hotspot_temp_max
                FROM "{table_name_testing}"'''
    
    df_training = pandas.read_sql_query(query_training, conn)
    df_validation =  pandas.read_sql_query(query_validation, conn)
    df_testing = pandas.read_sql_query(query_testing, conn)
    conn.close()
    

    # Specify inputs/outputs to system; load current and ambient are inputs (x), hotspot temp are the outputs (y)
    x_training = df_training[['phaseCurrentMax','phaseVoltageMax']].values
    y_training = df_training['hotspot_temp_max'].values

    x_validation =df_validation[['phaseCurrentMax','phaseVoltageMax']].values
    y_validation =df_validation['hotspot_temp_max'].values

    x_testing = df_testing[['phaseCurrentMax','phaseVoltageMax']].values
    y_testing = df_testing['hotspot_temp_max'].values

    # Scale features to ensure distance between HS temp, current and ambient isnt like 100 to 1, SVR very sensitive for input features
    scaler = MinMaxScaler()
    x_train_scaled = scaler.fit_transform(x_training)
    x_validation_scaled = scaler.transform(x_validation)
    x_test_scaled = scaler.transform(x_testing) #make sure test set has same scaling as training set
    
    # clean up definition/naming conventions down below
    X_train_full = x_train_scaled
    y_train_full = y_training
    
    
    #TODO========================================================================================================================================
     # --- Particle filter setup ---
    particles = numpy.zeros((n_particles, 3))
    

    # best_params = {'C': 3.5612691951343716, 'gamma': 0.005749232868996007, 'epsilon': 0.482992710224002}
    best_params = {'C': 3.5612691951343716, 'gamma': 0.09749232868996007, 'epsilon': 0.22992710224002}
  
    best_logC = numpy.log10(best_params['C'])
    best_logGamma = numpy.log10(best_params['gamma'])
    best_epsilon = best_params['epsilon']

    # Define small random noise around best values

    # logC_std = 2          # how wide to explore around best C
    # logGamma_std = 0.059   # how wide to explore around best gamma
    # epsilon_std = 0.01    # how wide to explore around best epsilon

    logC_std = 0.5
    logGamma_std = 0.05
    epsilon_std = 0.05


    particles[:,0] = rng.normal(best_logC, logC_std, n_particles)
    particles[:,1] = rng.normal(best_logGamma, logGamma_std, n_particles)
    particles[:,2] = rng.normal(best_epsilon, epsilon_std, n_particles)
    
    weights = numpy.ones(n_particles) / n_particles
    mse_history = []
    window_size = len(X_train_full) // n_steps
    
    start_time = time.time()
    
    for t in range(n_steps):
        start_idx = max(0, t * window_size // 2)
        end_idx = min(len(X_train_full), start_idx + window_size)
        X_window = X_train_full[start_idx:end_idx]
        y_window = y_train_full[start_idx:end_idx]
        
        # Predict / diffuse particles
        noise = rng.normal(0, process_noise_std, size=particles.shape)
        particles += noise
        particles[:,2] = numpy.clip(particles[:,2], 0.001, 1.0)
        
        # Evaluate likelihood/probability
        mses, likelihoods = [], []
        
        for p in particles:
            mse, lh = evaluate_particle(p, X_window, y_window)
            mses.append(mse)
            likelihoods.append(lh)
        
        mses = numpy.array(mses)
        likelihoods = numpy.array(likelihoods)
        weights = likelihoods / (likelihoods.sum() + 1e-12)
        
        # Resample particles, need to replace higher probablity particles with repeats of the same. ie switch/transform to uniform pdf
        cumulative = numpy.cumsum(weights)
        r = rng.random() / n_particles
        idxs = []
        j = 0

        for i in range(n_particles):
            u = r + i / n_particles
            while j < len(cumulative) - 1 and u > cumulative[j]:
                j += 1
            idxs.append(j)

        particles = particles[idxs]
        mse_history.append(mses.min())
    
    # Final parameters for SVR
    final_particle = numpy.average(particles, axis=0, weights=weights)
    final_params = {
        "C": 10**final_particle[0],
        "gamma": 10**final_particle[1],
        "epsilon": final_particle[2]
    }
    
    elapsed = time.time() - start_time
    print(f"\nParticle filtering completed in {elapsed:.1f}s")
    
    return final_params, mse_history, X_train_full, y_train_full, x_test_scaled, y_testing, scaler

#TODO============================================================================================================================#

#! Using PF results and created training/test datasets, create SVR model
def train_and_evaluate_svr(X_train, y_train, X_test, y_test, params):
    model = SVR(kernel='rbf', **params)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    test_mse = mean_squared_error(y_test, y_pred)
    return model, y_pred, test_mse

#?=====================--------------TESTING-SECTION-------------------===============================#
database = Database(dbpath="/home/eveuio/DataProcessing/transformerDB")

# serverThread = threading.Thread(target= pseudo_server, args = (database,), daemon=False)
# updateMetricsThread = threading.Thread(target=database.update_transformer_average_data, args = (), daemon = False)

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
createDataSet(transformer22A03, database)

#?-------------------------------POPULATE HISTORICAL AVERAGE TABLES---------------------------------------------------------------------

# final_params, mse_history, X_train, y_train, X_test, y_test, scaler = particle_filter_for_SVR(transformer=transformer22A03,database=database)
    
# print("\nEstimated SVR Hyperparameters:")
# print(final_params)

# model, y_pred, test_mse = train_and_evaluate_svr(X_train, y_train, X_test, y_test, final_params)
# print("\n")

# print("TRAIN y: mean,std,min,max:", numpy.mean(y_train), numpy.std(y_train), numpy.min(y_train), numpy.max(y_train))
# print("TEST  y: mean,std,min,max:", numpy.mean(y_test),  numpy.std(y_test),  numpy.min(y_test),  numpy.max(y_test))
# print("\n")

# # Model on train set
# y_train_pred = model.predict(X_train)
# print("Train MSE, MAE:", mean_squared_error(y_train, y_train_pred), mean_absolute_error(y_train, y_train_pred))
# print("Train y mean vs pred mean:", numpy.mean(y_train), numpy.mean(y_train_pred))
# print("Train residual mean:", numpy.mean(y_train - y_train_pred))
# print("\n")

# # Model on test set
# print("Test  MSE, MAE:", mean_squared_error(y_test, y_pred), mean_absolute_error(y_test, y_pred))
# print("Test  y mean vs pred mean:", numpy.mean(y_test), numpy.mean(y_pred))
# print("Test  residual mean:", numpy.mean(y_test - y_pred))
# print("\n")


# # --- Plots ---
# plt.figure(figsize=(7,4))
# plt.plot(mse_history, marker='o')
# plt.xlabel("Filtering Step")
# plt.ylabel("Best MSE")
# plt.title("Particle Filter Convergence")
# plt.grid(True)
# plt.tight_layout()
# plt.show(block=False)

# plt.figure(figsize=(10,5))
# plt.plot(y_test, label='Actual Hot-Spot Temp', color='black', linewidth=2)
# plt.plot(y_pred, label='Predicted Hot-Spot Temp', color='tab:blue', linestyle='--', linewidth=2)

# plt.xlabel("Test Sample Index")
# plt.ylabel("Hot-Spot Temperature (°C)")
# plt.title("Model Prediction vs Actual Hot-Spot Temperature")
# plt.legend()
# plt.grid(True, linestyle='--', alpha=0.6)
# plt.tight_layout()
# plt.show()