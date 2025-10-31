import numpy
import pandas
import sqlite3

from sklearn.svm import SVR
from sklearn.preprocessing import MinMaxScaler, RobustScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import GridSearchCV, KFold, TimeSeriesSplit
from scipy.stats import pearsonr
import joblib

from transformerFunctions import Transformer
from database import Database

import matplotlib.pyplot as plt
from datetime import datetime
#TODO:============================================CALCULATION-FUNCTIONS==================================================================#
#! Return avgAmbientTemp given load current percent
def avgAmbientTemp(loadCurrentPercent):
    #TODO: lightly loaded transformers 10 degrees above ambient room temp, heavily loaded transformers 30 degrees above ambient room temp
    ambientTemp = 23.8889 + loadCurrentPercent*(40.556-23.8889)
    return ambientTemp

    
#TODO:==========================================PARTICLE-FILTER-FUNCTIONS===============================================================# 
#! Evaluate particle using calculated C, gamma epsilon parameters. Return a Mean Square Error and Probability Weight
def kfold_optimize_initial_particle(X_train, y_train, n_splits=10):
    
    #TODO:=================================================================================
    param_grid = {
        "C": [5, 10, 20,30,40],
        "epsilon": [0.001, 0.01, 0.1, 0.5, 1],
        "gamma": [0.01, 0.1, 0.5, 1]
    }

    # TimeSeriesSplit preserves ordering, avoiding leakage
    tscv = TimeSeriesSplit(n_splits=n_splits)

    grid_search = GridSearchCV(
        estimator=SVR(),
        param_grid=param_grid,
        scoring='neg_mean_squared_error',
        cv=tscv,
        n_jobs=-1,   # use all CPU cores
        verbose=1
    )

    grid_search.fit(X_train, y_train)

    best_params = grid_search.best_params_
    best_mse = -grid_search.best_score_  # CV average
    best_likelihood = numpy.exp(-best_mse)

    # Convert to logC, logGamma, epsilon for consistency with particle filter init
    logC = numpy.log10(best_params['C'])
    logGamma = numpy.log10(best_params['gamma'])
    epsilon = best_params['epsilon']
    best_particle = numpy.array([logC, logGamma, epsilon])

    print(f"📊 GridSearchCV Best Particle → C={10**logC:.2f}, Gamma={10**logGamma:.2f}, epsilon={epsilon:.3f}")
    print(f"📉 Best MSE: {best_mse:.6f}")

    #TODO:=================================================================================


    return best_particle, best_mse, best_likelihood


def evaluate_particle(particle, X_train, y_train):
    # """
    # Evaluate a single particle's SVR parameters on the validation set.
    # """
    # logC, logGamma, epsilon = particle
    # C, gamma = 10**logC, 10**logGamma

    # model = SVR(kernel='rbf', C=C, gamma=gamma, epsilon=epsilon)
    # model.fit(X_train, y_train)

    # y_pred = model.predict(X_train)

    # mse = mean_squared_error(y_train, y_pred)
    # likelihood = numpy.exp(-mse)
    # return mse, likelihood

    logC, logGamma, epsilon = particle
    C, gamma = 10**logC, 10**logGamma

    # Ensure parameters stay in reasonable numeric ranges
    C = numpy.clip(C, 1e-3, 1e4)
    gamma = numpy.clip(gamma, 1e-4, 1e3)

    model = SVR(kernel='rbf', C=C, gamma=gamma, epsilon=epsilon)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_train)

    # Compute metrics
    mse = mean_squared_error(y_train, y_pred)

    # Combine MSE and correlation into a single "likelihood"
    # The (1 + corr)**2 term rewards models that follow trends.
    likelihood = numpy.exp(-mse) 

    return mse, likelihood

#! Calculate final SVR parameters based on historical data (C, gamma, epsilon)
def particle_filter_for_SVR(transformer:Transformer, database:Database):
    # --- Load data from training/testing sets---
    db_path=database.db_path
    
    table_name_training= f'''{transformer.name}_trainingData'''
    table_name_validation = f'''{transformer.name}_validationData'''
    table_name_testing = f'''{transformer.name}_testingData'''
    
    conn = sqlite3.connect(db_path)

    query_training = f'''SELECT DATETIME, 
                hotspot_temp_max,
                phaseCurrentMax,
                phaseVoltageMax,
                phaseCurrentLag1,
                phaseVoltageLag1,
                hotspot_lag1,
                T_ambient
                FROM "{table_name_training}"'''
    
    query_validation = f'''SELECT DATETIME, 
                hotspot_temp_max,
                phaseCurrentMax,
                phaseVoltageMax,
                phaseCurrentLag1,
                phaseVoltageLag1,
                hotspot_lag1,
                T_ambient
                FROM "{table_name_validation}"'''
    
    query_testing = f'''SELECT DATETIME, 
                hotspot_temp_max,
                phaseCurrentMax,
                phaseVoltageMax,
                phaseCurrentLag1,
                phaseVoltageLag1,
                hotspot_lag1,
                T_ambient
                FROM "{table_name_testing}"'''
    
    df_training = pandas.read_sql_query(query_training, conn)
    df_validation =  pandas.read_sql_query(query_validation, conn)
    df_testing = pandas.read_sql_query(query_testing, conn)
    conn.close()
    
    
    # Specify inputs/outputs to system; load current and ambient are inputs (x), hotspot temp are the outputs (y)
    # x_training = df_training[['phaseCurrentMax','T_ambient']].values
    x_training = df_training[['phaseCurrentMax', 'phaseVoltageMax', 'T_ambient','phaseCurrentLag1', 'phaseVoltageLag1', 'hotspot_lag1']].values
    y_training = df_training['hotspot_temp_max'].values

    x_validation =df_validation[['phaseCurrentMax', 'phaseVoltageMax', 'T_ambient','phaseCurrentLag1', 'phaseVoltageLag1', 'hotspot_lag1']].values
    y_validation =df_validation['hotspot_temp_max'].values

    x_testing = df_testing[['phaseCurrentMax', 'phaseVoltageMax', 'T_ambient','phaseCurrentLag1', 'phaseVoltageLag1', 'hotspot_lag1']].values
    y_testing = df_testing['hotspot_temp_max'].values

    # Scale features to ensure distance between HS temp, current and ambient isnt like 100 to 1, SVR very sensitive for input features
    scaler = MinMaxScaler()
    x_train_scaled = scaler.fit_transform(x_training)
    x_validation_scaled = scaler.transform(x_validation)
    x_test_scaled = scaler.transform(x_testing) #make sure test set has same scaling as training set

    #TODO=======================================================Work-in-Progress=========================================================#
    
    n_particles = 20
    n_iterations = 10
    
    rng = numpy.random.default_rng(67)
    #K-Fold to find starting particle
    best_particle_kfold, base_mse, base_likelihood = kfold_optimize_initial_particle(x_train_scaled, y_training)

    # #Initialize PF particles around K-Fold result 
    # logC_range = (best_particle_kfold[0]-1, best_particle_kfold[0]+1) 
    # logGamma_range = (best_particle_kfold[1]-1, best_particle_kfold[1]+1) 
    # epsilon_range = (max(0.001, best_particle_kfold[2]-0.1), best_particle_kfold[2]+0.1)

    # # Tighter around K-Fold
    logC_range     = (best_particle_kfold[0]-0.5, best_particle_kfold[0]+0.5)
    logGamma_range = (best_particle_kfold[1]-0.5, best_particle_kfold[1]+0.5)
    epsilon_range  = (max(0.001, best_particle_kfold[2]-0.05), best_particle_kfold[2]+0.05)

    # logC_range = (best_particle_kfold[0]-2, best_particle_kfold[0]+2)
    # logGamma_range = (best_particle_kfold[1]-2, best_particle_kfold[1]+2)
    # epsilon_range = (max(0.001, best_particle_kfold[2]-0.2), best_particle_kfold[2]+0.2)

    # logC_range = (best_particle_kfold[0]-3, best_particle_kfold[0]+3)
    # logGamma_range = (best_particle_kfold[1]-3, best_particle_kfold[1]+3)
    # epsilon_range = (max(0.001, best_particle_kfold[2]-0.3), best_particle_kfold[2]+0.3)




    particles = numpy.column_stack([
        rng.uniform(*logC_range, n_particles),
        rng.uniform(*logGamma_range, n_particles),
        rng.uniform(*epsilon_range, n_particles)
    ])
    weights = numpy.ones(n_particles) / n_particles
    mse_history = []

    best_particle = None
    best_mse = numpy.inf

    # 3️⃣ PF Iterations
    for iteration in range(n_iterations):
        mses, likelihoods = [], []

        for p in particles:
            mse, likelihood = evaluate_particle(p, x_train_scaled, y_training)
            mses.append(mse)
            likelihoods.append(likelihood)
            
            if mse < best_mse:
                best_mse = mse
                best_particle = p.copy()
                print(f"Particle Iteration {iteration+1}/{n_iterations} | Best MSE so far: {best_mse:.6f} | C: {10**best_particle[0]} | gamma: {10**best_particle[1]} | e: {best_particle[2]}")

        mses = numpy.array(mses)
        likelihoods = numpy.array(likelihoods)
        weights = likelihoods / numpy.sum(likelihoods)
        mse_history.append(best_mse)

        indices = rng.choice(range(n_particles), size=n_particles, p=weights)
        particles = particles[indices]

        # Add decaying Gaussian noise for exploration
        noise_scale = 0.1 * (0.9 ** iteration)  # decays each iteration
        particles += rng.normal(0, noise_scale, particles.shape)

        # Clip particles to ranges
        particles[:, 0] = numpy.clip(particles[:, 0], *logC_range)
        particles[:, 1] = numpy.clip(particles[:, 1], *logGamma_range)
        particles[:, 2] = numpy.clip(particles[:, 2], *epsilon_range)

        print(f"Iteration {iteration+1}/{n_iterations} | Best MSE so far: {best_mse:.6f}")
        print("\n")

    # Final hyperparameters
    final_params = {
        'C': 10**best_particle[0],
        'gamma': 10**best_particle[1],
        'epsilon': best_particle[2]
    }
    
    #TODO===============================================================Work-in-Progress=========================================================#
    
    return final_params, mse_history, x_train_scaled, y_training, x_test_scaled, y_testing, scaler
    # return x_train_scaled, y_training, x_test_scaled, y_testing, scaler
   
    

#TODO================================================TRAIN-AND-EVALUATE-SVR==================================================================#

#! Using PF results and created training/test datasets, create SVR model
def train_and_evaluate_svr(X_train, y_train, X_test, y_test, params):
    model = SVR(kernel='rbf', **params)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    test_mse = mean_squared_error(y_test, y_pred)
    return model, y_pred, test_mse

#?=====================--------------TESTING-SECTION-------------------===============================#
database = Database(db_path="/home/eveuio/DataProcessing/isolationFolder/transformerDB.db")

# serverThread = threading.Thread(target= pseudo_server, args = (database,), daemon=False)
# updateMetricsThread = threading.Thread(target=database.update_transformer_average_data, args = (), daemon = False)

# transformerEX02A3 = Transformer(name = "22B01",
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
#                                XR_Ratio=3.5)
transformer = Transformer(name = "21A05",
                              ratedCurrent_H=69.4,
                              ratedVoltage_H=12470,
                              ratedCurrent_L=1804,
                              ratedVoltage_L=480,
                               impedance=5.82,
                              windingMaterial="Aluminum",
                               thermalClass_rated=220,
                               avgWindingTempRise_rated=115,
                               weight_CoreAndCoil=3538,
                               weight_total=4536,
                               database=database,
                               age=19,
                               ratedKVA=1500,
                               XR_Ratio=3.5)

database.addTransformerTesting(transformer)
database.createDataSet(transformer=transformer)

# ?-------------------------------POPULATE HISTORICAL AVERAGE TABLES---------------------------------------------------------------------

final_params, mse_history, X_train, y_train, X_test, y_test, scaler = particle_filter_for_SVR(transformer=transformer,database=database)
# X_train, y_train, X_test, y_test, scaler = particle_filter_for_SVR(transformer=transformerEX02A3,database=database)

#!=============Best run so far==========#
# final_params = {
#     "C": 4,
#     'gamma': 0.05, 
#     'epsilon': 0.5 #keep
# }
#!=======================================#

print("\nEstimated SVR Hyperparameters:")
print(final_params)

model, y_pred, test_mse = train_and_evaluate_svr(X_train, y_train, X_test, y_test, final_params)

print("\n")

print("TRAIN y: mean,std,min,max:", numpy.mean(y_train), numpy.std(y_train), numpy.min(y_train), numpy.max(y_train))
print("TEST  y: mean,std,min,max:", numpy.mean(y_test),  numpy.std(y_test),  numpy.min(y_test),  numpy.max(y_test))
print("\n")

# Model on train set
y_train_pred = model.predict(X_train)
print("Train MSE, MAE:", mean_squared_error(y_train, y_train_pred), mean_absolute_error(y_train, y_train_pred))
print("Train y mean vs pred mean:", numpy.mean(y_train), numpy.mean(y_train_pred))
print("Train residual mean:", numpy.mean(y_train - y_train_pred))
print("\n")

# Model on test set
print("Test  MSE, MAE:", mean_squared_error(y_test, y_pred), mean_absolute_error(y_test, y_pred))
print("Test  y mean vs pred mean:", numpy.mean(y_test), numpy.mean(y_pred))
print("Test  residual mean:", numpy.mean(y_test - y_pred))
print("\n")

joblib.dump(model, f"/home/eveuio/DataProcessing/isolationFolder/HSModels/{transformer.name}_pf_svr_model_{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}.pkl")

# --- Plots ---
plt.figure(figsize=(7,4))
plt.plot(mse_history, marker='o')
plt.xlabel("Filtering Step")
plt.ylabel("Best MSE")
plt.title("Particle Filter Convergence")
plt.grid(True)
plt.tight_layout()
plt.show(block=False)

plt.figure(figsize=(10,5))
plt.plot(y_test, label='Actual Hot-Spot Temp', color='black', linewidth=2)
plt.plot(y_pred, label='Predicted Hot-Spot Temp', color='tab:blue', linestyle='--', linewidth=2)

plt.xlabel("Test Sample Index")
plt.ylabel("Hot-Spot Temperature (°C)")
plt.title("Model Prediction vs Actual Hot-Spot Temperature")
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()