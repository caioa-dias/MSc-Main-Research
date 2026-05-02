# -*- coding: utf-8 -*-
"""
========================================================================================================
Module: optimization_MHP_ANN
Author: Caio Dias Filho
Creation date: 2026-03-27
Last modification: 2026-03-27
Version: 1.0
========================================================================================================

OVERVIEW
--------


WORKFLOW
--------


SEARCH SPACE
------------

        
DEPENDENCIES
------------


OUTPUT FILES
------------


REPRODUCIBILITY
----------------
    

ASSUMPTIONS
------------
    

LIMITATIONS
------------


========================================================================================================
"""

# =======================================================================================================
#                                            USER INPUT SECTION
# =======================================================================================================

# --- Number of optimization trials ---
N_TRIALS = 200


# =======================================================================================================
#                                              CORE FUNCTION
# =======================================================================================================

# System libraries and error suppressing:
import os
import warnings
os.environ['TF_CPP_MIN_LOG_LEVEL']="2"
os.environ['TF_ENABLE_ONEDNN_OPTS']="0"
warnings.filterwarnings("ignore")

# Multithreading setup:
os.environ["MKL_NUM_THREADS"] = "8"
os.environ["OMP_NUM_THREADS"] = "8"
os.environ["TF_NUM_INTRAOP_THREADS"] = "8"
os.environ["TF_NUM_INTEROP_THREADS"] = "2"
import tensorflow
tensorflow.config.threading.set_intra_op_parallelism_threads(8)
tensorflow.config.threading.set_inter_op_parallelism_threads(2)

# Scientific libraries:
from functools import partial
tensorflow.get_logger().setLevel("ERROR")
import sklearn as skl
import pandas as pd
import numpy as np
import optuna
import keras
import gc

# Reproductibility setup
SEED = 42
os.environ['PYTHONHASHSEED'] = str(SEED)
tensorflow.random.set_seed(SEED)
np.random.seed(SEED)
print(f'\nGlobal seed set to: {SEED}')

def load_and_split_data(filepath: str):
    """

    """

    # --- Loading the dataset ---
    data = (pd.read_csv(filepath, sep=';')).copy()

    # --- Create an identifier for flow case (assuming 80 sections per case) ---
    wing_sections = 80
    idx = np.arange(len(data))
    data['flow_case'] = idx // wing_sections
    unique_cases = data['flow_case'].unique()
    train_cases, test_cases = skl.model_selection.train_test_split(unique_cases, test_size=0.2, 
        random_state=SEED, shuffle=True)
    
    # --- Separate the datasets ---
    data_train = data[data['flow_case'].isin(train_cases)]
    data_test = data[data['flow_case'].isin(test_cases)]

    return data_train, data_test

def preprocess_fold_data(data_train: pd.DataFrame, data_val: pd.DataFrame):
    """
    """

    # --- Defining input and target columns ---
    X_cols = data_train.columns[0:3]
    Cl_cols = data_train.columns[3]
    Cp_cols = data_train.columns[4:205]

    # --- Splitting the raw data into training and validation sets ---
    X_train_raw = data_train[X_cols].to_numpy()
    X_val_raw = data_val[X_cols].to_numpy()

    Cl_train_raw = data_train[Cl_cols].to_numpy().reshape(-1, 1)
    Cl_val_raw = data_val[Cl_cols].to_numpy().reshape(-1, 1)

    Cp_train_raw = data_train[Cp_cols].to_numpy()
    Cp_val_raw = data_val[Cp_cols].to_numpy()

    # --- Scaling the data ---
    X_scaler = skl.preprocessing.StandardScaler()
    X_train = X_scaler.fit_transform(X_train_raw)
    X_val = X_scaler.transform(X_val_raw)

    Cl_scaler = skl.preprocessing.StandardScaler()
    Cl_train = Cl_scaler.fit_transform(Cl_train_raw)
    Cl_val = Cl_scaler.transform(Cl_val_raw)

    Cp_scaler = skl.preprocessing.StandardScaler()
    Cp_train = Cp_scaler.fit_transform(Cp_train_raw)
    Cp_val = Cp_scaler.transform(Cp_val_raw)

    return X_train, Cl_train, Cp_train, X_val, Cl_val, Cp_val, X_scaler, Cl_scaler, Cp_scaler

def build_model(trial: optuna.Trial, input_dim: int, cl_output_dim: int, cp_output_dim: int, cl_loss: float, cp_loss: float):
    """
    """

    # --- Shared trunk hyperparameters ---
    n_shared_hidden_layers = trial.suggest_int(name='n_shared_hidden_layers', low=2, high=3, step=1)
    activation_shared = trial.suggest_categorical(name='activation_shared', choices=['swish', 'gelu'])

    # --- Cl head hyperparameters ---
    n_cl_hidden_layers = trial.suggest_int(name='n_cl_hidden_layers', low=1, high=2, step=1)
    activation_cl = trial.suggest_categorical(name='activation_cl', choices=['swish', 'gelu', 'relu'])

    # --- Cp head hyperparameters ---
    n_cp_hidden_layers = trial.suggest_int(name='n_cp_hidden_layers', low=1, high=2, step=1)
    activation_cp = trial.suggest_categorical(name='activation_cp', choices=['swish', 'gelu'])

    # --- Training hyperparameters ---
    learning_rate = trial.suggest_categorical(name='learning_rate', choices=[5e-4, 1e-3, 1.5e-3])
    
    # --- Defining model input layer ---
    inputs = keras.layers.Input(shape=(input_dim,), name='Input_Layer')
    x = inputs

    # --- Shared hidden layers ---
    for i in range(n_shared_hidden_layers):
        units = trial.suggest_int(name=f'shared_units_layer{i+1}', low=96, high=192, step=32)
        x = keras.layers.Dense(units=units, activation=activation_shared, kernel_initializer='glorot_uniform',
            name=f'Shared_Hidden_Layer{i+1}')(x)
        
    # --- Cl head hidden layers ---
    h1 = x
    for i in range(n_cl_hidden_layers):
        units = trial.suggest_int(name=f'cl_units_layer{i+1}', low=96, high=192, step=32)
        h1 = keras.layers.Dense(units=units, activation=activation_cl, kernel_initializer='glorot_uniform',
            name=f'Cl_Hidden_Layer{i+1}')(h1)
        
    out_cl = keras.layers.Dense(units=cl_output_dim, activation='linear', kernel_initializer='glorot_uniform',
        name='Cl_Output')(h1)

    # --- Cp head hidden layers ---
    h2 = x
    for i in range(n_cp_hidden_layers):
        units = trial.suggest_int(name=f'cp_units_layer{i+1}', low=96, high=192, step=32)
        h2 = keras.layers.Dense(units=units, activation=activation_cp, kernel_initializer='glorot_uniform',
            name=f'Cp_Hidden_Layer{i+1}')(h2)
        
    out_cp = keras.layers.Dense(units=cp_output_dim, activation='linear', kernel_initializer='glorot_uniform',
        name='Cp_Output')(h2)
    
    # --- Defining the model ---
    model = keras.Model(inputs=inputs, outputs=[out_cl, out_cp], name='low_fidelity_mhp_ann')

    # --- Compiling the model ---
    model.compile(loss={'Cl_Output': 'mse', 'Cp_Output': 'mse'}, optimizer=keras.optimizers.Nadam(learning_rate=learning_rate), 
        metrics={'Cl_Output': ['mae'], 'Cp_Output': ['mae']}, loss_weights={'Cl_Output': cl_loss, 
        'Cp_Output': cp_loss})
    
    return model

def predict_test(model: keras.Model, X_val: np.ndarray, Cl_val: np.ndarray, Cp_val: np.ndarray,
    Cl_scaler: skl.preprocessing.StandardScaler, Cp_scaler: skl.preprocessing.StandardScaler):
    """
    """

    # --- Predicting on the validation set ---
    y_pred = model.predict(X_val, verbose=0)

    # --- Separating the outputs ---
    Cl_pred = y_pred[0]
    Cp_pred = y_pred[1]

    # --- Reverses the scaling ---
    Cl_val_raw = Cl_scaler.inverse_transform(Cl_val)
    Cl_pred_raw = Cl_scaler.inverse_transform(Cl_pred)
    Cp_val_raw = Cp_scaler.inverse_transform(Cp_val)
    Cp_pred_raw = Cp_scaler.inverse_transform(Cp_pred)

    return Cl_val_raw, Cl_pred_raw, Cp_val_raw, Cp_pred_raw

def compute_cl_from_cp(cp_data: np.ndarray, airfoil_data: pd.DataFrame, AoA: float):
    """

    """

    # --- Converts all variables to arrays ---
    cp = np.asarray(cp_data)
    x = np.asarray(airfoil_data['x'].values)
    y = np.asarray(airfoil_data['y'].values)

    # --- Calculates panel geomeric deltas (dx, dy) ---
    panel_dx = np.diff(x)
    panel_dy = np.diff(y)

    # --- Computes the average Cp for each panel ---
    panel_cp = 0.5 * (cp[:-1] + cp[1:])

    # --- Integrates to find normal (Cn) and axial (Ca) force coefficients ---
    Cn = np.sum(panel_cp * panel_dx)
    Ca = -np.sum(panel_cp * panel_dy)

    # --- Project Cn and Ca into the lift coefficient (Cl) using the angle of attack ---
    alpha = np.radians(AoA)
    Cl = Cn * np.cos(alpha) - Ca * np.sin(alpha)

    return Cl

def compute_CL_from_cp(X_scaled: np.ndarray, Cl_raw: np.ndarray, Cp_raw: np.ndarray, X_scaler: skl.preprocessing.StandardScaler,
    airfoil_data: pd.DataFrame, AoA_effective: pd.DataFrame):
    """

    """

    # --- Defines the wing geometry data ---
    b = 0.766
    c = 0.2165
    S = 2*(b*c)

    # --- Defines useful list ---
    CL_list = []

    # --- Inverse transforms the input data and merge input and outputs ---
    X_raw = X_scaler.inverse_transform(X_scaled)
    y_raw = np.concatenate((Cl_raw, Cp_raw), axis=1)
    data = pd.DataFrame(np.hstack((X_raw, y_raw)))

    # --- Adds a flow case identifier for CL calculation ---
    wing_sections = 80
    idx = np.arange(len(data))
    data['flow_case'] = idx // wing_sections
    n_cases = len(data['flow_case'].unique())

    # --- Loops through each flow case ---
    for i in range(0, n_cases):
        # Defines useful list:
        cl_list = []
        # Extracts a specific case:
        case_data = data[data['flow_case'] == i]
        # Extracts the Cp data for this case:
        cp_case = case_data.iloc[:, 4:205].values
        # Extracts the AoA for this case:
        re_case = float(case_data.iloc[0, 0])
        aoa_case = float(case_data.iloc[0, 1])
        mask = (np.isclose(AoA_effective['Re'].astype(float), re_case, atol=1e-6) &
                np.isclose(AoA_effective['AoA'].astype(float), aoa_case, atol=1e-6))
        AoA_condition = AoA_effective.loc[mask].copy()

        # Loops through each wing section:
        for j in range(len(case_data)):
            cl = compute_cl_from_cp(cp_case[j, :], airfoil_data, AoA_condition.iloc[j]['AoA_Effective'])
            cl_list.append(cl)

        # Defines the y positions for each case:
        y = case_data.iloc[:, 2].values * b
        cl_list = np.array(cl_list)
        # Calculates the CL for this case:
        CL = (2/S) * np.trapz(cl_list*c, x=y)
        # Adds the CL to the list of CL for all cases:
        CL_list.append(CL)
    
    return np.array(CL_list)

def objective(trial: optuna.Trial, data_train: pd.DataFrame, airfoil_data: pd.DataFrame, 
    AoA_effective: pd.DataFrame):
    """

    """

    # --- Defines the loss weights ---
    cl_loss_weight = trial.suggest_float(name='loss_weight', low=0.2, high=0.8)
    cp_loss_weight = 1.0 - cl_loss_weight

    # --- Defines the lists to store results ---
    val_mse_loss = []
    CL_loss = []

    # --- Defines the K-fold cross-validation ---
    gkf = skl.model_selection.GroupKFold(n_splits=5)

    # --- Loops through each fold ---
    for fold, (train_idx, val_idx) in enumerate(gkf.split(data_train, 
        groups=data_train['flow_case'].to_numpy()), start=1):

        # Splits the train dataset for this fold:
        train_df = data_train.iloc[train_idx]
        val_df = data_train.iloc[val_idx]

        # Preprocesses the data for this fold:
        X_train, Cl_train, Cp_train, X_val, Cl_val, Cp_val, X_scaler, Cl_scaler, Cp_scaler = preprocess_fold_data(train_df, val_df)

        # Builds the model:
        model = build_model(trial, X_train.shape[1], Cl_train.shape[1], Cp_train.shape[1], cl_loss_weight, cp_loss_weight)

        # Trains the model:
        callbacks = [keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, min_delta=1e-6, restore_best_weights=True), 
            keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6, cooldown=5, min_delta=1e-5)]
        history = model.fit(X_train, {'Cl_Output':Cl_train, 'Cp_Output':Cp_train}, validation_data=(X_val, {'Cl_Output':Cl_val, 'Cp_Output':Cp_val}),
            epochs=500, callbacks=callbacks, batch_size=128, verbose=0)

        # Evaluates the model using validation data:
        Cl_val_raw, Cl_pred_raw, Cp_val_raw, Cp_pred_raw = predict_test(model, X_val, Cl_val, Cp_val, Cl_scaler, Cp_scaler)

        # Finds the validation loss and appends to the list:
        y_val_raw = np.concatenate((Cl_val_raw, Cp_val_raw), axis=1)
        y_pred_raw = np.concatenate((Cl_pred_raw, Cp_pred_raw), axis=1)
        mse_output = skl.metrics.mean_squared_error(y_val_raw, y_pred_raw)
        val_mse_loss.append(mse_output)

        # Calculates the wing lift coefficient error:
        CL_true = compute_CL_from_cp(X_val, Cl_val_raw, Cp_val_raw, X_scaler, airfoil_data, AoA_effective)
        CL_pred = compute_CL_from_cp(X_val, Cl_pred_raw, Cp_pred_raw, X_scaler, airfoil_data, AoA_effective)

        # Appends the CL error to the list:
        mse_CL = skl.metrics.mean_squared_error(CL_true, CL_pred)
        CL_loss.append(mse_CL)

        # --- Clears the memory ---
        tensorflow.keras.backend.clear_session()
        gc.collect()
        del model

        # --- Prompts the user ---
        print(f'\nFold {fold} completed.')

    # --- Calculates the objective functions based on the mean of the folds ---
    objective_mse_loss = np.mean(val_mse_loss)
    objective_CL_loss = np.mean(CL_loss)

    return objective_mse_loss, objective_CL_loss

def main(n_trials: int):
    """

    """

    # --- Defines the data path --- 
    print('\nLoading dataset into memory...\n')
    data_path = 'LowFidelity-ANN/utils/LowFidelity-PressureDistributionData.csv'
    airfoil_data_path = 'LowFidelity-ANN/utils/NACA23015.csv'
    aoa_effective_path = 'LowFidelity-ANN/utils/LowFidelity-AoAEffectiveData.csv'

    # --- Defines the airfoil data ---
    airfoil_data = pd.read_csv(airfoil_data_path, sep=',', names=['x', 'y'])
    AoA_effective = pd.read_csv(aoa_effective_path, sep=';')

    # --- Load and split the data ---
    data_train, _ = load_and_split_data(data_path)

    # --- Defines the optuna study ---
    study = optuna.create_study(directions=['minimize', 'minimize'], sampler=optuna.samplers.TPESampler(seed=SEED),
        study_name='mhp_ann_study', 
        storage='sqlite:///LowFidelity-ANN/MHP-MultiHeadPerceptron/optimization-results/mhp_ann_study.db', 
        load_if_exists=True)

    # --- Executes the hyperparameter optimization ---
    print('\nStarting hyperparameter optimization process...\n')
    objective_with_data = partial(objective, data_train=data_train, airfoil_data=airfoil_data, 
        AoA_effective=AoA_effective)
    study.optimize(objective_with_data, n_trials=n_trials, show_progress_bar=True)

    # --- Saves all the optimization trials ---
    df_trials = study.trials_dataframe()
    df_trials = df_trials.rename(columns={'values_0': 'MSE_loss', 'values_1': 'CL_loss'})
    df_trials.to_csv('LowFidelity-ANN/MHP-MultiHeadPerceptron/optimization-results/optimization_trials.csv', 
        index=False)
    print('Saved: LowFidelity-ANN/MHP-MultiHeadPerceptron/optimization-results/optimization_trials.csv\n')

    # --- Saves the optimization trials on the Pareto front ---
    pareto_idx = [t.number for t in study.best_trials]
    df_pareto = df_trials[df_trials['number'].isin(pareto_idx)].copy()
    df_pareto.to_csv('LowFidelity-ANN/MHP-MultiHeadPerceptron/optimization-results/optimization_pareto_trials.csv', 
        index=False)
    print('Saved: LowFidelity-ANN/MHP-MultiHeadPerceptron/optimization-results/optimization_pareto_trials.csv\n')

    print('Optimization completed.\n')

    return

if __name__== "__main__":
    main(N_TRIALS)