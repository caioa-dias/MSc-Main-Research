# -*- coding: utf-8 -*-
"""
========================================================================================================
Module: optimization_MLP_ANN
Author: Caio Dias Filho
Creation date: 2025-12-02
Last modification: 2026-04-09
Version: 2.0 (final)
========================================================================================================

OVERVIEW
--------
This module performs multi-objective hyperparameter optimization of a Multi-Layer Perceptron Artificial
Neural Network (MLP-ANN) designed to reconstruct low-fidelity aerodynamic data generated from low-fidelity
analysis.

The neural network receives aerodynamic input parameters and predicts:
    - Sectional lift coefficient (Cl)
    - Pressure coefficient distribution (Cp)

The optimization process simultaneously minimizes two objectives:
    1. The mean squared error (MSE) between the predicted and true aerodynamic outputs.
    2. The mean squared error in the reconstructed global lift coefficient (CL), obtained from the 
       predicted pressure coefficient distributions and corrected using the effective angle of attack 
       of each section.

This dual-objective formulation ensures that the optimized model is accurate both in terms of local 
aerodynamic reconstruction and global aerodynamic consistency.

    
WORKFLOW
--------
The optimization workflow implemented in this module consists of:
    - Loading the aerodynamic dataset.
    - Splitting the dataset into training and testing subsets.
    - Performing Group K-Fold cross-validation on the training set.
    - Preprocessing each fold independently using feature standardization.
    - Building neural network models based on Optuna-sampled hyperparameters.
    - Training each model with early stopping and adaptive learning-rate reduction.
    - Evaluating each fold using prediction-space MSE.
    - Reconstructing the global lift coefficient (CL) from Cp distributions.
    - Computing the CL reconstruction error for each fold.
    - Aggregating the objective values across folds.
    - Saving all optimization trials and Pareto-optimal trials.


SEARCH SPACE
------------
The hyperparameter search space includes:

Architecture parameters:
    - Number of hidden layers (2 to 6)
    - Number of units per hidden layer (64 to 256, step of 32)

Activation functions:
    - swish or gelu

Optimization parameters:
    - Optimizer:
        - Adam or Nadam
    - Learning rate:
        - 7.5e-4, 1.0e-3, 1.5e-3 or 3.0e-3

        
DEPENDENCIES
------------
Python libraries:
    - os 
    - warnings
    - functools
    - gc
    - numpy
    - pandas
    - scikit-learn
    - tensorflow / keras
    - optuna


OUTPUT FILES
------------
Optimization database:
    - LowFidelity-ANN/MLP-MultiLayerPerceptron/optimization-results/mlp_ann_study.db

Optimization trial history:
    - LowFidelity-ANN/MLP-MultiLayerPerceptron/optimization-results/optimization_trials.csv

Pareto-optimal trials:
    - LowFidelity-ANN/MLP-MultiLayerPerceptron/optimization-results/optimization_pareto_trials.csv


REPRODUCIBILITY
----------------
Reproducibility is enforced through:
    - Fixed random seed (SEED = 42)
    - Fixed numpy seed
    - Fixed TensorFlow seed
    - Fixed dataset split random_state
    - Persistent Optuna study storage using SQLite

These measures ensure that the data split and optimization sampler behavior remain consistent across 
runs, subject to hardware and backend determinism limitations.
    

ASSUMPTIONS
------------
The implementation assumes that:
    - The aerodynamic dataset is stored in CSV format using semicolon separators.
    - Each aerodynamic flow case contains exactly 80 spanwise wing sections.
    - The first three columns of the dataset correspond to:
        1. Reynolds number (Re)
        2. Angle of attack (AoA)
        3. Normalized spanwise position (y/b)
    - The remaining target columns contain sectional lift coefficient and pressure coefficient
    distribution data.
    - The airfoil geometry file contains the coordinates required for Cp integration. 
    - The effective angle-of-attack dataset contains matching Reynolds number and AoA entries for all
    flow cases present in the aerodynamic dataset.
    

LIMITATIONS
------------
Potential limitations of this module include:
    - The optimization is computationally expensive due to 5-fold cross-validation within each trial.
    - The CL reconstruction metric depends on the quality of the effective angle-of-attack dataset and 
    on the geometric consistency of the airfoil coordinates.
    - The method assumes a fixed wing discretization with 80 sections per aerodynamic case.
    - The study does not include regularization techniques such as dropout in the current search space.
    - The reproducibility of deep-learning optimization may still be affected by low-level backend
    operations depending on the execution environment.

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
tensorflow.get_logger().setLevel("ERROR")

# Scientific libraries:
from functools import partial
import sklearn as skl
import pandas as pd
import numpy as np
import optuna
import keras
import gc

# Reproducibility setup
SEED = 42
os.environ['PYTHONHASHSEED'] = str(SEED)
tensorflow.random.set_seed(SEED)
np.random.seed(SEED)
print(f'\nGlobal seed set to: {SEED}')

def load_and_split_data(filepath: str):
    """
    Load and split the aerodynamic dataset.

    This function reads the aerodynamic dataset from a CSV file, creates a flow-case identifier assuming
    a fixed number of wing sections per case, and splits the cases into training and testing subsets.

    The split is performed at the flow-case level to avoid data leakage between subsets.

    Parameters
    ----------
    filepath : str
        Path to the CSV file containing the aerodynamic dataset.

    Returns
    -------
    data_train : pd.DataFrame
        Training subset of the dataset

    data_test : pd.DataFrame
        Testing subset of the dataset.
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
    Preprocess the training and validation data of a cross-validation fold.

    This function separates input and target variables, applies standardization to both,
    and returns the scaled arrays together with the fitted scalers required for inverse
    transformation during post-processing.

    Parameters
    ----------
    data_train : pandas.DataFrame
        Training subset for the current fold.

    data_val : pandas.DataFrame
        Validation subset for the current fold.

    Returns
    -------
    X_train : numpy.ndarray
        Standardized training input data.

    y_train : numpy.ndarray
        Standardized training target data.

    X_val : numpy.ndarray
        Standardized validation input data.

    y_val : numpy.ndarray
        Standardized validation target data.

    y_scaler : sklearn.preprocessing.StandardScaler
        Fitted scaler used for the target variables.

    X_scaler : sklearn.preprocessing.StandardScaler
        Fitted scaler used for the input variables.
    """

    # --- Defining input and target columns ---
    X_cols = data_train.columns[0:3]
    y_cols = data_train.columns[3:205]

    # --- Splitting the raw data into training and validation sets ---
    X_train_raw = data_train[X_cols].to_numpy()
    X_val_raw = data_val[X_cols].to_numpy()

    y_train_raw = data_train[y_cols].to_numpy()
    y_val_raw = data_val[y_cols].to_numpy()

    # --- Scaling the data ---
    X_scaler = skl.preprocessing.StandardScaler()
    X_train = X_scaler.fit_transform(X_train_raw)
    X_val = X_scaler.transform(X_val_raw)

    y_scaler = skl.preprocessing.StandardScaler()
    y_train = y_scaler.fit_transform(y_train_raw)
    y_val = y_scaler.transform(y_val_raw)

    return X_train, y_train, X_val, y_val, y_scaler, X_scaler

def build_model(trial: optuna.Trial, input_dim: int, output_dim: int):
    """
    Build and compile a Multi-Layer Perceptron model from Optuna trial parameters.

    This function defines the neural network architecture according to the hyperparameters sampled by 
    Optuna, including the number of hidden layers, the number of units per layer, the activation function, 
    the optimizer, and the learning rate.

    Parameters
    ----------
    trial : optuna.Trial
        Optuna trial object used to sample the model hyperparameters.

    input_dim : int
        Number of input features.

    output_dim : int
        Number of output variables.

    Returns
    -------
    model : keras.Model
        Compiled Keras model ready for training.
    """

    # --- Defining the hyperparameters ---
    n_hidden_layers = trial.suggest_int(name='n_hidden_layers', low=2, high=6, step=1)
    activation = trial.suggest_categorical(name='activation', choices=['swish', 'gelu'])
    learning_rate = trial.suggest_categorical(name='learning_rate', choices=[7.5e-4, 1e-3, 1.5e-3, 3e-3])
    optimizer = trial.suggest_categorical(name='optimizer', choices=['Adam', 'Nadam'])

    # --- Optimizer definition ---
    if optimizer == 'Adam':
        optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    else:
        optimizer = keras.optimizers.Nadam(learning_rate=learning_rate)
    
    # --- Defining model input layer ---
    inputs = keras.layers.Input(shape=(input_dim,), name='Input_Layer')
    x = inputs

    # --- Defining model hidden layers ---
    for i in range(n_hidden_layers):
        x = keras.layers.Dense(units=trial.suggest_int(name=f'units_layer{i+1}', low=64, high=256, 
            step=32), activation=activation, kernel_initializer='glorot_uniform', 
            name=f'Hidden_Layer{i+1}')(x)

    # --- Defining model output layer ---
    outputs = keras.layers.Dense(units=output_dim, activation='linear', 
        kernel_initializer='glorot_uniform', name='Output_Layer')(x)
    
    # --- Defining the model ---
    model = keras.Model(inputs=inputs, outputs=outputs, name='low_fidelity_mlp_ann')

    # --- Compiling the model ---
    model.compile(loss='mse', optimizer=optimizer, metrics=['mae'])
    
    return model

def predict_test(model: keras.Model, X_val: np.ndarray, y_val: np.ndarray,
    y_scaler: skl.preprocessing.StandardScaler):
    """
    Generate predictions for the validation set and reverse the output scaling.

    This function uses the trained model to predict the validation targets, then applies the inverse 
    transformation of the target scaler to both the true and predicted values in order to recover their 
    physical scale.

    Parameters
    ----------
    model : keras.Model
        Trained Keras model used for prediction.

    X_val : numpy.ndarray
        Standardized validation input data.

    y_val : numpy.ndarray
        Standardized validation target data.

    y_scaler : sklearn.preprocessing.StandardScaler
        Fitted scaler used for the target variables.

    Returns
    -------
    y_val_raw : numpy.ndarray
        Validation target values in the original scale.

    y_pred_raw : numpy.ndarray
        Predicted target values in the original scale.
    """

    # --- Predicting on the validation set ---
    y_pred = model.predict(X_val, verbose=0)

    # --- Reverses the scaling ---
    y_val_raw = y_scaler.inverse_transform(y_val)
    y_pred_raw = y_scaler.inverse_transform(y_pred)

    return y_val_raw, y_pred_raw

def compute_cl_from_cp(cp_data: np.ndarray, airfoil_data: pd.DataFrame, AoA: float):
    """
    Compute the sectional lift coefficient from a pressure coefficient distribution.

    This function integrates the pressure coefficient distribution over the airfoil surface panels to 
    obtain the normal and axial force coefficients, which are then projected into the lift direction using 
    the specified angle of attack.

    Parameters
    ----------
    cp_data : numpy.ndarray
        Pressure coefficient distribution along the airfoil surface.

    airfoil_data : pandas.DataFrame
        Airfoil geometry coordinates containing the x and y positions of the surface points.

    AoA : float
        Angle of attack, in degrees, used for force projection.

    Returns
    -------
    Cl : float
        Sectional lift coefficient computed from the pressure distribution.
    """

    # --- Converts all variables to arrays ---
    cp = np.asarray(cp_data)
    x = np.asarray(airfoil_data['x'].values)
    y = np.asarray(airfoil_data['y'].values)

    # --- Calculates panel geometric deltas (dx, dy) ---
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

def compute_CL_from_cp(X_scaled: np.ndarray, y_raw: np.ndarray, X_scaler: skl.preprocessing.StandardScaler,
    airfoil_data: pd.DataFrame, AoA_effective: pd.DataFrame):
    """
    Reconstruct the global lift coefficient from sectional pressure distributions.

    This function converts the scaled input data back to physical scale, groups the data by aerodynamic 
    flow case, computes the sectional lift coefficient for each wing section using the corresponding 
    effective angle of attack, and integrates the sectional lift distribution across the wing span to 
    obtain the global lift coefficient.

    Parameters
    ----------
    X_scaled : numpy.ndarray
        Standardized input data.

    y_raw : numpy.ndarray
        Target data in the original physical scale.

    X_scaler : sklearn.preprocessing.StandardScaler
        Fitted scaler used to inverse transform the input variables.

    airfoil_data : pandas.DataFrame
        Airfoil geometry coordinates used for sectional lift computation.

    AoA_effective : pandas.DataFrame
        Dataset containing the effective angle of attack for each Reynolds number,
        geometric angle of attack, and spanwise section.

    Returns
    -------
    numpy.ndarray
        Array containing the reconstructed global lift coefficient for each flow case.
    """

    # --- Defines the wing geometry data ---
    b = 0.766
    c = 0.2165
    S = 2*(b*c)

    # --- Defines useful list ---
    CL_list = []

    # --- Inverse transforms the input data and merge input and outputs ---
    X_raw = X_scaler.inverse_transform(X_scaled)
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
    Evaluate one Optuna trial using multi-objective cross-validation.

    This function performs 5-fold Group K-Fold cross-validation on the training dataset. For each fold, 
    it preprocesses the data, builds the model from the sampled hyperparameters, trains the model, computes 
    the prediction-space mean squared error, reconstructs the global lift coefficient from the predicted Cp 
    distributions, and evaluates the corresponding CL reconstruction error.

    The final objective values are obtained as the mean of the fold-wise errors.

    Parameters
    ----------
    trial : optuna.Trial
        Optuna trial object containing the sampled hyperparameters.

    data_train : pandas.DataFrame
        Training dataset used in the optimization procedure.

    airfoil_data : pandas.DataFrame
        Airfoil geometry coordinates used for lift reconstruction.

    AoA_effective : pandas.DataFrame
        Effective angle-of-attack dataset used in the CL reconstruction process.

    Returns
    -------
    objective_mse_loss : float
        Mean validation MSE across the cross-validation folds.

    objective_CL_loss : float
        Mean global lift coefficient reconstruction MSE across the cross-validation folds.
    """

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
        X_train, y_train, X_val, y_val, y_scaler, X_scaler = preprocess_fold_data(train_df, val_df)

        # Builds the model:
        model = build_model(trial, X_train.shape[1], y_train.shape[1])

        # Trains the model:
        callbacks = [keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, min_delta=1e-6, 
            restore_best_weights=True), 
            keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6, 
            cooldown=5, min_delta=1e-5)]
        model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=500, batch_size=128,
            callbacks=callbacks, verbose=0)

        # Evaluates the model using validation data:
        y_val_raw, y_pred_raw = predict_test(model, X_val, y_val, y_scaler)

        # Finds the validation loss and appends to the list:
        mse_output = skl.metrics.mean_squared_error(y_val_raw, y_pred_raw)
        val_mse_loss.append(mse_output) 

        # Calculates the wing lift coefficient error:
        CL_true = compute_CL_from_cp(X_val, y_val_raw, X_scaler, airfoil_data, AoA_effective)
        CL_pred = compute_CL_from_cp(X_val, y_pred_raw, X_scaler, airfoil_data, AoA_effective)

        # Appends the CL error to the list:
        mse_CL = skl.metrics.mean_squared_error(CL_true, CL_pred)
        CL_loss.append(mse_CL)

        # Clears the memory:
        tensorflow.keras.backend.clear_session()
        gc.collect()
        del model

        # Prompts the user:
        print(f'Fold {fold} completed.')

    # --- Calculates the objective functions based on the mean of the folds ---
    print('\n')
    objective_mse_loss = np.mean(val_mse_loss)
    objective_CL_loss = np.mean(CL_loss)

    return objective_mse_loss, objective_CL_loss

def main(n_trials: int):
    """
    Execute the complete hyperparameter optimization workflow.

    This function loads the required datasets, initializes the Optuna study, executes the multi-objective
    optimization process, and saves both the full trial history and the Pareto-optimal trials to disk.

    Parameters
    ----------
    n_trials : int
        Number of Optuna trials to execute.

    Returns
    -------
    None
        This function does not return any value.
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
        study_name='mlp_ann_study', 
        storage='sqlite:///LowFidelity-ANN/MLP-MultiLayerPerceptron/optimization-results/mlp_ann_study.db', 
        load_if_exists=True)

    # --- Executes the hyperparameter optimization ---
    print('\nStarting hyperparameter optimization process...\n')
    objective_with_data = partial(objective, data_train=data_train, airfoil_data=airfoil_data, 
        AoA_effective=AoA_effective)
    study.optimize(objective_with_data, n_trials=n_trials, show_progress_bar=True)

    # --- Saves all the optimization trials ---
    df_trials = study.trials_dataframe()
    df_trials = df_trials.rename(columns={'values_0': 'MSE_loss', 'values_1': 'CL_loss'})
    df_trials.to_csv('LowFidelity-ANN/MLP-MultiLayerPerceptron/optimization-results/optimization_trials.csv', 
        index=False)
    print('Saved: LowFidelity-ANN/MLP-MultiLayerPerceptron/optimization-results/optimization_trials.csv\n')

    # --- Saves the optimization trials on the Pareto front ---
    pareto_idx = [t.number for t in study.best_trials]
    df_pareto = df_trials[df_trials['number'].isin(pareto_idx)].copy()
    df_pareto.to_csv('LowFidelity-ANN/MLP-MultiLayerPerceptron/optimization-results/optimization_pareto_trials.csv', 
        index=False)
    print('Saved: LowFidelity-ANN/MLP-MultiLayerPerceptron/optimization-results/optimization_pareto_trials.csv\n')

    print('Optimization completed.\n')

    return

if __name__== "__main__":
    main(N_TRIALS)