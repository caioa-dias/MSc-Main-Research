# -*- coding: utf-8 -*-
"""
========================================================================================================
Module: tuning_MLP_ANN
Author: Caio Dias Filho
Creation date: 2026-04-03
Last modification: 2026-04-09
Version: 2.0 (final)
========================================================================================================

OVERVIEW
--------
This module performs hyperparameter tuning of a low-fidelity Multi-Layer Perceptron Artificial Neural
Network (MLP-ANN) using a grid seach approach.

The model is trained to reconstruct aerodynamic quantites from low fidelity data, predicting:

    - Sectional lift coefficient (Cl)
    - Pressure coefficient distribution (Cp)

The tuning process evaluates combinations of:

    - Batch size
    - Weight initialization strategy

Two objective functions are computed:

    1. Prediction-space Mean Squared Error (MSE)
    2. Global lift coefficient (CL) reconstruction Mean Squared Error (MSE)

The CL metric is obtained by integrating the predicted pressure coefficient distribution using effective
angle-of-attack corrections.


WORKFLOW
--------
The workflow implemented in this module consists of:

    - Loading and splitting the dataset at flow-case level
    - Performing K-fold cross-validation
    - Preprocessing each fold independently using feature standardization
    - Training the neural network for each hyperparameter combination
    - Evaluating prediction accuracy (MSE)
    - Reconstructing global lift coefficient (CL) from Cp distributions
    - Computing CL reconstruction error
    - Aggregating results across folds
    - Saving all results and extracting the Pareto-optimal solutions


SEARCH SPACE
------------
The hyperparameter search space includes:

    - Batch size:
        32, 64, 128, 256

    - Kernel initializer:
        - glorot_uniform
        - glorot_normal
        - he_uniform
        - he_normal

        
DEPENDENCIES
------------
Python libraries:
    - os
    - warning
    - numpy
    - pandas
    - scikit-learn
    - tensorflow / keras
    - itertools
    - gc


OUTPUT FILES
------------
Grid search results:
    
    - LowFidelity-ANN/optimization-results/mlp_ann/grid_search_results.csv

Pareto-optimal solutions:

    - LowFidelity-ANN/optimization-results/mlp_ann/grid_search_pareto_results.csv

    
REPRODUCIBILITY
---------------
Reproductibility is ensured through:

    - Fixed global random seed (SEED = 42)
    - Fixed NumPy seeds
    - Fixed TensorFlow seeds
    - Deterministic dataset splitting


ASSUMPTIONS
-----------
The implementation assumes:

    - Each aerodynamic case contains 80 spanwise sections
    - The dataset follows the structure:
        Columns 0–2 : input variables (Re, AoA, y/b)
        Columns 3+  : target variables (Cl and Cp)
    - Airfoil geometry is available for Cp integration
    - Effective angle-of-attack data is consistent with the aerodynamic dataset
    - The weight initialization strategies are compatible with the activation function obtained from the
      overall architecture optimization.


LIMITATIONS
-----------
Potential limitations include:

    - Exhaustive grid search increses computatiional cost
    - No adaptive search strategy (e.g., Bayesian optimization)
    - Fixed neural network architecture (already optimized)
    - CL reconstructionaccuracy depends on AoA_effective dataset quality

========================================================================================================
"""

# =======================================================================================================
#                                            USER INPUT SECTION
# =======================================================================================================

# --- Search space for hyperparameter tuning ---
BATCH_SIZES = [32, 64, 128, 256]
INITIALIZERS = ['glorot_uniform', 'glorot_normal', 'he_uniform', 'he_normal']


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
import sklearn as skl
import pandas as pd
import numpy as np
import itertools
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
    Load and split the aerodynamic dataset into training and testing sets.

    this function reads the dataset, assigns a flow-case identifier assuming a fixed number of wing sections
    per case, and performs a case-wise split to avoid data leakage.

    Parameters
    ----------
    filepath : str
        Path to the dataset file.

    Returns
    -------
    data_train : pd.DataFrame
        Training dataset.

    data_test : pd.DataFrame
        Testing dataset.
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
    Preprocess data for a cross-validation fold.

    This function separates input and target variables, applies standardzation, and returns scaled arrays
    along with fitted scalers.

    Parameters
    ----------
    data_train : pd.DataFrame
        Training dataset for the fold.

    data_val : pd.DataFrame
        Validation dataset for the fold.

    Returns
    -------
    X_train : np.ndarray
        Standardized training inputs.

    y_train : np.ndarray
        Standardized training targets.

    X_val : np.ndarray
        Standardized validation inputs.

    y_val : np.ndarray
        Standardized validation targets.

    y_scaler : skl.preprocessing.StandardScaler
        Scaler for target variables.

    X_scaler : skl.preprocessing.StandardScaler
        Scaler for input variables.
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

def build_model(initializer: str, input_dim: int, output_dim: int):
    """
    Build and compile the MLP neural network.

    The architecture is fixed and uses multiple dense layers with GELU activation. The kernel initializer
    is provided as a hyperparameter.

    Parameters
    ----------
    initializer : str
        Weight initialization method.

    input_dim : int
        Number of input features.
    
    output_dim : int
        Number of output features.

    Returns
    -------
    model : keras.Model
        Compiled neural network model.
    """

    # --- Defining model input layer ---
    inputs = keras.layers.Input(shape=(input_dim,), name='Input_Layer')
    x = inputs

    # --- Defining model hidden layers ---
    x = keras.layers.Dense(units=96, activation='gelu', kernel_initializer=initializer, 
        name='Hidden_Layer1')(x)
    x = keras.layers.Dense(units=160, activation='gelu', kernel_initializer=initializer, 
        name='Hidden_Layer2')(x)
    x = keras.layers.Dense(units=160, activation='gelu', kernel_initializer=initializer, 
        name='Hidden_Layer3')(x)
    x = keras.layers.Dense(units=192, activation='gelu', kernel_initializer=initializer, 
        name='Hidden_Layer4')(x)
    x = keras.layers.Dense(units=224, activation='gelu', kernel_initializer=initializer, 
        name='Hidden_Layer5')(x)
    
    # --- Defining model output layer ---
    outputs = keras.layers.Dense(units=output_dim, activation='linear', 
        kernel_initializer=initializer, name='Output_Layer')(x)
    
    # --- Defining the model ---
    model = keras.Model(inputs=inputs, outputs=outputs, name='low_fidelity_mlp_ann')

    # --- Compiling the model ---
    model.compile(loss='mse', optimizer=keras.optimizers.Nadam(learning_rate=3e-3), 
        metrics=['mae']) 

    return model

def predict_test(model: keras.Model, X_val: np.ndarray, y_val: np.ndarray,
    y_scaler: skl.preprocessing.StandardScaler):
    """
    Generate predictions and convert them back to physical scale.

    Parameters
    ----------
    model : keras.Model
        Trained neural network model.
    
    X_val : np.ndarray
        Standardized validation inputs.

    y_val : np.ndarray
        Standardized validation targets.
    
    y_scaler : skl.preprocessing.StandardScaler
        Target scaler used for inverse transformation.

    Returns
    -------
    y_val_raw : np.ndarray
        True values in physical scale.

    y_pred_raw : np.ndarray
        Predicted values in physical scale.
    """

    # --- Predicting on the validation set ---
    y_pred = model.predict(X_val, verbose=0)

    # --- Reverses the scaling ---
    y_val_raw = y_scaler.inverse_transform(y_val)
    y_pred_raw = y_scaler.inverse_transform(y_pred)

    return y_val_raw, y_pred_raw

def compute_cl_from_cp(cp_data: np.ndarray, airfoil_data: pd.DataFrame, AoA: float):
    """
    Compute sectional lift coefficient from pressure coefficient distribution.

    Parameters
    ----------
    cp_data : np.ndarray
        Pressure coefficient distribution.

    airfoil_data : pd.DataFrame
        Airfoil geometry coordinates.

    AoA : float
        Angle of attacks in degrees.

    Returns
    -------
    Cl : float
        Sectional lift coefficient.
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

    # --- Projects Cn and Ca into the lift coefficient (Cl) using the angle of attack ---
    alpha = np.radians(AoA)
    Cl = Cn * np.cos(alpha) - Ca * np.sin(alpha)

    return Cl

def compute_CL_from_cp(X_scaled: np.ndarray, y_raw: np.ndarray, X_scaler: skl.preprocessing.StandardScaler,
    airfoil_data: pd.DataFrame, AoA_effective: pd.DataFrame):
    """
    Reconstruct the global lift coefficient (CL) from Cp distributions.

    This function computes sectional lift coefficients using effective angles of attack and integrates
    them along the span.

    Parameters
    ----------
    X_scaled : np.ndarray
        Standardized inputs.
    
    y_raw : np.ndarray
        Target values in physical scale.

    X_scaler : skl.preprocessing.StandardScaler
        Input scaler.

    airfoil_data : pd.DataFrame
        Airfoil geometry coordinates.

    AoA_effective : pd.DataFrame
        Effective angle of attack data for each flow case.

    Returns
    -------
    CL_list : np.ndarray
        List of reconstructed global lift coefficients for each flow case.    
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

def objective(batch_size: int, initializer: str, data_train: pd.DataFrame, airfoil_data: pd.DataFrame,
    AoA_effective: pd.DataFrame):
    """
    Evaluate one hyperparameter combination using cross-validation.

    This function performs 5-fold Group K-Fold cross-validation, trains the model, computes prediction MSE,
    reconstructs CL, and evaluates CL error.

    Parameters
    ----------
    batch_size : int
        Batch size used during training.

    initializer : str
        Weight initialization method.

    data_train : pd.DataFrame
        Training dataset.

    airfoil_data : pd.DataFrame
        Airfoil geometry coordinates.

    AoA_effective : pd.DataFrame
        Effective angle of attack data for each flow case.

    Returns
    -------
    objective_mse_loss : float
        Mean MSE loss across folds.

    objective_CL_loss : float
        Mean CL reconstruction MSE across folds.
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
        model = build_model(initializer, X_train.shape[1], y_train.shape[1])

        # Trains the model:
        callbacks = [keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, min_delta=1e-6,
            restore_best_weights=True),
            keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6,
            cooldown=5, min_delta=1e-5)]
        model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=500, batch_size=batch_size,
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
        CL_loss.append((mse_CL))

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

def extract_pareto_front(data_results: pd.DataFrame, objective1: str, objective2: str):
    """
    Extract Pareto-optimal solutions from the grid search results.

    A solution is considered Pareto-optimal if no other solution improves one objective without worsening 
    the other.
    
    Parameters
    ----------
    data_results : pd.DataFrame
        DataFrame containing the grid search results with objective values.

    objective1 : str
        Name of the first objective.

    objective2 : str
        Name of the second objective.

    Returns
    -------
    pareto_front : pd.DataFrame
        DataFrame containing only the Pareto-optimal solutions.
    """

    # --- Extracts the non-dominated solutions ---
    pareto_mask = []

    # -- Loops through each solution and checks if it is dominated by any other solution ---
    for i, row_i in data_results.iterrows():
        dominated = False
    
        for j, row_j in data_results.iterrows():
            if i == j:
                continue

            # Checks if the current solution is dominated by the other solution:
            condition1 = row_j[objective1] <= row_i[objective1]
            condition2 = row_j[objective2] <= row_i[objective2]
            strict = (row_j[objective1] < row_i[objective1]) or (row_j[objective2] < row_i[objective2])

            # If the current solution is dominated by the other solution, break the loop:
            if condition1 and condition2 and strict:
                dominated = True
                break
        
        # --- If the current solution is not dominated, add it to the list ---
        pareto_mask.append(not dominated)

    return data_results[pareto_mask].copy()

def main(batch_sizes: list, initializers: list):
    """
    Execute the grid search hyperparameter tuning process.

    This function evaluates all combinations of batch size and initializer, computes both objective 
    functions, saves results, and extracts the Pareto front.

    Parameters
    ----------
    batch_sizes : list
        List of batch sizes to evaluate.

    initializers : list
        List of initializers to evaluate.

    Returns
    -------
    None
        Saves results to CSV files and prints progress to the user.
    """

    # --- Defines the data path ---
    print('\nLoading dataset into memory...\n')
    data_path = 'LowFidelity-ANN/utils/LowFidelity-PressureDistributionData.csv'
    airfoil_data_path = 'LowFidelity-ANN/utils/NACA23015.csv'
    aoa_effective_path = 'LowFidelity-ANN/utils/LowFidelity-AoAEffectiveData.csv'

    # --- Defines the airfoil data ---
    airfoil_data = pd.read_csv(airfoil_data_path, sep=',', names=['x', 'y'])
    AoA_effective = pd.read_csv(aoa_effective_path, sep=';')

    # --- Loads and splits the data ---
    data_train, _ = load_and_split_data(data_path)

    # --- Defines the grid search combinations ---
    results = []
    combinations = list(itertools.product(batch_sizes, initializers))
    print(f'Starting grid search with {len(combinations)} combinations...\n')

    # --- Runs the grid search ---
    for run_id, (batch_size, initializer) in enumerate(combinations, start=1):

        # Promps the user:
        print(f'Running combination {run_id}/{len(combinations)}: batch_size={batch_size}, initializer={initializer}')

        # Evaluates the objectives:
        objective_mse_loss, objective_CL_loss = objective(batch_size, initializer, 
            data_train, airfoil_data, AoA_effective)
        
        # Appends the results to the list:
        results.append({'run_id': run_id, 'batch_size': batch_size, 'kernel_initializer': initializer,
            'MSE_loss': objective_mse_loss, 'CL_loss': objective_CL_loss})
        
        # Promps the user:
        print(f'Finished combination {run_id}/{len(combinations)}')

    # --- Saves the results to a CSV file ---
    data_results = pd.DataFrame(results)
    data_results.to_csv('LowFidelity-ANN/optimization-results/mlp_ann/tuning_results.csv', index=False)
    print('Saved: LowFidelity-ANN/MLP-MultiLayerPerceptron/optimization-results/tuning_results.csv\n')

    # --- Saves the Pareto front results to a CSV file ---
    data_pareto = extract_pareto_front(data_results, objective1='MSE_loss', objective2='CL_loss')
    data_pareto.to_csv('LowFidelity-ANN/optimization-results/mlp_ann/tuning_pareto_results.csv', index=False)
    print('Saved: LowFidelity-ANN/MLP-MultiLayerPerceptron/optimization-results/tuning_pareto_results.csv\n')

    # --- Prompts the user ---
    print('Grid search completed.\n')

    return

if __name__ == "__main__":
    main(BATCH_SIZES, INITIALIZERS)