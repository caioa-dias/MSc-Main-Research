# -*- coding: utf-8 -*-
"""
========================================================================================================
Module: optimization_potential_ann
Author: Caio Dias Filho
Creation date: 2025-12-02
Last modification: 2026-02-12
Version: 1.2
========================================================================================================

OVERVIEW
--------
This module performs the hyperparameter optimization of a low-fidelity numerical Artificial Neural Network
(ANN) designed to predict:

    - Sectional lift coefficient (Cl)
    - Chordwise pressure coefficient distribution (Cp)

The ANN is trained using a structured aerodynamic dataset generated from potential flow simulations.
Hyperparameter optimization is conducted using the Optuna framework with Tree-structured Parzen Estimator
(TPE) sampling and Median Pruning strategy.

The workflow ensures research-grade reproducibility and produces publication-ready artifacts, including:

    - Trial history export
    - Optimization convergence plots
    - Hyperparameter importance analysis (fANOVA)
    - Structured optimization report

    
WORKFLOW
--------
The complete optimization pipeline consists of:

    1) Load aerodynamic dataset (.csv)
    2) Preprocess and scale input/output variables
    3) Split dataset based on wing condition groups
    4) Define ANN architecture search space:
        - Network depth
        - Layer widths
        - Activation function
        - Regularization strategy
        - Weight initialization
        - Optimizer
        - Learning rate
        - Batch size
    5) Train models across multiple seeds to reduce stochastic bias
    6) Apply pruning during training
    7) Evaluate mean validation loss
    8) Save results, plots and textual optimization report

    
SEARCH SPACE
------------
The optimization includes:

    Architecture:
        - n_layers ∈ [2, 9]
        - units_layer_i ∈ [32, 256] (step=32)

    Train configuration:
        - optimizer ∈ {Adam, Nadam}
        - learning_rate ∈ {1e-4, 1e-3, 1e-2, 1e-1}
        - batch_size ∈ [32, 128] (step=32)

    Regularization:
        - reg_type ∈ {Dropout, Batch Normalization}
        - dropout_rate ∈ [0.05, 0.30] (if applicable)

    Initialization:
        - weight_initializer ∈ {glorot_uniform, he_uniform, lecun_normal}

        
DEPENDENCIES
------------
System libraries:
    - warnings
    - logging
    - os
    - time
    - random
    - functools
    - typing

Scientific libraries:
    - numpy
    - pandas
    - matplotlib
    - seaborn
    - sklearn

Deep learning libraries:
    - tensorflow
    - keras

Optimization libraries:
    - optuna

    
OUTPUT FILES
------------
CSV:
    - Potential-ANN/optimization-results/trials-data.csv
    - Potential-ANN/optimization-results/parameter-importance.csv

Figures (600 DPI, publication-ready):
    - Potential-ANN/optimization-results/loss_optimization_history_single.pdf
    - Potential-ANN/optimization-results/loss_optimization_history_double.pdf
    - Potential-ANN/optimization-results/parameter_importance_single.pdf
    - Potential-ANN/optimization-results/parameter_importance_double.pdf

Report:
    - Potential-ANN/optimization-results/optimization-report.txt

    
COMPUTATIONAL COST
------------------
Total runtime scales approximately with:

    N_trials x N_seeds x N_epochs x Model_complexity

where model complexity depends on:
    - Network depth
    - Layer widths
    - Batch size

Pruning reduces expected runtime by early termination of underperforming trials.


REPRODUCIBILITY
----------------
Global reproducibility is enforced through:

    - PYTHONHASHSEED
    - numpy random seed
    - tensorflow random seed
    - optuna sampler seed
    - fANOVA seed for importance evaluation

Multiple seed are used per trial to reduce stochastic variance and avoid "lucky shot" bias.


ASSUMPTIONS
------------
    - The dataset 'Potential-PressureDistributionData.csv' exists.
    - Data format matches preprocessing expectantions:
        [Re, AoA, y/b, Cl, Cp_1, ..., Cp_201]
    - Output directories already exist.
    - GPU/CPU environment is correctly configured for TensorFlow.

    
LIMITATIONS
------------
    - Training is computationally intensive for large trial counts.
    
=================================================================================================
"""

# =======================================================================================================
#                                            USER INPUT SECTION
# =======================================================================================================

# --- Number of optimization trials ---
N_TRIALS = 2


# =======================================================================================================
#                                              CORE FUNCTION
# =======================================================================================================

# System libraries:
import warnings
import logging
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
warnings.filterwarnings("ignore")
import time
import random
from functools import partial
from typing import Tuple

# Scientific libraries:
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler
from sklearn.model_selection import train_test_split

# Deep learning libraries:
import tensorflow as tf
from keras.layers import Dense, Input, Dropout, BatchNormalization
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.optimizers import Adam, Nadam
from keras.backend import clear_session
from keras.models import Sequential, clone_model
logging.getLogger('tensorflow').setLevel(logging.ERROR)
tf.get_logger().setLevel(logging.ERROR)

# Optimization libraries:
import optuna
from optuna.integration import TFKerasPruningCallback

# Reproductibility setup
SEED = 42
os.environ['PYTHONHASHSEED'] = str(SEED)
tf.random.set_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)
print(f'\nGlobal seed set to: {SEED}')

def load_and_preprocess_data(data_raw:pd.DataFrame, random_state:int):
    """
    Preprocess the aerodynamic dataset and generate a scaled training set.

    This function performs structured preprocessing of the aerodynamic dataset by grouping wing sections into
    aerodynamic conditions to prevent data leakage across spanwise sections of the same wing configuration.

    The procedure inclues:
        1. Group-based train split (by wing condition)
        2. Feature/target separation
        3. Independent scaling of:
            - Input features (MinMaxScaler)
            - Sectional lift coefficient Cl (MinMaxScaler)
            - Pressure coefficient distribution Cp (MaxAbsScaler)

    Parameters
    ----------
    data_raw : pd.DataFrame
        Raw dataset containing:
            [Re, AoA, y/b, Cl, Cp_1, ..., Cp_201]

    random_state : int
        Random seed used to shuffle wing confition groups.

    Returns
    -------
    X_train : np.ndarray
        Scaled input features array of shape (n_samples, 3).

    Y_train : np.ndarray
        Scaled target array of shape (n_samples, 1 + n_cp_points).

    Notes
    -----
    - Data is split by wing condition rather than by individual rows to ensure physical consistency and prevent
      data leakage.
    - Test set is intentionally not returned since optimization focueses on validation loss. 
    - Scaling is fitted only on training data.
    """

    # Create an index array to assign grouped wing conditions
    # Each aerodynamic configuration contains a fixed number of 80 spanwise sections
    data = data_raw.copy()
    wing_sections = 80
    indices = np.arange(len(data))
    data['wing_condition'] = indices // wing_sections

    # Split aerodynamic conditions instead of individual samples
    # This avoids leakage between spanwise sections of the same wing
    unique_conditions = data['wing_condition'].unique()

    train_conds, test_conds = train_test_split(
        unique_conditions, 
        test_size=0.2, 
        random_state=random_state,
        shuffle=True)
    
    # Select only training aerodynamic configurations
    data_train = data[data['wing_condition'].isin(train_conds)].drop(columns=['wing_condition'])

    # Separate input features (Re, AoA, y/b)
    input_cols = data_train.columns[0:3]
    X_train = data_train[input_cols].values

    # Separate target variables:
    Y_cl_train = data_train.iloc[:, 3].values.reshape(-1, 1)
    Y_cp_train = data_train.iloc[:, 4:].values

    # --------------------------------------------------------------------------------------------------
    # Feature Scaling
    # --------------------------------------------------------------------------------------------------

    # Scale input variables to [0, 1] range
    # This improves optimization stability and gradient convergence
    scaler_x = MinMaxScaler()
    X_train = scaler_x.fit_transform(X_train)
    
    # Independently scale Cl to [0, 1] range
    # Cl has different physical magnitude compared to Cp
    cl_scaler = MinMaxScaler()
    Y_cl_train = cl_scaler.fit_transform(Y_cl_train)
    
    # Scale Cp distribution using MaxAbsScaler
    # Cp contains negative values; preserving relative sign is important
    cp_scaler = MaxAbsScaler()
    Y_cp_train = cp_scaler.fit_transform(Y_cp_train)
    
    # Concatenate the targets back together:
    Y_train = np.hstack([Y_cl_train, Y_cp_train])

    return X_train, Y_train

def prepare_datasets_for_seeds(data_raw: pd.DataFrame, seeds: list):
    """
    Generate preprocessed datasets for multiple random seeds.

    This function prepares independent training datasets using different random splits to reduce stochastic bias
    during hyperparameter optimization.

    Parameters
    ----------
    data_raw : pd.DataFrame
        Raw dataset containing:
            [Re, AoA, y/b, Cl, Cp_1, ..., Cp_201].

    seeds : list
        List of random seeds used to generate independent splits.
    
    Returns
    -------
    datasets : dict
        Dictionary mapping each seed to a tuple:
            seed -> (X_train, Y_train)
    
    Notes
    -----
    - Each seed produces a different grouping-based split of aerodynamic conditions, increasing robustness of
      validation performance estimation.
    """

    # Initialize container to store preprocessed datasets indexed by seed:
    datasets = {}

    # Iterate over predefined seeds to create independent grouped splits:
    for seed in seeds:

        # Perform grouped split and scaling using the specified random state:
        X_train, Y_train = load_and_preprocess_data(data_raw, random_state=seed)

        # Store processed datasets to ensure deterministic reuse during trials:
        datasets[seed] = (X_train, Y_train)

    # Return dictionary containing all precomputed training datasets:
    return datasets

def build_model(trial: optuna.Trial, input_shape: Tuple[int], output_shape: int):
    """
    Construct the ANN architecture for a given Optuna trial

    The architecture search space includes:
        - Network depth
        - Layer widths
        - Activation function
        - Regularization strategy
        - Weight initialization

    Parameters
    ----------
    trial : optuna.Trial
        Optunal trial object responsible for sampling hyperparameters.

    input_shape : Tuple[int]
        Shape of the input layer (number of input features).

    output_shape : int
        Total number of outputs (Cl + Cp distribution).

    Returns
    -------
    model : keras.Sequential
        Uncompiled Keras sequential model.

    Notes
    -----
    - Regularization is applied after each hidden layer.
    - Weight intialization is consistent across all layers in a trial.
    - Output layer uses linear activation for regression.
    """

    # Initialize sequential container for fully-conected feedfoward network:
    model = Sequential()

    # Sample architectural depth controlling model capacity:
    n_layers = trial.suggest_int('n_layers', 2, 9)

    # Sample shared hyperparameters applied consistently across hidden layers:
    activation = trial.suggest_categorical('activation', ['swish', 'tanh', 'gelu'])
    initializer_name = trial.suggest_categorical('weight_initializer', ['glorot_uniform', 'he_uniform', 'lecun_normal'])

    # Sample regularization strategy to control overfitting behavior:
    reg_type = trial.suggest_categorical('reg_type', ['Dropout', 'Batch Normalization'])

    if reg_type == 'Dropout':
        dropout_rate = trial.suggest_float('dropout_rate', 0.05, 0.3, step=0.05)

    # Construct hidden layers according to sampled depth:
    for i in range(n_layers-1):

        # Add fully conected layer with sampled configuration:
        if i == 0:
            # First layer requires explicit input shape specification
            model.add(Input(shape=input_shape))
        else:
            # Sample layer width defning representational power:
            units = trial.suggest_int(f'units_layer{i}', 32, 256, step=32)
            model.add(Dense(units, activation=activation, kernel_initializer=initializer_name))

        # Apply selected regularization mechanism after each hidden layer:
        if reg_type == 'Dropout':
            model.add(Dropout(dropout_rate))
        if reg_type == 'Batch Normalization':
            model.add(BatchNormalization())

    # Add linear output layer for multi-target regression:
    model.add(Dense(output_shape, activation='linear', kernel_initializer=initializer_name))

    return model

def objective(trial: optuna.Trial, datasets: dict, seeds: list):
    """
    Objective function for hyperparameter optimization.

    This function evaluates a sampled hyperparameter configuration by:

        1. Building the base ANN architecture
        2. Training the model across multiple predefined seeds
        3. Computing the mean validation loss
        4. Applying pruning based on validation performance

    Parameters
    ----------
    trial : optuna.Trial
        Optunal trial object.

    datasets : dict
        Dictionary mapping seed -> (X_train, Y_train).

    seeds : list[int]
        List of seeds used for robustness evaluation.

    Returns
    -------
    mean_loss : float
        Mean minimum validation loss across all seeds.

    Methodology
    -----------
    - Each seed corresponds to an independent grouped data split.
    - Model architecture is built once and cloned for each seed.
    - Pruning is applied only on the first seed to avoid over-pruning.
    - Early stopping and learning rate reduction are used to stabilize training.

    Notes
    -----
    - Using multiple seeds mitigate the risk of selecting hyperparameters that perform well due to favorable
      stochastic initialization.
    """

    # Sample optimization hyperparameters related to training dynamics:
    optimizer_name = trial.suggest_categorical('optimizer', ['Adam', 'Nadam'])
    lr = trial.suggest_categorical('learning_rate', [1e-4, 1e-3, 1e-2, 1e-1])
    batch_size = trial.suggest_int('batch_size', 32, 128, step=32)

    losses = []

    # Iterate over predefined seeds to reduce stochastic bias:
    # Each seed corresponds to an independent grouped data split
    for i, seed in enumerate(seeds):

        # Ensure deterministic behavior for this specific seed:
        tf.keras.backend.clear_session()
        tf.random.set_seed(seed)
        np.random.seed(seed)

        X_train, Y_train = datasets[seed]

        # Build model architecture sampled for this trial:
        # Architecture is idential across seeds, only weight differ
        model = build_model(trial, input_shape=(X_train.shape[1],), output_shape=Y_train.shape[1])
                            
        # Instantiate optimizer dynamically to allow learning rate tuning:
        if optimizer_name == 'Adam':
            optimizer = Adam(learning_rate=lr)
        else:
            optimizer = Nadam(learning_rate=lr)
        
        model.compile(loss='mean_squared_error', optimizer=optimizer, metrics=['mae'])

        # Callbacks to stabilize and regularize training:
        callbacks = [EarlyStopping(monitor='val_loss', patience=30, restore_best_weights=True),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=15, min_lr=1e-6)]
        
        # Pruning is applied only to the first seed to avoid execcive computational overhead:
        if i == 0: 
            callbacks.append(TFKerasPruningCallback(trial, 'val_loss'))
        
        history = model.fit(X_train, Y_train, validation_split=0.2, epochs=200, verbose=0, 
            batch_size=batch_size, callbacks=callbacks)
        
        # Extract minimum validation loss achieved during training
        losses.append(min(history.history['val_loss']))

    # Aggregate performance across seeds
    mean_loss = float(np.mean(losses))
         
    return mean_loss

def save_results(study: optuna.Study, execution_time: float):
    """
    Perform comprehensive post-processing of an Optuna hyperparameters optimization study.

    This function executes a complete post-optimization workflow designed for research-grade reproducibility
    and direct inclusion in scientific publications. The procedure includes structured data export, convergence 
    visualization, hyperparameter importance analysis, and generation of a formal optimization report.

    Workflow
    --------
    1. Export raw trial history to .csv format.
    2. Generate loss convergence plots formatted for:
        - Single-column layout (.pdf)
        - Double-column layout (.pdf)
    3. Compute hyperparameter importance using the Fanova evaluator.
    4. Export hyperparameter importance data to .csv format.
    5. Generate hyperparameter importance plots formatted for:
        - Single-column layout (.pdf)
        - Double-column layout (.pdf)
    6. Generate a structured textual optimization report summarizing:
        - Best objective value
        - Number of total and pruned trials
        - Best hyperparameter configuration
        - Relative hyperparameter importance ranking

    All figures are generated at high resolution (600 DPI) using serif fonts (Times New Roman) and STIX math
    rendering to ensure compatibility with academic journal submission standards.

    Parameters
    ----------
    study : optuna.Study
        Complete Optuna study object containing all optimization trials.
        The study must include:
            - Completed trial values
            - Best parameters
            - Trial states (COMPLETED / PRUNED)

    execution_time : float
        Total optimization runtime in seconds.

    Returns
    -------
    None
        This function does not return objects. It generates and saves multiple artifacts to disk.

    Output Files
    ------------
    .csv:
        - Potential-ANN/optimization-results/trials-data.csv
        - Potential-ANN/optimization-results/parameter-importance.csv
    
    Figures (optimization-plots/, 600 DPI .pdf):
        - loss_optimization_history_double.png
        - loss_optimization_history_single.png
        - parameter_importance_double.png
        - parameter_importance_single.png

    Report:
        - Potential-ANN/optimization-results/optimization-report.txt
    
    Side Effects
    ------------
    - Reads internal trial data from the Optuna study.
    - Writes multiple .csv, .png, and .txt files to disk.
    - Overwrites exising files with identical names.
    - Uses global variable 'SEED' for fANOVA reproducibility.

    Computational Notes
    ------------------
    - Hyperparameter importance computation scales with:
        O(N_trials * N_parameters)
    - Plot generation cost is negliable relative to optimization runtime.
    - File I/O operations may become significant for very large studies.

    Assumptions
    -----------
    - The study contains at least one completed trial.
    - The objective is minimization (cumulative minimum is used).
    - Required directories already exist.
    - 'SEED' is defined globally for reproducibility.

    Reproducibility
    --------------
    Importance evaluation uses:
        optuna.importance.FanovaImportanceEvaluator(seed=SEED)

    Ensuring deterministic importance ranking when the same study and seed are provided.
    """
    
    # ===================================================================================================
    # 1. SINGLE-COLUMN VISUALIZATION: LOSS OPTIMIZATION HISTORY
    # ===================================================================================================
    # Setting plot parameters:
    with plt.rc_context({'font.family': 'serif', 'font.serif': ['Times New Roman'], 
        'mathtext.fontset': 'stix'}):

        # --- Data ---
        data = study.trials_dataframe()
        data.to_csv('Potential-ANN/optimization-results/trials-data.csv', index=False)
        data['best_value'] = data['value'].cummin()

        # --- Figure size: single-column --- 
        fig, ax = plt.subplots(figsize=(3.35, 2.55))

        # --- Plot ---
        ax.scatter(data['number'] + 1, data['value'], alpha=1.0, s=6, facecolors='none', edgecolors='#000000', 
            label='Trial mean validation loss', linewidth=0.4)
        
        ax.plot(data['number'] + 1, data['best_value'], color='#DC143C', linewidth=0.8, label='Best validation loss')

        # --- Labels ---
        ax.set_xlabel('Trial number', fontsize=9, fontname='Times New Roman')
        ax.set_ylabel('Mean validation loss (MSE)', fontsize=9, fontname='Times New Roman')

        # --- Ticks ---
        ax.tick_params(axis='both', which='major', labelsize=7, width=0.8, direction='in')
        ax.ticklabel_format(axis='y', style='sci', scilimits=(0, 0), useMathText=True)
        ax.yaxis.get_offset_text().set_fontsize(7)

        # --- Grid ---
        ax.grid(True, which='major', linestyle='--', linewidth=0.5, alpha=0.4)

        # --- Limits ---
        ax.set_xlim(1, data['number'].max() + 1)
        ax.set_ylim(0, 7.5e-3)

        # --- Legend ---
        ax.legend(fontsize=7, fancybox=False, edgecolor='black', loc='upper right')

        # --- Save ---
        plt.tight_layout(pad=0.6)
        plt.savefig('Potential-ANN/optimization-results/loss_optimization_history_single.pdf', dpi=600)
        plt.close()
        print(f'\nPlot saved as Potential-ANN/optimization-results/loss_optimization_history_single.pdf')


    # ===================================================================================================
    # 2. DOUBLE-COLUMN VISUALIZATION: LOSS OPTIMIZATION HISTORY
    # ===================================================================================================
    # Setting plot parameters:
    with plt.rc_context({'font.family': 'serif', 'font.serif': ['Times New Roman'], 
        'mathtext.fontset': 'stix'}):

        # --- Figure size: double-column --- 
        fig, ax = plt.subplots(figsize=(6.7, 4.8))

        # --- Plot ---
        ax.scatter(data['number'] + 1, data['value'], alpha=1.0, s=16, facecolors='none', edgecolors='#000000', 
            label='Trial mean validation loss', linewidth=0.5)
        
        ax.plot(data['number'] + 1, data['best_value'], color='#DC143C', linewidth=1.6, label='Best validation loss')

        # --- Labels ---
        ax.set_xlabel('Trial number', fontsize=11, fontname='Times New Roman')
        ax.set_ylabel('Mean validation loss (MSE)', fontsize=11, fontname='Times New Roman')

        # --- Ticks ---
        ax.tick_params(axis='both', which='major', labelsize=10, width=0.8, direction='in')
        ax.ticklabel_format(axis='y', style='sci', scilimits=(0, 0), useMathText=True)
        ax.yaxis.get_offset_text().set_fontsize(9)

        # --- Grid ---
        ax.grid(True, which='major', linestyle='--', linewidth=0.6, alpha=0.4)

        # --- Limits ---
        ax.set_xlim(1, data['number'].max() + 1)
        ax.set_ylim(0, 7.5e-3)

        # --- Legend ---
        ax.legend(fontsize=9, fancybox=False, edgecolor='black', loc='upper right')

        # --- Save ---
        plt.tight_layout(pad=0.6)
        plt.savefig('Potential-ANN/optimization-results/loss_optimization_history_double.pdf', dpi=600)
        plt.close()
        print(f'Plot saved as Potential-ANN/optimization-results/loss_optimization_history_double.pdf\n')


    # ===================================================================================================
    # 3. SINGLE-COLUMN VISUALIZATION: PARAMETER IMPORTANCE
    # ===================================================================================================
    # Setting plot parameters:
    with plt.rc_context({'font.family': 'serif', 'font.serif': ['Times New Roman'], 
        'mathtext.fontset': 'stix'}):

        # --- Data ---
        importance = optuna.importance.get_param_importances(study, evaluator=optuna.importance.FanovaImportanceEvaluator(seed=SEED))
        data = pd.DataFrame(list(importance.items()), columns=['Hyperparameter', 'Importance'])
        data.to_csv('Potential-ANN/optimization-results/parameter-importance.csv', index=False)
        new_name = {'activation': r'Activation function', 'n_layers': r'Network depth', 'reg_type': r'Regularization strategy',
            'optimizer': r'Optimizer', 'batch_size': r'Batch size', 'learning_rate': r'Learning rate', 
            'units_layer1': r'2nd layer width', 'units_layer2': r'3rd layer width', 'units_layer3': r'4th layer width', 
            'units_layer4': r'5th layer width', 'units_layer5': r'6th layer width', 'units_layer6': r'7th layer width', 
            'units_layer7': r'8th layer width', 'dropout_rate': r'Dropout rate', 'weight_initializer': r'Weight initialization'}
        data['Parameters'] = data['Hyperparameter'].map(new_name).fillna(data['Hyperparameter'])

        # --- Figure size: single-column --- 
        fig, ax = plt.subplots(figsize=(3.35, 2.55))

        # --- Plot ---
        ax = sns.barplot(data=data, x='Importance', y='Parameters', color='lightgray', edgecolor='#000000')

        # --- Labels ---
        ax.set_xlabel('Relative importance', fontsize=9, fontname='Times New Roman')
        ax.set_ylabel(' ', fontsize=9, fontname='Times New Roman')

        # --- Ticks ---
        ax.tick_params(axis='both', which='major', labelsize=8, width=0.8, direction='in')

        # --- Grid ---
        ax.grid(True, which='major', linestyle='--', linewidth=0.5, alpha=0.4)

        # --- Limits ---
        ax.set_xlim(0, data['Importance'].max() * 1.15)

        # --- Save ---
        plt.tight_layout(pad=0.6)
        plt.subplots_adjust(left=0.355)
        plt.savefig('Potential-ANN/optimization-results/parameter_importance_single.pdf', dpi=600)
        plt.close()
        print(f'Plot saved as Potential-ANN/optimization-results/parameter_importance_single.pdf')


    # ===================================================================================================
    # 4. DOUBLE-COLUMN VISUALIZATION: PARAMETER IMPORTANCE
    # ===================================================================================================
    # Setting plot parameters:
    with plt.rc_context({'font.family': 'serif', 'font.serif': ['Times New Roman'], 
        'mathtext.fontset': 'stix'}):

        # --- Figure size: double-column paper --- 
        fig, ax = plt.subplots(figsize=(6.7, 4.8))

        # --- Plot ---
        ax = sns.barplot(data=data, x='Importance', y='Parameters', color='lightgray', edgecolor='#000000')

        # --- Labels ---
        ax.set_xlabel('Relative importance', fontsize=11, fontname='Times New Roman')
        ax.set_ylabel(' ', fontsize=11, fontname='Times New Roman')

        # --- Ticks ---
        ax.tick_params(axis='both', which='major', labelsize=10, width=0.8, direction='in')

        # --- Grid ---
        ax.grid(True, which='major', linestyle='--', linewidth=0.6, alpha=0.4)

        # --- Limits ---
        ax.set_xlim(0, data['Importance'].max() * 1.15)

        # --- Save ---
        plt.tight_layout(pad=0.6)
        plt.subplots_adjust(left=0.22)
        plt.savefig('Potential-ANN/optimization-results/parameter_importance_double.pdf', dpi=600)
        plt.close()
        print(f'Plot saved as Potential-ANN/optimization-results/parameter_importance_double.pdf\n')


    # ===================================================================================================
    # 5. TEXT FILE: MODEL OPTIMIZATION REPORT
    # ===================================================================================================
    # --- Data ---
    data = study.trials_dataframe()

    # --- Textual Content ---
    content = f"""
PARAMETER OPTIMIZATION REPORT - LF_NUMERICAL_ANN
===========================================================
Date: {time.strftime("%Y-%m-%d %H:%M:%S")}
Execution Time: {execution_time:.2f} seconds

1. BEST MODEL RESULTS
----------------------------------
Best Mean Validation Loss (MSE): {study.best_value:.6f}
Total Trials: {len(study.trials)}
Pruned Trials: {len(data[data['state'] == 'PRUNED'].copy())}
                
Best Hyperparameters:
{"\n".join([f"- {k:<25}: {v}" for k, v in study.best_params.items()])}

2. HYPERPARAMETER IMPORTANCE
----------------------------------
{"\n".join([f"- {k:<25}: {v:.4f}" for k, v in importance.items()])}
==========================================================
"""

    # --- Save ---
    with open('Potential-ANN/optimization-results/optimization-report.txt', "w", encoding="utf-8") as f:
        f.write(content)
    print("Report saved as 'Potential-ANN/optimization-results/optimization-report.txt'\n")

    return

def main(n_trials: int):
    """
    Execute the full hyperparameter optimization workflow.

    Parameters
    ----------
    n_trials : int
        Number of optimization trials to perform.

    Workflow
    --------
        1. Load aerodynamic dataset
        2. Prepare datasets for multiple seeds
        3. Create Optuna study with TPE sampler and Median pruner
        4. Run optimization
        5. Save plots, reports and structured results

    Returns
    -------
    None

    Notes
    -----
    - Total runtime scales approcimately with:
        n_trials x n_seeds x training_time_per_model
    """

    # Load raw aerodynamic dataset generated from potential flow simulations:
    print('\nLoading dataset into memory...')
    data_raw = pd.read_csv('Potential-PressureDistributionData.csv', sep=';')

    # Define fixed random seeds used to reduce stochastic bias in validation performance:
    # Multiple seeds provide robustness against favorable random initialization effects.
    seeds = [3, 42, 123]

    # Precompute grouped and scaled datasets once to avoid redundant preprocessing during each trial, significantly
    # reducing total computational cost:
    datasets = prepare_datasets_for_seeds(data_raw, seeds)

    # Initialize Optuna study with:
    #   - TPE sampler
    #   - Median pruner
    # Fixed seed ensures deterministic sampling behavior.
    study = optuna.create_study(direction='minimize', sampler=optuna.samplers.TPESampler(seed=SEED), 
        pruner=optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=50, interval_steps=5), 
        study_name='Optimization-Potential-ANN')

    # Measure wall-clock optimization time for reporting and reproducibility:
    start_time = time.time()

    # Execute hyperparameter optimization:
    # Each trial evaluates architecure + training configuration acrsoss multiple independent dataset splits
    print('\nStarting optimization process...\n')
    objective_with_data = partial(objective, datasets=datasets, seeds=seeds)
    study.optimize(objective_with_data, n_trials=n_trials)

    # Compute total execution time for structured reporting:
    execution_time = time.time() - start_time

    # Perform structured post-processing:
    #   - Export trials dataframe
    #   - Generate convergence plots
    #   - Compute hyperparameter importance (fANOVA)
    #   - Save textual optimization summary
    save_results(study, execution_time)
    print('Optimization completed.\n')

    return

if __name__== "__main__":

    main(N_TRIALS)