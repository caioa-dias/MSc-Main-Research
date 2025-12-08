# -*- coding: utf-8 -*-
"""
--------------------------------------------------------------------------------------------------------
Function:               LF_numerical_ann_optimization
Author:                 Caio Dias Filho
Creation date:          2025-12-02
Last modification:      2025-12-08
Version:                1.0

Description:
    This script performs the architecture optimization of an Artificial Neural Network (ANN) that predicts
    the pressure coefficient (Cp) distribution over wing sections and its corresponding sectional lift
    coefficient (Cl). The optimization is performed using the Optuna library, and as the results returns
    the best hyperparameters found, and the importance of each hyperparameter is calculated.
            
Dependencies:
    - warnings
    - logging
    - os
    - matplotlib
    - typing
    - pathlib
    - seaborn
    - pandas
    - numpy
    - time
    -random
    - sklearn
    - tensorflow (keras)
    - optuna

Future implementations:
    >>> ALL IMPLEMENTATIONS DONE!
--------------------------------------------------------------------------------------------------------
"""

# System and logging configuration
import warnings
import logging
import os

os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
warnings.filterwarnings("ignore")

# Standard libraries
from matplotlib import pyplot as plt
from typing import Tuple
from pathlib import Path
import seaborn as sns
import pandas as pd
import numpy as np
import time
import random

# Machine learning & metrics
from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler
from sklearn.model_selection import train_test_split

# Deep learning (Keras/TensorFlow)
import tensorflow as tf
from keras.layers import Dense, Input, Dropout, BatchNormalization
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.optimizers import Adam, Nadam
from keras.backend import clear_session
from keras.models import Sequential, clone_model

logging.getLogger('tensorflow').setLevel(logging.ERROR)
tf.get_logger().setLevel(logging.ERROR)

# Hyperparameter optimization
import optuna
from optuna.integration import TFKerasPruningCallback

# Reproductibility setup
SEED = 42
os.environ['PYTHONHASHSEED'] = str(SEED)
tf.random.set_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)
print(f'\nGlobal seed set to: {SEED}')



def load_and_preprocess_data(filepath:str, random_state:int):
    """
    Loads the dataset ans splits into training and testing sets based on wing condiitions, aiming 
    to improve the model's generalization capability. Also splits the dataset into features (X) and
    targets (Y) datasets, and applies scales the data, in which the features and 'Cl' is scaled using
    MinMaxScaler and 'Cp' is scaled using MaxAbsScaler.
    
    Args:
        filepath: Path to the .csv file containing the dataset.
        random_state: Seed for random shuffle.

    Returns:
        X_train (np.ndarray): Array containing the scaled training features.
        Y_train (np.ndarray): Array containing the scaled training targets.
    """

    # 1. Loading the dataset:
    data_path = Path.cwd() / filepath
    data = pd.read_csv(data_path, sep=',')

    # 2. Create identifier for wing condition (assuming 80 sections per condition):
    wing_sections = 80
    indices = np.arange(len(data))
    data['wing_condition'] = indices // wing_sections
    wing_conditions = data['wing_condition'].unique()
    train_conds, test_conds = train_test_split(wing_conditions, test_size=0.2, random_state=random_state)

    # 3. Separate DataFrames and drop auxiliary column:
    data_train = data[data['wing_condition'].isin(train_conds)].drop(columns=['wing_condition'])

    # 4. Defining the input columns:
    input_cols = data_train.columns[0:3]

    # 5. Splitting the input data into the training dataset (test dataset isn't used):
    X_train = data_train[input_cols].values

    # 6. Defining the target columns and splitting into training dataset:
    Y_cl_train = data_train.iloc[:, 3].values.reshape(-1, 1)
    Y_cp_train = data_train.iloc[:, 4:].values

    # 7. Scaling the inputs using MinMaxScaler:
    scaler_x = MinMaxScaler()
    X_train = scaler_x.fit_transform(X_train)
    
    # 8. Scaling the 'Cl' using MinMaxScaler:
    cl_scaler = MinMaxScaler()
    Y_cl_train = cl_scaler.fit_transform(Y_cl_train)
    
    # 9. Scaling the 'Cp' using MaxAbsScaler:
    cp_scaler = MaxAbsScaler()
    Y_cp_train = cp_scaler.fit_transform(Y_cp_train)
    
    # 10. Concatenate the targets back together:
    Y_train = np.hstack([Y_cl_train, Y_cp_train])

    return X_train, Y_train

def build_model(trial: optuna.Trial, input_shape: Tuple[int], output_shape: int):
    """
    Builds and compiles the Artificial Neural Network architecture.

    Args:
        trial: Optuna trial object.
        input_shape: Shape of the input layer.
        output_shape: Number of outputs.

    Returns:
        model (Sequential): Artificial Neural Network model.
    """

    # Defining the model architecture for a singular trial:
    model = Sequential()

    # Defining general parameters:
    n_layers = trial.suggest_categorical('n_layers', [2, 3, 4, 5, 6, 7, 8, 9])
    reg_type = trial.suggest_categorical('reg_type', ['Dropout', 'Batch Normalization'])
    activation = trial.suggest_categorical('activation', ['swish', 'tanh', 'gelu'])

    # Defining the dropout rate if it is selected as the regularization strategy:
    if reg_type == 'Dropout':
        dropout_rate = trial.suggest_categorical('dropout_rate', [0.1, 0.2])

    # Input layer:
    model.add(Input(shape=input_shape))

    # Hidden layers loop:
    for i in range(n_layers):
        units = trial.suggest_categorical(f'units_layer{i}', [32, 64, 96, 128, 160, 192, 224, 256])
        model.add(Dense(units, activation=activation))

        if reg_type == 'Dropout':
                model.add(Dropout(dropout_rate))
        if reg_type == 'Batch Normalization':
            model.add(BatchNormalization())

    # Output layer:
    model.add(Dense(output_shape, activation='linear'))

    return model

def objective(trial: optuna.Trial):
    """
    Objective function for the Optuna hyperparameter optimization.

    Args:
        trial: Optuna trial object.

    Returns:
        mean_loss (float): Mean validation loss value for the current trial, considering training 
        with 3 different predefined seeds. Aimimng to avoid lucky shots.
    """

    # 1. Cleaning the session memory:
    clear_session()

    # 2. Loading and preprocessing the data to build the base model:
    X_train, Y_train = load_and_preprocess_data('Numerical-PressureDistributionData.csv', 
            random_state=42)
    base_model = build_model(trial, input_shape=(X_train.shape[1],), output_shape=Y_train.shape[1])

    # 3. Defining standard parameters:
    seeds = [3, 42, 69]
    losses = []

    # 4. Looping over the SEEDS:
    for seed in seeds:

        # Loading and preprocessing the data:
        X_train, Y_train = load_and_preprocess_data('Numerical-PressureDistributionData.csv', 
            random_state=seed)

        # Cloning the base model (copying its architecture without weight values):
        model = clone_model(base_model)

        # Optimizer selection:
        optimizer = trial.suggest_categorical('optimizer', ['Adam', 'Nadam'])
        lr = trial.suggest_categorical('learning_rate', [1e-5, 1e-4, 1e-3])

        if optimizer == 'Adam':
            optimizer = Adam(learning_rate=lr)
        if optimizer == 'Nadam':
            optimizer = Nadam(learning_rate=lr)

        # Compiling the model:
        model.compile(loss='mean_squared_error', optimizer=optimizer, metrics=['mae'])

        # Defining the model callbacks:
        callbacks = [EarlyStopping(monitor='val_loss', patience=30, restore_best_weights=True),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=15, min_lr=1e-6)]
        
        # Adding the pruning callback possibility only for the first seed:
        if seed == seeds[0]: 
            callbacks.append(TFKerasPruningCallback(trial, 'val_loss'))
        
        # Training the model:
        history = model.fit(X_train, Y_train, validation_split=0.2, epochs=200, verbose=0, 
            batch_size=trial.suggest_categorical('batch_size', [32, 64, 128]), 
            callbacks=callbacks)
        
        # Evaluating the model using the test dataset:
        losses.append(min(history.history['val_loss']))
    
    # 5. Calculating the mean loss for the three seeds:
    mean_loss = np.mean(losses)

    # 6. Cleaning the session memory:
    del model
    clear_session()
         
    return mean_loss

def plot_parameter_impact(study: optuna.Study, param_name: str, x_label: str, save_path:str):
    """
    Plots the relationship between the parameter and the loss value. Automatic selects the plot style based
    on the parameter type (categorical or numerical).

    Args:
        study: Optuna study object.
        param_name: Name of the parameter to plot.
        x_label: Label for the x-axis.
        save_path: Path to save the plot.

    Returns:
        None. Saves the plot as .png file.
    """

    # Setting plot parameters:
    with plt.rc_context({'font.family': 'serif', 'font.serif': ['Times New Roman'], 
        'mathtext.fontset': 'stix'}):

        # Preparing the data, filtering for complete trials:
        data = study.trials_dataframe()
        data = data[data['state'] == 'COMPLETE'].copy()
        parameter = f'params_{param_name}'

        if parameter not in data.columns:
            print(f"Parameter {param_name} not found in the study.\n")
            return
        
        # Filters only the necessary columns and removes NaNs:
        data_plot = data[[parameter, 'value']].dropna()

        # Setting the figure parameters:
        plt.figure(figsize=(8,6))
        ax = plt.gca()
        plt.xlabel(x_label, fontsize=12, fontname='Times New Roman', labelpad=5)
        plt.ylabel('Mean Validation Loss (MSE)', fontsize=12, fontname='Times New Roman', labelpad=5)
        plt.ticklabel_format(style='sci', scilimits=(0, 0), axis='y', useMathText=True)
        plt.tick_params(axis='x', labelsize=12)
        plt.tick_params(axis='y', labelsize=12)
        plt.ylim(0, 5e-3)
        plt.grid(True, which='both', ls='--', alpha=0.6)
        ax.yaxis.get_offset_text().set_fontsize(12)
        
        # Plotting the curves: 
        sns.boxplot(data=data_plot, x=parameter, y='value', color='#4682B4', linecolor='black', width=0.5, 
            showfliers=False, linewidth=1.2)

        # Saving the plot:
        plt.tight_layout()
        plt.savefig(save_path, dpi=300)
        plt.close()
        print(f'Plot saved as {save_path}\n')

        return

def save_results(study: optuna.Study, execution_time: float):
    """
    Generates the optimization history and parameter importance plots, and saves a optimization report.

    Args:
        study: Optuna study object.
        execution_time: Time taken to run the optimization.

    Returns:
        None. Saves the plots as .png files and the optimization report as .txt file.
    """
    
    # =============================================================================================================
    # 1. VISUALIZATION: LOSS OPTIMIZATION HISTORY
    # =============================================================================================================
    # Setting plot parameters:
    with plt.rc_context({'font.family': 'serif', 'font.serif': ['Times New Roman'], 
        'mathtext.fontset': 'stix'}):

        # Preparing the data, filtering for complete trials:
        data = study.trials_dataframe()
        data = data[data['state'] == 'COMPLETE'].copy()
        data['number'] = data.number
        data['value'] = data.value
        data['best_value'] = data['value'].cummin()

        # Setting the figure parameters:
        plt.figure(figsize=(8,6))
        ax = plt.gca()
        plt.xlabel('Trial Number', fontsize=12, fontname='Times New Roman')
        plt.ylabel('Mean Validation Loss (MSE)', fontsize=12, fontname='Times New Roman')
        plt.ticklabel_format(style='sci', scilimits=(0, 0), axis='y', useMathText=True)
        plt.tick_params(axis='x', labelsize=12)
        plt.tick_params(axis='y', labelsize=12)
        plt.xlim(0, max(data['number']))
        plt.ylim(0, 5e-3)
        plt.grid(True, which='both', ls='--', alpha=0.6)
        ax.yaxis.get_offset_text().set_fontsize(12)

        # Plotting the curves:
        plt.scatter(data['number'], data['value'], alpha=1, s=20, color='#000080', 
            label='Trial Mean Validation Loss', edgecolors='None')
        plt.plot(data['number'], data['best_value'], color='#DC143C', linewidth=1, 
            label='Minimum Mean Validation Loss')
        plt.legend(loc='upper right', frameon=True, facecolor='white', edgecolor='black', 
            prop={'family': 'Times New Roman', 'size': 12})

        # Saving the plot:
        plt.tight_layout()
        plt.savefig('LF_numerical_ann/plots/optimization/loss_optimization_history.png', dpi=300)
        plt.close()
        print(f'Plot saved as LF_numerical_ann/plots/optimization/loss_optimization_history.png\n')


    # =============================================================================================================
    # 2. VISUALIZATION: PARAMETER IMPORTANCE
    # =============================================================================================================
    # Setting plot parameters:
    with plt.rc_context({'font.family': 'serif', 'font.serif': ['Times New Roman'], 
        'mathtext.fontset': 'stix'}):

        # Preparing the data, filtering for complete trials:
        importance = optuna.importance.get_param_importances(study, 
            evaluator=optuna.importance.FanovaImportanceEvaluator(seed=SEED))
        data = pd.DataFrame(list(importance.items()), columns=['Hyperparameter', 'Importance'])

        # Renaming the parameters for visualization:
        rename = {'activation': r'Activation Function',
            'n_layers': r'Network Depth',
            'reg_type': r'Regularization Strategy',
            'optimizer': r'Optimizer',
            'batch_size': r'Batch Size',
            'learning_rate': r'Learning Rate',
            'units_layer0': r'1st Layer Neurons',
            'units_layer1': r'2nd Layer Neurons',
            'units_layer2': r'3rd Layer Neurons',
            'units_layer3': r'4th Layer Neurons',
            'units_layer4': r'5th Layer Neurons',
            'units_layer5': r'6th Layer Neurons',
            'dropout_rate': r'Dropout Rate'}
        
        data['Parameters'] = data['Hyperparameter'].map(rename).fillna(data['Hyperparameter'])

        # Setting the figure parameters:
        plt.figure(figsize=(8,6))
        plt.xlabel('Relative Importance', fontsize=12, fontname='Times New Roman')
        plt.ylabel(' ', fontsize=12, fontname='Times New Roman')
        plt.xlim(0, data['Importance'].max()*1.15)
        plt.grid(True, which='both', ls='--', alpha=0.6)

        # Plotting the curves:
        ax = sns.barplot(data=data, x='Importance', y='Parameters', color='#000080', edgecolor='black')
        for tick in ax.get_xticklabels() + ax.get_yticklabels():
            tick.set_fontname('Times New Roman')

        # Saving the plot:
        plt.tight_layout()
        plt.savefig('LF_numerical_ann/plots/optimization/hyperparameter_importance.png', dpi=300)
        plt.close()
        print(f'Plot saved as LF_numerical_ann/plots/optimization/hyperparameter_importance.png\n')


    # =============================================================================================================
    # 3. TEXT FILE: MODEL OPTIMIZATION REPORT
    # =============================================================================================================
    # Generate the model's optimization report:
    data = study.trials_dataframe()
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

    # Saving the report:
    with open('LF_numerical_ann/reports/optimization_report.txt', "w", encoding="utf-8") as f:
        f.write(content)
    print("Report saved as 'LF_numerical_ann/reports/optimization_report.txt'\n")

    return

def main(n_trials: int):
    """
    Main execution workflow: Loads the data, calls the optimization function and saves the plots and results.
    """

    # 1. Starts the timer for recording execution time:
    start_time = time.time()

    # 2. Calls the optimization function:
    print('\nStarting optimization process...\n')
    study = optuna.create_study(direction='minimize', sampler=optuna.samplers.TPESampler(seed=SEED), 
        pruner=optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=50, interval_steps=5), study_name='LF_numerical_ann_optimization')
    study.optimize(objective, n_trials=n_trials)

    # 3. Finishes the timer:
    end_time = time.time()
    execution_time = end_time - start_time

    # 4. Saves results:
    plot_parameter_impact(study, 'activation', 'Activation Function', 
        'LF_numerical_ann/plots/optimization/param_impact_activation.png')
    plot_parameter_impact(study, 'n_layers', 'Network Depth', 
        'LF_numerical_ann/plots/optimization/param_impact_network_depth.png')
    plot_parameter_impact(study, 'reg_type', 'Regularization Strategy', 
        'LF_numerical_ann/plots/optimization/param_impact_reg_type.png')
    plot_parameter_impact(study, 'optimizer', 'Optimizer', 
        'LF_numerical_ann/plots/optimization/param_impact_optimizer.png')
    plot_parameter_impact(study, 'batch_size', 'Batch Size', 
        'LF_numerical_ann/plots/optimization/param_impact_batch_size.png')
    plot_parameter_impact(study, 'learning_rate', 'Learning Rate', 
        'LF_numerical_ann/plots/optimization/param_impact_learning_rate.png')
    plot_parameter_impact(study, 'units_layer0', '1st Layer Neurons', 
        'LF_numerical_ann/plots/optimization/param_impact_1st_layer_neurons.png')
    plot_parameter_impact(study, 'units_layer1', '2nd Layer Neurons', 
        'LF_numerical_ann/plots/optimization/param_impact_2nd_layer_neurons.png')
    plot_parameter_impact(study, 'units_layer2', '3rd Layer Neurons', 
        'LF_numerical_ann/plots/optimization/param_impact_3rd_layer_neurons.png')
    plot_parameter_impact(study, 'units_layer3', '4th Layer Neurons', 
        'LF_numerical_ann/plots/optimization/param_impact_4th_layer_neurons.png')
    plot_parameter_impact(study, 'units_layer4', '5th Layer Neurons', 
        'LF_numerical_ann/plots/optimization/param_impact_5th_layer_neurons.png')
    plot_parameter_impact(study, 'units_layer5', '6th Layer Neurons', 
        'LF_numerical_ann/plots/optimization/param_impact_6th_layer_neurons.png')
    plot_parameter_impact(study, 'dropout_rate', 'Dropout Rate', 
        'LF_numerical_ann/plots/optimization/param_impact_dropout_rate.png')
    save_results(study, execution_time)
    print('Optimization completed.\n')

    return



if __name__== "__main__":
    # 1. Sets the number of trials:
    N_TRIALS = 10

    # 2. Calls the main function:
    main(N_TRIALS)