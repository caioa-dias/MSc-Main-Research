# -*- coding: utf-8 -*-
"""
========================================================================================================
Module: train_MLP_ANN
Author: Caio Dias Filho
Creation date: 2025-11-28
Last modification: 2026-04-09
Version: 2.0 (final)
========================================================================================================

OVERVIEW
--------
This module implements the complete training, evaluation, and analysis pipeline for a low-fidelity Multi-
Layer Perceptron Artificial Neural Network (MLP-ANN) applied to aerodynamic reconstruction using low-
fidelity data.

The neural network is trained to predict:

    - Sectional lift coefficient (CL)
    - Pressure coefficient distribution (Cp)

The module integrates all stages of the modeling workflow, including:

    - Structured dataset loading and splitting at flow-case level
    - Feature and target normalization using standardization
    - Neural network construction and supervised training
    - Model evaluation using regression metrics
    - Prediction on unseen test data
    - Generation of diagnostics and publication-quality plots
    - Error analysis across target and input domains
    - Export of trained model, scalers, and performance report

The objective is to provide both accurate predictions and a comprehensive analysis of model behavior 
across the aerodynamic parameter space.


DEPENDENCIES 
------------
Python libraries:

    - os
    - warnings
    - time
    - gc
    - numpy
    - pandas
    - matplotlib
    - seaborn
    - scikit-learn
    - tensorflow / keras
    - joblib


INPUT DATA FORMAT
-----------------
The input dataset must satisfy the following structure:

    - CSV file with colon separators (';')
    - Each aerodynamic case contains exactly 80 spanwise sections
    - Columns:
        0-2   : input features
            - Reynolds number (Re)
            - Angle of attack (AoA)
            - Normalized spanwise position (y/b)
        3-204 : target variables
            - Sectional lift coefficient (CL)
            - Pressure coefficient distribution (Cp)

The dataset is split at the flow-case level to prevent data leakage between training, validation, and 
testing subsets.


OUTPUT FILES
------------
Generated outputs include:

Model:
    - LowFidelity-ANN/MLP-MultiLayerPerceptron/MLP-ANN.keras

Scalers:
    - LowFidelity-ANN/MLP-MultiLayerPerceptron/scalers/mlp_ann_X_scaler.pkl
    - LowFidelity-ANN/MLP-MultiLayerPerceptron/scalers/mlp_ann_y_scaler.pkl

Plots:
    - Data distribution in parameter space
    - Training history (MSE and MAE)
    - Prediction scatter plots
    - Error envelope plots (by target, input, and chordwise position)
    - Error distribution plots

Report:
    - LowFidelity-ANN/reports/performance-report-mlp-ann.txt

Dataset export:
    - Raw test dataset for reproducibility


NOTES
-----
- Reproducibility is ensured through fixed random seeds.
- Early stopping is used to prevent overfitting.
- Learning rate is adaptively reduced during training.
- All plots follow publication-ready formatting standards.
- Error analysis is performed at multiple levels:
    - Global (overall prediction)
    - Target-wise (Cl vs Cp)
    - Input-conditioned (Re, AoA, y/b)
    - Spatial (chordwise Cp distribution)

========================================================================================================
"""

# =======================================================================================================
#                                            USER INPUT SECTION
# =======================================================================================================

# --- Dataset path ---
data_path = 'LowFidelity-ANN/utils/LowFidelity-PressureDistributionData.csv'


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
import tensorflow
tensorflow.get_logger().setLevel("ERROR")
import matplotlib
from matplotlib import pyplot as plt
import sklearn as skl
import seaborn as sns
import pandas as pd
import numpy as np
import joblib
import keras
import time
import gc

# Reproducibility setup:
SEED = 42
os.environ['PYTHONHASHSEED'] = str(SEED)
tensorflow.random.set_seed(SEED)
np.random.seed(SEED)
print(f'\nGlobal seed set to: {SEED}')

def load_and_split_data(filepath: str):
    """
    Load and split the aerodynamic dataset into training, validation, and test sets.

    The dataset is divided at the flow-case level to ensure that all spanwise sections belonging to the 
    same aerodynamic condition remain within the same subset, preventing data leakage.

    Parameters
    ----------
    filepath : str
        Path to the dataset file.

    Returns
    -------
    data_train : pandas.DataFrame
        Training dataset.

    data_val : pandas.DataFrame
        Validation dataset.

    data_test : pandas.DataFrame
        Test dataset.
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
    train_cases, val_cases = skl.model_selection.train_test_split(train_cases, test_size=0.1,
        random_state=SEED, shuffle=True)
    
    # --- Separate the datasets ---
    data_train = data[data['flow_case'].isin(train_cases)]
    data_val = data[data['flow_case'].isin(val_cases)].drop(columns=['flow_case'])
    data_test = data[data['flow_case'].isin(test_cases)]

    return data_train, data_val, data_test

def preprocess_data(data_train: pd.DataFrame, data_val: pd.DataFrame, data_test: pd.DataFrame):
    """
    Preprocess the dataset for neural network training.

    This function separates input features and target variables, applies standardization independently to 
    each component, and stores the fitted scalers for later use.

    Parameters
    ----------
    data_train : pandas.DataFrame
        Training dataset.

    data_val : pandas.DataFrame
        Validation dataset.

    data_test : pandas.DataFrame
        Test dataset.

    Returns
    -------
    X_train, X_val, X_test : numpy.ndarray
        Standardized input features.

    y_train, y_val, y_test : numpy.ndarray
        Standardized target variables.

    X_scaler, y_scaler : sklearn.preprocessing.StandardScaler
        Fitted scalers for input and target variables.
    """

    # --- Defining input and target columns ---
    X_cols = data_train.columns[0:3]
    y_cols = data_train.columns[3:205]

    # --- Splitting the raw features data into training and testing sets ---
    X_train_raw = data_train[X_cols].to_numpy()
    X_val_raw = data_val[X_cols].to_numpy()
    X_test_raw = data_test[X_cols].to_numpy()

    # --- Splitting the raw targets data into training and testing data ---
    y_train_raw = data_train[y_cols].to_numpy()
    y_val_raw = data_val[y_cols].to_numpy()
    y_test_raw = data_test[y_cols].to_numpy()

    # --- Scaling the features data ---
    X_scaler = skl.preprocessing.StandardScaler()
    X_train = X_scaler.fit_transform(X_train_raw)
    X_val = X_scaler.transform(X_val_raw)
    X_test = X_scaler.transform(X_test_raw)

    # --- Scaling the targets data ---
    y_scaler = skl.preprocessing.StandardScaler()
    y_train = y_scaler.fit_transform(y_train_raw)
    y_val = y_scaler.transform(y_val_raw)
    y_test = y_scaler.transform(y_test_raw)

    # --- Saving the scalers ---
    joblib.dump(X_scaler, 'LowFidelity-ANN/MLP-MultiLayerPerceptron/scalers/mlp_ann_X_scaler.pkl')
    joblib.dump(y_scaler, 'LowFidelity-ANN/MLP-MultiLayerPerceptron/scalers/mlp_ann_y_scaler.pkl')

    return X_train, X_val, X_test, y_train, y_val, y_test, X_scaler, y_scaler

def build_model(input_dim: int, output_dim: int):
    """
    Build and compile the Multi-Layer Perceptron neural network.

    The architecture consists of multiple fully connected layers using GELU activation functions and 
    Glorot-normal weight initialization.

    Parameters
    ----------
    input_dim : int
        Number of input features.

    output_dim : int
        Number of output variables.

    Returns
    -------
    model : keras.Model
        Compiled neural network model.
    """

    # --- Defining the model input layer ---
    inputs = keras.layers.Input(shape=(input_dim,), name='Input_Layer')

    # --- Defining the model hidden layers ---
    x = keras.layers.Dense(units=96, activation='gelu', kernel_initializer='glorot_normal', 
        name='Hidden_Layer_1')(inputs)

    x = keras.layers.Dense(units=160, activation='gelu', kernel_initializer='glorot_normal', 
        name='Hidden_Layer_2')(x)

    x = keras.layers.Dense(units=160, activation='gelu', kernel_initializer='glorot_normal', 
        name='Hidden_Layer_3')(x)

    x = keras.layers.Dense(units=192, activation='gelu', kernel_initializer='glorot_normal', 
        name='Hidden_Layer_4')(x)
    
    x = keras.layers.Dense(units=224, activation='gelu', kernel_initializer='glorot_normal', 
        name='Hidden_Layer_5')(x)

    # --- Defining the model output layer ---
    outputs = keras.layers.Dense(units=output_dim, activation='linear', 
        kernel_initializer='glorot_normal', name='Output_Layer')(x)

    # --- Defining the model ---
    model = keras.Model(inputs=inputs, outputs=outputs, name='low_fidelity_mlp_ann')

    # --- Compiling the model ---
    model.compile(loss='mse', optimizer=keras.optimizers.Nadam(learning_rate=0.003), metrics=['mae'])

    return model

def predict_test(model: keras.Model, X_test: np.ndarray, y_test: np.ndarray,
    y_scaler: skl.preprocessing.StandardScaler):
    """
    Generate predictions on the test set and convert them to physical scale.

    Parameters
    ----------
    model : keras.Model
        Trained neural network.

    X_test : numpy.ndarray
        Standardized test inputs.

    y_test : numpy.ndarray
        Standardized test targets.

    y_scaler : sklearn.preprocessing.StandardScaler
        Target scaler used for inverse transformation.

    Returns
    -------
    y_test_raw : numpy.ndarray
        True target values in physical scale.

    y_pred_raw : numpy.ndarray
        Predicted target values in physical scale.
    """

    # --- Predicting on the test set ---
    y_pred = model.predict(X_test, verbose=0)

    # --- Reverses the scaling ---
    y_test_raw = y_scaler.inverse_transform(y_test)
    y_pred_raw = y_scaler.inverse_transform(y_pred)

    return y_test_raw, y_pred_raw

def plot_data_split_parameter_space(X_train_raw: np.ndarray, X_val_raw: np.ndarray, 
    X_test_raw: np.ndarray, layout: str):
    """
    Visualize the distribution of training, validation, and test datasets in the aerodynamic parameter 
    space. The plot displays Reynolds number versus angle of attack, highlighting how the dataset is 
    partitioned.

    Parameters
    ----------
    X_train_raw, X_val_raw, X_test_raw : numpy.ndarray
        Input data in physical scale.

    layout : str
        Plot layout configuration.

    Returns
    -------
    None
        The function saves the figure to disk and does not return any value.
    """

    # --- Plot layout configurations ---
    plot_layout = {'single_column': {'width': 5.025, 'height': 3.06, 'main_lw': 0.8, 'sec_lw': 0.4, 
            'scatter': 2, 'label_fs': 12, 'lp': 5, 'sec_fs': 10, 'tick': 0.4, 'grid_alpha': 0.2, 
            'm_scale': 3}, 
        'double_column': {'width': 10.05, 'height': 5.76, 'main_lw': 1.2, 'sec_lw': 0.6, 'scatter': 10, 
            'label_fs': 19, 'lp': 10, 'sec_fs': 15, 'tick': 0.5, 'grid_alpha':0.3, 'm_scale': 4}}
    
    cfg = plot_layout[layout]
    # --- Setting figure size:
    width, height = cfg['width'], cfg['height']
    # --- Setting plot element size:
    main_lw, sec_lw, scatter = cfg['main_lw'], cfg['sec_lw'], cfg['scatter']
    # --- Setting text font size:
    label_fs, sec_fs = cfg['label_fs'], cfg['sec_fs']
    # --- Setting aesthetic parameters:
    lp, tick, grid_alpha, m_scale = cfg['lp'], cfg['tick'], cfg['grid_alpha'], cfg['m_scale']

    # Setting plot parameters:
    with plt.rc_context({'font.family': 'serif', 'font.serif': ['Times New Roman'],
        'mathtext.fontset': 'stix'}):

        # --- Figure size: single-column ---
        fig, ax = plt.subplots(figsize=(width, height))

        # --- Plot ---
        ax.scatter(X_train_raw[:,0], X_train_raw[:,1], alpha=0.35, s=scatter, marker='.',
            facecolor='#1F77B4', label='Training and validation dataset')
        ax.scatter(X_val_raw[:,0], X_val_raw[:,1], alpha=0.75, s=scatter, marker='.',
            facecolor='#1F77B4')
        ax.scatter(X_test_raw[:,0], X_test_raw[:,1], alpha=0.95, s=scatter, marker='s',
            facecolor='#FF7F0E', label='Test dataset')

        # --- Labels ---
        ax.set_xlabel(r'Reynolds number ($Re$)', fontsize=label_fs, fontname='Times New Roman', 
            labelpad=lp)
        ax.set_ylabel(r'Angle of attack ($\alpha$) [$^\circ$]', fontsize=label_fs, 
            fontname='Times New Roman', labelpad=lp)
        
        # --- Ticks ---
        for spine in ax.spines.values():
            spine.set_linewidth(0.5)
        ax.tick_params(axis='both', which='major', labelsize=sec_fs, width=tick, direction='in')
        ax.xaxis.set_major_locator(matplotlib.ticker.MultipleLocator(2))

        # --- Grid ---
        ax.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=grid_alpha)

        # --- Limits ---
        ax.set_xlim([0, 1100000])
        ax.set_ylim([-5, 13])
        formatter = matplotlib.ticker.ScalarFormatter(useMathText=True)
        formatter.set_scientific(True)
        formatter.set_powerlimits((5, 5))
        ax.xaxis.set_major_formatter(formatter)
        ax.set_xticks([0, 2e5, 4e5, 6e5, 8e5, 1e6])
        ax.xaxis.get_offset_text().set_fontsize(sec_fs)
        ax.xaxis.get_offset_text().set_y(1.02)

        # --- Legend ---
        legend = ax.legend(fontsize=sec_fs, fancybox=False, edgecolor='#000000', loc='upper center', 
            bbox_to_anchor=(0.5, -0.25), markerscale=m_scale, ncol=2)
        if layout == 'single_column':
            legend.get_frame().set_linewidth(0.5)

        # --- Save plot ---
        plt.tight_layout(pad=0.8)
        plt.savefig(f'LowFidelity-ANN/MLP-MultiLayerPerceptron/plots/data_split_parameter_space_{layout}.png', dpi=600)
        plt.close()
        print(f'Saved: LowFidelity-ANN/MLP-MultiLayerPerceptron/plots/data_split_parameter_space_{layout}.png\n')

    return

def plot_training_history(history: dict, metric: str, layout: str):
    """
    Plot training and validation history for a selected metric.

    Parameters
    ----------
    history : keras.callbacks.History
        Training history.

    metric : str
        Metric to plot ('mse' or 'mae').

    layout : str
        Plot layout configuration.

    Returns
    -------
    None
        The function saves the figure to disk and does not return any value.
    """


    # --- Load data for MSE metric plot ---
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(1, len(loss) + 1)
    best_epoch_loss = np.argmin(val_loss) + 1
    best_val_loss = val_loss[best_epoch_loss - 1]

    # --- Load data for MAE metric plot ---
    mae = history.history['mae']
    val_mae = history.history['val_mae']
    best_val_mae = val_mae[best_epoch_loss - 1] 

    # --- Plot layout configurations ---
    plot_layout = {'single_column': {'width': 3.35, 'height': 2.55, 'main_lw': 0.8, 'sec_lw': 0.4, 
            'scatter': 10, 'label_fs': 12, 'lp': 5, 'sec_fs': 10, 'tick': 0.4, 'grid_alpha': 0.2}, 
        'double_column': {'width': 6.7, 'height': 4.8, 'main_lw': 1.2, 'sec_lw': 0.6, 'scatter': 20, 
            'label_fs': 18, 'lp': 10, 'sec_fs': 14, 'tick': 0.5, 'grid_alpha':0.3}}
    
    cfg = plot_layout[layout]
    # --- Setting figure size:
    width, height = cfg['width'], cfg['height']
    # --- Setting plot element size:
    main_lw, sec_lw, scatter = cfg['main_lw'], cfg['sec_lw'], cfg['scatter']
    # --- Setting text font size:
    label_fs, sec_fs = cfg['label_fs'], cfg['sec_fs']
    # --- Setting aesthetic parameters:
    lp, tick, grid_alpha = cfg['lp'], cfg['tick'], cfg['grid_alpha']

    # Setting plot parameters:
    with plt.rc_context({'font.family': 'serif', 'font.serif': ['Times New Roman'],
        'mathtext.fontset': 'stix'}):

        # --- Figure size: single-column ---
        fig, ax = plt.subplots(figsize=(width, height))

        if metric == 'loss':
            # --- Plot ---
            plt.plot(epochs, loss, label='Training Loss', color='#1B3A6F', linewidth=main_lw)
            plt.plot(epochs, val_loss, label='Validation Loss', color='#B22222', linewidth=main_lw, 
                linestyle='--')
            plt.axvline(x=best_epoch_loss, color='#B22222', linestyle='--', alpha=0.4, linewidth=sec_lw)
            plt.scatter(best_epoch_loss, best_val_loss, label=f'Best loss epoch ({best_epoch_loss})',
                alpha=1.0, s=scatter, facecolors='#B22222', edgecolors='#000000', linewidth=0.3, 
                zorder=5)
        
            # --- Labels ---
            ax.set_xlabel('Epochs', fontsize=label_fs, fontname='Times New Roman', labelpad=lp)
            ax.set_ylabel('Mean squared error (MSE)', fontsize=label_fs, fontname='Times New Roman', 
                labelpad=lp)

        elif metric == 'mae':
            # --- Plot ---            
            plt.plot(epochs, mae, label='Training MAE', color="#303134", linewidth=main_lw)
            plt.plot(epochs, val_mae, label='Validation MAE', color='#B22222', linewidth=main_lw, 
                linestyle='--')
            plt.axvline(x=best_epoch_loss, color='#B22222', linestyle='--', alpha=0.4, linewidth=sec_lw)
            plt.scatter(best_epoch_loss, best_val_mae, label=f'Best loss epoch ({best_epoch_loss})',
                alpha=1.0, s=scatter, facecolors='#B22222', edgecolors='#000000', linewidth=0.3, 
                zorder=5)
        
            # --- Labels ---
            ax.set_xlabel('Epochs', fontsize=label_fs, fontname='Times New Roman', labelpad=lp)
            ax.set_ylabel('Mean absolute error (MAE)', fontsize=label_fs, fontname='Times New Roman', 
                labelpad=lp)

        # --- Ticks ---
        for spine in ax.spines.values():
            spine.set_linewidth(0.5)
        ax.tick_params(axis='both', which='major', labelsize=sec_fs, width=tick, direction='in')
        ax.ticklabel_format(axis='y', style='sci', scilimits=(0,0), useMathText=True)
        ax.xaxis.get_offset_text().set_fontsize(sec_fs)
        ax.yaxis.get_offset_text().set_fontsize(sec_fs)

        # --- Grid ---
        ax.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=grid_alpha)

        # --- Limits ---
        ax.set_xlim([1, len(epochs)])
        if metric == 'mse':
            ax.set_ylim([0, 0.03])
        elif metric == 'mae':
            ax.set_ylim([0, 0.1])

        # --- Legend ---
        legend = ax.legend(fontsize=sec_fs, fancybox=False, edgecolor='#000000', loc='upper right')
        if layout == 'single_column':
            legend.get_frame().set_linewidth(0.5)

        # --- Save plot ---
        plt.tight_layout(pad=0.6)
        plt.savefig(f'LowFidelity-ANN/MLP-MultiLayerPerceptron/plots/train_{metric}_history_{layout}.png', dpi=600)
        plt.close()
        print(f'Saved: LowFidelity-ANN/MLP-MultiLayerPerceptron/plots/train_{metric}_history_{layout}.png\n')

    return

def plot_test_predictions(y_test_raw: np.ndarray, y_pred_raw: np.ndarray, target: str, layout: str):
    """
    Plot predicted versus true values for a selected target on the test set.

    This function generates a scatter plot comparing predicted and reference values in physical scale. It 
    can be used for the full output space or separately for the sectional lift coefficient and pressure 
    coefficient distribution. Standard regression metrics are displayed within the figure.

    Parameters
    ----------
    y_test_raw : np.ndarray
        Ground truth target values in physical scale.

    y_pred_raw : np.ndarray
        Predicted target values in physical scale.

    target : str
        Target subset to be visualized. Accepted values are:
            - 'overall': complete output space (Cl + Cp)
            - 'cl': sectional lift coefficient only
            - 'cp': pressure coefficient distribution only

    layout : str
        Figure layout format. Accepted values are:
            - 'single_column': compact format for single-column figures
            - 'double_column': extended format for double-column figures

    Returns
    -------
    None
        The function saves the figure to disk and does not return any value.

    Notes
    -----
    - A dashed diagonal line is included to indicate the ideal prediction line.
    - Performance metrics shown in the figure include R², MAE, MSE, and RMSE.
    """

    # --- Splits the targets data ---
    cl_test = y_test_raw[:,0].reshape(-1, 1)
    cl_pred = y_pred_raw[:,0].reshape(-1, 1)
    cp_test = y_test_raw[:,1:]
    cp_pred = y_pred_raw[:,1:]

    if target == 'overall':
        r2 = skl.metrics.r2_score(y_test_raw, y_pred_raw)
        mae = skl.metrics.mean_absolute_error(y_test_raw, y_pred_raw)
        mse = skl.metrics.mean_squared_error(y_test_raw, y_pred_raw)
        rmse = np.sqrt(mse)
        min_axis = min(np.min(y_test_raw), np.min(y_pred_raw))
        max_axis = max(np.max(y_test_raw), np.max(y_pred_raw))

    elif target == 'cl':
        r2 = skl.metrics.r2_score(cl_test, cl_pred)
        mae = skl.metrics.mean_absolute_error(cl_test, cl_pred)
        mse = skl.metrics.mean_squared_error(cl_test, cl_pred)
        rmse = np.sqrt(mse)
        min_axis = min(np.min(cl_test), np.min(cl_pred))
        max_axis = max(np.max(cl_test), np.max(cl_pred))

    elif target == 'cp':
        r2 = skl.metrics.r2_score(cp_test, cp_pred)
        mae = skl.metrics.mean_absolute_error(cp_test, cp_pred)
        mse = skl.metrics.mean_squared_error(cp_test, cp_pred)
        rmse = np.sqrt(mse)
        min_axis = min(np.min(cp_test), np.min(cp_pred))
        max_axis = max(np.max(cp_test), np.max(cp_pred))

    # --- Plot layout configurations ---
    plot_layout = {'single_column': {'width': 3.35, 'height': 2.55, 'main_lw': 0.8, 'sec_lw': 0.4, 
            'scatter': 2, 'label_fs': 12, 'lp': 5, 'sec_fs': 10, 'tick': 0.4, 'grid_alpha': 0.2, 
            'm_scale': 1}, 
        'double_column': {'width': 6.7, 'height': 4.8, 'main_lw': 1.2, 'sec_lw': 0.6, 'scatter': 6, 
            'label_fs': 18, 'lp': 10, 'sec_fs': 14, 'tick': 0.5, 'grid_alpha':0.3, 'm_scale': 2}}
    
    cfg = plot_layout[layout]
    # --- Setting figure size:
    width, height = cfg['width'], cfg['height']
    # --- Setting plot element size:
    main_lw, sec_lw, scatter = cfg['main_lw'], cfg['sec_lw'], cfg['scatter']
    # --- Setting text font size:
    label_fs, sec_fs = cfg['label_fs'], cfg['sec_fs']
    # --- Setting aesthetic parameters:
    lp, tick, grid_alpha, m_scale = cfg['lp'], cfg['tick'], cfg['grid_alpha'], cfg['m_scale']

    # Setting plot parameters:
    with plt.rc_context({'font.family': 'serif', 'font.serif': ['Times New Roman'],
        'mathtext.fontset': 'stix'}):

        # --- Figure size: single-column ---
        fig, ax = plt.subplots(figsize=(width, height))

        if target == 'overall':
            # --- Plot ---
            plt.scatter(cl_test, cl_pred, alpha=0.8, s=scatter, facecolors='#D55E00', 
                edgecolors='#000000', linewidth=0.1, label=r'Lift coefficient ($C_l$)')
            plt.scatter(cp_test, cp_pred, alpha=0.8, s=scatter, facecolors='#0072B2', 
                edgecolors='#000000', linewidth=0.1, label=r'Pressure coefficient ($C_p$)')
            plt.plot([min_axis, max_axis], [min_axis, max_axis], color='#B22222', linestyle='--', 
                linewidth=main_lw, zorder=5)
            
            # --- Legend ---
            legend = ax.legend(fontsize=sec_fs, fancybox=False, edgecolor='#000000', loc='lower right', 
                markerscale=m_scale)
            legend.get_frame().set_linewidth(0.5)
    
        elif target == 'cl':
            # --- Plot ---
            plt.scatter(cl_test, cl_pred, alpha=0.8, s=scatter, facecolors='#D55E00', 
                edgecolors='#000000', linewidth=0.1, label=r'Lift coefficient ($C_l$)')
            plt.plot([min_axis, max_axis], [min_axis, max_axis], color='#B22222', linestyle='--', 
                linewidth=main_lw, zorder=5)
        
        elif target == 'cp':
            # --- Plot ---
            plt.scatter(cp_test, cp_pred, alpha=0.8, s=scatter, facecolors='#0072B2', 
                edgecolors='#000000', linewidth=0.1, label=r'Pressure coefficient ($C_p$)')
            plt.plot([min_axis, max_axis], [min_axis, max_axis], color='#B22222', linestyle='--', 
                linewidth=main_lw, zorder=5)

        # --- Labels ---
        ax.set_xlabel('True values', fontsize=label_fs, fontname='Times New Roman', labelpad=lp)
        ax.set_ylabel('Predicted values', fontsize=label_fs, fontname='Times New Roman', labelpad=lp)

        # --- Ticks ---
        for spine in ax.spines.values():
            spine.set_linewidth(0.5)
        ax.tick_params(axis='both', which='major', labelsize=sec_fs, width=tick, direction='in')

        # --- Grid ---
        ax.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=grid_alpha)

        # --- Limits ---
        ax.set_xlim([min_axis, max_axis])
        ax.set_ylim([min_axis, max_axis])

        # --- Legend ---
        def format_sci_math(value: float, precision: int = 2) -> str:
            if value == 0:
                return "0"
            exponent = int(np.floor(np.log10(abs(value))))
            mantissa = value / 10**exponent
            return rf"${mantissa:.{precision}f} \times 10^{{{exponent}}}$"
        textstr = (f"R² = {r2:.5f}\n"
                + rf"MAE = {format_sci_math(mae)}" + "\n"
                + rf"MSE = {format_sci_math(mse)}" + "\n"
                + rf"RMSE = {format_sci_math(rmse)}")
        props = dict(boxstyle='square, pad=0.35', facecolor='white', alpha=1.0, edgecolor='k', 
            linewidth=0.5)
        ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=sec_fs, verticalalignment='top', 
            bbox=props, linespacing=1.5, fontname='Times New Roman')

        # --- Save plot ---
        plt.tight_layout(pad=0.6)
        plt.savefig(f'LowFidelity-ANN/MLP-MultiLayerPerceptron/plots/test_predictions_{target}_{layout}.png', dpi=600)
        plt.close()
        print(f'Saved: LowFidelity-ANN/MLP-MultiLayerPerceptron/plots/test_predictions_{target}_{layout}.png\n')

    return

def plot_test_error_envelope_by_target(y_test_raw: np.ndarray, y_pred_raw: np.ndarray, target: str, 
        layout: str):
    """
    Plot the absolute prediction error envelope as a function of the target value.

    This function evaluates how the absolute prediction error varies across the range of the selected 
    target variable. The data are grouped into bins according to the true target value, and the resulting 
    envelope displays the median error together with the interquartile range.

    Parameters
    ----------
    y_test_raw : np.ndarray
        Ground truth target values in physical scale.

    y_pred_raw : np.ndarray
        Predicted target values in physical scale.

    target : str
        Target to be analyzed. Accepted values are:
            - 'cl': sectional lift coefficient
            - 'cp': pressure coefficient distribution

    layout : str
        Figure layout format. Accepted values are:
            - 'single_column': compact format for single-column figures
            - 'double_column': extended format for double-column figures

    Returns
    -------
    None
        The function saves the figure to disk and does not return any value.

    Notes
    -----
    - The solid curve represents the median absolute error.
    - The shaded region corresponds to the interquartile range (25th to 75th percentile).
    - For Cp, the analysis is performed over the flattened pressure-coefficient values.
    """

    # --- Splits the targets data ---
    cl_test = y_test_raw[:,0].reshape(-1, 1)
    cl_pred = y_pred_raw[:,0].reshape(-1, 1)
    cp_test = y_test_raw[:,1:]
    cp_pred = y_pred_raw[:,1:]

    if target == 'cl':
        test_flat = cl_test.flatten()
        pred_flat = cl_pred.flatten()
        fill_color = '#D55E00'

    elif target == 'cp':
        test_flat = cp_test.flatten()
        pred_flat = cp_pred.flatten()
        fill_color = '#0072B2'

    # --- Dataframe ---
    data = pd.DataFrame({'True': test_flat, 'Abs_Error': np.abs(test_flat - pred_flat)})
    # Define the number of bins and associate points:
    bins = np.linspace(data['True'].min(), data['True'].max(), 101)
    data['Bin'] = pd.cut(data['True'], bins=bins, include_lowest=True)
    # Group the data by bin and calculate the quartiles:
    data_stats = data.groupby('Bin', observed=True)['Abs_Error'].agg(median='median', 
        q25=lambda x: np.percentile(x, 25), q75=lambda x: np.percentile(x, 75)).reset_index()
    # Compute bin centers:
    data_stats['Bin_center'] = data_stats['Bin'].apply(lambda x: x.mid)
    # Remove NaN values:
    data_stats = data_stats.dropna()
    # Smooth the quartile curves:
    data_stats['median'] = data_stats['median'].rolling(window=3, center=True, min_periods=1).mean()
    data_stats['q25'] = data_stats['q25'].rolling(window=3, center=True, min_periods=1).mean()
    data_stats['q75'] = data_stats['q75'].rolling(window=3, center=True, min_periods=1).mean()
    # Calculates minimum and maximum values
    min_axis = min(data['True'])
    max_axis = max(data['True'])

    # --- Plot layout configurations ---
    plot_layout = {'single_column': {'width': 3.35, 'height': 2.55, 'main_lw': 0.8, 'sec_lw': 0.4, 
            'scatter': 2, 'label_fs': 12, 'lp': 5, 'sec_fs': 10, 'tick': 0.4, 'grid_alpha': 0.2, 
            'm_scale': 1}, 
        'double_column': {'width': 6.7, 'height': 4.8, 'main_lw': 1.2, 'sec_lw': 0.6, 'scatter': 6, 
            'label_fs': 18, 'lp': 10, 'sec_fs': 14, 'tick': 0.5, 'grid_alpha':0.3, 'm_scale': 2}}
    
    cfg = plot_layout[layout]
    # --- Setting figure size:
    width, height = cfg['width'], cfg['height']
    # --- Setting plot element size:
    main_lw = cfg['main_lw']
    # --- Setting text font size:
    label_fs, sec_fs = cfg['label_fs'], cfg['sec_fs']
    # --- Setting aesthetic parameters:
    lp, tick, grid_alpha = cfg['lp'], cfg['tick'], cfg['grid_alpha']

    # Setting plot parameters:
    with plt.rc_context({'font.family': 'serif', 'font.serif': ['Times New Roman'],
        'mathtext.fontset': 'stix'}):

        # --- Figure size: single-column ---
        fig, ax = plt.subplots(figsize=(width, height))

        # --- Plot ---
        ax.plot(data_stats['Bin_center'], data_stats['median'], color='#000000', linewidth=main_lw,
            label='Median absolute error', zorder=2)
        ax.fill_between(data_stats['Bin_center'], data_stats['q25'], data_stats['q75'], color=fill_color,
            alpha=0.3, linewidth=0, label='Interquartile range (IQR)', zorder=1)

        # --- Labels ---
        if target == 'cl': 
            ax.set_xlabel(r'Sectional lift coefficient ($C_l$)', fontsize=label_fs, 
                fontname='Times New Roman', labelpad=lp)
            ax.set_ylabel(r'$C_l$ absolute prediction error', fontsize=label_fs, 
                fontname='Times New Roman', labelpad=lp)
        elif target == 'cp':
            ax.set_xlabel(r'Pressure coefficient ($C_p$)', fontsize=label_fs, 
                fontname='Times New Roman', labelpad=lp)
            ax.set_ylabel(r'$C_p$ mean absolute prediction error', fontsize=label_fs, 
                fontname='Times New Roman', labelpad=lp)

        # --- Ticks ---
        for spine in ax.spines.values():
            spine.set_linewidth(0.5)
        ax.tick_params(axis='both', which='major', labelsize=sec_fs, width=tick, direction='in')
        ax.ticklabel_format(axis='y', style='sci', scilimits=(0,0), useMathText=True)
        ax.xaxis.get_offset_text().set_fontsize(sec_fs)
        ax.yaxis.get_offset_text().set_fontsize(sec_fs)

        # --- Grid ---
        ax.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=grid_alpha)

        # --- Limits ---
        ax.set_xlim([min_axis, max_axis])
        ax.set_ylim(0, 0.005)

        # --- Legend ---
        legend = ax.legend(fontsize=sec_fs, fancybox=False, edgecolor='#000000', loc='upper right')
        if layout == 'single_column':
            legend.get_frame().set_linewidth(0.5)

        # --- Save plot ---
        plt.tight_layout(pad=0.6)
        plt.savefig(f'LowFidelity-ANN/MLP-MultiLayerPerceptron/plots/test_error_envelope_{target}_by_{target}_{layout}.png', dpi=600)
        plt.close()
        print(f'Saved: LowFidelity-ANN/MLP-MultiLayerPerceptron/plots/test_error_envelope_{target}_by_{target}_{layout}.png\n')

    return

def plot_test_target_error_envelope_by_input(X_test_raw: np.ndarray, y_test_raw: np.ndarray, 
    y_pred_raw: np.ndarray, target: str, input: str, layout: str):
    """
    Plot the prediction error envelope as a function of an input variable.

    This function investigates how prediction error varies with respect to a selected aerodynamic input
    parameter. Samples are grouped into bins along the chosen input axis, and the resulting plot shows 
    the median absolute error together with the interquartile range.

    For the sectional lift coefficient, the error is computed directly for each sample. For the pressure 
    coefficient distribution, the mean absolute error across the Cp vector is used for each sample.

    Parameters
    ----------
    X_test_raw : np.ndarray
        Test input data in physical scale.

    y_test_raw : np.ndarray
        Ground truth target values in physical scale.

    y_pred_raw : np.ndarray
        Predicted target values in physical scale.

    target : str
        Target to be analyzed. Accepted values are:
            - 'cl': sectional lift coefficient
            - 'cp': pressure coefficient distribution

    input : str
        Input variable to be used on the horizontal axis. Accepted values are:
            - 're': Reynolds number
            - 'aoa': angle of attack
            - 'yb': normalized spanwise position

    layout : str
        Figure layout format. Accepted values are:
            - 'single_column': compact format for single-column figures
            - 'double_column': extended format for double-column figures

    Returns
    -------
    None
        The function saves the figure to disk and does not return any value.

    Notes
    -----
    - The median error is displayed as a solid line.
    - The shaded envelope corresponds to the interquartile range.
    - This plot is intended to diagnose conditional model performance across the
      aerodynamic input space.
    """

    # --- Splits the targets data ---
    cl_test = y_test_raw[:,0].reshape(-1, 1)
    cl_pred = y_pred_raw[:,0].reshape(-1, 1)
    cp_test = y_test_raw[:,1:]
    cp_pred = y_pred_raw[:,1:]

    if target == 'cl':
        abs_error = np.abs(cl_test - cl_pred)
        abs_error = abs_error.flatten()
        color = '#D55E00'
    elif target == 'cp':
        abs_error = np.mean(np.abs(cp_test - cp_pred), axis=1)
        abs_error = abs_error.flatten()
        color = '#0072B2'

    if input == 're':
        input_data = X_test_raw[:,0]
    elif input == 'aoa':
        input_data = X_test_raw[:,1]
    elif input == 'yb':
        input_data = X_test_raw[:,2]
    
    # --- Dataframe ---
    data = pd.DataFrame({'Input': input_data, 'Abs_Error': abs_error})
    # Define the number of bins and associate points:
    bins = np.linspace(data['Input'].min(), data['Input'].max(), 41)
    data['Bin'] = pd.cut(data['Input'], bins=bins, include_lowest=True)
    # Group the data by bin and calculate the quartiles:
    data_stats = data.groupby('Bin', observed=True)['Abs_Error'].agg(median='median', 
        q25=lambda x: np.percentile(x, 25), q75=lambda x: np.percentile(x, 75)).reset_index()
    # Compute bin centers:
    data_stats['Bin_center'] = data_stats['Bin'].apply(lambda x: x.mid)
    # Remove NaN values:
    data_stats = data_stats.dropna()
    # Smooth the quartile curves:
    data_stats['median'] = data_stats['median'].rolling(window=3, center=True, min_periods=1).mean()
    data_stats['q25'] = data_stats['q25'].rolling(window=3, center=True, min_periods=1).mean()
    data_stats['q75'] = data_stats['q75'].rolling(window=3, center=True, min_periods=1).mean()
    # Calculates minimum and maximum values
    min_axis = min(data['Input'])
    max_axis = max(data['Input'])

    # --- Plot layout configurations ---
    plot_layout = {'single_column': {'width': 3.35, 'height': 2.55, 'main_lw': 0.8, 'sec_lw': 0.4, 
            'scatter': 2, 'label_fs': 12, 'lp': 5, 'sec_fs': 10, 'tick': 0.4, 'grid_alpha': 0.2, 
            'm_scale': 1}, 
        'double_column': {'width': 6.7, 'height': 4.8, 'main_lw': 1.2, 'sec_lw': 0.6, 'scatter': 6, 
            'label_fs': 18, 'lp': 10, 'sec_fs': 14, 'tick': 0.5, 'grid_alpha':0.3, 'm_scale': 2}}
    
    cfg = plot_layout[layout]
    # --- Setting figure size:
    width, height = cfg['width'], cfg['height']
    # --- Setting plot element size:
    main_lw = cfg['main_lw']
    # --- Setting text font size:
    label_fs, sec_fs = cfg['label_fs'], cfg['sec_fs']
    # --- Setting aesthetic parameters:
    lp, tick, grid_alpha = cfg['lp'], cfg['tick'], cfg['grid_alpha']

    # Setting plot parameters:
    with plt.rc_context({'font.family': 'serif', 'font.serif': ['Times New Roman'],
        'mathtext.fontset': 'stix'}):

        # --- Figure size: single-column ---
        fig, ax = plt.subplots(figsize=(width, height))

        # --- Plot ---
        ax.plot(data_stats['Bin_center'], data_stats['median'], color='#000000', linewidth=main_lw,
            label='Median absolute error', zorder=2)
        ax.fill_between(data_stats['Bin_center'], data_stats['q25'], data_stats['q75'], color=color,
            alpha=0.25, linewidth=0, label='25th-75th percentile interval', zorder=1)

        # --- Labels ---
        if input == 're': 
            ax.set_xlabel(r'Reynolds number ($Re$)', fontsize=label_fs, fontname='Times New Roman', 
                labelpad=lp)
        elif input == 'aoa':
            ax.set_xlabel(r'Angle of attack ($\alpha$) [$^\circ$]', fontsize=label_fs, 
                fontname='Times New Roman', labelpad=lp)
        elif input == 'yb':
            ax.set_xlabel(r'Normalized spanwise position ($y/b$)', fontsize=label_fs, 
                fontname='Times New Roman', labelpad=lp)
        if target == 'cl':
            ax.set_ylabel('$C_l$ absolute prediction error', fontsize=label_fs, 
                fontname='Times New Roman', labelpad=lp)
        elif target == 'cp':
            ax.set_ylabel('$C_p$ mean absolute prediction error', fontsize=label_fs,
                fontname='Times New Roman', labelpad=lp)

        # --- Ticks ---
        # --- Ticks ---
        for spine in ax.spines.values():
            spine.set_linewidth(0.5)
        ax.tick_params(axis='both', which='major', labelsize=sec_fs, width=tick, direction='in')
        ax.ticklabel_format(axis='y', style='sci', scilimits=(0,0), useMathText=True)
        ax.xaxis.get_offset_text().set_fontsize(sec_fs)
        ax.yaxis.get_offset_text().set_fontsize(sec_fs)

        # --- Grid ---
        ax.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=grid_alpha)

        # --- Limits ---
        ax.set_xlim([min_axis, max_axis])
        ax.set_ylim(0, 0.005)
        if input == 're':
            formatter = matplotlib.ticker.ScalarFormatter(useMathText=True)
            formatter.set_scientific(True)
            formatter.set_powerlimits((5, 5))
            ax.xaxis.set_major_formatter(formatter)
            ax.xaxis.set_major_locator(matplotlib.ticker.MultipleLocator(1e5))
            ax.xaxis.get_offset_text().set_fontsize(sec_fs)
            ax.xaxis.get_offset_text().set_x(1.02)

        # --- Legend ---
        legend = ax.legend(fontsize=sec_fs, fancybox=False, edgecolor='#000000', loc='upper right')
        if layout == 'single_column':
            legend.get_frame().set_linewidth(0.5)

        # --- Save plot ---
        plt.tight_layout(pad=0.6)
        plt.savefig(f'LowFidelity-ANN/MLP-MultiLayerPerceptron/plots/test_{target}_error_envelope_by_{input}_{layout}.png', dpi=600)
        plt.close()
        print(f'Saved: LowFidelity-ANN/MLP-MultiLayerPerceptron/plots/test_{target}_error_envelope_by_{input}_{layout}.png\n')

    return

def plot_test_cp_error_envelope_by_chord(y_test_raw: np.ndarray, y_pred_raw: np.ndarray, surface: str, 
    layout: str):
    """
    Plot the chordwise Cp prediction error envelope for a selected wing surface.

    This function evaluates how the absolute prediction error in pressure coefficient varies along the 
    normalized chordwise coordinate. The Cp distribution is separated into suction-side and pressure-side 
    subsets using a predefined chordwise mapping. The resulting envelope plot shows the median error and 
    interquartile range as a function of chordwise position.

    Parameters
    ----------
    y_test_raw : np.ndarray
        Ground truth target values in physical scale.

    y_pred_raw : np.ndarray
        Predicted target values in physical scale.

    surface : str
        Wing surface to be analyzed. Accepted values are:
            - 'suction': suction side
            - 'pressure': pressure side

    layout : str
        Figure layout format. Accepted values are:
            - 'single_column': compact format for single-column figures
            - 'double_column': extended format for double-column figures

    Returns
    -------
    None
        The function saves the figure to disk and does not return any value.

    Notes
    -----
    - The chordwise coordinates are loaded from an external Cp mapping file.
    - The envelope is based on the distribution of absolute errors at each chordwise region.
    - This analysis is useful for identifying localized reconstruction deficiencies
      along the airfoil surface.
    """
    # --- Loads pressure measurement chordwise positions ---
    cp_mapping_path = 'LowFidelity-ANN/utils/LowFidelity-CpMapping.csv'
    cp_position = pd.read_csv(cp_mapping_path, sep=';', usecols=['X_coord']).to_numpy().ravel()

    # --- Splits the pressure data ---
    cp_test = y_test_raw[:,1:]
    cp_pred = y_pred_raw[:,1:]
    cp_test_suction = cp_test[:, :104]
    cp_test_pressure = cp_test[:, 103:]
    cp_pred_suction = cp_pred[:, :104]
    cp_pred_pressure = cp_pred[:, 103:]

    # --- Splits the coordinates data ---
    cp_suction_pos = cp_position[:104]
    cp_pressure_pos = cp_position[103:]

    # --- Flattens the data ---
    cp_suction_pos = np.tile(np.ravel(cp_suction_pos), cp_test_suction.shape[0])
    cp_pressure_pos = np.tile(np.ravel(cp_pressure_pos), cp_test_pressure.shape[0])
    cp_test_suction = cp_test_suction.flatten()
    cp_test_pressure = cp_test_pressure.flatten()
    cp_pred_suction = cp_pred_suction.flatten()
    cp_pred_pressure = cp_pred_pressure.flatten()

    if surface == 'suction':
        abs_error = np.abs(cp_test_suction - cp_pred_suction)
        cp_pos = cp_suction_pos
    elif surface == 'pressure':
        abs_error = np.abs(cp_test_pressure - cp_pred_pressure)
        cp_pos = cp_pressure_pos

    # --- Dataframe ---
    data = pd.DataFrame({'Position': cp_pos, 'Abs_Error': abs_error})
    # Define the number of bins and associate points:
    bins = np.linspace(data['Position'].min(), data['Position'].max(), 41)
    data['Bin'] = pd.cut(data['Position'], bins=bins, include_lowest=True)
    # Group the data by bin and calculate the quartiles:
    data_stats = data.groupby('Bin', observed=True)['Abs_Error'].agg(median='median', 
        q25=lambda x: np.percentile(x, 25), q75=lambda x: np.percentile(x, 75)).reset_index()
    # Compute bin centers:
    data_stats['Bin_center'] = data_stats['Bin'].apply(lambda x: x.mid)
    # Remove NaN values:
    data_stats = data_stats.dropna()
    # Smooth the quartile curves:
    data_stats['median'] = data_stats['median'].rolling(window=3, center=True, min_periods=1).mean()
    data_stats['q25'] = data_stats['q25'].rolling(window=3, center=True, min_periods=1).mean()
    data_stats['q75'] = data_stats['q75'].rolling(window=3, center=True, min_periods=1).mean()

    # --- Plot layout configurations ---
    plot_layout = {'single_column': {'width': 3.35, 'height': 2.55, 'main_lw': 0.8, 'sec_lw': 0.4, 
            'scatter': 2, 'label_fs': 12, 'lp': 5, 'sec_fs': 10, 'tick': 0.4, 'grid_alpha': 0.2, 
            'm_scale': 1}, 
        'double_column': {'width': 6.7, 'height': 4.8, 'main_lw': 1.2, 'sec_lw': 0.6, 'scatter': 6, 
            'label_fs': 18, 'lp': 10, 'sec_fs': 14, 'tick': 0.5, 'grid_alpha':0.3, 'm_scale': 2}}
    
    cfg = plot_layout[layout]
    # --- Setting figure size:
    width, height = cfg['width'], cfg['height']
    # --- Setting plot element size:
    main_lw = cfg['main_lw']
    # --- Setting text font size:
    label_fs, sec_fs = cfg['label_fs'], cfg['sec_fs']
    # --- Setting aesthetic parameters:
    lp, tick, grid_alpha = cfg['lp'], cfg['tick'], cfg['grid_alpha']

    # Setting plot parameters:
    with plt.rc_context({'font.family': 'serif', 'font.serif': ['Times New Roman'],
        'mathtext.fontset': 'stix'}):

        # --- Figure size: single-column ---
        fig, ax = plt.subplots(figsize=(width, height))

        # --- Plot ---
        ax.plot(data_stats['Bin_center'], data_stats['median'], color='#000000', linewidth=main_lw,
            label='Median absolute error', zorder=2)
        ax.fill_between(data_stats['Bin_center'], data_stats['q25'], data_stats['q75'], color='#0072B2',
            alpha=0.25, linewidth=0, label='25th-75th percentile interval', zorder=1)

        # --- Labels ---
        if surface == 'suction': 
            ax.set_xlabel(r'Normalized chordwise position ($x/c$) on suction side', fontsize=label_fs, 
                fontname='Times New Roman', labelpad=lp)
        elif surface == 'pressure':
            ax.set_xlabel(r'Normalized chordwise position ($x/c$) on pressure side', fontsize=label_fs, 
                fontname='Times New Roman', labelpad=lp)
        ax.set_ylabel('$C_p$ absolute prediction error', fontsize=label_fs, fontname='Times New Roman', 
            labelpad=lp)

        # --- Ticks ---
        for spine in ax.spines.values():
            spine.set_linewidth(0.5)
        ax.tick_params(axis='both', which='major', labelsize=sec_fs, width=tick, direction='in')
        ax.ticklabel_format(axis='y', style='sci', scilimits=(0,0), useMathText=True)
        ax.xaxis.get_offset_text().set_fontsize(sec_fs)
        ax.yaxis.get_offset_text().set_fontsize(sec_fs)

        # --- Grid ---
        ax.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=grid_alpha)

        # --- Limits ---
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 0.005)

        # --- Legend ---
        legend = ax.legend(fontsize=sec_fs, fancybox=False, edgecolor='#000000', loc='upper right')
        if layout == 'single_column':
            legend.get_frame().set_linewidth(0.5)

        # --- Save plot ---
        plt.tight_layout(pad=0.6)
        plt.savefig(f'LowFidelity-ANN/MLP-MultiLayerPerceptron/plots/test_cp_error_envelope_{surface}_side_{layout}.png', dpi=600)
        plt.close()
        print(f'Saved: LowFidelity-ANN/MLP-MultiLayerPerceptron/plots/test_cp_error_envelope_{surface}_side_{layout}.png\n')

    return

def plot_test_error_distribution_by_target(y_test_raw: np.ndarray, y_pred_raw: np.ndarray, target: str, 
    layout: str):
    """
    Plot the distribution of prediction errors for a selected target.

    This function generates a histogram with kernel density estimation (KDE) to visualize the statistical 
    distribution of model errors for the selected target. For the sectional lift coefficient, the absolute 
    error is computed directly. For the pressure coefficient distribution, the mean absolute error across the
    Cp vector is computed for each sample.

    Parameters
    ----------
    y_test_raw : np.ndarray
        Ground truth target values in physical scale.

    y_pred_raw : np.ndarray
        Predicted target values in physical scale.

    target : str
        Target to be analyzed. Accepted values are:
            - 'cl': sectional lift coefficient
            - 'cp': pressure coefficient distribution

    layout : str
        Figure layout format. Accepted values are:
            - 'single_column': compact format for single-column figures
            - 'double_column': extended format for double-column figures

    Returns
    -------
    None
        The function saves the figure to disk and does not return any value.

    Notes
    -----
    - The histogram is normalized as a probability density.
    - The KDE curve is included to provide a smoother representation of the
      underlying error distribution.
    """

    # --- Splits the targets data ---
    cl_test = y_test_raw[:,0].reshape(-1, 1)
    cl_pred = y_pred_raw[:,0].reshape(-1, 1)
    cp_test = y_test_raw[:,1:]
    cp_pred = y_pred_raw[:,1:]

    if target == 'cl':
        abs_error = np.abs(cl_test - cl_pred)
        abs_error = abs_error.flatten()
        color = '#D55E00'
    elif target == 'cp':
        abs_error = np.mean(np.abs(cp_test - cp_pred), axis=1)
        abs_error = abs_error.flatten()
        color = '#0072B2'

    # --- Plot layout configurations ---
    plot_layout = {'single_column': {'width': 3.35, 'height': 2.55, 'main_lw': 0.8, 'sec_lw': 0.4, 
            'scatter': 2, 'label_fs': 12, 'lp': 5, 'sec_fs': 10, 'tick': 0.4, 'grid_alpha': 0.2, 
            'm_scale': 1}, 
        'double_column': {'width': 6.7, 'height': 4.8, 'main_lw': 1.2, 'sec_lw': 0.6, 'scatter': 6, 
            'label_fs': 18, 'lp': 10, 'sec_fs': 14, 'tick': 0.5, 'grid_alpha':0.3, 'm_scale': 2}}
    
    cfg = plot_layout[layout]
    # --- Setting figure size:
    width, height = cfg['width'], cfg['height']
    # --- Setting plot element size:
    main_lw = cfg['main_lw']
    # --- Setting text font size:
    label_fs, sec_fs = cfg['label_fs'], cfg['sec_fs']
    # --- Setting aesthetic parameters:
    lp, tick, grid_alpha = cfg['lp'], cfg['tick'], cfg['grid_alpha']

    # Setting plot parameters:
    with plt.rc_context({'font.family': 'serif', 'font.serif': ['Times New Roman'],
        'mathtext.fontset': 'stix'}):

        # --- Figure size: single-column ---
        fig, ax = plt.subplots(figsize=(width, height))

        # --- Plot ---
        sns.histplot(abs_error, bins=41, stat='density', kde=True, color=color, edgecolor='#000000',
            alpha=0.6, line_kws={'color': '#000000', 'linewidth': main_lw})

        # --- Labels ---
        if target == 'cl':
            ax.set_xlabel('$C_l$ absolute prediction error', fontsize=label_fs, 
                fontname='Times New Roman', labelpad=lp)
        if target == 'cp':
            ax.set_xlabel('$C_p$ mean absolute prediction error', fontsize=label_fs, 
                fontname='Times New Roman', labelpad=lp)
        ax.set_ylabel('Probability density', fontsize=label_fs, fontname='Times New Roman', labelpad=lp)

        # --- Ticks ---
        for spine in ax.spines.values():
            spine.set_linewidth(0.5)
        ax.tick_params(axis='both', which='major', labelsize=sec_fs, width=tick, direction='in')

        # --- Grid ---
        ax.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=grid_alpha)

        # --- Limits ---
        ax.set_xlim(0, np.max(abs_error))
        ax.set_ylim(0, 500)

        # --- Save plot ---
        plt.tight_layout(pad=0.6)
        plt.savefig(f'LowFidelity-ANN/MLP-MultiLayerPerceptron/plots/test_{target}_error_distribution_{layout}.png', dpi=600)
        plt.close()
        print(f'Saved: LowFidelity-ANN/MLP-MultiLayerPerceptron/plots/test_{target}_error_distribution_{layout}.png\n')

    return

def save_report(model: keras, history, fit_losses: list, execution_time: float, X_test_raw: np.ndarray, 
    y_test_raw: np.ndarray, y_pred_raw: np.ndarray, X_train: np.ndarray, X_val: np.ndarray):
    """
    Generate and save a comprehensive performance report of the trained model.

    The report includes:

        - Training configuration
        - Dataset statistics
        - Overall and per-target evaluation metrics (Cl and Cp)
        - Final training and validation losses
        - Model architecture summary

    Performance is evaluated using:

        - Coefficient of determination (R²)
        - Mean Absolute Error (MAE)
        - Mean Squared Error (MSE)
        - Root Mean Squared Error (RMSE)

    Parameters
    ----------
    model : keras.Model
        Trained neural network.

    history : keras.callbacks.History
        Training history object.

    fit_losses : list
        Final training and validation losses.

    execution_time : float
        Total execution time.

    Returns
    -------
    None
    """

    # --- Splits the targets data ---
    cl_test = y_test_raw[:,0].reshape(-1, 1)
    cl_pred = y_pred_raw[:,0].reshape(-1, 1)
    cp_test = y_test_raw[:,1:]
    cp_pred = y_pred_raw[:,1:]

    r2_overall = skl.metrics.r2_score(y_test_raw, y_pred_raw)
    mae_overall = skl.metrics.mean_absolute_error(y_test_raw, y_pred_raw)
    mse_overall = skl.metrics.mean_squared_error(y_test_raw, y_pred_raw)
    rmse_overall = np.sqrt(mse_overall)

    r2_cl = skl.metrics.r2_score(cl_test, cl_pred)
    mae_cl = skl.metrics.mean_absolute_error(cl_test, cl_pred)
    mse_cl = skl.metrics.mean_squared_error(cl_test, cl_pred)
    rmse_cl = np.sqrt(mse_cl)

    r2_cp = skl.metrics.r2_score(cp_test, cp_pred)
    mae_cp = skl.metrics.mean_absolute_error(cp_test, cp_pred)
    mse_cp = skl.metrics.mean_squared_error(cp_test, cp_pred)
    rmse_cp = np.sqrt(mse_cp)

    # --- Generate the performance report ---
    stringlist = []
    model.summary(print_fn=lambda x: stringlist.append(x))
    model_structure = "\n".join(stringlist)
    
    content = f"""
PERFORMANCE REPORT - potential_mlp_ann
==========================================================
Date: {time.strftime("%Y-%m-%d %H:%M:%S")}
Execution Time: {execution_time:.2f} seconds


1. CONFIGURATION
----------------------------------
Epochs: {len(history.history['loss'])}
Input Shape: {X_test_raw.shape}
Outputs: {y_test_raw.shape}


2. DATASET SUMMARY
----------------------------------
Total samples: {X_train.shape[0] + X_val.shape[0] + X_test_raw.shape[0]}
Total flow cases: {(X_train.shape[0] + X_val.shape[0] + X_test_raw.shape[0])//80}
Sections per flow case: 80

Training samples: {X_train.shape[0]}
Training flow cases: {X_train.shape[0]//80}

Validation samples: {X_val.shape[0]}
Validation flow cases: {X_val.shape[0]//80}

Testing samples: {X_test_raw.shape[0]}
Testing flow cases: {X_test_raw.shape[0]//80}


3. OVERALL MODEL TEST PERFORMANCE
----------------------------------
R² Score: {r2_overall:.6f}
MAE:      {mae_overall:.6f}
MSE:      {mse_overall:.6f}
RMSE:     {rmse_overall:.6f}


4. CL TEST PERFORMANCE
----------------------------------
R² Score: {r2_cl:.6f}
MAE:      {mae_cl:.6f}
MSE:      {mse_cl:.6f}
RMSE:     {rmse_cl:.6f}


5. CP TEST PERFORMANCE
----------------------------------
R² Score: {r2_cp:.6f}
MAE:      {mae_cp:.6f}
MSE:      {mse_cp:.6f}
RMSE:     {rmse_cp:.6f}


6. FINAL LOSS RESULTS
----------------------------------
Best epoch: {np.argmin(history.history['loss']) + 1}
Final Loss (Training): {fit_losses[0]:.6f}
Final MAE (Training): {fit_losses[1]:.6f}
Final Loss (Validation): {fit_losses[2]:.6f}
Final MAE (Validation): {fit_losses[3]:.6f}


7. MODEL ARCHITECTURE
----------------------------------
{model_structure}
==========================================================
"""
    # Saving the report:
    with open('LowFidelity-ANN/MLP-MultiLayerPerceptron/train-results/performance-report-MLP-ANN.txt', "w", 
        encoding="utf-8") as f: f.write(content)
    print(f"Saved: LowFidelity-ANN/MLP-MultiLayerPerceptron/train-results/performance-report-MLP-ANN.txt\n")

    return

def main(data_path: str):
    """
    Execute the complete training and evaluation pipeline.

    This function orchestrates the entire workflow:

        - Data loading and splitting
        - Preprocessing and normalization
        - Model construction and training
        - Model evaluation
        - Test set prediction
        - Diagnostic visualization
        - Performance report generation

    Parameters
    ----------
    data_path : str
        Path to the aerodynamic dataset.

    Returns
    -------
    None
        The function saves the figure to disk and does not return any value.
    """

    # --- Starts the timer for recording the execution time ---
    start_time = time.time()

    # --- Loads and split the dataset ---
    print('\nStarting the Multi-Layer Perceptron ANN training process...\n')
    print(f'Loading data from: {data_path}\n')
    data_train, data_val, data_test = load_and_split_data(data_path)
    print('Data loaded and split successfully.\n')

    # --- Preprocessing the data for the final model training ---
    print('Preprocessing data for the training...\n')
    X_train, X_val, X_test, y_train, y_val, y_test, X_scaler, y_scaler = preprocess_data(data_train, 
        data_val, data_test)
    
    # --- Builds the model ---
    print('Building model architecture...\n')
    model = build_model(X_train.shape[1], y_train.shape[1])

    # --- Trains the model ---
    print('Starting training...\n')
    callbacks = [keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, min_delta=1e-6,
            restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6,
            cooldown=5, min_delta=1e-5),
        keras.callbacks.ModelCheckpoint('LowFidelity-ANN/MLP-MultiLayerPerceptron/train-results/MLP-ANN.keras', 
            monitor='val_loss', save_best_only=True)]
    history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=500, callbacks=callbacks, 
        batch_size=32, verbose=0)
    print("Saved: LowFidelity-ANN/MLP-MultiLayerPerceptron/train-results/MLP-ANN.keras\n")

    # --- Evaluates the model ---
    print('Evaluating the model...\n')
    train_loss, train_mae = model.evaluate(X_train, y_train, verbose=0)
    val_loss, val_mae = model.evaluate(X_val, y_val, verbose=0)
    fit_losses = [train_loss, train_mae, val_loss, val_mae]

    # --- Predicting on the test set ---
    print('Predicting on the test set...\n')
    y_test_raw, y_pred_raw = predict_test(model, X_test, y_test, y_scaler)

    # --- Reverses the input scaling transformation ---
    X_train_raw = X_scaler.inverse_transform(X_train)
    X_val_raw = X_scaler.inverse_transform(X_val)
    X_test_raw = X_scaler.inverse_transform(X_test)

    # --- Finishes the timer ---
    end_time = time.time()
    execution_time = end_time - start_time

    # --- Plots the parameter space distribution ---
    plot_data_split_parameter_space(X_train_raw, X_val_raw, X_test_raw, layout='single_column')
    plot_data_split_parameter_space(X_train_raw, X_val_raw, X_test_raw, layout='double_column')

    # --- Plots the training history ---
    plot_training_history(history, metric='mse', layout='single_column')
    plot_training_history(history, metric='mae', layout='single_column')

    # --- Saves test predictions plots ---
    plot_test_predictions(y_test_raw, y_pred_raw, target='cl', layout='single_column')
    plot_test_predictions(y_test_raw, y_pred_raw, target='cp', layout='single_column')
    plot_test_predictions(y_test_raw, y_pred_raw, target='overall', layout='single_column')

    # --- Saves the test error envelope plots ---
    plot_test_error_envelope_by_target(y_test_raw, y_pred_raw, target='cl', layout='single_column')
    plot_test_error_envelope_by_target(y_test_raw, y_pred_raw, target='cp', layout='single_column')
    plot_test_target_error_envelope_by_input(X_test_raw, y_test_raw, y_pred_raw, target = 'cl', 
        input='re', layout='single_column')
    plot_test_target_error_envelope_by_input(X_test_raw, y_test_raw, y_pred_raw, target = 'cl', 
        input='aoa', layout='single_column')
    plot_test_target_error_envelope_by_input(X_test_raw, y_test_raw, y_pred_raw, target = 'cl', 
        input='yb', layout='single_column')
    plot_test_target_error_envelope_by_input(X_test_raw, y_test_raw, y_pred_raw, target = 'cp', 
        input='re', layout='single_column')
    plot_test_target_error_envelope_by_input(X_test_raw, y_test_raw, y_pred_raw, target = 'cp', 
        input='aoa', layout='single_column')
    plot_test_target_error_envelope_by_input(X_test_raw, y_test_raw, y_pred_raw, target = 'cp', 
        input='yb', layout='single_column')
    plot_test_cp_error_envelope_by_chord(y_test_raw, y_pred_raw, surface='suction', layout='single_column')
    plot_test_cp_error_envelope_by_chord(y_test_raw, y_pred_raw, surface='pressure', layout='single_column')

    # --- Saves the test error distribution plots ---
    plot_test_error_distribution_by_target(y_test_raw, y_pred_raw, target='cl', layout='single_column')
    plot_test_error_distribution_by_target(y_test_raw, y_pred_raw, target='cp', layout='single_column')

    # --- Saves the text performance report ---
    save_report(model, history, fit_losses, execution_time, X_test_raw, y_test_raw, y_pred_raw, X_train, X_val)

    # --- Saves the raw test dataset ---
    data_test.to_csv('LowFidelity-ANN/MLP-MultiLayerPerceptron/raw_test_dataset.csv', sep=',', index=False)    

    # --- Finishes the process ---
    print("Process completed successfully.\n")

    return

if __name__ == '__main__':
    main(data_path)