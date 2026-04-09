# -*- coding: utf-8 -*-
"""
========================================================================================================
Module: train_potential_mlp_pinn
Author: Caio Dias Filho
Creation date: 2026-03-17
Last modification: 2026-03-17
Version: 1.0
========================================================================================================

OVERVIEW
--------


DEPENDENCIES 
------------

    
OUTPUT FILES
------------


NOTES
-----


========================================================================================================
"""

# =======================================================================================================
#                                            USER INPUT SECTION
# =======================================================================================================

# --- Dataset path ---
data_path = 'Potential-ANN/utils/Potential-PressureDistributionData.csv'

# --- Airfoil data path ---
airfoil_data_path = 'Potential-ANN/utils/NACA23015.csv'

# --- Physical loss weight ---
lambda_physical = 0.1


# =======================================================================================================
#                                              CORE FUNCTION
# =======================================================================================================

# System libraries and error suppresing:
import os
import warnings
os.environ['TF_CPP_MIN_LOG_LEVEL']="2"
os.environ['TF_ENABLE_ONEDNN_OPTS']="0"
warnings.filterwarnings("ignore")

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

# Reproductibility setup:
SEED = 42
os.environ['PYTHONHASHSEED'] = str(SEED)
tensorflow.random.set_seed(SEED)
np.random.seed(SEED)
print(f'Global seed set to: {SEED}')

def load_and_split_data(filepath: str):
    """
    """

    # --- Loading the dataset ---
    data = (pd.read_csv(filepath, sep=';')).copy()

    # --- Create identifier for flow case (assuming 80 sections per case) ---
    wing_sections = 80
    idx = np.arange(len(data))
    data['flow_case'] = idx // wing_sections
    unique_cases = data['flow_case'].unique()
    train_cases, test_cases = skl.model_selection.train_test_split(unique_cases, test_size=0.2, 
        random_state=42, shuffle=True)
    train_cases, val_cases = skl.model_selection.train_test_split(train_cases, test_size=0.2, 
        random_state=42, shuffle=True)

    # --- Separate the datasets and drop auxiliary column ---
    data_train = data[data['flow_case'].isin(train_cases)].drop(columns=['flow_case'])
    data_val = data[data['flow_case'].isin(val_cases)].drop(columns=['flow_case'])
    data_test = data[data['flow_case'].isin(test_cases)].drop(columns=['flow_case'])

    return data_train, data_val, data_test

def preprocess_data(data_train: pd.DataFrame, data_val: pd.DataFrame, data_test: pd.DataFrame):
    """
    """

    # --- Defining input and target columns ---
    X_cols = data_train.columns[0:3]
    y_cols = data_train.columns[3:205]

    # --- Splitting the raw features data into training, validation, and testing sets ---
    X_train_raw = data_train[X_cols].to_numpy()
    X_val_raw = data_val[X_cols].to_numpy()
    X_test_raw = data_test[X_cols].to_numpy()

    # --- Splitting the angle of attack feature into training, validation, and testing sets ---
    AoA_train_raw = X_train_raw[:, 1].reshape(-1, 1)
    AoA_val_raw = X_val_raw[:, 1].reshape(-1, 1)

    # --- Splitting the raw targets data into training, validation, and testing data ---
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

    # --- Appends the AoA feature to the target training data ---
    y_train = np.hstack((y_train, AoA_train_raw))
    y_val = np.hstack((y_val, AoA_val_raw))

    # --- Saving the scalers ---
    joblib.dump(X_scaler, 'Potential-PINN/scalers/train-mlp-pinn/mlp_pinn_X_scaler.pkl')
    joblib.dump(y_scaler, 'Potential-PINN/scalers/train-mlp-pinn/mlp_pinn_y_scaler.pkl')

    return X_train, X_val, X_test, y_train, y_val, y_test, X_scaler, y_scaler

def build_pinn_loss(panel_dx: np.ndarray, panel_dy: np.ndarray, lambda_physical: float, 
    y_scaler: skl.preprocessing.StandardScaler):
    """
    """

    # --- Converting to tensorflow constants ---
    panel_dx = tensorflow.constant(panel_dx, dtype=tensorflow.float32)
    panel_dy = tensorflow.constant(panel_dy, dtype=tensorflow.float32)

    y_mean = y_scaler.mean_.astype(np.float32)
    y_scale = y_scaler.scale_.astype(np.float32)

    Cl_mean = tensorflow.constant(y_mean[0:1].reshape(1, 1), dtype=tensorflow.float32)
    Cl_scale = tensorflow.constant(y_scale[0:1].reshape(1, 1), dtype=tensorflow.float32)

    Cp_mean = tensorflow.constant(y_mean[1:].reshape(1, -1), dtype=tensorflow.float32)
    Cp_scale = tensorflow.constant(y_scale[1:].reshape(1, -1), dtype=tensorflow.float32)

    def pinn_loss(y_true, y_pred):
        """
        """
        
        # --- Casting to float32 ---
        y_true = tensorflow.cast(y_true, dtype=tensorflow.float32)
        y_pred = tensorflow.cast(y_pred, dtype=tensorflow.float32)

        # --- Defines the true supervised values ---
        Cl_true = y_true[:, 0:1]
        Cp_true = y_true[:, 1:-1]
        AoA_deg = y_true[:, -1:]

        # --- Defines the predicted supervised values ---
        Cl_pred = y_pred[:, 0:1]
        Cp_pred = y_pred[:, 1:]

        # --- Defines the supervised loss ---
        mse_Cl = tensorflow.reduce_mean(tensorflow.square(Cl_true - Cl_pred))
        mse_Cp = tensorflow.reduce_mean(tensorflow.square(Cp_true - Cp_pred))
        data_loss = mse_Cl + mse_Cp

        # --- Reescales the predicted supervised values ---
        Cl_pred_physics = Cl_pred * Cl_scale + Cl_mean
        Cp_pred_physics = Cp_pred * Cp_scale + Cp_mean

        # --- Reconstructs Cl from Cp ---
        # Computes the average Cp for each panel:
        panel_Cp = 0.5 * (Cp_pred_physics[:, :-1] + Cp_pred_physics[:, 1:])
        # Integrates to find normal (Cn) and axial (Ca) force coefficients:
        Cn = tensorflow.reduce_sum(panel_Cp * panel_dx, axis=1, keepdims=True)
        Ca = -tensorflow.reduce_sum(panel_Cp * panel_dy, axis=1, keepdims=True)
        # Defines the angle of attack in radians:
        AoA_rad = AoA_deg * (np.pi / 180.0)
        # Projects Cn and Ca into the lift coefficient (Cl) using the angle of attack:
        Cl_from_Cp_physics = Cn * tensorflow.cos(AoA_rad) - Ca * tensorflow.sin(AoA_rad)
        # Defines the physical loss:
        phys_loss = tensorflow.reduce_mean(tensorflow.square(Cl_pred_physics - Cl_from_Cp_physics))

        return data_loss + lambda_physical * phys_loss
    
    return pinn_loss

def pinn_mae_metric(y_true: np.ndarray, y_pred: np.ndarray):
    """
    """

    # --- Casting to float32 ---
    y_true = tensorflow.cast(y_true, dtype=tensorflow.float32)
    y_pred = tensorflow.cast(y_pred, dtype=tensorflow.float32)

    # --- Defines the true supervised values, excluding the angle of attack ---
    supervised_true = y_true[:, :-1]

    return tensorflow.reduce_mean(tensorflow.abs(supervised_true - y_pred))

def build_model(input_dim: int, output_dim: int, panel_dx: np.ndarray, panel_dy: np.ndarray, 
    lambda_physical: float, y_scaler: skl.preprocessing.StandardScaler):
    """
    """

    # --- Defining the model input layer ---
    inputs = keras.layers.Input(shape=(input_dim,), name='Input_Layer')

    # --- Defining the model hidden layers ---
    x = keras.layers.Dense(units=96, activation='gelu', kernel_initializer='glorot_uniform', 
        name='Hidden_Layer_1')(inputs)
    #x = keras.layers.Dropout(0.05, name='Dropout_1')(x)

    x = keras.layers.Dense(units=256, activation='gelu', kernel_initializer='glorot_uniform', 
        name='Hidden_Layer_2')(x)
    #x = keras.layers.Dropout(0.05, name='Dropout_2')(x)

    x = keras.layers.Dense(units=32, activation='gelu', kernel_initializer='glorot_uniform', 
        name='Hidden_Layer_3')(x)
    #x = keras.layers.Dropout(0.05, name='Dropout_3')(x)

    x = keras.layers.Dense(units=160, activation='gelu', kernel_initializer='glorot_uniform', 
        name='Hidden_Layer_4')(x)
    
    x = keras.layers.Dense(units=32, activation='gelu', kernel_initializer='glorot_uniform', 
        name='Hidden_Layer_5')(x)

    # --- Defining the model output layer ---
    outputs = keras.layers.Dense(units=output_dim, activation='linear', kernel_initializer='glorot_uniform', 
        name='Output_Layer')(x)

    # --- Defining the model ---
    model = keras.Model(inputs=inputs, outputs=outputs, name='potential_mlp_ann')

    # --- Defining the physical loss and MAE metric ---
    physical_loss = build_pinn_loss(panel_dx=panel_dx, panel_dy=panel_dy, lambda_physical=lambda_physical,
        y_scaler=y_scaler)
    pinn_mae_metric.__name__ = 'mae'

    # --- Compiling the model ---
    model.compile(loss=physical_loss, optimizer=keras.optimizers.Adam(learning_rate=0.002733), 
        metrics=[pinn_mae_metric])

    return model

def predict_test(model: keras.Model, X_test: np.ndarray, y_test: np.ndarray, 
    y_scaler: skl.preprocessing.StandardScaler):
    """
    """

    # --- Predicting on the test set ---
    y_pred = model.predict(X_test, verbose=0)

    # --- Reverses the scaling ---
    y_test_raw = y_scaler.inverse_transform(y_test)
    y_pred_raw = y_scaler.inverse_transform(y_pred)

    return y_test_raw, y_pred_raw

def plot_training_history(history: dict, metric: str, layout: str):
    """
    """

    # --- Load data for MSE metric plot ---
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(1, len(loss) + 1)
    best_epoch = np.argmin(val_loss) + 1
    best_val_loss = min(val_loss)

    # --- Load data for MAE metric plot ---
    mae_loss = history.history['mae']
    val_mae_loss = history.history['val_mae']
    best_mae_val_loss = val_mae_loss[best_epoch - 1]

    # --- Plot layout configurations ---
    plot_layout = {'single_column': {'width': 3.35, 'height': 2.55, 'main_lw': 0.8, 'sec_lw': 0.4, 
            'scatter': 10, 'label_fs': 8, 'lp': 5, 'sec_fs': 6, 'tick': 0.4, 'grid_alpha': 0.2}, 
        'double_column': {'width': 6.7, 'height': 4.8, 'main_lw': 1.2, 'sec_lw': 0.6, 'scatter': 20, 
            'label_fs': 18, 'lp': 10, 'sec_fs': 14, 'tick': 0.5, 'grid_alpha':0.3}}
    
    cfg = plot_layout[layout]
    # --- Setting figure size:
    width, height = cfg['width'], cfg['height']
    # --- Setting plot element size:
    main_lw, sec_lw, scatter = cfg['main_lw'], cfg['sec_lw'], cfg['scatter']
    # --- Setting text font size:
    label_fs, sec_fs = cfg['label_fs'], cfg['sec_fs']
    # --- Setting aestetic parameters:
    lp, tick, grid_alpha = cfg['lp'], cfg['tick'], cfg['grid_alpha']

    # Setting plot paramters:
    with plt.rc_context({'font.family': 'serif', 'font.serif': ['Times New Roman'],
        'mathtext.fontset': 'stix'}):

        # --- Figure size: single-column ---
        fig, ax = plt.subplots(figsize=(width, height))

        if metric == 'mse':
            # --- Plot ---
            plt.semilogy(epochs, loss, label='Training MSE', color='#1B3A6F', linewidth=main_lw)
            plt.semilogy(epochs, val_loss, label='Validation MSE', color='#B22222', linestyle='--',
                linewidth=main_lw)
            plt.axvline(x=best_epoch, color='#000000', linestyle='--', alpha=0.4, linewidth=sec_lw)
            plt.scatter(best_epoch, best_val_loss, label=f'Best validation epoch ({best_epoch})',
                alpha=1.0, s=scatter, facecolors='#E63946', edgecolors='#000000', linewidth=0.3, 
                zorder=5)
        
            # --- Labels ---
            ax.set_xlabel('Epochs', fontsize=label_fs, fontname='Times New Roman', labelpad=lp)
            ax.set_ylabel('Mean squared error (MSE)', fontsize=label_fs, fontname='Times New Roman', 
                labelpad=lp)

        elif metric == 'mae':
            # --- Plot ---
            plt.semilogy(epochs, mae_loss, label='Training MAE', color='#1B3A6F', linewidth=main_lw)
            plt.semilogy(epochs, val_mae_loss, label='Validation MAE', color='#B22222', linestyle='--',
                linewidth=main_lw)
            plt.axvline(x=best_epoch, color='#000000', linestyle='--', alpha=0.4, linewidth=sec_lw)
            plt.scatter(best_epoch, best_mae_val_loss, label=f'Best validation epoch ({best_epoch})',
                alpha=1.0, s=scatter, facecolors='#E63946', edgecolors='#000000', linewidth=0.3, 
                zorder=5)
        
            # --- Labels ---
            ax.set_xlabel('Epochs', fontsize=label_fs, fontname='Times New Roman', labelpad=lp)
            ax.set_ylabel('Mean absolute error (MAE)', fontsize=label_fs, fontname='Times New Roman', 
                labelpad=lp)

        # --- Ticks ---
        for spine in ax.spines.values():
            spine.set_linewidth(0.5)
        ax.tick_params(axis='both', which='major', labelsize=sec_fs, width=tick, direction='in')

        # --- Grid ---
        ax.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=grid_alpha)

        # --- Limits ---
        ax.set_xlim([1, len(epochs)])

        # --- Legend ---
        legend = ax.legend(fontsize=sec_fs, fancybox=False, edgecolor='#000000', loc='upper right')
        if layout == 'single_column':
            legend.get_frame().set_linewidth(0.5)

        # --- Save plot ---
        plt.tight_layout(pad=0.6)
        plt.savefig(f'Potential-PINN/plots/train-mlp-pinn/train_{metric}_history_{layout}.png', dpi=600)
        plt.close()
        print(f'Saved: Potential-PINN/plots/train-mlp-pinn/train_{metric}_history_{layout}.png\n')

    return

def plot_test_predictions(y_test_raw: np.ndarray, y_pred_raw: np.ndarray, target: str, layout: str):
    """
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
            'scatter': 2, 'label_fs': 8, 'lp': 5, 'sec_fs': 6, 'tick': 0.4, 'grid_alpha': 0.2, 
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
    # --- Setting aestetic parameters:
    lp, tick, grid_alpha, m_scale = cfg['lp'], cfg['tick'], cfg['grid_alpha'], cfg['m_scale']

    # Setting plot paramters:
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
        textstr = (f"R² = {r2:.5f}\n"
                f"MAE = {mae:.5f}\n"
                f"MSE = {mse:.5f}\n"
                f"RMSE = {rmse:.5f}")
        props = dict(boxstyle='square, pad=0.35', facecolor='white', alpha=1.0, edgecolor='k', 
            linewidth=0.5)
        ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=sec_fs, verticalalignment='top', 
            bbox=props, linespacing=1.5, fontname='Times New Roman')

        # --- Save plot ---
        plt.tight_layout(pad=0.6)
        plt.savefig(f'Potential-PINN/plots/train-mlp-pinn/test_predictions_{target}_{layout}.png', dpi=600)
        plt.close()
        print(f'Saved: Potential-PINN/plots/train-mlp-pinn/test_predictions_{target}_{layout}.png\n')

    return

def plot_test_error_envelope_by_target(y_test_raw: np.ndarray, y_pred_raw: np.ndarray, target: str, 
        layout: str):
    """
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
            'scatter': 2, 'label_fs': 8, 'lp': 5, 'sec_fs': 6, 'tick': 0.4, 'grid_alpha': 0.2, 
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
    # --- Setting aestetic parameters:
    lp, tick, grid_alpha = cfg['lp'], cfg['tick'], cfg['grid_alpha']

    # Setting plot paramters:
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

        # --- Grid ---
        ax.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=grid_alpha)

        # --- Limits ---
        ax.set_xlim([min_axis, max_axis])
        ax.set_ylim(0, 0.075)

        # --- Legend ---
        legend = ax.legend(fontsize=sec_fs, fancybox=False, edgecolor='#000000', loc='upper right')
        if layout == 'single_column':
            legend.get_frame().set_linewidth(0.5)

        # --- Save plot ---
        plt.tight_layout(pad=0.6)
        plt.savefig(f'Potential-PINN/plots/train-mlp-pinn/test_error_envelope_{target}_by_{target}_{layout}.png', dpi=600)
        plt.close()
        print(f'Saved: Potential-PINN/plots/train-mlp-pinn/test_error_envelope_{target}_by_{target}_{layout}.png\n')

    return

def plot_test_target_error_envelope_by_input(X_test_raw: np.ndarray, y_test_raw: np.ndarray, 
    y_pred_raw: np.ndarray, target: str, input: str, layout: str):
    """
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
            'scatter': 2, 'label_fs': 8, 'lp': 5, 'sec_fs': 6, 'tick': 0.4, 'grid_alpha': 0.2, 
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
    # --- Setting aestetic parameters:
    lp, tick, grid_alpha = cfg['lp'], cfg['tick'], cfg['grid_alpha']

    # Setting plot paramters:
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
        for spine in ax.spines.values():
            spine.set_linewidth(0.5)
        ax.tick_params(axis='both', which='major', labelsize=sec_fs, width=tick, direction='in')

        # --- Grid ---
        ax.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=grid_alpha)

        # --- Limits ---
        ax.set_xlim([min_axis, max_axis])
        ax.set_ylim(0, 0.075)
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
        plt.savefig(f'Potential-PINN/plots/train-mlp-pinn/test_{target}_error_envelope_by_{input}_{layout}.png', dpi=600)
        plt.close()
        print(f'Saved: Potential-PINN/plots/train-mlp-pinn/test_{target}_error_envelope_by_{input}_{layout}.png\n')

    return

def plot_test_cp_error_envelope_by_chord(y_test_raw: np.ndarray, y_pred_raw: np.ndarray, surface: str, 
    layout: str):
    """
    """
    # --- Loads pressure measurement chordwise positions ---
    cp_mapping_path = 'Potential-ANN/utils/cp_mapping.csv'
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
            'scatter': 2, 'label_fs': 8, 'lp': 5, 'sec_fs': 6, 'tick': 0.4, 'grid_alpha': 0.2, 
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
    # --- Setting aestetic parameters:
    lp, tick, grid_alpha = cfg['lp'], cfg['tick'], cfg['grid_alpha']

    # Setting plot paramters:
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
        ax.xaxis.set_major_locator(matplotlib.ticker.MultipleLocator(0.2))

        # --- Grid ---
        ax.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=grid_alpha)

        # --- Limits ---
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 0.075)

        # --- Legend ---
        legend = ax.legend(fontsize=sec_fs, fancybox=False, edgecolor='#000000', loc='upper right')
        if layout == 'single_column':
            legend.get_frame().set_linewidth(0.5)

        # --- Save plot ---
        plt.tight_layout(pad=0.6)
        plt.savefig(f'Potential-PINN/plots/train-mlp-pinn/test_cp_error_envelope_{surface}_side_{layout}.png', dpi=600)
        plt.close()
        print(f'Saved: Potential-PINN/plots/train-mlp-pinn/test_cp_error_envelope_{surface}_side_{layout}.png\n')

    return

def plot_test_error_distribution_by_target(y_test_raw: np.ndarray, y_pred_raw: np.ndarray, target: str, 
    layout: str):
    """
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
            'scatter': 2, 'label_fs': 8, 'lp': 5, 'sec_fs': 6, 'tick': 0.4, 'grid_alpha': 0.2, 
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
    # --- Setting aestetic parameters:
    lp, tick, grid_alpha = cfg['lp'], cfg['tick'], cfg['grid_alpha']

    # Setting plot paramters:
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
        plt.savefig(f'Potential-PINN/plots/train-mlp-pinn/test_{target}_error_distribution_{layout}.png', dpi=600)
        plt.close()
        print(f'Saved: Potential-PINN/plots/train-mlp-pinn/test_{target}_error_distribution_{layout}.png\n')

    return

def plot_data_split_parameter_space(X_train_raw: np.ndarray, X_val_raw: np.ndarray, X_test_raw: np.ndarray, 
    layout: str):
    """
    """

    # --- Plot layout configurations ---
    plot_layout = {'single_column': {'width': 5.025, 'height': 3.06, 'main_lw': 0.8, 'sec_lw': 0.4, 
            'scatter': 3, 'label_fs': 8, 'lp': 5, 'sec_fs': 6, 'tick': 0.4, 'grid_alpha': 0.2, 
            'm_scale': 2}, 
        'double_column': {'width': 10.05, 'height': 5.76, 'main_lw': 1.2, 'sec_lw': 0.6, 'scatter': 12, 
            'label_fs': 18, 'lp': 10, 'sec_fs': 14, 'tick': 0.5, 'grid_alpha':0.3, 'm_scale': 3}}
    
    cfg = plot_layout[layout]
    # --- Setting figure size:
    width, height = cfg['width'], cfg['height']
    # --- Setting plot element size:
    main_lw, sec_lw, scatter = cfg['main_lw'], cfg['sec_lw'], cfg['scatter']
    # --- Setting text font size:
    label_fs, sec_fs = cfg['label_fs'], cfg['sec_fs']
    # --- Setting aestetic parameters:
    lp, tick, grid_alpha, m_scale = cfg['lp'], cfg['tick'], cfg['grid_alpha'], cfg['m_scale']

    # Setting plot paramters:
    with plt.rc_context({'font.family': 'serif', 'font.serif': ['Times New Roman'],
        'mathtext.fontset': 'stix'}):

        # --- Figure size: single-column ---
        fig, ax = plt.subplots(figsize=(width, height))

        # --- Plot ---
        ax.scatter(X_train_raw[:,0], X_train_raw[:,1], alpha=0.8, s=scatter,
            facecolor='#4C72B0', label='Training dataset')
        ax.scatter(X_val_raw[:,0], X_val_raw[:,1], alpha=0.8, s=scatter, marker='^',
            facecolor='#DD8452', label='Validation dataset')
        ax.scatter(X_test_raw[:,0], X_test_raw[:,1], alpha=0.8, s=scatter, marker='s',
            facecolor='#55A868', label='Test dataset')

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
            bbox_to_anchor=(0.5, -0.25), markerscale=m_scale, ncol=3)
        if layout == 'single_column':
            legend.get_frame().set_linewidth(0.5)

        # --- Save plot ---
        plt.tight_layout(pad=0.8)
        plt.savefig(f'Potential-PINN/plots/train-mlp-pinn/data_split_parameter_space_{layout}.png', dpi=600)
        plt.close()
        print(f'Saved: Potential-PINN/plots/train-mlp-pinn/data_split_parameter_space_{layout}.png\n')

    return

def save_report(model: keras, history, fit_losses: list, execution_time: float, X_test_raw: np.ndarray, 
    y_test_raw: np.ndarray, y_pred_raw: np.ndarray, X_train: np.ndarray, X_val: np.ndarray):
    """
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
PERFORMANCE REPORT - potential_mlp_pinn
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
R² Score: {r2_overall:.5f}
MAE:      {mae_overall:.6f}
MSE:      {mse_overall:.6f}
RMSE:     {rmse_overall:.6f}


4. CL TEST PERFORMANCE
----------------------------------
R² Score: {r2_cl:.5f}
MAE:      {mae_cl:.6f}
MSE:      {mse_cl:.6f}
RMSE:     {rmse_cl:.6f}


5. CP TEST PERFORMANCE
----------------------------------
R² Score: {r2_cp:.5f}
MAE:      {mae_cp:.6f}
MSE:      {mse_cp:.6f}
RMSE:     {rmse_cp:.6f}


6. FINAL LOSS RESULTS
----------------------------------
Best epoch: {np.argmin(history.history['val_loss']) + 1}
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
    with open('Potential-PINN/reports/performance-report-mlp-pinn.txt', "w", encoding="utf-8") as f:
        f.write(content)
    print(f"Saved: Potential-PINN/reports/performance-report-mlp-pinn.txt\n")

    return

def main(data_path: str, airfoil_data_path: str, lambda_physical: float):
    """
    """

    # --- Starts the timer for recording the execution time ---
    start_time = time.time()

    # --- Loads and split the dataset ---
    print("\nStarting the Multi-Layer Perceptron ANN training process...\n")
    print(f"Loading data from: {data_path}\n")
    data_train, data_val, data_test = load_and_split_data(data_path)
    print("Data loaded and split successfully.\n")

    # --- Loads the airfoil data ---
    print(f"Loading airfoil data from: {airfoil_data_path}\n") 
    airfoil_data = pd.read_csv(airfoil_data_path, sep=',', names=['x', 'y'])
    panel_dx = np.diff(airfoil_data['x'].values).astype(np.float32)
    panel_dy = np.diff(airfoil_data['y'].values).astype(np.float32)
    print("Airfoil data loaded successfully.\n")

    # --- Preprocessing the data ---
    print("Preprocessing data...\n")
    X_train, X_val, X_test, y_train, y_val, y_test, X_scaler, y_scaler = preprocess_data(data_train, data_val, data_test)
    print(f"Input Shape: {X_train.shape}, Output Shape: {y_train.shape}.\n")

    # --- Builds the model ---
    print("Building model architecture...\n")
    model = build_model(input_dim=X_train.shape[1], output_dim=y_train.shape[1]-1, panel_dx=panel_dx, 
        panel_dy=panel_dy, lambda_physical=lambda_physical, y_scaler=y_scaler)

    # --- Trains the model ---
    print("Starting training...\n")
    callbacks = [keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, min_delta=1e-6,restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6, 
            cooldown=5),
        keras.callbacks.ModelCheckpoint('Potential-ANN/trained-models/mlp-ann.keras', monitor='val_loss', 
            save_best_only=True)]
    history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=500, callbacks=callbacks, 
        batch_size=128, verbose=0)
    print("Saved: Potential-PINN/trained-models/mlp-pinn.keras\n")

    # --- Evaluates the model ---
    print("Evaluating the model...\n")
    train_loss, train_mae = model.evaluate(X_train, y_train, verbose=0)
    val_loss, val_mae = model.evaluate(X_val, y_val, verbose=0)
    fit_losses = [train_loss, train_mae, val_loss, val_mae]
    y_test_raw, y_pred_raw = predict_test(model, X_test, y_test, y_scaler)

    # --- Reverses the input scaling transformation ---
    X_train_raw = X_scaler.inverse_transform(X_train)
    X_val_raw = X_scaler.inverse_transform(X_val)
    X_test_raw = X_scaler.inverse_transform(X_test)

    # --- Finishes the timer ---
    end_time = time.time()
    execution_time = end_time - start_time

    # --- Saves the parameter space distribution ---
    plot_data_split_parameter_space(X_train_raw, X_val_raw, X_test_raw, layout='single_column')
    plot_data_split_parameter_space(X_train_raw, X_val_raw, X_test_raw, layout='double_column')

    # --- Saves training history plots ---
    plot_training_history(history, metric='mse', layout='single_column')
    plot_training_history(history, metric='mae', layout='single_column')

    # --- Saves test predictions plots ---
    plot_test_predictions(y_test_raw, y_pred_raw, target='cl', layout='single_column')
    plot_test_predictions(y_test_raw, y_pred_raw, target='cp', layout='single_column')
    plot_test_predictions(y_test_raw, y_pred_raw, target='overall', layout='single_column')

    # --- Saves the test error envelope plots ---
    plot_test_error_envelope_by_target(y_test_raw, y_pred_raw, target='cl', layout='single_column')
    plot_test_error_envelope_by_target(y_test_raw, y_pred_raw, target='cp', layout='single_column')
    plot_test_target_error_envelope_by_input(X_test_raw, y_test_raw, y_pred_raw, target = 'cl', input='re', 
        layout='single_column')
    plot_test_target_error_envelope_by_input(X_test_raw, y_test_raw, y_pred_raw, target = 'cl', input='aoa', 
        layout='single_column')
    plot_test_target_error_envelope_by_input(X_test_raw, y_test_raw, y_pred_raw, target = 'cl', input='yb', 
        layout='single_column')
    plot_test_target_error_envelope_by_input(X_test_raw, y_test_raw, y_pred_raw, target = 'cp', input='re', 
        layout='single_column')
    plot_test_target_error_envelope_by_input(X_test_raw, y_test_raw, y_pred_raw, target = 'cp', input='aoa', 
        layout='single_column')
    plot_test_target_error_envelope_by_input(X_test_raw, y_test_raw, y_pred_raw, target = 'cp', input='yb', 
        layout='single_column')
    plot_test_cp_error_envelope_by_chord(y_test_raw, y_pred_raw, surface='suction', layout='single_column')
    plot_test_cp_error_envelope_by_chord(y_test_raw, y_pred_raw, surface='pressure', layout='single_column')

    # --- Saves the test error distribution plots ---
    plot_test_error_distribution_by_target(y_test_raw, y_pred_raw, target='cl', layout='single_column')
    plot_test_error_distribution_by_target(y_test_raw, y_pred_raw, target='cp', layout='single_column')

    # --- Saves the text performance report ---
    save_report(model, history, fit_losses, execution_time, X_test_raw, y_test_raw, y_pred_raw, X_train, X_val)

    print("Process completed successfully.\n")

    return

if __name__ == '__main__':
    main(data_path, airfoil_data_path, lambda_physical)