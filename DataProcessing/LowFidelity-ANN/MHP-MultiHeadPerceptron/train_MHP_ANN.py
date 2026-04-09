
"""
========================================================================================================
Module: train_MHP_ANN
Author: Caio Dias Filho
Creation date: 2026-03-19
Last modification: 2026-03-27
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
data_path = 'LowFidelity-ANN/utils/LowFidelity-PressureDistributionData.csv'


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
print(f'\nGlobal seed set to: {SEED}')

def load_and_split_data(filepath: str):
    """
    
    """

    # --- Loading the dataset ---
    data = (pd.read_csv(filepath, sep=';')).copy()

    # --- Create identifier for flow case (assuming 80 sections per case)
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
    Cl_cols = data_train.columns[3]
    Cp_cols = data_train.columns[4:205]

    # --- Splitting the raw features data into training, validation, and testing sets ---
    X_train_raw = data_train[X_cols].to_numpy()
    X_val_raw = data_val[X_cols].to_numpy()
    X_test_raw = data_test[X_cols].to_numpy()

    # --- Splitting the raw lift coefficient data into training, validation, and testing sets ---
    Cl_train_raw = data_train[Cl_cols].to_numpy().reshape(-1, 1)
    Cl_val_raw = data_val[Cl_cols].to_numpy().reshape(-1, 1)
    Cl_test_raw = data_test[Cl_cols].to_numpy().reshape(-1, 1)

    # --- Splitting the raw pressure coefficient data into training, validation, and testing sets ---
    Cp_train_raw = data_train[Cp_cols].to_numpy()
    Cp_val_raw = data_val[Cp_cols].to_numpy()
    Cp_test_raw = data_test[Cp_cols].to_numpy()

    # --- Scaling the features data ---
    X_scaler = skl.preprocessing.StandardScaler()
    X_train = X_scaler.fit_transform(X_train_raw)
    X_val = X_scaler.transform(X_val_raw)
    X_test = X_scaler.transform(X_test_raw)

    # --- Scaling the lift coefficient data ---
    Cl_scaler = skl.preprocessing.StandardScaler()
    Cl_train = Cl_scaler.fit_transform(Cl_train_raw)
    Cl_val = Cl_scaler.transform(Cl_val_raw)
    Cl_test = Cl_scaler.transform(Cl_test_raw)

    # --- Scaling the pressure coefficient data ---
    Cp_scaler = skl.preprocessing.StandardScaler()
    Cp_train = Cp_scaler.fit_transform(Cp_train_raw)
    Cp_val = Cp_scaler.transform(Cp_val_raw)
    Cp_test = Cp_scaler.transform(Cp_test_raw)

    # --- Saving the scalers ---
    joblib.dump(X_scaler, 'LowFidelity-ANN/MHP-MultiHeadPerceptron/scalers/mhp_ann_X_scaler.pkl')
    joblib.dump(Cl_scaler, 'LowFidelity-ANN/MHP-MultiHeadPerceptron/scalers/mhp_ann_Cl_scaler.pkl')
    joblib.dump(Cp_scaler, 'LowFidelity-ANN/MHP-MultiHeadPerceptron/scalers/mhp_ann_Cp_scaler.pkl')

    return X_train, X_val, X_test, Cl_train, Cl_val, Cl_test, Cp_train, Cp_val, Cp_test, X_scaler, Cl_scaler, Cp_scaler

def build_model(input_dim: int, cl_output_dim: int, cp_output_dim: int):
    """
    
    """

    # --- Defining the model input layer ---
    inputs = keras.layers.Input(shape=(input_dim,), name='Input_Layer')

    # --- Defining the model hidden layers ---
    x = keras.layers.Dense(units=224, activation='gelu', kernel_initializer='glorot_uniform', 
        name='Hidden_Layer_1')(inputs)
    x = keras.layers.Dense(units=224, activation='gelu', kernel_initializer='glorot_uniform', 
        name='Hidden_Layer_2')(x)
    x = keras.layers.Dense(units=192, activation='gelu', kernel_initializer='glorot_uniform', 
        name='Hidden_Layer_3')(x)
    x = keras.layers.Dense(units=256, activation='gelu', kernel_initializer='glorot_uniform', 
        name='Hidden_Layer_4')(x)
    
    # --- Defining the model lift coefficient head ---
    h1 = keras.layers.Dense(units=160, activation='relu', kernel_initializer='glorot_uniform',
        name='Cl_Layer_1')(x)
    h1 = keras.layers.Dense(units=224, activation='relu', kernel_initializer='glorot_uniform',
        name='Cl_Layer_2')(h1)
    h1 = keras.layers.Dense(units=128, activation='relu', kernel_initializer='glorot_uniform',
        name='Cl_Layer_3')(h1)
    out_cl = keras.layers.Dense(units=cl_output_dim, activation='linear', kernel_initializer='glorot_uniform',
        name='Cl_Output')(h1)
    
    # --- Defining the model pressure coefficient head ---
    h2 = keras.layers.Dense(units=224, activation='swish', kernel_initializer='glorot_uniform',
        name='Cp_Layer_1')(x)
    out_cp = keras.layers.Dense(units=cp_output_dim, activation='linear', kernel_initializer='glorot_uniform',
        name='Cp_Output')(h2)
    
    # --- Defining the model ---
    model = keras.Model(inputs=inputs, outputs=[out_cl, out_cp], name='low_fidelity_mhp_ann')

    # --- Compiling the model ---
    model.compile(loss={'Cl_Output': 'mse', 'Cp_Output': 'mse'}, optimizer=keras.optimizers.Nadam(learning_rate=0.001), 
        metrics={'Cl_Output': ['mae'], 'Cp_Output': ['mae']}, loss_weights={'Cl_Output': 0.2, 'Cp_Output': 0.2})
    
    return model
    
def predict_test(model: keras.Model, X_test: np.ndarray, Cl_test: np.ndarray, Cp_test: np.ndarray, 
    Cl_scaler:skl.preprocessing.StandardScaler, Cp_scaler:skl.preprocessing.StandardScaler):
    """
    
    """

    # --- Predicting on the test set ---
    y_pred = model.predict(X_test, verbose=0)

    # --- Separating the outputs ---
    Cl_pred = y_pred[0]
    Cp_pred = y_pred[1]

    # --- Reverses the scaling ---
    Cl_test_raw = Cl_scaler.inverse_transform(Cl_test)
    Cl_pred_raw = Cl_scaler.inverse_transform(Cl_pred)
    Cp_test_raw = Cp_scaler.inverse_transform(Cp_test)
    Cp_pred_raw = Cp_scaler.inverse_transform(Cp_pred)

    return Cl_test_raw, Cl_pred_raw, Cp_test_raw, Cp_pred_raw

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
        plt.savefig(f'LowFidelity-ANN/MHP-MultiHeadPerceptron/plots/data_split_parameter_space_{layout}.png', dpi=600)
        plt.close()
        print(f'Saved: LowFidelity-ANN/MHP-MultiHeadPerceptron/plots/data_split_parameter_space_{layout}.png\n')

    return

def plot_training_history(history: dict, output: str, metric: str, layout: str):
    """
    
    """

    # --- Load data for complete MSE metric plot ---
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(1, len(loss) + 1)
    best_epoch_loss = np.argmin(val_loss) + 1
    best_val_loss = val_loss[best_epoch_loss - 1]

    # --- Load data for Cl MSE metric plot ---
    cl_loss = history.history['Cl_Output_loss']
    cl_val_loss = history.history['val_Cl_Output_loss']
    cl_best_epoch = np.argmin(cl_val_loss) + 1
    best_cl_val_loss = cl_val_loss[cl_best_epoch - 1]

    # --- Load data for Cp MSE metric plot ---
    cp_loss = history.history['Cp_Output_loss']
    cp_val_loss = history.history['val_Cp_Output_loss']
    cp_best_epoch = np.argmin(cp_val_loss) + 1
    best_cp_val_loss = cp_val_loss[cp_best_epoch - 1]

    # --- Load data for Cl MAE metric plot ---
    cl_mae = history.history['Cl_Output_mae']
    cl_val_mae = history.history['val_Cl_Output_mae']
    cl_best_epoch_mae = np.argmin(cl_val_mae) + 1
    best_cl_val_mae = cl_val_mae[cl_best_epoch_mae - 1]

    # --- Load data for Cp MAE metric plot ---
    cp_mae = history.history['Cp_Output_mae']
    cp_val_mae = history.history['val_Cp_Output_mae']
    cp_best_epoch_mae = np.argmin(cp_val_mae) + 1
    best_cp_val_mae = cp_val_mae[cp_best_epoch_mae - 1]

    # --- Plot layout configuration ---
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

        if output == 'overall' and metric == 'mse':
            # --- Plot ---
            plt.semilogy(epochs, loss, label='Training', color='#1B3A6F', linewidth=main_lw)
            plt.semilogy(epochs, val_loss, label='Validation', color='#B22222', linestyle='--',
                linewidth=main_lw)
            plt.axvline(x=best_epoch_loss, color='#B22222', linestyle='--', alpha=0.4, linewidth=sec_lw)
            plt.scatter(best_epoch_loss, best_val_loss, label=f'Best validation loss (epoch {best_epoch_loss})',
                alpha=1.0, s=scatter, facecolors='#B22222', edgecolors='#000000', linewidth=0.3,
                zorder=5)

            # --- Labels ---
            ax.set_xlabel('Epochs', fontsize=label_fs, fontname='Times New Roman', labelpad=lp)
            ax.set_ylabel('Total Loss (MSE)', fontsize=label_fs, fontname='Times New Roman', labelpad=lp)

        elif output == 'cl' and metric == 'mse':
            # --- Plot ---
            plt.semilogy(epochs, cl_loss, label='Training', color='#1B3A6F', linewidth=main_lw)
            plt.semilogy(epochs, cl_val_loss, label='Validation', color='#B22222', linestyle='--',
                linewidth=main_lw)
            plt.axvline(x=cl_best_epoch, color='#B22222', linestyle='--', alpha=0.4, linewidth=sec_lw)
            plt.scatter(cl_best_epoch, best_cl_val_loss, label=f'Best validation loss (epoch {cl_best_epoch})',
                alpha=1.0, s=scatter, facecolors='#B22222', edgecolors='#000000', linewidth=0.3,
                zorder=5)

            # --- Labels ---
            ax.set_xlabel('Epochs', fontsize=label_fs, fontname='Times New Roman', labelpad=lp)
            ax.set_ylabel(r'MSE (Lift Coefficient, $C_l$)', fontsize=label_fs, fontname='Times New Roman', labelpad=lp)

        elif output == 'cp' and metric == 'mse':
            # --- Plot ---
            plt.semilogy(epochs, cp_loss, label='Training', color='#1B3A6F', linewidth=main_lw)
            plt.semilogy(epochs, cp_val_loss, label='Validation', color='#B22222', linestyle='--',
                linewidth=main_lw)
            plt.axvline(x=cp_best_epoch, color='#B22222', linestyle='--', alpha=0.4, linewidth=sec_lw)
            plt.scatter(cp_best_epoch, best_cp_val_loss, label=f'Best validation loss (epoch {cp_best_epoch})',
                alpha=1.0, s=scatter, facecolors='#B22222', edgecolors='#000000', linewidth=0.3,
                zorder=5)

            # --- Labels ---
            ax.set_xlabel('Epochs', fontsize=label_fs, fontname='Times New Roman', labelpad=lp)
            ax.set_ylabel(r'MSE (Pressure Coefficient, $C_p$)', fontsize=label_fs, fontname='Times New Roman', labelpad=lp)

        elif output == 'cl' and metric == 'mae':
            # --- Plot ---
            plt.semilogy(epochs, cl_mae, label='Training', color='#1B3A6F', linewidth=main_lw)
            plt.semilogy(epochs, cl_val_mae, label='Validation', color='#B22222', linestyle='--',
                linewidth=main_lw)
            plt.axvline(x=cl_best_epoch_mae, color='#B22222', linestyle='--', alpha=0.4, linewidth=sec_lw)
            plt.scatter(cl_best_epoch_mae, best_cl_val_mae, label=f'Best validation loss (epoch {cl_best_epoch_mae})',
                alpha=1.0, s=scatter, facecolors='#B22222', edgecolors='#000000', linewidth=0.3,
                zorder=5)

            # --- Labels ---
            ax.set_xlabel('Epochs', fontsize=label_fs, fontname='Times New Roman', labelpad=lp)
            ax.set_ylabel(r'MAE (Lift Coefficient, $C_l$)', fontsize=label_fs, fontname='Times New Roman', labelpad=lp)

        elif output == 'cp' and metric == 'mae':
            # --- Plot ---
            plt.semilogy(epochs, cp_mae, label='Training', color='#1B3A6F', linewidth=main_lw)
            plt.semilogy(epochs, cp_val_mae, label='Validation', color='#B22222', linestyle='--',
                linewidth=main_lw)
            plt.axvline(x=cp_best_epoch_mae, color='#B22222', linestyle='--', alpha=0.4, linewidth=sec_lw)
            plt.scatter(cp_best_epoch_mae, best_cp_val_mae, label=f'Best validation loss (epoch {cp_best_epoch_mae})',
                alpha=1.0, s=scatter, facecolors='#B22222', edgecolors='#000000', linewidth=0.3,
                zorder=5)

            # --- Labels ---
            ax.set_xlabel('Epochs', fontsize=label_fs, fontname='Times New Roman', labelpad=lp)
            ax.set_ylabel(r'MAE (Pressure Coefficient, $C_p$)', fontsize=label_fs, fontname='Times New Roman', labelpad=lp)

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
        plt.savefig(f'LowFidelity-ANN/MHP-MultiHeadPerceptron/plots/train_{output}_{metric}_history_{layout}.png', dpi=600)
        plt.close()
        print(f'Saved: LowFidelity-ANN/MHP-MultiHeadPerceptron/plots/train_{output}_{metric}_history_{layout}.png\n')

    return

def plot_test_predictions(Cl_test_raw: np.ndarray, Cl_pred_raw: np.ndarray, Cp_test_raw: np.ndarray, 
    Cp_pred_raw: np.ndarray, target: str, layout: str):
    """
    
    """

    # --- Merges the outputs ---
    y_test_raw = np.concatenate((Cl_test_raw, Cp_test_raw), axis=1)
    y_pred_raw = np.concatenate((Cl_pred_raw, Cp_pred_raw), axis=1)

    if target == 'overall':
        r2 = skl.metrics.r2_score(y_test_raw, y_pred_raw)
        mae = skl.metrics.mean_absolute_error(y_test_raw, y_pred_raw)
        mse = skl.metrics.mean_squared_error(y_test_raw, y_pred_raw)
        rmse = np.sqrt(mse)
        min_axis = min(np.min(y_test_raw), np.min(y_pred_raw))
        max_axis = max(np.max(y_test_raw), np.max(y_pred_raw))
    
    elif target == 'cl':
        r2 = skl.metrics.r2_score(Cl_test_raw, Cl_pred_raw)
        mae = skl.metrics.mean_absolute_error(Cl_test_raw, Cl_pred_raw)
        mse = skl.metrics.mean_squared_error(Cl_test_raw, Cl_pred_raw)
        rmse = np.sqrt(mse)
        min_axis = min(np.min(Cl_test_raw), np.min(Cl_pred_raw))
        max_axis = max(np.max(Cl_test_raw), np.max(Cl_pred_raw))

    elif target == 'cp':
        r2 = skl.metrics.r2_score(Cp_test_raw, Cp_pred_raw)
        mae = skl.metrics.mean_absolute_error(Cp_test_raw, Cp_pred_raw)
        mse = skl.metrics.mean_squared_error(Cp_test_raw, Cp_pred_raw)
        rmse = np.sqrt(mse)
        min_axis = min(np.min(Cp_test_raw), np.min(Cp_pred_raw))
        max_axis = max(np.max(Cp_test_raw), np.max(Cp_pred_raw))

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
            plt.scatter(Cl_test_raw, Cl_pred_raw, alpha=0.8, s=scatter, facecolors='#D55E00', 
                edgecolors='#000000', linewidth=0.1, label=r'Lift coefficient ($C_l$)')
            plt.scatter(Cp_test_raw, Cp_pred_raw, alpha=0.8, s=scatter, facecolors='#0072B2', 
                edgecolors='#000000', linewidth=0.1, label=r'Pressure coefficient ($C_p$)')
            plt.plot([min_axis, max_axis], [min_axis, max_axis], color='#B22222', linestyle='--', 
                linewidth=main_lw, zorder=5)
            
            # --- Legend ---
            legend = ax.legend(fontsize=sec_fs, fancybox=False, edgecolor='#000000', loc='lower right', 
                markerscale=m_scale)
            legend.get_frame().set_linewidth(0.5)
    
        elif target == 'cl':
            # --- Plot ---
            plt.scatter(Cl_test_raw, Cl_pred_raw, alpha=0.8, s=scatter, facecolors='#D55E00', 
                edgecolors='#000000', linewidth=0.1, label=r'Lift coefficient ($C_l$)')
            plt.plot([min_axis, max_axis], [min_axis, max_axis], color='#B22222', linestyle='--', 
                linewidth=main_lw, zorder=5)
        
        elif target == 'cp':
            # --- Plot ---
            plt.scatter(Cp_test_raw, Cp_pred_raw, alpha=0.8, s=scatter, facecolors='#0072B2', 
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
        plt.savefig(f'LowFidelity-ANN/MHP-MultiHeadPerceptron/plots/test_predictions_{target}_{layout}.png', dpi=600)
        plt.close()
        print(f'Saved: LowFidelity-ANN/MHP-MultiHeadPerceptron/plots/test_predictions_{target}_{layout}.png\n')

    return

def plot_test_error_envelope_by_target(Cl_test_raw: np.ndarray, Cl_pred_raw: np.ndarray, Cp_test_raw: np.ndarray, 
    Cp_pred_raw: np.ndarray, target: str, layout: str):
    """
    
    """

    if target == 'cl':
        test_flat = Cl_test_raw.flatten()
        pred_flat = Cl_pred_raw.flatten()
        fill_color = '#D55E00'

    elif target == 'cp':
        test_flat = Cp_test_raw.flatten()
        pred_flat = Cp_pred_raw.flatten()
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
        plt.savefig(f'LowFidelity-ANN/MHP-MultiHeadPerceptron/plots/test_error_envelope_{target}_by_{target}_{layout}.png', dpi=600)
        plt.close()
        print(f'Saved: LowFidelity-ANN/MHP-MultiHeadPerceptron/plots/test_error_envelope_{target}_by_{target}_{layout}.png\n')

    return

def plot_test_target_error_envelope_by_input(X_test_raw: np.ndarray, Cl_test_raw: np.ndarray, 
    Cp_test_raw: np.ndarray, Cl_pred_raw: np.ndarray, Cp_pred_raw: np.ndarray, target: str, input: str, 
    layout: str):
    """

    """

    if target == 'cl':
        abs_error = np.abs(Cl_test_raw - Cl_pred_raw)
        abs_error = abs_error.flatten()
        color = '#D55E00'
    elif target == 'cp':
        abs_error = np.mean(np.abs(Cp_test_raw - Cp_pred_raw), axis=1)
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
        plt.savefig(f'LowFidelity-ANN/MHP-MultiHeadPerceptron/plots/test_{target}_error_envelope_by_{input}_{layout}.png', dpi=600)
        plt.close()
        print(f'Saved: LowFidelity-ANN/MHP-MultiHeadPerceptron/plots/test_{target}_error_envelope_by_{input}_{layout}.png\n')

    return

def plot_test_cp_error_envelope_by_chord(Cp_test_raw: np.ndarray, Cp_pred_raw: np.ndarray, surface: str, 
    layout: str):
    """

    """
    # --- Loads pressure measurement chordwise positions ---
    cp_mapping_path = 'LowFidelity-ANN/utils/LowFidelity-CpMapping.csv'
    cp_position = pd.read_csv(cp_mapping_path, sep=';', usecols=['X_coord']).to_numpy().ravel()

    # --- Splits the pressure data ---
    cp_test_suction = Cp_test_raw[:, :104]
    cp_test_pressure = Cp_test_raw[:, 103:]
    cp_pred_suction = Cp_pred_raw[:, :104]
    cp_pred_pressure = Cp_pred_raw[:, 103:]

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
        plt.savefig(f'LowFidelity-ANN/MHP-MultiHeadPerceptron/plots/test_cp_error_envelope_{surface}_side_{layout}.png', dpi=600)
        plt.close()
        print(f'Saved: LowFidelity-ANN/MHP-MultiHeadPerceptron/plots/test_cp_error_envelope_{surface}_side_{layout}.png\n')

    return

def plot_test_error_distribution_by_target(Cl_test_raw: np.ndarray, Cl_pred_raw: np.ndarray, 
    Cp_test_raw: np.ndarray, Cp_pred_raw: np.ndarray, target: str, layout: str):
    """
    """

    if target == 'cl':
        abs_error = np.abs(Cl_test_raw - Cl_pred_raw)
        abs_error = abs_error.flatten()
        color = '#D55E00'
    elif target == 'cp':
        abs_error = np.mean(np.abs(Cp_test_raw - Cp_pred_raw), axis=1)
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
        plt.savefig(f'LowFidelity-ANN/MHP-MultiHeadPerceptron/plots/test_{target}_error_distribution_{layout}.png', dpi=600)
        plt.close()
        print(f'Saved: LowFidelity-ANN/MHP-MultiHeadPerceptron/plots/test_{target}_error_distribution_{layout}.png\n')

    return

def save_report(model: keras, history: dict, fit_losses: list, execution_time: float, X_test_raw: np.ndarray, 
    Cl_test_raw: np.ndarray, Cl_pred_raw: np.ndarray, Cp_test_raw: np.ndarray, Cp_pred_raw: np.ndarray, 
    X_train: np.ndarray, X_val: np.ndarray):
    """
    """

    # --- Merges the outputs ---
    y_test_raw = np.concatenate((Cl_test_raw, Cp_test_raw), axis=1)
    y_pred_raw = np.concatenate((Cl_pred_raw, Cp_pred_raw), axis=1)
    # --- Splits the targets data ---

    r2_overall = skl.metrics.r2_score(y_test_raw, y_pred_raw)
    mae_overall = skl.metrics.mean_absolute_error(y_test_raw, y_pred_raw)
    mse_overall = skl.metrics.mean_squared_error(y_test_raw, y_pred_raw)
    rmse_overall = np.sqrt(mse_overall)

    r2_cl = skl.metrics.r2_score(Cl_test_raw, Cl_pred_raw)
    mae_cl = skl.metrics.mean_absolute_error(Cl_test_raw, Cl_pred_raw)
    mse_cl = skl.metrics.mean_squared_error(Cl_test_raw, Cl_pred_raw)
    rmse_cl = np.sqrt(mse_cl)

    r2_cp = skl.metrics.r2_score(Cp_test_raw, Cp_pred_raw)
    mae_cp = skl.metrics.mean_absolute_error(Cp_test_raw, Cp_pred_raw)
    mse_cp = skl.metrics.mean_squared_error(Cp_test_raw, Cp_pred_raw)
    rmse_cp = np.sqrt(mse_cp)

    # --- Generate the performance report ---
    stringlist = []
    model.summary(print_fn=lambda x: stringlist.append(x))
    model_structure = "\n".join(stringlist)
    
    content = f"""
PERFORMANCE REPORT - low_fidelity_mhp_ann
==========================================================
Date: {time.strftime("%Y-%m-%d %H:%M:%S")}
Execution Time: {execution_time:.2f} seconds


1. CONFIGURATION
----------------------------------
Epochs: {len(history.history['loss'])}
Input Shape: {X_test_raw.shape}
Total Outputs: {y_test_raw.shape}
Cl Outputs: {Cl_test_raw.shape}
Cp Outputs: {Cp_test_raw.shape}


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
Final Total Loss (Training): {fit_losses[0]:.6f}
Final Total MSE Loss (Validation): {fit_losses[5]:.6f}

Final Cl MSE Loss (Training): {fit_losses[1]:.6f}
Final Cl MSE Loss (Validation): {fit_losses[6]:.6f}
Final Cl MAE (Training): {fit_losses[3]:.6f}
Final Cl MAE (Validation): {fit_losses[8]:.6f}

Final Cp MSE Loss (Training): {fit_losses[2]:.6f}
Final Cp MSE Loss (Validation): {fit_losses[7]:.6f}
Final Cp MAE (Training): {fit_losses[4]:.6f}
Final Cp MAE (Validation): {fit_losses[9]:.6f}


7. MODEL ARCHITECTURE
----------------------------------
{model_structure}
==========================================================
"""
    # Saving the report:
    with open('LowFidelity-ANN/MHP-MultiHeadPerceptron/performance-report-mhp-ann.txt', "w", encoding="utf-8") as f:
        f.write(content)
    print(f"Saved: LowFidelity-ANN/MHP-MultiHeadPerceptron/performance-report-mhp-ann.txt\n")

    return

def main(data_path: str):

    # --- Starts the timer for recording the execution time ---
    start_time = time.time()

    # --- Loads and split the dataset ---
    print("\nStarting the Multi-Head Perceptron ANN training process...\n")
    print(f"Loading data from: {data_path}\n")
    data_train, data_val, data_test = load_and_split_data(data_path)
    print("Data loaded and split successfully.\n")

    # --- Preprocessing the data ---
    print("Preprocessing data...\n")
    X_train, X_val, X_test, Cl_train, Cl_val, Cl_test, Cp_train, Cp_val, Cp_test, X_scaler, Cl_scaler, Cp_scaler = preprocess_data(data_train, data_val, data_test)
    print(f"Input Shape: {X_train.shape[1]}\nLift Coefficient Output Shape: {Cl_train.shape[1]}\nPressure Coefficient Output Shape: {Cp_train.shape[1]}\n")

    # --- Builds the model ---
    print("Building model architecture...\n")
    model = build_model(X_train.shape[1], Cl_train.shape[1], Cp_train.shape[1])

    # --- Trains the model ---
    print("Starting training...\n")
    callbacks = [keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, min_delta=1e-6, restore_best_weights=True),
                 keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6, cooldown=5, min_delta=1e-5),
                 keras.callbacks.ModelCheckpoint('LowFidelity-ANN/MHP-MultiHeadPerceptron/MHP-ANN.keras', monitor='val_loss', save_best_only=True)]
    history = model.fit(X_train, {'Cl_Output':Cl_train, 'Cp_Output':Cp_train}, validation_data=(X_val, {'Cl_Output':Cl_val, 'Cp_Output':Cp_val}),
        epochs=500, callbacks=callbacks, batch_size=128, verbose=0)
    print("Saved: LowFidelity-ANN/MHP-MultiHeadPerceptron/MHP-ANN.keras\n")

    # --- Train evaluation ---
    print("Evaluating the model...\n")
    train_evaluation = model.evaluate(X_train, {'Cl_Output':Cl_train, 'Cp_Output':Cp_train}, verbose=0, return_dict=True)
    train_loss = train_evaluation['loss']
    Cl_train_loss = train_evaluation['Cl_Output_loss']
    Cp_train_loss = train_evaluation['Cp_Output_loss']
    Cl_train_mae = train_evaluation['Cl_Output_mae']
    Cp_train_mae = train_evaluation['Cp_Output_mae']

    # --- Validation evaluation ---
    val_evaluation = model.evaluate(X_val, {'Cl_Output':Cl_val, 'Cp_Output':Cp_val}, verbose=0, return_dict=True)
    val_loss = val_evaluation['loss']
    Cl_val_loss = val_evaluation['Cl_Output_loss']
    Cp_val_loss = val_evaluation['Cp_Output_loss']
    Cl_val_mae = val_evaluation['Cl_Output_mae']
    Cp_val_mae = val_evaluation['Cp_Output_mae']

    # --- Assembling the losses ---
    fit_losses = [train_loss, Cl_train_loss, Cp_train_loss, Cl_train_mae, Cp_train_mae,
                  val_loss, Cl_val_loss, Cp_val_loss, Cl_val_mae, Cp_val_mae]

    # --- Predicting on the test set ---
    print("Predicting on the test set...\n")
    Cl_test_raw, Cl_pred_raw, Cp_test_raw, Cp_pred_raw = predict_test(model, X_test, Cl_test, Cp_test, Cl_scaler, Cp_scaler)

    # --- Reverses the input scaling transformation ---
    X_train_raw = X_scaler.inverse_transform(X_train)
    X_val_raw = X_scaler.inverse_transform(X_val)
    X_test_raw = X_scaler.inverse_transform(X_test)

    # --- Finishes the timer ---
    end_time = time.time()
    execution_time = end_time - start_time

    # --- Saves the parameter space distribution ---
    plot_data_split_parameter_space(X_train_raw, X_val_raw, X_test_raw, layout='single_column')

    # --- Saves training history plots ---
    plot_training_history(history, output='overall', metric='mse', layout='single_column')
    plot_training_history(history, output='cl', metric='mse', layout='single_column')
    plot_training_history(history, output='cl', metric='mae', layout='single_column')
    plot_training_history(history, output='cp', metric='mse', layout='single_column')
    plot_training_history(history, output='cp', metric='mae', layout='single_column')

    # --- Saves test prediction plots ---
    plot_test_predictions(Cl_test_raw, Cl_pred_raw, Cp_test_raw, Cp_pred_raw, target='overall', layout='single_column')
    plot_test_predictions(Cl_test_raw, Cl_pred_raw, Cp_test_raw, Cp_pred_raw, target='cl', layout='single_column')
    plot_test_predictions(Cl_test_raw, Cl_pred_raw, Cp_test_raw, Cp_pred_raw, target='cp', layout='single_column')

    # --- Saves the test error envelope plots by target ---
    plot_test_error_envelope_by_target(Cl_test_raw, Cl_pred_raw, Cp_test_raw, Cp_pred_raw, target='cl', layout='single_column')
    plot_test_error_envelope_by_target(Cl_test_raw, Cl_pred_raw, Cp_test_raw, Cp_pred_raw, target='cp', layout='single_column')

    # --- Saves the test error envelop plots by input ---
    plot_test_target_error_envelope_by_input(X_test_raw, Cl_test_raw, Cp_test_raw, Cl_pred_raw, Cp_pred_raw,
        target='cl', input='re', layout='single_column')
    plot_test_target_error_envelope_by_input(X_test_raw, Cl_test_raw, Cp_test_raw, Cl_pred_raw, Cp_pred_raw,
        target='cl', input='aoa', layout='single_column')
    plot_test_target_error_envelope_by_input(X_test_raw, Cl_test_raw, Cp_test_raw, Cl_pred_raw, Cp_pred_raw,
        target='cl', input='yb', layout='single_column')
    plot_test_target_error_envelope_by_input(X_test_raw, Cl_test_raw, Cp_test_raw, Cl_pred_raw, Cp_pred_raw,
        target='cp', input='re', layout='single_column')
    plot_test_target_error_envelope_by_input(X_test_raw, Cl_test_raw, Cp_test_raw, Cl_pred_raw, Cp_pred_raw,
        target='cp', input='aoa', layout='single_column')
    plot_test_target_error_envelope_by_input(X_test_raw, Cl_test_raw, Cp_test_raw, Cl_pred_raw, Cp_pred_raw,
        target='cp', input='yb', layout='single_column')
    
    # --- Saves the test error envelope plots by chord ---
    plot_test_cp_error_envelope_by_chord(Cp_test_raw, Cp_pred_raw, surface='suction', layout='single_column')
    plot_test_cp_error_envelope_by_chord(Cp_test_raw, Cp_pred_raw, surface='pressure', layout='single_column')

    # --- Saves the test error distribution plots ---
    plot_test_error_distribution_by_target(Cl_test_raw, Cl_pred_raw, Cp_test_raw, Cp_pred_raw, target='cl', layout='single_column')
    plot_test_error_distribution_by_target(Cl_test_raw, Cl_pred_raw, Cp_test_raw, Cp_pred_raw, target='cp', layout='single_column')

    # --- Saves the text performance report ---
    save_report(model, history, fit_losses, execution_time, X_test_raw, Cl_test_raw, Cl_pred_raw, Cp_test_raw, Cp_pred_raw, X_train, X_val)

    print("Process completed successfully.\n")

    return

if __name__ == "__main__":
    main(data_path)