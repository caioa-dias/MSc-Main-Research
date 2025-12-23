# -*- coding: utf-8 -*-
"""
--------------------------------------------------------------------------------------------------------
Function:               LF_numerical_ann_final
Author:                 Caio Dias Filho
Creation date:          2025-11-28
Last modification:      2025-11-29
Version:                1.0

Description:
    This script performs the training of an Artificial Neural Network (ANN) to predict the pressure 
    coefficient (Cp) distribution over wing sections and its corresponding sectional lift coefficient 
    (Cl). It handles data loading, preprocessing (scaling), model training, evaluation, and artifact 
    saving (model, scalers, and reports).
            
Dependencies:
    - matplotlib
    - typing
    - pathlib
    - seaborn
    - pandas
    - numpy
    - joblib
    - time
    - sklearn
    - tensorflow (keras)

Future implementations:
    >>> Implement K-Fold cross-validation.
    >>> Not working properly.
--------------------------------------------------------------------------------------------------------
"""

# Standard libraries
from matplotlib import pyplot as plt
from typing import Tuple
from pathlib import Path
import seaborn as sns
import pandas as pd
import numpy as np
import joblib
import time

# Machine learning & metrics
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler
from sklearn.model_selection import train_test_split

# Deep learning (Keras/TensorFlow)
import tensorflow as tf
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, History
from keras.models import Sequential
from keras.layers import Dense, Input, Dropout, BatchNormalization
from keras.optimizers import Adam, Nadam



def load_and_split_data(filepath: str, test_size: float, random_state: int):
    """
    Loads the dataset and splits it into training and testing sets based on wing conditions, aiming
    to improve the model's generalization capability.

    Args:
        filepath: Path to the .csv file containing the dataset.
        test_size: Fraction of the dataset to be used for testing.
        random_state: Seed for random shuffle.

    Returns:
        data_train (pd.DataFrame): DataFrame containing the training dataset.
        data_test (pd.DataFrame): DataFrame containing the testing dataset.
    """

    # 1. Loading the dataset:
    data_path = Path.cwd() / filepath
    data = pd.read_csv(data_path, sep=',')
    
    # 2. Create identifier for wing condition (assuming 80 sections per condition):
    wing_sections = 80
    indices = np.arange(len(data))
    data['wing_condition'] = indices // wing_sections
    wing_conditions = data['wing_condition'].unique()
    train_conds, test_conds = train_test_split(wing_conditions, test_size=test_size, random_state=random_state)
    
    # 3. Separate DataFrames and drop auxiliary column:
    data_train = data[data['wing_condition'].isin(train_conds)].drop(columns=['wing_condition'])
    data_test = data[data['wing_condition'].isin(test_conds)].drop(columns=['wing_condition'])
    
    return data_train, data_test

def preprocess_data(data_train: pd.DataFrame, data_test: pd.DataFrame):
    """
    Splits the dataset into features (X) and targets (Y) datasets, and applies scales the data, in
    which the features and 'Cl' is scaled using MinMaxScaler and 'Cp' is scaled using MaxAbsScaler.

    Args:
        data_train: DataFrame containing the training dataset.
        data_test: DataFrame containing the testing dataset.

    Returns:
        X_train_scaled (np.ndarray): Array containing the scaled training features.
        Y_train_scaled (np.ndarray): Array containing the scaled training targets.
        X_test_scaled (np.ndarray): Array containing the scaled testing features.
        Y_test_scaled (np.ndarray): Array containing the scaled testing targets.
        cl_scaler (MinMaxScaler): Scaler for 'Cl' targets.
        cp_scaler (MaxAbsScaler): Scaler for 'Cp' targets. 
    """

    # 1. Defining the input columns:
    input_cols = data_train.columns[0:3]
    
    # 2. Splitting the input data into training and testing datasets:
    X_train = data_train[input_cols].values
    X_test = data_test[input_cols].values
    
    # 3. Defining the target columns and splitting into training and testing datasets:
    Y_cl_train = data_train.iloc[:, 3].values.reshape(-1, 1)
    Y_cp_train = data_train.iloc[:, 4:].values
    
    Y_cl_test = data_test.iloc[:, 3].values.reshape(-1, 1)
    Y_cp_test = data_test.iloc[:, 4:].values
    
    # 4. Scaling the inputs using MinMaxScaler:
    scaler_x = MinMaxScaler()
    X_train_scaled = scaler_x.fit_transform(X_train)
    X_test_scaled = scaler_x.transform(X_test)
    
    # 5. Scaling the 'Cl' using MinMaxScaler:
    cl_scaler = MinMaxScaler()
    Y_cl_train_scaled = cl_scaler.fit_transform(Y_cl_train)
    Y_cl_test_scaled = cl_scaler.transform(Y_cl_test)
    
    # 6. Scaling the 'Cp' using MaxAbsScaler:
    cp_scaler = MaxAbsScaler()
    Y_cp_train_scaled = cp_scaler.fit_transform(Y_cp_train)
    Y_cp_test_scaled = cp_scaler.transform(Y_cp_test)
    
    # 7. Concatenate the targets back together:
    Y_train_scaled = np.hstack([Y_cl_train_scaled, Y_cp_train_scaled])
    Y_test_scaled = np.hstack([Y_cl_test_scaled, Y_cp_test_scaled])
    
    # 8. Saving the scalers:
    joblib.dump(scaler_x, 'LF_numerical_pinn/scalers/scaler_x.pkl')
    joblib.dump(cl_scaler, 'LF_numerical_pinn/scalers/scaler_cl.pkl')
    joblib.dump(cp_scaler, 'LF_numerical_pinn/scalers/scaler_cp.pkl')

    scaler_params = {
        'cl_scale': cl_scaler.scale_[0],  #Fator de escala do MinMaxScaler
        'cl_min': cl_scaler.min_[0],      #Offset do MinMax Scaler
        'cp_scale': cp_scaler.scale_  #Fator máximo do MaxAbsScaler
    }
    
    return X_train_scaled, Y_train_scaled, X_test_scaled, Y_test_scaled, cl_scaler, cp_scaler, scaler_params

def build_model(input_shape: Tuple[int], output_shape: int, scaler_params:dict, integration_weights:np.ndarray):
    """
    Builds and compiles the Artificial Neural Network architecture.

    Args:
        input_shape: Shape of the input layer.
        output_shape: Number of outputs.

    Returns:
        model (Sequential): Artificial Neural Network model.
    """

    def physics_informed_loss(y_true, y_pred):
        cl_pred = y_pred[:, 0:1]
        cp_pred = y_pred[:, 1:]

        mse_loss = tf.reduce_mean(tf.square(y_true - y_pred))

        cl_scale = tf.cast(scaler_params['cl_scale'], dtype=tf.float32)
        cl_min = tf.cast(scaler_params['cl_min'], dtype=tf.float32)

        cp_scale = tf.cast(scaler_params['cp_scale'], dtype=tf.float32)

        cl_pred_phys = (cl_pred - cl_min) / cl_scale

        cp_pred_phys = cp_pred * cp_scale

        weights = tf.constant(integration_weights, dtype=tf.float32)
        weights = tf.reshape(weights, (1, -1))

        cl_integrated = -1 * tf.reduce_sum(cp_pred_phys * weights, axis=1, keepdims=True)

        physical_loss = tf.reduce_mean(tf.square(cl_pred_phys - cl_integrated))

        lambda_phys = 0.1

        return mse_loss + (lambda_phys*physical_loss)

    # 1. Defining the model architecture:
    model = Sequential([
        Input(shape=input_shape),
        
        Dense(256, activation='swish', name='Hidden_Layer_1'),
        BatchNormalization(), 
        #Dropout(0.2, name='Dropout_1'),
        
        Dense(224, activation='swish', name='Hidden_Layer_2'),
        BatchNormalization(),
        #Dropout(0.2, name='Dropout_2'),
        
        Dense(128, activation='swish', name='Hidden_Layer_3'),
        BatchNormalization(),
        
        Dense(output_shape, activation='linear', name='Output_Layer')
    ], name='LF_numerical_ann')
    
    # 2. Compiling the model:
    model.compile(loss=physics_informed_loss, optimizer=Nadam(learning_rate=0.01), metrics=['mae'])

    return model

def save_results(model: Sequential, history: History, X_test: np.ndarray, Y_test: np.ndarray, Y_pred: np.ndarray, 
    execution_time: float):
    """
    Generates the training history, predictions and model sensitivity plots, and saves a performance report.

    Args:
        model: Artificial Neural Network sequential model.
        history: Model training history.
        X_test: Array containing the testing features.
        Y_test: Array containing the testing targets.
        Y_pred: Array containing the predicted targets.
        execution_time: Time taken to train the model.

    Returns:
        None. Saves the plots as .png files and the performance report as .txt file.
    """

    # =============================================================================================================
    # 1. VISUALIZATION: TRAINING HISTORY (MSE)
    # =============================================================================================================
    # Setting defined parameters:
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(1, len(loss) + 1)
    best_epoch = np.argmin(val_loss) + 1
    best_val_loss = min(val_loss)

    # Setting the figure parameters:
    plt.figure(figsize=(8,6))
    plt.suptitle("Model Training History: Loss Evolution (Mean Squared Error)", fontsize=14, fontname='Times New Roman', 
        fontweight='bold')
    plt.xlabel("Epochs", fontsize=12, fontname='Times New Roman')
    plt.ylabel("Loss [Log Scale]", fontsize=12, fontname='Times New Roman')
    plt.xlim(0, len(epochs)+1)
    plt.grid(True, which='both', ls='--', alpha=0.6)

    # Plotting the curves:
    plt.semilogy(epochs, loss, label='Training', color='#000080', linewidth=2)
    plt.semilogy(epochs, val_loss, label='Validation', color='#DC143C', linewidth=2)
    plt.axvline(x=best_epoch, color='gray', linestyle='--', alpha=0.5, linewidth=1)
    plt.scatter(best_epoch, best_val_loss, color='#DC143C', s=40, edgecolor='black', 
        label=f'Best epoch: {best_epoch}', zorder=5)
    plt.legend(loc='upper right', frameon=True, facecolor='white', edgecolor='black', 
        prop={'family': 'Times New Roman', 'size': 12})
    
    # Saving the plot:
    plt.tight_layout()
    plt.savefig('LF_numerical_pinn/plots-training/training_history_loss.png', dpi=300)
    plt.close()
    print(f"Plot saved as LF_numerical_ann/plots/training_history_loss.png\n")


    # =============================================================================================================
    # 2. VISUALIZATION: TRAINING HISTORY (MAE)
    # =============================================================================================================
    # Setting defined parameters:
    mae_loss = history.history['mae']
    mae_val_loss = history.history['val_mae']
    best_mae_val_loss = min(mae_val_loss)

    # Setting the figure parameters:
    plt.figure(figsize=(8,6))
    plt.suptitle("Model Training History: Mean Absolute Error (MAE) Evolution", fontsize=14, fontname='Times New Roman', 
        fontweight='bold')
    plt.xlabel("Epochs", fontsize=12, fontname='Times New Roman')
    plt.ylabel("MAE", fontsize=12, fontname='Times New Roman')
    plt.xlim(0, len(epochs)+1)
    plt.grid(True, which='both', ls='--', alpha=0.6)

    # Plotting the curves:
    plt.plot(epochs, mae_loss, label='Training', color='#000080', linewidth=2)
    plt.plot(epochs, mae_val_loss, label='Validation', color='#DC143C', linewidth=2)
    plt.axvline(x=best_epoch, color='gray', linestyle='--', alpha=0.5, linewidth=1)
    plt.scatter(best_epoch, best_mae_val_loss, color='#DC143C', s=40, edgecolor='black', 
        label=f'Best epoch: {best_epoch}', zorder=5)
    plt.legend(loc='upper right', frameon=True, facecolor='white', edgecolor='black', 
        prop={'family': 'Times New Roman', 'size': 12})
    
    # Saving the plot:
    plt.tight_layout()
    plt.savefig('LF_numerical_pinn/plots-training/training_history_mae.png', dpi=300)
    plt.close()
    print(f"Plot saved as LF_numerical_ann/plots/training_history_mae.png\n")


    # =============================================================================================================
    # 3. VISUALIZATION: PREDICTIONS VS. REAL (OVERALL MODEL)
    # =============================================================================================================
    # Calculates the metrics for the overall model:
    r2_overall = r2_score(Y_test, Y_pred)
    mae_overall = mean_absolute_error(Y_test, Y_pred)
    mse_overall = mean_squared_error(Y_test, Y_pred)
    rmse_overall = np.sqrt(mse_overall) 

    # Setting defined parameters:
    min_val = min(np.min(Y_test), np.min(Y_pred))
    max_val = max(np.max(Y_test), np.max(Y_pred))

    # Setting the figure parameters:
    plt.figure(figsize=(8,8))
    ax = plt.gca()
    ax.set_aspect('equal')
    plt.suptitle("Model Predictions vs. Real Values (Overall Model)", fontsize=14, fontname='Times New Roman', 
        fontweight='bold')
    plt.xlabel("True Values", fontsize=12, fontname='Times New Roman')
    plt.ylabel("Predicted Values", fontsize=12, fontname='Times New Roman')
    plt.xlim([min_val, max_val])
    plt.ylim([min_val, max_val])
    plt.grid(True, which='both', ls='--', alpha=0.6)

    # Plotting the points:
    plt.scatter(Y_test, Y_pred, facecolors='#4682B4', edgecolors='#1C3144', linewidth=0.6, s=15, alpha=1.0)
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2)
    textstr = '\n'.join((
        r'$R^2 = %.4f$' % (r2_overall, ),
        r'$MAE = %.4f$' % (mae_overall, ),
        r'$MSE = %.4f$' % (mse_overall, ),
        r'$RMSE = %.4f$' % (rmse_overall, )))
    props = dict(boxstyle='square, pad=0.5', facecolor='white', alpha=1.0, edgecolor='k')
    ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=12,
            verticalalignment='top', bbox=props, linespacing=1.5, fontname='Times New Roman')

    # Saving the plot:
    plt.tight_layout()
    plt.savefig('LF_numerical_pinn/plots-training/predictions_overall.png', dpi=300)
    plt.close()
    print(f"Plot saved as LF_numerical_ann/plots/predictions_overall.png\n")


    # =============================================================================================================
    # 4. VISUALIZATION: PREDICTIONS VS. REAL (LIFT COEFFICIENT)
    # =============================================================================================================
    # Splits the lift coefficient data from the target dataset:
    cl_test = Y_test[:, 0].reshape(-1, 1)
    cl_pred = Y_pred[:, 0].reshape(-1, 1)
    
    # Calculates the metrics for the overall model:
    r2_cl = r2_score(cl_test, cl_pred)
    mae_cl = mean_absolute_error(cl_test, cl_pred)
    mse_cl = mean_squared_error(cl_test, cl_pred)
    rmse_cl = np.sqrt(mse_cl) 

    # Setting defined parameters:
    min_val = min(np.min(cl_test), np.min(cl_pred))
    max_val = max(np.max(cl_test), np.max(cl_pred))

    # Setting the figure parameters:
    plt.figure(figsize=(8,8))
    ax = plt.gca()
    ax.set_aspect('equal')
    plt.suptitle("Model Predictions vs. Real Values (Lift Coefficient)", fontsize=14, fontname='Times New Roman', 
        fontweight='bold')
    plt.xlabel("True Values", fontsize=12, fontname='Times New Roman')
    plt.ylabel("Predicted Values", fontsize=12, fontname='Times New Roman')
    plt.xlim([min_val, max_val])
    plt.ylim([min_val, max_val])
    plt.grid(True, which='both', ls='--', alpha=0.6)

    # Plotting the points:
    plt.scatter(cl_test, cl_pred, facecolors='#4682B4', edgecolors='#1C3144', linewidth=0.6, s=15, alpha=1.0)
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2)
    textstr = '\n'.join((
        r'$R^2 = %.4f$' % (r2_cl, ),
        r'$MAE = %.4f$' % (mae_cl, ),
        r'$MSE = %.4f$' % (mse_cl, ),
        r'$RMSE = %.4f$' % (rmse_cl, )))
    props = dict(boxstyle='square, pad=0.5', facecolor='white', alpha=1.0, edgecolor='k')
    ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=12,
            verticalalignment='top', bbox=props, linespacing=1.5, fontname='Times New Roman')

    # Saving the plot:
    plt.tight_layout()
    plt.savefig('LF_numerical_pinn/plots-training/predictions_cl.png', dpi=300)
    plt.close()
    print(f"Plot saved as LF_numerical_ann/plots/predictions_cl.png\n")


    # =============================================================================================================
    # 5. VISUALIZATION: PREDICTIONS VS. REAL (PRESSURE COEFFICIENT)
    # =============================================================================================================
    # Splits the lift coefficient data from the target dataset:
    cp_test = Y_test[:, 1:]
    cp_pred = Y_pred[:, 1:]
    
    # Calculates the metrics for the overall model:
    r2_cp = r2_score(cp_test, cp_pred)
    mae_cp = mean_absolute_error(cp_test, cp_pred)
    mse_cp = mean_squared_error(cp_test, cp_pred)
    rmse_cp = np.sqrt(mse_cp) 

    # Setting defined parameters:
    min_val = min(np.min(cp_test), np.min(cp_pred))
    max_val = max(np.max(cp_test), np.max(cp_pred))

    # Setting the figure parameters:
    plt.figure(figsize=(8,8))
    ax = plt.gca()
    ax.set_aspect('equal')
    plt.suptitle("Model Predictions vs. Real Values (Pressure Coefficient)", fontsize=14, fontname='Times New Roman', 
        fontweight='bold')
    plt.xlabel("True Values", fontsize=12, fontname='Times New Roman')
    plt.ylabel("Predicted Values", fontsize=12, fontname='Times New Roman')
    plt.xlim([min_val, max_val])
    plt.ylim([min_val, max_val])
    plt.grid(True, which='both', ls='--', alpha=0.6)

    # Plotting the points:
    plt.scatter(cp_test, cp_pred, facecolors='#4682B4', edgecolors='#1C3144', linewidth=0.6, s=15, alpha=1.0)
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2)
    textstr = '\n'.join((
        r'$R^2 = %.4f$' % (r2_cp, ),
        r'$MAE = %.4f$' % (mae_cp, ),
        r'$MSE = %.4f$' % (mse_cp, ),
        r'$RMSE = %.4f$' % (rmse_cp, )))
    props = dict(boxstyle='square, pad=0.5', facecolor='white', alpha=1.0, edgecolor='k')
    ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=12,
            verticalalignment='top', bbox=props, linespacing=1.5, fontname='Times New Roman')

    # Saving the plot:
    plt.tight_layout()
    plt.savefig('LF_numerical_pinn/plots-training/predictions_cp.png', dpi=300)
    plt.close()
    print(f"Plot saved as LF_numerical_ann/plots/predictions_cp.png\n")


    # =============================================================================================================
    # 6. VISUALIZATION: MAE DISTRIBUTION (LIFT COEFFICIENT)
    # =============================================================================================================
    # Flattens the data (necessary for the Seaborn plot):
    cl_test_flat = cl_test.flatten()
    cl_pred_flat = cl_pred.flatten()

    # Defines the DataFrame and discretization for the bar plot:
    df_cl_mae = pd.DataFrame({'True_Cl': cl_test_flat, 'Abs_Error': np.abs(cl_test_flat - cl_pred_flat)})
    bars = np.arange(-0.4, 1.8, 0.2)
    df_cl_mae['Cl_Range'] = pd.cut(df_cl_mae['True_Cl'], bins=bars)

    # Setting the figure parameters:
    plt.figure(figsize=(10,6))
    sns.set_style("whitegrid")
    ax = sns.barplot(data=df_cl_mae, x='Cl_Range', y='Abs_Error', color='#4682B4', edgecolor='#1C3144',
        linewidth=0.8, errorbar=None)
    ax.set_xticks(ax.get_xticks())
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right', rotation_mode='anchor', fontsize=10)
    plt.suptitle("Model Sensitivity: Prediction Error across Sectional Lift Coefficient Ranges", fontsize=14, 
        fontname='Times New Roman', fontweight='bold')
    plt.xlabel("Sectional Lift Coefficient Value Range", fontsize=12, fontname='Times New Roman')
    plt.ylabel("Mean Absolute Error (MAE)", fontsize=12, fontname='Times New Roman')
    ax.set_axisbelow(True)
    plt.grid(axis='y', linestyle='--', alpha=0.6)

    # Saving the plot:
    plt.tight_layout()
    plt.savefig('LF_numerical_pinn/plots-training/model_sensitivity_cl.png', dpi=300)
    plt.close()
    print(f"Plot saved as LF_numerical_ann/plots/model_sensitivity_cl.png\n")


    # =============================================================================================================
    # 7. VISUALIZATION: MAE DISTRIBUTION (PRESSURE COEFFICIENT)
    # =============================================================================================================
    # Flattens the data (necessary for the Seaborn plot):
    cp_test_flat = cp_test.flatten()
    cp_pred_flat = cp_pred.flatten()

    # Defines the DataFrame and discretization for the bar plot:
    df_cp_mae = pd.DataFrame({'True_Cp': cp_test_flat, 'Abs_Error': np.abs(cp_test_flat - cp_pred_flat)})
    bars = np.arange(-4.0, 1.0, 0.5)
    df_cp_mae['Cp_Range'] = pd.cut(df_cp_mae['True_Cp'], bins=bars)

    # Setting the figure parameters:
    plt.figure(figsize=(10,6))
    sns.set_style("whitegrid")
    ax = sns.barplot(data=df_cp_mae, x='Cp_Range', y='Abs_Error', color='#4682B4', edgecolor='#1C3144',
        linewidth=0.8, errorbar=None)
    ax.set_xticks(ax.get_xticks())
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right', rotation_mode='anchor', fontsize=10)
    plt.suptitle("Model Sensitivity: Prediction Error across Pressure Coefficient Ranges", fontsize=14, 
        fontname='Times New Roman', fontweight='bold')
    plt.xlabel("Pressure Coefficient Value Range", fontsize=12, fontname='Times New Roman')
    plt.ylabel("Mean Absolute Error (MAE)", fontsize=12, fontname='Times New Roman')
    ax.set_axisbelow(True)
    plt.grid(axis='y', linestyle='--', alpha=0.6)

    # Saving the plot:
    plt.tight_layout()
    plt.savefig('LF_numerical_pinn/plots-training/model_sensitivity_cp.png', dpi=300)
    plt.close()
    print(f"Plot saved as LF_numerical_ann/plots/model_sensitivity_cp.png\n")


    # =============================================================================================================
    # 8. TEXT FILE: MODEL PERFORMANCE REPORT
    # =============================================================================================================
    # Generate the model's performance report
    stringlist = []
    model.summary(print_fn=lambda x: stringlist.append(x))
    model_structure = "\n".join(stringlist)
    
    content = f"""
PERFORMANCE REPORT - LF_NUMERICAL_ANN
==========================================================
Date: {time.strftime("%Y-%m-%d %H:%M:%S")}
Execution Time: {execution_time:.2f} seconds

1. CONFIGURATION
----------------------------------
Epochs: {len(history.history['loss'])}
Batch Size: 64
Optimizer: Adam
Input Shape: {X_test.shape}
Outputs: Cl + (120*Cp)

2. OVERALL MODEL TEST PERFORMANCE
----------------------------------
R² Score: {r2_overall:.5f}
MAE:      {mae_overall:.6f}
MSE:      {mse_overall:.6f}
RMSE:     {rmse_overall:.6f}

3. CL TEST PERFORMANCE
----------------------------------
R² Score: {r2_cl:.5f}
MAE:      {mae_cl:.6f}
MSE:      {mse_cl:.6f}
RMSE:     {rmse_cl:.6f}

4. CP TEST PERFORMANCE
----------------------------------
R² Score: {r2_cp:.5f}
MAE:      {mae_cp:.6f}
MSE:      {mse_cp:.6f}
RMSE:     {rmse_cp:.6f}

5. FINAL LOSS RESULTS
----------------------------------
Final Loss (Training): {history.history['loss'][-1]:.6f}
Final Loss (Validation): {history.history['val_loss'][-1]:.6f}

6. MODEL ARCHITECTURE
----------------------------------
{model_structure}
==========================================================
"""
    
    # Saving the report:
    with open('LF_numerical_ann/reports/performance_report_LF_numerical_ann.txt', "w", encoding="utf-8") as f:
        f.write(content)
    print(f"Report saved as {'LF_numerical_pinn/reports/performance_report_LF_numerical_ann.txt'}\n")

    return

def calculate_non_uniform_weights(x_coords):
    x = np.ravel(x_coords)
    dx = np.diff(x)
    weights = np.zeros_like(x)

    weights[0] = 0.5 * dx[0]
    weights[1:-1] = 0.5 * (dx[:-1] + dx[1:])
    weights[-1] = 0.5 * dx[-1]

    return weights

def main(data_path: str):
    """
    Main execution workflow: Loads data, processes it, trains the ANN, and evaluates performance.
    """

    chord_df = pd.read_csv('Numerical-ChordwiseAddress.csv', sep=';', header=None, dtype=float)
    x_coords = np.array(chord_df)
    x_coords = x_coords.flatten()

    integration_weights = calculate_non_uniform_weights(x_coords)

    # 1. Starts the timer for recording execution time:
    start_time = time.time()
    
    # 2. Loads and split the dataset:
    print("\nStarting ANN training process...\n")
    print(f"Loading data from: {data_path}\n")
    data_train, data_test = load_and_split_data(data_path, test_size=0.2, random_state=21)
    print("Data loaded and split successfully.\n")
    
    # 3. Preprocesses the data:
    print("Preprocessing data (scaling)...\n")
    X_train, Y_train, X_test, Y_test, cl_scaler, cp_scaler, scaler_params = preprocess_data(data_train, data_test)
    print(f"Input Shape: {X_train.shape}, Output Shape: {Y_train.shape}\n")
    
    # 4. Builds the model:
    print("Building model architecture..\n.")
    model = build_model(input_shape=(X_train.shape[1],), output_shape=Y_train.shape[1],
        scaler_params=scaler_params, integration_weights=integration_weights)
    
    # 5. Trains the model:
    print("\nStarting training...\n")
    callbacks = [EarlyStopping(monitor='val_loss', patience=30, restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=15, min_lr=1e-6),
        ModelCheckpoint('LF_numerical_ann/LF_numerical_ann.keras', monitor='val_loss', save_best_only=True)]
    
    history = model.fit(X_train, Y_train, validation_split=0.2, epochs=500, 
        batch_size=128, verbose=0, callbacks=callbacks)
    
    # 6. Evaluates the model:
    print("Evaluating model on test set...\n")
    Y_pred_scaled = model.predict(X_test)

    # 7. Splits the target variables for reverse transformation:
    Y_test_cl_scaled = Y_test[:, 0].reshape(-1, 1)
    Y_test_cp_scaled = Y_test[:, 1:]
    Y_pred_cl_scaled = Y_pred_scaled[:, 0].reshape(-1, 1)
    Y_pred_cp_scaled = Y_pred_scaled[:, 1:]

    # 8. Reverses the scaling:
    Y_test_cl = cl_scaler.inverse_transform(Y_test_cl_scaled)
    Y_test_cp = cp_scaler.inverse_transform(Y_test_cp_scaled)
    Y_pred_cl = cl_scaler.inverse_transform(Y_pred_cl_scaled)
    Y_pred_cp = cp_scaler.inverse_transform(Y_pred_cp_scaled)

    # 9. Recombines the target variables back together:
    Y_test = np.hstack([Y_test_cl, Y_test_cp])
    Y_pred = np.hstack([Y_pred_cl, Y_pred_cp])
    
    # 10. Finishes the timer:
    end_time = time.time()
    execution_time = end_time - start_time
    
    # 11. Saves results:
    save_results(model, history, X_test, Y_test, Y_pred, execution_time)
    print("Process completed successfully.\n")

    return



if __name__ == "__main__":
    main('Numerical-PressureDistributionData.csv')