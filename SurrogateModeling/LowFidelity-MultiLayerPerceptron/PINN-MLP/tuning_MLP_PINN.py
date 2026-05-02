# -*- coding: utf-8 -*-
"""
========================================================================================================
Module: tuning_MLP_PINN
Author: Caio Dias Filho
Creation date: 2026-05-02
Last modification: 2026-05-02
Version: 1.0 (final)
========================================================================================================

OVERVIEW
--------
This module performs hyperparameter tuning of a Physics-Informed Multi-Layer Perceptron Artificial Neural
Network (MLP-PINN) using a grid search strategy.

The model predicts:

    - Sectional lift coefficient (Cl)
    - Pressure coefficient distribution (Cp)

The PINN formulation includes a physics-informed loss function that enforces consistency between the 
predicted sectional lift coefficient and the lift coefficient reconstructed from the predicted pressure
coefficient distribution through aerodynamic integration.

The tuning process evaluates combinations of:

    - Batch size
    - Kernel initializer

Each configuration is assessed using two objectives:

    - Prediction Mean Squared Error (MSE)
    - Global lift coefficient (CL) reconstruction Mean Squared Error


WORKFLOW
--------
The implemented workflow consists of:

    - Loading and merging aerodynamic data with the effective angle-of-attack dataset.
    - Splitting data into training and testing subsets at flow-case level.
    - Generating all grid search combinations.
    - Performing Group K-Fold cross-validation for each configuration.
    - Preprocessing each fold with feature and target standardization.
    - Training PINN models with fixed architecture and selected tuning parameters.
    - Evaluating prediction accuracy on validation folds.
    - Reconstructing global lift coefficient (CL) from Cp distributions.
    - Computing CL reconstruction error.
    - Aggregating fold-wise results.
    - Extracting Pareto-optimal configurations.
    - Saving tuning results and Pareto front results.
    

SEARCH SPACE
------------
The tuning search space includes:

    - Batch size:
        [32, 64, 128, 256]

    - Kernel initializer:
        ['glorot_uniform', 'glorot_normal', 'he_uniform', 'he_normal']


DEPENDENCIES
------------
Python libraries:

    - os
    - warnings
    - gc
    - itertools
    - numpy
    - pandas
    - scikit-learn
    - tensorflow / keras


OUTPUT FILES
------------
Tuning results:

    - PINN-MLP/optimization-results/tuning_results.csv

Pareto-optimal tuning results:

    - PINN-MLP/optimization-results/tuning_pareto_results.csv


REPRODUCIBILITY
---------------
Reproducibility is ensured through:

    - Fixed global seed (SEED = 42)
    - NumPy and TensorFlow seed control
    - Deterministic dataset splitting
    - Fixed grid seatch combinations


ASSUMPTIONS
-----------
The implementation assumes that:

    - Each flow case contains 80 spanwise sections.
    - Cp distributions follow a consistent chordwise discretization.
    - Airfoil geometry is consistent with Cp data.
    - Effective AoA data correctly matches the aerodynamic conditions.
    - The fixed PINN architecture was previously selected from optimization studies.
    - Cp integration provides a physically meaningful reconstruction of sectional and global lift.


LIMITATIONS
-----------
Potential limitations include:

    - Grid search is limited to batch size and kernel initializer.
    - The neural network architecture and physical loss function remain fixed.
    - Computational cost is increased by cross-validation and physics-informed training.
    - Results depend on the quality of the effective angle-of-attack dataset.
    - CL reconstruction may be affected by integration and discretization errors.

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

# Reproducibility setup
SEED = 42
os.environ['PYTHONHASHSEED'] = str(SEED)
tensorflow.random.set_seed(SEED)
np.random.seed(SEED)
print(f'\nGlobal seed set to: {SEED}')

def load_and_split_data(filepath: str, AoA_effective: pd.DataFrame):
    """
    Load and split the dataset with effective angle-of-attack integration.

    This function merges the aerodynamic dataset with the effective angle-of-attack dataset and splits it 
    into training and testing subsets based on flow cases.
    
    Parameters
    ----------
    filepath : str
        Path to the aerodynamic dataset.

    AoA_effective : pandas.DataFrame
        Effective angle-of-attack dataset.

    Returns
    -------
    data_train : pandas.DataFrame
        Training dataset.

    data_test : pandas.DataFrame
        Testing dataset.
    """

    # --- Loading the dataset ---
    data = (pd.read_csv(filepath, sep=';')).copy()

    # --- Adding the effective angle of attack ---
    key_cols = ['Re', 'AoA', 'y', 'cl']
    data = data.merge(AoA_effective[key_cols + ['AoA_Effective']], on=key_cols, how='left',
        validate='one_to_one')

    # --- Create identifier for flow case (assuming 80 sections per case) ---
    wing_sections = 80
    idx = np.arange(len(data))
    data['flow_case'] = idx // wing_sections
    unique_cases = data['flow_case'].unique()
    train_cases, test_cases = skl.model_selection.train_test_split(unique_cases, test_size=0.2,
        random_state=SEED, shuffle=True)
    
    # --- Separate the datasets and drop auxiliary column ---
    data_train = data[data['flow_case'].isin(train_cases)]
    data_test = data[data['flow_case'].isin(test_cases)]

    return data_train, data_test

def preprocess_fold_data(data_train: pd.DataFrame, data_val: pd.DataFrame):
    """
    Preprocess data for PINN training.

    This function standardizes inputs and targets and appends the effective angle of attack to the target
    vector for use in the physical loss.

    Parameters
    ----------
    data_train : pandas.DataFrame
        Training dataset for the fold.

    data_val : pandas.DataFrame
        Validation dataset for the fold.

    Returns
    -------
    X_train : numpy.ndarray
        Standardized input features for training.

    y_train : numpy.ndarray
        Standardized targets for training.

    X_val : numpy.ndarray
        Standardized input features for validation.

    y_val : numpy.ndarray
        Standardized targets for validation.

    y_scaler : sklearn.preprocessing.StandardScaler
        Scaler object used to standardize targets.

    X_scaler : sklearn.preprocessing.StandardScaler
        Scaler object used to standardize inputs.
    """

    # --- Defining input and target columns ---
    X_cols = ['Re', 'AoA', 'y']
    y_cols = ['cl'] + [col for col in data_train.columns[4:] if col != 'AoA_Effective' and 
        col != 'flow_case']

    # --- Splitting the raw features into training and validation sets ---
    X_train_raw = data_train[X_cols].to_numpy()
    X_val_raw = data_val[X_cols].to_numpy()

    # --- Splitting the effective angle of attack into training and validation sets ---
    AoA_train_raw = data_train[['AoA_Effective']].to_numpy()
    AoA_val_raw = data_val[['AoA_Effective']].to_numpy()

    # --- Splitting the raw targets into training and validation sets ---
    y_train_raw = data_train[y_cols].to_numpy()
    y_val_raw = data_val[y_cols].to_numpy()

    # --- Scaling the features data ---
    X_scaler = skl.preprocessing.StandardScaler()
    X_train = X_scaler.fit_transform(X_train_raw)
    X_val = X_scaler.transform(X_val_raw)

    # --- Scaling the targets data ---
    y_scaler = skl.preprocessing.StandardScaler()
    y_train = y_scaler.fit_transform(y_train_raw)
    y_val = y_scaler.transform(y_val_raw)

    # --- Appends the AoA feature to the target training and validation data ---
    y_train = np.hstack((y_train, AoA_train_raw))
    y_val = np.hstack((y_val, AoA_val_raw))

    return X_train, y_train, X_val, y_val, y_scaler, X_scaler

def build_pinn_loss(panel_dx: np.ndarray, panel_dy: np.ndarray, lambda_physical: float,
    y_scaler: skl.preprocessing.StandardScaler):
    """
    Construct the physics-informed loss function.

    This function builds a custom loss function that combines supervised prediction error with a physical
    consistency constraint. The physical term enforces agreement between the predicted sectional lift
    coefficient and the lift coefficient reconstructed from the predicted pressure coefficient 
    distribution.

    Parameters
    ----------
    panel_dx : numpy.ndarray
        Panel coordinate differences in the x-direction.

    panel_dy : numpy.ndarray
        Panel coordinate differences in the y-direction.

    lambda_physical : float
        Weighting factor applied to the physical consistency loss.

    y_scaler : sklearn.preprocessing.StandardScaler
        Target scaler used to recover Cl and Cp values in physical scale inside the loss function.

    Returns
    -------
    pinn_loss : callable
        TensorFlow-compatible physics-informed loss function.
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

    def pinn_loss(y_true: np.ndarray, y_pred: np.ndarray):
        """
        Compute the physics-informed training loss.

        This function evaluates the total PINN loss as the sum of a supervised data loss and a weighted
        physical consistency loss. The supervised loss compares the predicted outputs with the true 
        scaled aerodynamic targets, while the physical loss enforces consistency between the predicted 
        sectional lift coefficient and the lift coefficient reconstructed from the predicted Cp distribution.

        Parameters
        ----------
        y_true : numpy.ndarray
            True target array. The last column contains the effective angle of attack used only for the
            physical loss term.

        y_pred : numpy.ndarray
            Predicted target array containing the scaled Cl and Cp outputs.

        Returns
        -------
        loss : tensorflow.Tensor
            Total physics-informed loss used for model training.
        """

        # --- Casting to float32 ---
        y_true = tensorflow.cast(y_true, dtype=tensorflow.float32)
        y_pred = tensorflow.cast(y_pred, dtype=tensorflow.float32)

        # --- Defines the true supervised values ---
        AoA_deg = y_true[:, -1:]

        # --- Defines the predicted supervised values ---
        Cl_pred = y_pred[:, 0:1]
        Cp_pred = y_pred[:, 1:]

        # --- Defines the supervised loss as a global MSE ---
        supervised_true = y_true[:, :-1]
        supervised_pred = y_pred
        data_loss = tensorflow.reduce_mean(tensorflow.square(supervised_true - supervised_pred))

        # --- Reescales the predicted supervised values ---
        Cl_pred_raw = Cl_pred * Cl_scale + Cl_mean
        Cp_pred_raw = Cp_pred * Cp_scale + Cp_mean

        # --- Computes the average Cp for each panel ---
        panel_Cp = 0.5 * (Cp_pred_raw[:, :-1] + Cp_pred_raw[:, 1:])

        # --- Integrates to find normal (Cn) and axial (Ca) force coefficients ---
        Cn = tensorflow.reduce_sum(panel_Cp * panel_dx, axis=1, keepdims=True)
        Ca = -tensorflow.reduce_sum(panel_Cp * panel_dy, axis=1, keepdims=True)

        # --- Defines the angle of attack in radians ---
        AoA_rad = AoA_deg * (np.pi / 180.0)

        # --- Projects Cn and Ca into the lift coefficient (Cl) using the angle of attack ---
        Cl_from_Cp_raw = Cn * tensorflow.cos(AoA_rad) - Ca * tensorflow.sin(AoA_rad)

        # --- Defines the physical loss ---
        physical_loss = tensorflow.reduce_mean(tensorflow.square(Cl_pred_raw - Cl_from_Cp_raw))

        # --- Defines the training loss ---
        loss = data_loss + lambda_physical * physical_loss

        return loss
    
    return pinn_loss

def pinn_supervised_mse(y_true: np.ndarray, y_pred: np.ndarray):
    """
    Compute the supervised mean squared error for the PINN model.

    This metric evaluates only the supervised prediction error, excluding the effective angle of attack
    appended to the target array for physical-loss computation.

    Parameters
    ----------
    y_true : numpy.ndarray
        True target array containing scaled aerodynamic targets and the effective angle of attack in the
        last column.

    y_pred : numpy.ndarray
        Predicted target array containing scaled aerodynamic outputs.

    Returns
    -------
    supervised_mse : tensorflow.Tensor
        Mean squared error between supervised targets and predictions.
    """

    # --- Casting float 32 ---
    y_true = tensorflow.cast(y_true, dtype=tensorflow.float32)
    y_pred = tensorflow.cast(y_pred, dtype=tensorflow.float32)

    # --- Defines the true supervised values, excluding the angle of attack ---
    supervised_true = y_true[:, :-1]

    # --- Calculates the supervised mean squared error ---
    supervised_mse = tensorflow.reduce_mean(tensorflow.square(supervised_true - y_pred))

    return supervised_mse

def pinn_supervised_mae(y_true: np.ndarray, y_pred: np.ndarray):
    """
    Compute the mean absolute error for the PINN model. 

    This metric evaluates the absolute error only for the supervised aerodynamic outputs, excluding the
    effective angle of attack appended to the target array for physical-loss computation.

    Parameters
    ----------
    y_true : numpy.ndarray
        True target array containing scaled aerodynamic targets and the effective angle of attack in the
        last column.

    y_pred : numpy.ndarray
        Predicted target array containing scaled aerodynamic outputs.

    Returns
    -------
    supervised_mae : tensorflow.Tensor
        Mean absolute error between supervised targets and predictions.
    """

    # --- Casting to float32 ---
    y_true = tensorflow.cast(y_true, dtype=tensorflow.float32)
    y_pred = tensorflow.cast(y_pred, dtype=tensorflow.float32)

    # --- Defines the true supervised values, excluding the angle of attack ---
    supervised_true = y_true[:, :-1]

    # --- Calculates the mean absolute error ---
    supervised_mae = tensorflow.reduce_mean(tensorflow.abs(supervised_true - y_pred))

    return supervised_mae

def build_physical_metric(panel_dx: np.ndarray, panel_dy: np.ndarray, 
    y_scaler: skl.preprocessing.StandardScaler):
    """
    Build the physical consistency metric for the PINN model.

    This function constructs a metric that measures the mean squared discrepancy between the predicted
    sectional lift coefficient and the lift coefficient reconstructed from the predicted pressure
    coefficient distribution.

    Parameters
    ----------
    panel_dx : numpy.ndarray
        Panel coordinate differences in the x-direction.

    panel_dy : numpy.ndarray
        Panel coordinate differences in the y-direction.

    y_scaler : sklearn.preprocessing.StandardScaler
        Target scaler used to recover Cl and Cp values in physical scale.

    Returns
    -------
    physical_mse : callable
        TensorFlow-compatible metric function returning the physical consistency MSE.
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

    def physical_mse(y_true: np.ndarray, y_pred: np.ndarray):
        """
        Compute the physical consistency mean squared error.

        This metric evaluates the discrepancy between the predicted sectional lift coefficient and the
        sectional lift coefficient reconstructed from the predicted pressure coefficient distribution.

        The reconstruction is performed by integrating Cp over the airfoil surface and projecting the
        resulting normal and axial force coefficients into the lift direction using the effective angle
        of attack stored in the last column of y_true.

        Parameters
        ----------
        y_true : numpy.ndarray
            True target array. The last column contains the effective angle of attack used for aerodynamic
            force projection.

        y_pred : numpy.ndarray
            Predicted target array containing scaled Cl and Cp outputs.

        Returns
        -------
        physical_mse : tensorflow.Tensor
            Mean squared physical consistency error in physical scale.
        """

        # --- Casting to float32 ---
        y_true = tensorflow.cast(y_true, dtype=tensorflow.float32)
        y_pred = tensorflow.cast(y_pred, dtype=tensorflow.float32)

        # --- Defines the true and predicted values ---
        Cl_pred = y_pred[:, 0:1]
        Cp_pred = y_pred[:, 1:]
        AoA_deg = y_true[:, -1:]

        # --- Reescales the predicted values ---
        Cl_pred_raw = Cl_pred * Cl_scale + Cl_mean
        Cp_pred_raw = Cp_pred * Cp_scale + Cp_mean

        # --- Computes the average Cp for each panel ---
        panel_Cp = 0.5 * (Cp_pred_raw[:, :-1] + Cp_pred_raw[:, 1:])

        # --- Integrates to find normal (Cn) and axial (Ca) force coefficients ---
        Cn = tensorflow.reduce_sum(panel_Cp * panel_dx, axis=1, keepdims=True)
        Ca = -tensorflow.reduce_sum(panel_Cp * panel_dy, axis=1, keepdims=True)

        # --- Defines the angle of attack in radians ---
        AoA_rad = AoA_deg * (np.pi / 180.0)

        # --- Projects Cn and Ca into the lift coefficient (Cl) using the angle of attack ---
        Cl_from_Cp_raw = Cn * tensorflow.cos(AoA_rad) - Ca * tensorflow.sin(AoA_rad)

        # --- Defines the physical loss ---
        return tensorflow.reduce_mean(tensorflow.square(Cl_pred_raw - Cl_from_Cp_raw))

    return physical_mse

def build_model(initializer: str, input_dim: int, output_dim: int, panel_dx: np.ndarray,
    panel_dy: np.ndarray, y_scaler: skl.preprocessing.StandardScaler):
    """
    Build and compile the tuned Physics-Informed MLP model.

    This function defines a fixed PINN architecture and compiles it using the selected kernel initializer.
    The model is trained with a physics-informed loss function that combines supervised prediction error
    and Cp-based lift consistency.

    Parameters
    ----------
    initializer : str
        Kernel initializer used in all dense layers.

    input_dim : int
        Number of input features.

    output_dim : int
        Number of supervised aerodynamic output variables.

    panel_dx : numpy.ndarray
        Panel coordinate differences in the x-direction.

    panel_dy : numpy.ndarray
        Panel coordinate differences in the y-direction.

    y_scaler : sklearn.preprocessing.StandardScaler
        Target scaler used inside the physics-informed loss and physical metric.

    Returns
    -------
    model : keras.Model
        Compiled physics-informed MLP model.
    """

    # --- Defining model input layer ---
    inputs = keras.layers.Input(shape=(input_dim,), name='Input_Layer')
    x = inputs

    # --- Defining model hidden layers ---
    x = keras.layers.Dense(units=96, activation='gelu', kernel_initializer=initializer, 
        name='Hidden_Layer_1')(inputs)
    x = keras.layers.Dense(units=96, activation='gelu', kernel_initializer=initializer, 
        name='Hidden_Layer_2')(x)
    x = keras.layers.Dense(units=256, activation='gelu', kernel_initializer=initializer, 
        name='Hidden_Layer_3')(x)
    x = keras.layers.Dense(units=192, activation='gelu', kernel_initializer=initializer, 
        name='Hidden_Layer_4')(x)
    x = keras.layers.Dense(units=256, activation='gelu', kernel_initializer=initializer, 
        name='Hidden_Layer_5')(x)
    x = keras.layers.Dense(units=160, activation='gelu', kernel_initializer=initializer, 
        name='Hidden_Layer_6')(x)
    
    # --- Defining the model output layer ---
    outputs = keras.layers.Dense(units=output_dim, activation='linear',
        kernel_initializer=initializer, name='Output_Layer')(x)
  
    # --- Defining the model ---
    model = keras.Model(inputs=inputs, outputs=outputs, name='low_fidelity_mlp_pinn')

    # --- Defining the physical loss and MAE metric ---
    loss = build_pinn_loss(panel_dx, panel_dy, lambda_physical=0.094743, y_scaler=y_scaler)
    physical_metric = build_physical_metric(panel_dx, panel_dy, y_scaler)
    
    # --- Renaming the metrics ---
    pinn_supervised_mae.__name__ = 'supervised_mae'
    pinn_supervised_mse.__name__ = 'supervised_mse'
    physical_metric.__name__ = 'physical_mse'

    # --- Compiling the model ---
    model.compile(loss=loss, optimizer=keras.optimizers.Nadam(learning_rate=0.003), 
        metrics=[pinn_supervised_mse, pinn_supervised_mae, physical_metric])

    return model

def predict_validation(model: keras.Model, X_val: np.ndarray, y_val: np.ndarray,
    y_scaler: skl.preprocessing.StandardScaler):
    """
    Generate validation predictions and convert outputs back to physical scale.

    This function predicts the validation targets using the trained PINN model, removes the effective
    angle-of-attack column from the true target array, and applies inverse target scaling to both
    reference and predicted outputs.

    Parameters
    ----------
    model : keras.Model
        Trained PINN model.

    X_val : numpy.ndarray
        Scaled validation input data.

    y_val : numpy.ndarray
        Scaled validation target data with the effective angle of attack appended as the last column.

    y_scaler : sklearn.preprocessing.StandardScaler
        Target scaler used to recover physical-scale aerodynamic outputs.

    Returns
    -------
    y_val_raw : numpy.ndarray
        Reference validation outputs in physical scale.

    y_pred_raw : numpy.ndarray
        Predicted validation outputs in physical scale.
    """

    # --- Predicting on the validation set ---
    y_pred = model.predict(X_val, verbose=0)

    # --- Reverses the scaling ---
    y_val_raw = y_scaler.inverse_transform(y_val[:, :-1])
    y_pred_raw = y_scaler.inverse_transform(y_pred)

    return y_val_raw, y_pred_raw

def compute_cl_from_cp(cp_data: np.ndarray, airfoil_data: pd.DataFrame, AoA: float):
    """
    Compute sectional lift coefficient from pressure coefficient distribution.

    This function integrates the pressure coefficient distribution over the airfoil surface to obtain
    normal and axial force coefficients, which are then projected into the lift direction using the
    provided angle of attack.

    Parameters
    ----------
    cp_data : numpy.ndarray
        Pressure coefficient distribution along the airfoil surface.

    airfoil_data : pandas.DataFrame
        Airfoil geometry coordinates containing x and y columns.

    AoA : float
        Effective angle of attack in degrees.

    Returns
    -------
    Cl : float
        Sectional lift coefficient reconstructed from the Cp distribution.
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
    Compute global lift coefficient from sectional Cp distributions.

    This function reconstructs the wing lift coefficient by computing sectional lift coefficients from Cp
    distributions and integrating them along the span.

    For each flow case, the corresponding effective angle of attack is retrieved from the effective
    angle-of-attack dataset and used in the Cp-to-Cl integration.

    Parameters
    ----------
    X_scaled : numpy.ndarray
        Scaled input data.

    y_raw : numpy.ndarray
        Aerodynamic outputs in physical scale.

    X_scaler : sklearn.preprocessing.StandardScaler
        Input scaler used to recover physical-scale input variables.

    airfoil_data : pandas.DataFrame
        Airfoil geometry coordinates used for sectional force integration.

    AoA_effective : pandas.DataFrame
        Dataset containing effective angle-of-attack values for each aerodynamic condition and spanwise
        section.

    Returns
    -------
    CL : numpy.ndarray
        Global lift coefficient values reconstructed for each flow case.
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

        # Defines the y position for each case:
        y = case_data.iloc[:, 2].values * b
        cl_list = np.array(cl_list)
        # Calculates the CL for this case:
        CL = (2/S) * np.trapz(cl_list*c, x=y)
        # Adds the CL to the list of CL for all cases:
        CL_list.append(CL)

    return np.array(CL_list)

def objective(batch_size: int, initializer: str, data_train: pd.DataFrame, airfoil_data: pd.DataFrame,
    AoA_effective: pd.DataFrame, panel_dx: np.ndarray, panel_dy: np.ndarray):
    """
    Evaluate one grid search configuration using physics-informed cross-validation.

    This function performs Group K-Fold cross-validation for a given combination of batch size and kernel
    initializer. For each fold, it preprocesses the data, trains the PINN model, evaluates prediction
    accuracy, reconstructs the global lift coefficient, and computes the corresponding CL reconstruction
    error.

    Parameters
    ----------
    batch_size : int
        Batch size used during model training.

    initializer : str
        Kernel initializer used in the PINN architecture.

    data_train : pandas.DataFrame
        Training dataset used for cross-validation.

    airfoil_data : pandas.DataFrame
        Airfoil geometry coordinates used for Cp integration.

    AoA_effective : pandas.DataFrame
        Effective angle-of-attack dataset used for CL reconstruction.

    panel_dx : numpy.ndarray
        Panel coordinate differences in the x-direction.

    panel_dy : numpy.ndarray
        Panel coordinate differences in the y-direction.

    Returns
    -------
    objective_mse_loss : float
        Mean validation prediction MSE across cross-validation folds.

    objective_CL_loss : float
        Mean global lift coefficient reconstruction MSE across cross-validation folds.
    """

    # --- Defines the list to store results ---
    val_mse_loss = []
    CL_loss = []

    # --- Defines the K-Fold cross-validation ---
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
        model = build_model(initializer, X_train.shape[1], y_train.shape[1]-1, panel_dx, panel_dy, y_scaler)

        # Trains the model:
        callbacks = [keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, min_delta=1e-6,
            restore_best_weights=True),
            keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6,
            cooldown=5, min_delta=1e-5)]
        model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=500, batch_size=batch_size,
            callbacks=callbacks, verbose=0)
        
        # Evaluates the model using validation data:
        y_val_raw, y_pred_raw = predict_validation(model, X_val, y_val, y_scaler)

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
    Execute the PINN grid search tuning workflow.

    This function loads the aerodynamic dataset, airfoil geometry, and effective angle-of-attack reference
    data, evaluates all combinations of batch size and kernel initializer, saves the full tuning results,
    and extracts the Pareto-optimal configurations.

    Parameters
    ----------
    batch_sizes : list
        List of batch sizes to evaluate.

    initializers : list
        List of kernel initializers to evaluate.

    Returns
    -------
    None
        The function does not return any value.
    """

    # --- Defines the data path ---
    print('\nLoading dataset into memory...\n')
    data_path = 'utils/LowFidelity-PressureDistributionData.csv'
    airfoil_data_path = 'utils/NACA23015.csv'
    aoa_effective_path = 'utils/LowFidelity-AoAEffectiveData.csv'

    # --- Defines the airfoil data ---
    airfoil_data = pd.read_csv(airfoil_data_path, sep=',', names=['x', 'y'])
    panel_dx = np.diff(airfoil_data['x'].values).astype(np.float32)
    panel_dy = np.diff(airfoil_data['y'].values).astype(np.float32)
    AoA_effective = pd.read_csv(aoa_effective_path, sep=';')

    # --- Load and split the data ---
    data_train, _ = load_and_split_data(data_path, AoA_effective)
    
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
            data_train, airfoil_data, AoA_effective, panel_dx, panel_dy)
                
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

if __name__ == '__main__':
    main(BATCH_SIZES, INITIALIZERS)