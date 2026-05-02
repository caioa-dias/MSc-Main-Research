# -*- coding: utf-8 -*-
"""
========================================================================================================
Module: experimental_data_final_assembly
Author: Caio Dias Filho
Creation date: 2026-04-28
Last modification: 2026-04-28
Version: 2.0 (final)
========================================================================================================

OVERVIEW
--------
This module performs the final assembly and reconstruction of the experimental aerodynamic dataset from
previously post-processed partial files generated for different sensor configurations.

The pipeline consolidates all partial datasets into a unified database, applies operating-condition
filtering , aligns the experimental data with the low-fidelity referece discretization, and reconstructs
sectional lift coefficients using physics-based integration.

Two datasets are generated:

    1. Sparse experimental dataset:
        - Original pressure tap measurements after filtering
        - Preserves experimental resolution

    2. Interpolated dataset:
        - Pressure distributions mapped to low-fidelity chordwise discretization
        - Enables consistency with data-driven and multi-fidelity models

The final interpolated dataset contains:

    - Reynolds number (Re)
    - Angle of attack (AoA)
    - Spanwise position (y)
    - Sectional lift coefficient (Cl)
    - Surface pressure coefficient distribution (Cp)

The sectional lift coefficient is computed by integrating the pressure distribution using an effective 
angle of attack obtained from a low-fidelity reference dataset.


WORKFLOW
--------
The implemented workflow consists of:

    - Reading all processed partial datasets
    - Concatenating all sensor configurations into a unified dataframe
    - Sorting the dataset by Reynolds number, angle of attack, and spanwise position
    - Renaming Cp columns using chordwise coordinates
    - Applying operating-condition filters for linear aerodynamic regime consistency
    - Exporting the sparse experimental dataset
    - Interpolating Cp distributions to match low-fidelity discretization (PCHIP)
    - Searching the closest effective angle-of-attack reference case
    - Computing sectional lift coefficient from Cp distributions
    - Appending cl values to the dataset
    - Exporting the final interpolated high-fidelity dataset

        
DEPENDENCIES
------------
Python libraries:

    - pandas
    - numpy
    - scipy
    - glob
    - os

External files:

    - Post-processed partial datasets (.csv)
    - Airfoil geometry file (.csv)
    - Effective angle-of-attack reference dataset (.csv)
    - Low-fidelity pressure distribution dataset (.csv)


OUTPUT FILES
------------
Sparse dataset:

    - HighFidelity-Aviation-Sparse-PressureDistributionData.csv

Final interpolated dataset:

    - HighFidelity-Aviation-PressureDistributionData.csv


ASSUMPTIONS
-----------
The implementation assumes that:

    - All partial datasets share the same structure and are correctly post-processed.
    - Each aerodynamic case corresponds to a unique combination of Re, AoA, and y.
    - Chordwise pressure taps follow a consistent ordering across datasets.
    - The low-fidelity dataset provides a valid reference discretization.
    - The effective angle-of-attack dataset adequately covers the required conditions.
    - Nearest-neighbor matching in (Re, AoA, y) space provides a valid approximation for AoA_effective.
    - Airfoil geometry is consistent with the Cp distributions.


LIMITATIONS
-----------
Potential limitations include:

    - Final dataset quality depends on the quality of the input partial datasets.
    - Nearest-case matching may introduce approximation errors in sparse regions.
    - Interpolation may smooth localized pressure features.
    - Linear-regime filtering is based on predefined thresholds.
    - The workflow assumes a fixed chordwise discretization.

========================================================================================================
"""

# =======================================================================================================
#                                            USER INPUT SECTION
# =======================================================================================================

# --- Postprocessed data folder name ---
data_folder_path = 'pipeline-output'

# --- Airfoil data file path ---
airfoil_data_path = 'utils/NACA23015.csv'

# --- Effective angle of attack file path ---
AoA_effective_data_path = 'utils/LowFidelity-AoAEffectiveData.csv'

# --- Low-fidelity dataset file path ---
low_fidelity_data_path = 'utils/LowFidelity-PressureDistributionData.csv'

# --- Sparse experimental dataset output file path ---
high_fidelity_sparse_data_path = 'utils/HighFidelity-Aviation-Sparse-PressureDistributionData.csv'

# =======================================================================================================
#                                              CORE FUNCTION
# =======================================================================================================

from scipy.interpolate import PchipInterpolator
import pandas as pd
import numpy as np
import glob
import os

def interpolate_pressure_dataset(high_fidelity_data_path: str, low_fidelity_data_path: str):
    """
    Interpolate high-fidelity pressure coefficient distributions to a target discretization.

    This function maps experimental high-fidelity Cp distributions to the chordwise discretization used
    in the low-fidelity dataset. The interpolation is performed separately on the upper and lower airfoil
    surfaces using a shape-preserving Piecewise Cubic Hermite Interpolating Polynomial (PCHIP).

    The procedure ensures:

        - Preservation of physical trends in Cp distributions;
        - Consistency between high-fidelity and low-fidelity datasets;
        - Compatibility for multi-fidelity learning frameworks.

    Parameters
    ----------
    high_fidelity_data_path : str
        Path to the high-fidelity pressure distribution dataset.

    low_fidelity_data_path : str
        Path to the low-fidelity dataset used as interpolation reference.

    Returns
    -------
    data_interpolated : pd.DataFrame
        Dataset containing high-fidelity Cp distributions interpolated to the low-fidelity chordwise 
        discretization, preserving Re, AoA, and y values.
    """

    def read_x_from_header(filepath: str, start_idx: int):
        """
        Extract chordwise coordinates from dataset headers.

        This function reads the first line of a dataset file and extracts the chordwise positions (x/c)
        encoded in the column headers, starting from a given index.

        Parameters
        ----------
        filepath : str
            Path to the dataset file.

        start_idx : int
            Index from which the Cp columns begin.

        Returns
        ------
        x_coords: numpy.ndarray
            Array containing the chordwise positions.
        """

        # --- Reads the file without converting into a pandas dataframe ---
        with open(filepath, 'r', encoding='utf-8') as f:
            header_line = f.readline().strip()

        # --- Extracts the pressure coefficient distribution column headers ---
        headers_raw = header_line.split(';')
        x_coords = np.array(headers_raw[start_idx:], dtype=float)

        return x_coords
    
    # --- Reads the physical x/c coordinates directly from original headers ---
    x_high_fidelity = read_x_from_header(high_fidelity_data_path, 5)
    x_low_fidelity = read_x_from_header(low_fidelity_data_path, 4)

    # --- Loads the datasets as dataframes ---
    high_fidelity_data = pd.read_csv(high_fidelity_data_path, sep=';')
    low_fidelity_data = pd.read_csv(low_fidelity_data_path, sep=';')

    # --- Identifies pressure coefficient columns from the pandas-loaded datasets ---
    high_fidelity_cp_cols = list(high_fidelity_data.columns[5:])
    low_fidelity_cp_cols = list(low_fidelity_data.columns[4:])

    # --- Identifies the leading edge position ---
    le_idx_high_fidelity = np.argmin(x_high_fidelity)
    le_idx_low_fidelity = np.argmin(x_low_fidelity)

    # --- Splits the original high fidelity data into upper and lower surfaces ---
    x_high_fidelity_upper = x_high_fidelity[:le_idx_high_fidelity + 1]
    x_high_fidelity_lower = x_high_fidelity[le_idx_high_fidelity:]

    upper_sort_idx = np.argsort(x_high_fidelity_upper)
    lower_sort_idx = np.argsort(x_high_fidelity_lower)

    x_high_fidelity_upper = x_high_fidelity_upper[upper_sort_idx]
    x_high_fidelity_lower = x_high_fidelity_lower[lower_sort_idx]

    # --- Splits the target low fidelity data into upper and lower surfaces ---
    x_low_fidelity_upper = x_low_fidelity[:le_idx_low_fidelity + 1]
    x_low_fidelity_lower = x_low_fidelity[le_idx_low_fidelity:]

    # --- Defines the matrix to store the interpolated data ---
    cp_interp_matrix = []

    # --- Interpolates the high fidelity data ---
    for _, row in high_fidelity_data.iterrows():

        # Extract the cp values:
        cp_high_fidelity = row[high_fidelity_cp_cols].astype(float).values

        # Splits into upper and lower surfaces pressure coefficient:
        cp_high_fidelity_upper = cp_high_fidelity[:le_idx_high_fidelity + 1]
        cp_high_fidelity_lower = cp_high_fidelity[le_idx_high_fidelity:]

        # Sorts the upper and lower surfaces pressure coefficient values:
        cp_high_fidelity_upper = cp_high_fidelity_upper[upper_sort_idx]
        cp_high_fidelity_lower = cp_high_fidelity_lower[lower_sort_idx]

        # Interpolates the upper surface:
        upper_interpolator = PchipInterpolator(x_high_fidelity_upper, cp_high_fidelity_upper, 
            extrapolate=True)

        # Interpolates the lower surface:
        lower_interpolator = PchipInterpolator(x_high_fidelity_lower, cp_high_fidelity_lower, 
            extrapolate=True)

        # Creates the interpolated pressure coefficient vector:
        cp_interp = np.empty_like(x_low_fidelity, dtype=float)

        # Interpolates the high fidelity data to the low fidelity resolution:
        cp_interp[:le_idx_low_fidelity + 1] = upper_interpolator(x_low_fidelity_upper)
        cp_interp[le_idx_low_fidelity:] = lower_interpolator(x_low_fidelity_lower)

        cp_interp_matrix.append(cp_interp)

    # --- Creates the interpolated dataframe ---
    cp_interp_matrix = np.asarray(cp_interp_matrix)
    cp_interp_data = pd.DataFrame(cp_interp_matrix, columns=low_fidelity_cp_cols)
    data_interpolated = pd.concat([high_fidelity_data[['Re', 'AoA', 'y']].reset_index(drop=True), 
        cp_interp_data], axis=1)
    
    return data_interpolated

def find_effective_angle_case(AoA_effective_data_path: str, Re_target: float, AoA_target: float, 
    y_target: float):
    """
    Retrieve the effective angle of attack for a given aerodynamic condition.

    This function identifies the closest matching flow case from the effective angle-of-attack dataset using 
    a standarized Euclidean distance based on:

        - Reynolds number (Re)
        - Angle of attack (AoA)
        - Spanwise position (y)
    
    The selected effective angle of attack is used for consistent aerodynamic force reconstruction.

    Parameters
    ----------
    AoA_effective_data_path : str
        Path to the effective angle-of-attack dataset.

    Re_target : float
        Target Reynolds number.

    AoA_target : float
        Target angle of attack, in degrees.

    y_target : float
        Target spanwise position.

    Returns
    -------
    AoA_effective : float
        Effective angle of attack associated with the closest matching flow case.
    """

        # --- Loads the low fidelity effective angle of attack data ---
    AoA_effective_data = pd.read_csv(AoA_effective_data_path, sep=';')

    # --- Evaluates the standardized euclidean distance to find the closes flow case ---
    distance = np.sqrt(((AoA_effective_data['AoA'] - AoA_target) / AoA_effective_data['AoA'].std())**2 +
                       ((AoA_effective_data['Re'] - Re_target) / AoA_effective_data['Re'].std())**2 +
                       ((AoA_effective_data['y'] - y_target) / AoA_effective_data['y'].std())**2)
    
    # --- Locates and saves the closes flow case ---
    closest_case = AoA_effective_data.loc[distance.idxmin()]
    
    # --- Converts the case to a numeric data array ---
    closest_case = np.array(pd.to_numeric(closest_case, errors='coerce'))

    # --- Saves the AoA effective value for this case ---
    AoA_effective = closest_case[4]

    return AoA_effective

def compute_cl_from_cp(cp_data: np.ndarray, airfoil_data_path: str, AoA_effective: float):
    """
    Compute sectional lift coefficient from a pressure coefficient distribution.

    This function integrates the pressure coefficient distribution over the airfoil surface panels to 
    obtain normal and axial force coefficients, which are then projected into the lift direction using
    the effective angle of attack.

    Parameters
    ----------
    cp_data : numpy.ndarray
        Pressure coefficient distribution along the airfoil surface.

    airfoil_data_path : str
        Path to the airfoil geometry file.

    AoA_effetive : float
        Effective angle of attack, in degrees.

    Returns 
    -------
    cl : float
        Sectional lift coefficient reconstructed from the Cp distribution.
    """

    # --- Loads the airfoil data ---
    airfoil_data = pd.read_csv(airfoil_data_path, sep=',', names=['x', 'y'])

    # --- Converts all variables to arrays ---
    x = np.asarray(airfoil_data['x'].values)
    y = np.asarray(airfoil_data['y'].values)
    cp = np.asarray(cp_data)

    # --- Calculates panel geometric deltas (dx, dy) ---
    panel_dx = np.diff(x)
    panel_dy = np.diff(y)

    # --- Computes the average pressure coefficient (Cp) for each panel ---
    panel_cp = 0.5 * (cp[:-1] + cp[1:])

    # --- Integrates to find normal (Cn) and axial (Ca) force coefficients ---
    Cn = np.sum(panel_cp * panel_dx)
    Ca = -np.sum(panel_cp * panel_dy)

    # --- Projects Cn and Ca into the lift coefficient (Cl) using the angle of attack ---
    alpha = np.radians(AoA_effective)
    Cl = Cn * np.cos(alpha) - Ca * np.sin(alpha)

    return Cl

def main(data_folder_path: str):
    """
    Execute the final experimental dataset assembly pipeline.

    This function performs the complete dataset reconstruction workflow:

        - Reads all post-processed partial datasets.
        - Concatenates all configurations into a unified dataset.
        - Applies hierarchical sorting (Re, AoA, y).
        - Renames Cp columns using chordwise coordinates.
        - Filters operating conditions to match the linear aerodynamic regime.
        - Saves the sparse experimental dataset.
        - Interpolates Cp distributions to low-fidelity discretization.
        - Computes sectional lift coefficients from Cp distributions.
        - Appends cl values to the dataset.
        - Exports the final high-fidelity aerodynamic dataset.

    Parameters
    ----------
    data_folder_path : str
        Path to the folder containing the processed partial datasets.

    Returns
    -------
    None
        The function saves the assembled dataset to disk and does not return any value.
    """

    # --- Prompts the user ---
    print('\nStarting data final assembly pipeline...\n')

    # --- Reads all the post processed data files and concatenates them ---
    data_files = glob.glob(os.path.join(data_folder_path, '*.csv'))
    raw_data = [pd.read_csv(f) for f in data_files]
    pressure_data = pd.concat(raw_data, ignore_index=True)

    # --- Sorts hierarchically the data by Re, AoA, and y ---
    pressure_data = pressure_data.sort_values(by=['Re', 'AoA', 'y'], 
        ascending=[True, True, True]).reset_index(drop=True)
    
    # --- Renames the headers for the final configuration ---
    chordwise_pos = np.array([0.91685912, 0.82101617, 0.72623557, 0.63353349, 0.5439261, 0.45842956, 
        0.37792148, 0.30337182, 0.23551963, 0.17510393, 0.12281755, 0.07926097, 0.04489607, 0.02004619, 
        0.00503464, 0, 0.00503464, 0.02004619, 0.04489607, 0.07926097, 0.12281755, 0.17510393, 0.23551963, 
        0.30337182, 0.37792148, 0.45842956, 0.5439261, 0.63353349, 0.72623557, 0.82101617, 0.91685912])
    pressure_data.columns = ['Re', 'AoA', 'y', 'CL', 'CD'] + list(np.round(chordwise_pos, 5))

    # --- Applies the filter for the linear lift curve region, mathcing the low fidelity analysis ---
    # Converts the Re and AoA data to numeric:
    pressure_data['Re'] = pd.to_numeric(pressure_data['Re'], errors='coerce')
    pressure_data['AoA'] = pd.to_numeric(pressure_data['AoA'], errors='coerce')

    # Defines the filter:
    filter = (((pressure_data["Re"] <= 120000) & (pressure_data["AoA"].between(-4, 9))) |
        ((pressure_data["Re"] > 120000) & (pressure_data["Re"] <= 300000) & 
        (pressure_data["AoA"].between(-4, 10))) | ((pressure_data["Re"] > 300000) & 
        (pressure_data["Re"] <= 560000) & (pressure_data["AoA"].between(-4, 11))) | 
        (pressure_data["Re"] > 560000))

    # Applies the filter:
    pressure_data = pressure_data[filter].reset_index(drop=True)

    # --- Saves the sparse experimental pressure coefficient data ---
    pressure_data.to_csv(high_fidelity_sparse_data_path, sep=';', index=False) 

    # --- Interpolates the high fidelity pressure coefficient data to the low fidelity discretization---
    pressure_data = interpolate_pressure_dataset(high_fidelity_sparse_data_path, low_fidelity_data_path)

    # --- Appends the lift coefficient for each case to the dataset ---
    cl_values = []
    for _, case in pressure_data.iterrows():
        cp = case[3:].astype(float).values
        AoA_effective = find_effective_angle_case(AoA_effective_data_path, case['Re'], case['AoA'], 
            case['y'])
        cl = compute_cl_from_cp(cp, airfoil_data_path, AoA_effective)
        cl_values.append(cl)
    pressure_data.insert(3, 'cl', cl_values)
    #pressure_data.drop(columns=['CL', 'CD'], inplace=True)

    # --- Exports the final dataset ---
    pressure_data.to_csv('HighFidelity-Aviation-PressureDistributionData.csv', index=False, sep=';')
    print('Final experimental dataset saved as HighFidelity-Aviation-PressureDistributionData.csv\n')
    
    return

if __name__ == '__main__':
    main(data_folder_path)