# -*- coding: utf-8 -*-
"""
========================================================================================================
Module: potential_convergence_visualization
Author: Caio Dias Filho
Creation date: 2026-02-23
Last modification: 2026-02-25
Version: 1.0
========================================================================================================

OVERVIEW
--------
This module visualizes the convergence behavior of the potential flow simulations by comparing the
calculated sectional lift coefficient (Cl) integrated from pressure coefficient (Cp) distributions 
against the true target Cl values.


WORKFLOW
--------
The complete visualization pipeline consists of:

    1) Load unconverged and converged aerodynamic datasets (.csv)
    2) Load reference airfoil geometry data
    3) Compute Cl from Cp integration across all observations
    4) Evaluate the absolute error between predicted and true Cl values
    5) Generate a structured, publication-ready scatter plot visualizing convergence threshold

    
DEPENDENCIES
------------
Python libraries:
    - matplotlib
    - tqdm
    - pandas
    - numpy

    
OUTPUT FILES
------------
Figures (600 DPI, publication-ready):
    - potential_convergence_visualization.pdf

    
ASSUMPTIONS
------------
- The datasets 'Potential-UnconvergedData.csv' and 'Potential-PressureDistributionData-Filtered.csv' 
  are available in the current working directory.
- The airfoil geometry data 'utils/NACA23015.csv' is available in the current working directory.
- Data format matches preprocessing expectations.

========================================================================================================
"""

# =======================================================================================================
#                                              CORE FUNCTION
# =======================================================================================================

from matplotlib import pyplot as plt
import matplotlib.ticker as ticker
from tqdm import tqdm
import pandas as pd
import numpy as np

def compute_cl_from_cp(cp_data:np.ndarray, airfoil_data:pd.DataFrame, AoA:float):

    # 1. Calculte panel geometric deltas (dx, dy):
    panel_dx = np.diff(airfoil_data['x'].values)
    panel_dy = np.diff(airfoil_data['y'].values)

    # 2. Compute the average Cp for each panel:
    panel_cp = (cp_data[:-1] + cp_data[1:])/2

    # 3. Integrate to find normal (Cn) and axial (Ca) force coefficients:
    airfoil_cn = np.sum(panel_cp * panel_dx)
    airfoil_ca = -np.sum(panel_cp * panel_dy)

    # 4. Project Cn and Ca into the lift coefficient (Cl) using the angle of attack:
    calc_cl = airfoil_cn * np.cos(np.radians(AoA)) - airfoil_ca * np.sin(np.radians(AoA))

    return calc_cl

def evaluate_absolute_error(dataset:pd.DataFrame, airfoil_data:pd.DataFrame):

    # 1. Initialize the results list:
    results = []
    
    # 2. Iterate through all the observations in the dataset to compute errors:
    for i in tqdm(range(len(dataset)), desc="Evaluating Cl absolute error...", unit="case"):
        # Extract paramaters for the current case:
        cp_values = dataset.iloc[i, 4:].values
        Re = dataset.iloc[i, 0]
        AoA = dataset.iloc[i, 1]
        y = dataset.iloc[i, 2]
        true_cl = dataset.iloc[i, 3]

        # Recompute Cl and calculate the absolute error:
        calc_cl = compute_cl_from_cp(cp_values, airfoil_data, AoA)
        abs_error = np.abs(true_cl - calc_cl)

        results.append([Re, AoA, y, true_cl, calc_cl,abs_error])

    # 3. Assemble and return the results DataFrame:
    results_df = pd.DataFrame(results, columns=['Re', 'AoA', 'y', 'true_cl', 'calc_cl', 'absolute_error'])

    return results_df

def generate_convergence_plot(unconverged_results: pd.DataFrame, converged_results: pd.DataFrame):
    """
    Generate and save a publication-ready scatter plot visualing the convergence regions.

    Parameters
    ----------
    unconverged_results : pd.DataFrame
        DataFrame containing the validation metrics for unconverged cases.
        Must contain 'AoA' and 'absolute_error' columns.

    converged_results : pd.DataFrame
        DataFrame containing the validation metrics for converged cases.
        Must contain 'AoA' and 'absolute_error' columns.

    Returns
    -------
    None

    Side Effects
    ------------
    - Creates and configures a Matplotlib figure object in memory.
    - Writes a high-resolution image file (PDF, 600 DPI) to the local disk.
    - Prints status messages to the standard output.
    """

    print('\nGenerating convergence visualization plot...\n')

    # Setting plot parameters:
    with plt.rc_context({'font.family': 'serif', 'font.serif': ['Times New Roman'],
    'mathtext.fontset': 'stix'}):

        # --- Figure size: double column ---
        fig, ax = plt.subplots(figsize=(6.7, 4.8))

        # --- Plot converged vs unconverged solutions ---
        ax.scatter(converged_results['AoA'], converged_results['absolute_error'], alpha=1, s=8, 
            facecolors='#4682B4', edgecolors='#000000', label='Converged Solutions', linewidth=0.25)
        ax.scatter(unconverged_results['AoA'], unconverged_results['absolute_error'], alpha=1, s=8, 
            facecolors='#DC143C', edgecolors='#000000', label='Unconverged Solutions', linewidth=0.25, marker='s')

        # --- Plot convergence thresholds and regions ---
        ax.axhspan(1e-7, 0.0095, facecolor='#228B22', alpha=0.08, label='Converged Region')
        ax.axhspan(0.0095, 10, facecolor='#DC143C', alpha=0.08, label='Unconverged Region')
        ax.hlines(0.01, xmin=-5, xmax=19, linewidth=0.4, color='k', linestyle='--')

        # --- Labels ---
        ax.set_xlabel(r'Angle of Attack ($AoA$) [$^\circ$]', fontsize=11, fontname='Times New Roman')
        ax.set_ylabel(r'Absolute Error $(|\Delta C_l|)$', fontsize=11, fontname='Times New Roman')

        # --- Ticks ---
        ax.tick_params(axis='both', which='major', labelsize=9, width=0.8, direction='in')
        ax.yaxis.set_major_locator(ticker.LogLocator(base=10.0, numticks=10))
        ax.yaxis.set_minor_locator(ticker.LogLocator(base=10.0, subs='auto', numticks=10))
        ax.set_yscale('log')

        # --- Grid ---
        ax.grid(True, which='major', linestyle='--', linewidth=0.6, alpha=0.4, color='gray')
        ax.grid(True, which='minor', linestyle='--', linewidth=0.3, alpha=0.2, color='gray')

        # --- Limits ---
        ax.set_xlim([-5, 19])
        ax.set_ylim([1e-7, 1.5])

        # --- Legend ---
        ax.legend(fontsize=9, fancybox=False, edgecolor='black', loc='upper left', markerscale=1.5)
        ax.text(18.8, 0.011, 'Threshold = 0.01', fontsize=9, fontname='Times New Roman', ha='right', fontweight='bold')
        
        # --- Save ---
        plt.tight_layout(pad=0.6)
        plt.savefig('potential_convergence_visualization.pdf', dpi=600)
        plt.close()
        print('Plot saved as "potential_convergence_visualization.pdf"\n')

    return

def main():
    """
    Execute the full convergence visualization workflow.

    Workflow
    --------
        1) Define file paths and configuration variables.
        2) Safely load unconverged, converged, and airfoil datasets into memory.
        3) Evaluate Cl absolute errors for both aerodynamic datasets.
        4) Trigger the generation of the convergence visualization plot.

    Parameters
    ----------
    None

    Returns
    -------
    None

    Side Effects
    ------------
    - Reads multiple input datasets (.csv) from the local filesystem.
    - Prints execution progress to the standard output.
    - Calls the 'generate_convergence_plot' function, which results in disk write operations.
    """

    # --- Workflow configuration ---
    FILE_UNCONVERGED = 'Potential-UnconvergedData.csv'
    FILE_CONVERGED = 'Potential-PressureDistributionData-Filtered.csv'
    FILE_AIRFOIL = 'utils/NACA23015.csv'

    # --- Data loading ---
    print('\nLoading datasets into memory...\n')
    unconverged_df = pd.read_csv(FILE_UNCONVERGED, sep=';')
    converged_df = pd.read_csv(FILE_CONVERGED, sep=';')
    airfoil_data = pd.read_csv(FILE_AIRFOIL, sep=',', names=['x','y'])

    # --- Error evaluation ---
    print('Evaluating Cl absolute errors for the unconverged results...')
    unconverged_results = evaluate_absolute_error(unconverged_df, airfoil_data)
    unconverged_results = unconverged_results.dropna().reset_index(drop=True)
    print('\nEvaluating Cl absolute errors for the converged results...')
    converged_results = evaluate_absolute_error(converged_df, airfoil_data)
    converged_results = converged_results.dropna().reset_index(drop=True)

    # --- Plot generation ---
    generate_convergence_plot(unconverged_results, converged_results)

    return

if __name__ == "__main__":
    main()