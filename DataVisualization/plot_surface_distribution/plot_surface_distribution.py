# -*- coding: utf-8 -*-
"""
========================================================================================================
Module: plot_lift_distribution
Author: Caio Dias Filho
Creation date: 2026-02-18
Last modification: 2026-02-18
Version: 1.0
========================================================================================================

OVERVIEW
--------
    
DEPENDENCIES
-----------

INPUT FILES
-----------    

OUTPUT FILES
-----------

NOTES
------

========================================================================================================
"""

# =======================================================================================================
#                                            USER INPUT SECTION
# =======================================================================================================

# --- Reynolds number definition ---
Re = 100000
AoA = 10

# =======================================================================================================
#                                              CORE FUNCTION
# =======================================================================================================

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import csv
from scipy.interpolate import RectBivariateSpline
from matplotlib.colors import TwoSlopeNorm
import matplotlib.cm as cm
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from tkinter import filedialog
import tkinter as tk


def select_data_file():
    """
    Open a graphical file selection dialog to choose the Optuna trials dataset.

    This function uses Tkinter to prompt the user to select a .csv file containing the results of a
    hyperparameter optimization study.

    Returns
    -------
    str
        Absolute path to the selected file.

    Side Effects
    ------------
    - Opens a graphical file dialog window.
    - Requires a GUI-enables environment.

    Notes
    -----
    - The selected file must correspond to an exported Optuna trials dataset.
    - The function does not validate the file content.
    """

    # Open a file selection dialog:
    root = tk.Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename(title="Select the trials data file", 
        filetypes=[("CSV Files", "*.csv")])

    return file_path



def plot_surface_distribution(data_path: str):
    """
    Generate and export publication-quality lift distribution plots (single- and double-column) for a 
    selected Reynolds number.

    This function reads aerodynamic dataset, filters the data for the Reynolds number defined in the USER
    INPUT SECTION, selects representative angles of attack, reconstructs the full-span lift distribution
    by mirroring half-span data, and generates two scientific-style figures:
        1) Single-column format
        2) Double-column format

    Each visualization includes:
        - Sectional lift coefficient distribution (Cl vs. y/b)
        - Symmetric full-span reconstruction
        - Angle-of-attack annotations
        - Journal-fourmatted typography and layout
        - Grid styling consistent with publication standards

    Parameters
    ----------
    data_path : str
        Path to the .csv aerodynamic dataset.

    Returns
    -------
    None
        The function saves two .pdf figures (single- and double-column).

    Side Effects
    ------------
    - Reads aerodynamic data from disk.
    - Filters dataset based on the global Reynolds number.
    - Generates two high-resolution .pdf figures (600 dpi).
    - Prints confirmation messages to console.

    Computational Cost
    ------------------
    Approximately proportional to:
        N_data_points x N_selected_AoA

    Notes
    -----
    - The Reynolds number is controlled by the global variable 'Re'.
    - If the dataset does not contain the selected Reynolds number, the resulting plots will be empty.
    - The half-span data are mirrored using array concatenation to reconstruct the full-span distribution.
    - Single-column width follows ~3.35 in standard.
    - Double-column width follows ~6.7 in standard.
    - .pdf output preserves vector graphics quality.
    """

    # ===================================================================================================
    # 1. SINGLE-COLUMN VISUALIZATION
    # ===================================================================================================
    # Setting plot parameters:
    with plt.rc_context({'font.family': 'serif', 'font.serif': ['Times New Roman'], 
        'mathtext.fontset': 'stix'}):

        # --- Data ---
        with open(data_path, 'r', encoding='utf-8') as file:
            reader = csv.reader(file, delimiter=';')
            og_columns = next(reader)

        data = pd.read_csv(data_path, sep=';', skiprows=1, header=None)
        data.columns = og_columns

        filtered_cond = data[(data['Re'] == Re) & (data['AoA'] == AoA)]

        airfoil = pd.read_csv('plot_surface_distribution/utils/NACA23015.dat', sep=',', names = ['x', 'y'])
        x_sectional = (airfoil['x'].values)*0.2165
        z_sectional = (airfoil['y'].values)*0.15

        y_orig = filtered_cond['y'].values
        cp_data_orig = filtered_cond.iloc[:,4:].to_numpy()

        sort_idx = np.argsort(y_orig)
        y_orig_sorted = y_orig[sort_idx]
        cp_data_sorted = cp_data_orig[sort_idx, :]

        idx_real = np.linspace(0, 1, cp_data_orig.shape[1])
        idx_target = np.linspace(0, 1, len(x_sectional))

        y_target = np.linspace(y_orig_sorted.min(), 0.766, 120)

        f_interp_2d = RectBivariateSpline(y_orig_sorted, idx_real, cp_data_sorted)
        cp_data = f_interp_2d(y_target, idx_target).T

        X = np.tile(x_sectional, (len(y_target), 1)).T
        Y = np.tile(y_target, (len(x_sectional), 1))
        Z = np.tile(z_sectional, (len(y_target), 1)).T

        fig = plt.figure(figsize=(6.7, 3))
        ax = fig.add_subplot(111, projection='3d')
        ax.set_box_aspect([0.2165, 0.766, 0.022])

        #texto_info = f"Re = {100000:.2e}\n$\\alpha$ = {12}°"
        #ax.text2D(0.05, 0.90, texto_info, transform=ax.transAxes, 
                #fontsize=12, color='black',
                #bbox=dict(facecolor='white', alpha=0.8, edgecolor='lightgray', boxstyle='round,pad=0.5'))

        ax.set_axis_off()

        norm = TwoSlopeNorm(vmin = cp_data.min(), vcenter=0, vmax=1)
        cmap = cm.RdYlBu_r
        face_colors = cmap(norm(cp_data))

        ax.plot_surface(X, Y, Z, facecolors=face_colors, edgecolor='k', linewidth=0.05, shade=False)

        v_root = np.array([X[:,0], Y[:, 0], Z[:, 0]]).T
        ax.add_collection3d(Poly3DCollection([v_root], facecolors='darkgray', edgecolors='black', alpha=1.0))
        v_tip = np.array([X[:,-1], Y[:, -1], Z[:, -1]]).T
        ax.add_collection3d(Poly3DCollection([v_tip], facecolors='darkgray', edgecolors='black', alpha=1.0))

        m = (cm.ScalarMappable(cmap=cmap, norm=norm))
        m.set_array([])
        cbar = plt.colorbar(m, ax=ax, shrink=0.5, aspect=12)
        cbar.set_label('Pressure Coefficient ($C_p$)', labelpad=10, fontname='Times New Roman', fontsize=9)

        ax.view_init(elev=30, azim=-130)
            
        # --- Save ---
        plt.tight_layout(pad=0.6)
        plt.savefig('plot_surface_distribution/lift_distribution_single.pdf', dpi=600)
        plt.close()
        print(f'\nPlot saved as plot_lift_distribution/lift_distribution_single.pdf.')


        return









def main():
    """
    Execute the lift distribution visualization workflow.

    This function coordinates the complete aerodynamic post-processing pipeline:
        1) Prompt the user to select the aerodynamic dataset (.csv).
        2) Filter data for the predefined Reynolds number.
        3) Generate both single-column and double-column lift distribution figures.
        4) Save the resulting figures in .pdf format.

    Returns
    -------
    None

    Side Effects
    ------------
    - Opens a graphical file selection dialog.
    - Reads aerodynamic dataset from disk.
    - Saves two .pdf figures to disk.
    - Prints execution status messages.

    Notes
    -----
    - The Reynolds number analyzed is defined in the USER INPUT SECTION.
    - The output files are saved inside:
        plot_lift_distribution/
    - Intended for aerodynamic post-processing and manuscript preparation. 
    - The script assumes symmetric wing geometry when mirroring the data.
    """

    # --- Prompt user to select dataset ---
    data_path = select_data_file()

    # --- Generate lift distribution figure ---
    plot_surface_distribution(data_path)

    return

if __name__ == "__main__":
    main()