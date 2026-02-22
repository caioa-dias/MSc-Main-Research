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
This module generates publication-quality spanwise lift distribution visualizations (sectional lift
coefficient (Cl) vs. normalized chordwise position (y/b)) for a given Reynolds number (Re).

For each dataset, the script produces:
    1) A single-column paper layout figure
    2) A double-column paper layout figure

Both figures follow scientific journal formatting standards and are exported in vector-based .pdf format, 
suitable for direct inclusion in academic manuscripts.

The workflow consists of:
    - Selecting a .csv aerodynamic dataset
    - Filtering the data by a predefined Reynold number
    - Selecting representative angles of attack
    - Mirroring the half-spa data to reconstruct full-span distribution
    - Generating publication-ready figures
    - Exporting single- and double-column .pdf visualizations

    
DEPENDENCIES
-----------
Python libraries:
    - matplotlib
    - tkinter
    - pandas 
    - numpy 


INPUT FILES
-----------
- .csv aerodynamic dataset containing at least the following columns:
    - 'Re': Reynolds number
    - 'AoA': Angle of attack (degrees)
    - 'y': normalized chordwise position (half-span)
    - 'cl': sectional lift coefficient
    

OUTPUT FILES
-----------
Two figures are generated ans saved inside:

    plot_lift_distribution/
    - lift_distribution_single.pdf
    - lift_distribution_double.pdf

Figures are exported in .pdf format (600 dpi), preserving vector graphic quality for publication.


NOTES
------
- The Reynolds number is defined in the USER INPUT SECTION.
- Only data corresponding to the selected Reynolds number are plotted.
- The half-span data are mirrored to reconstruct a symmetric full-span distribution.
- Selected angles of attack are subsampled for visual clarity.
- The script does not perform aerodynamic computations; it only visualizes existing results.
- For aerodynamic data generated using the vortex-lattice solver from AVL (Athena Vortex Lattice), the 
  sectional lift coefficient (Cl) does not vary eith Reynolds number. Consequently, changing Re in the 
  USER INPUT SECTION will not modify the lift distribution results unless the dataset itself was
  generared from different aerodynamic models.

========================================================================================================
"""

# =======================================================================================================
#                                            USER INPUT SECTION
# =======================================================================================================

# --- Reynolds number definition ---
Re = 1000000


# =======================================================================================================
#                                              CORE FUNCTION
# =======================================================================================================

import matplotlib.pyplot as plt
from tkinter import filedialog
import tkinter as tk
import pandas as pd
import numpy as np

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

def plot_lift_distribution(data_path: str):
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
        data = pd.read_csv(data_path, sep=';')
        # Filters by a single Reynolds number:
        data_Re = data[data['Re'] == Re]
        # Selects specific angles of attack:
        AoAs = sorted(data_Re['AoA'].unique())
        AoAs_plot = AoAs[::3]
        # Adds the final angle of attack:
        if AoAs[-1] not in AoAs_plot:
            AoAs_plot.append(AoAs[-1])

        # --- Figure size: single-column ---
        fig, ax = plt.subplots(figsize=(3.35, 2.55))

        # --- Plot ---
        for AoA in AoAs_plot:
            # Filters for a specific angle of attack:
            group = data_Re[data_Re['AoA'] == AoA]
            y = group['y'].values
            cl = group['cl'].values
            # Mirrors the plot:
            y = np.concatenate((-y[::-1], y))
            cl = np.concatenate((cl[::-1], cl))
            # Plots the lift distribution:
            ax.plot(y, cl, color='#000000', linewidth=0.7)
            # Plots the AoA annotation:
            plt.text(0, cl[np.argmin(np.abs(y))]+0.0005, f'$\\alpha$ = {AoA}°', fontsize=5, ha='center', va='bottom')
            # Plots the Reynolds number annotation:
            mantissa, exponent = f"{Re:.1e}".split("e")
            exponent = int(exponent)
            re_text = r"$Re_c = {:.1f} \times 10^{{{}}}$".format(float(mantissa), exponent)
            ax.text(-0.95, 1.68, re_text, fontsize=7, ha='left', va='top')
                     
        # --- Labels ---
        ax.set_xlabel(f'Normalized spanwise position ($y/b$)', fontsize=9, fontname='Times New Roman')
        ax.set_ylabel(f'Sectional lift coefficient ($C_l$)', fontsize=9, fontname='Times New Roman')

        # --- Ticks ---
        ax.tick_params(axis='both', which='major', labelsize=7, width=0.8, direction='in')

        # --- Grid ---
        ax.grid(True, which='major', linestyle='--', linewidth=0.5, alpha=0.4)

        # --- Limits ---
        ax.set_xlim([-1, 1])
        ax.set_ylim([-0.15, 1.75])

        # --- Save ---
        plt.tight_layout(pad=0.6)
        plt.savefig('plot_lift_distribution/lift_distribution_single.pdf', dpi=600)
        plt.close()
        print(f'\nPlot saved as plot_lift_distribution/lift_distribution_single.pdf.')


    # ===================================================================================================
    # 2. DOUBLE-COLUMN VISUALIZATION
    # ===================================================================================================
    # Setting plot parameters:
    with plt.rc_context({'font.family': 'serif', 'font.serif': ['Times New Roman'], 
        'mathtext.fontset': 'stix'}):

        # --- Figure size: single-column ---
        fig, ax = plt.subplots(figsize=(6.7, 4.8))

        # --- Plot ---
        for AoA in AoAs_plot:
            # Filters for a specific angle of attack:
            group = data_Re[data_Re['AoA'] == AoA]
            y = group['y'].values
            cl = group['cl'].values
            # Mirrors the plot:
            y = np.concatenate((-y[::-1], y))
            cl = np.concatenate((cl[::-1], cl))
            # Plots the lift distribution:
            ax.plot(y, cl, color='#000000', linewidth=1.6)
            # Plots the AoA annotation:
            ax.text(0, cl[np.argmin(np.abs(y))]+0.0005, f'$\\alpha$ = {AoA}°', fontsize=10, ha='center', va='bottom')
            # Plots the Reynolds number annotation:
            mantissa, exponent = f"{Re:.1e}".split("e")
            exponent = int(exponent)
            re_text = r"$Re_c = {:.1f} \times 10^{{{}}}$".format(float(mantissa), exponent)
            ax.text(-0.95, 1.68, re_text, fontsize=10, ha='left', va='top')
                     
        # --- Labels ---
        ax.set_xlabel(f'Normalized spanwise position ($y/b$)', fontsize=11, fontname='Times New Roman')
        ax.set_ylabel(f'Sectional lift coefficient ($C_l$)', fontsize=11, fontname='Times New Roman')

        # --- Ticks ---
        ax.tick_params(axis='both', which='major', labelsize=10, width=0.8, direction='in')

        # --- Grid ---
        ax.grid(True, which='major', linestyle='--', linewidth=0.6, alpha=0.4)

        # --- Limits ---
        ax.set_xlim([-1, 1])
        ax.set_ylim([-0.15, 1.75])

        # --- Save ---
        plt.tight_layout(pad=0.6)
        plt.savefig('plot_lift_distribution/lift_distribution_double.pdf', dpi=600)
        plt.close()
        print(f'\nPlot saved as plot_lift_distribution/lift_distribution_double.pdf.\n')

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
    plot_lift_distribution(data_path)

    return

if __name__ == "__main__":
    main()