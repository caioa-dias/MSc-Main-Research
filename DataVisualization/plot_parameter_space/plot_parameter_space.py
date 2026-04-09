# -*- coding: utf-8 -*-
"""
========================================================================================================
Module: plot_parameter_space
Author: Caio Dias Filho
Creation date: 2026-02-25
Last modification: 2026-02-26
Version: 1.0
========================================================================================================

OVERVIEW
--------
This module generates publication-quality visualizations of the parameter space (sampling points) used
in aerodynamic studies, supporting multi-fidelity datasets.

It specifically visualizes the relationship between the Angle of Attack (AoA) and the Reynolds Number
(Re) to evaluate the distribution and density of the samples observations across different fidelity
levels.

For each dataset, the script produces:
    1) A single-column paper layout figure (3.35 in width)
    2) A double-column paper layout figure (6.7 in width)

Both figures follow scientific joutnal formatting standards and are exported in high-resolution formats
suitable for academic manuscripts.


DEPENDENCIES
------------
Python libraries:
    - matplotlib
    - tkinter
    - pandas

INPUT FILES
------------
- .csv dataset containing at least the following columns:
    - 'Re': Reynolds number
    - 'AoA': Angle of attack (degrees)


OUTPUT FILES
-----------
Two figures are generated ans saved inside:

    plot_parameter_space/
    - parameter_space_single.png (600 dpi)
    - parameter_space_double.png (600 dpi)

========================================================================================================
"""

# =======================================================================================================
#                                           USER INPUT SECTION
# =======================================================================================================

# --- Plot aesthetics ---
# Data label: 'Low-Fidelity Observations', 'Medium-Fidelity Observations', 'High-Fidelity Observations':
data_label = ['Low-fidelity flow cases']

# Marker color for different fidelities:
marker_color = ['#000080']

# Output file name:
output_file = 'low_fidelity_parameter_space'

# Control for multi-fidelity visualization (if more than one fidelity, add labels and colors on the list):
two_fidelities = False
three_fidelities = False



# =======================================================================================================
#                                              CORE FUNCTIONS
# =======================================================================================================

from matplotlib.ticker import MultipleLocator
import matplotlib.pyplot as plt
from tkinter import filedialog
import tkinter as tk
import pandas as pd
import matplotlib


def select_data_file():
    """
    Open graphical file selection dialogs to choose one or more aerodynamic datasets.

    This function uses Tkinter to prompt the user to select .csv files. Depending on the fidelity settings
    (two or three fidelities), it will open subsequent dialogs to colect all necessary data paths.

    Returns
    -------
    file_list : list
        A list of strings containing the absolute paths to the selected files.

    Side Effects
    ------------
    - Opens one or more graphical file dialog windows.
    - Requires a GUI-enables environment.
    """

    # Initialize the list to store file paths:
    file_list = []

    # Initialize Tkinter root once to avoid multiple instances:
    root = tk.Tk()
    root.withdraw()

    # 1. Select the mandatory primary dataset:
    path_1 = filedialog.askopenfilename(title="Select the primary dataset",
        filetypes=[("CSV Files", "*.csv")])
    file_list.append(path_1)

    # 2. Select a secondary dataset if two or three fidelities are enables:
    if (two_fidelities or three_fidelities) == True:
        path_2 = filedialog.askopenfilename(title="Select the secondary dataset",
            filetypes=[("CSV Files", "*.csv")])
        file_list.append(path_2)

    # 3. Select a tertiary dataset if three fidelities are enables:
    if three_fidelities == True:
        path_3 = filedialog.askopenfilename(title="Select the tertiary dataset",
            filetypes=[("CSV Files", "*.csv")])
        file_list.append(path_3)

    # Close the root window properly:
    root.destroy()

    return file_list

def plot_parameter_space(data_path: list):
    """
    Generate and export publication-quality parameter space plots (single- and double-column) supporting
    multi-fidelity datasets.

    This function reads one or more aerodynamic datasets and creates scatter plots showing the distribution
    of Angle of Attack (AoA) vs. Reynolds Number (Re). If multiple fidelities are enables via global flags,
    the function layers the observations using distinct marker styles to visualize the sampling density 
    across different fidelity levels.

    The function generates two scientific-style figures:
        1) Single-column format (3.35 in width).
        2) Double-column format (6.7 in width).

    Parameters
    ----------
    data_path : list
        List of absolute paths (str) to the .csv aerodynamic datasets.

    Returns
    -------
    None
        The function saves two .png figures to the 'plot_parameter_space/' directory.

    Side Effects
    ------------
    - Reads multiple aerodynamic datasets from disk.
    - Generates high-resolution visualizations (600 dpi).
    - Prints export confirmation messages to the console.

    Notes
    -----
    - Global variables 'two_fidelities' and 'three_fidelities' control the layering logic.
    - Marker aesthetics (colors and labels) are pulled from the 'marker_color' and 'data_label' global
    lists.
    - Formatting uses Times New Roman font and Stix fonts to meet journal standards.
    """

    # ===================================================================================================
    # 1. SINGLE-COLUMN VISUALIZATION
    # ===================================================================================================
    # Setting plot parameters:
    with plt.rc_context({'font.family': 'serif', 'font.serif': ['Times New Roman'], 
        'mathtext.fontset': 'stix'}):

        # --- Data ---
        dataset = pd.read_csv(data_path[0], sep=';')

        # --- Figure size: single-column ---
        fig, ax = plt.subplots(figsize=(3.35, 2.55))

        # --- Plot ---
        ax.scatter(dataset['AoA'], dataset['Re'], alpha=0.8, s=2, 
            facecolors=marker_color[0], edgecolors='#000000', linewidth=0.01, label=data_label[0])
        
        if (two_fidelities or three_fidelities) == True:
            ax.scatter(dataset['AoA'], dataset['Re'], alpha=0.8, s=2, 
                facecolors=marker_color[1], edgecolors='#000000', linewidth=0.01, label=data_label[1])
            
        if three_fidelities == True:
            ax.scatter(dataset['AoA'], dataset['Re'], alpha=0.8, s=2, 
                facecolors=marker_color[2], edgecolors='#000000', linewidth=0.01, label=data_label[2])

        # --- Labels ---
        ax.set_ylabel(r'Reynolds number ($Re$)', fontsize=8, fontname='Times New Roman', labelpad=10)
        ax.set_xlabel(r'Angle of attack ($\alpha$) [$^\circ$]', fontsize=8, fontname='Times New Roman', labelpad=10)

        # --- Ticks ---
        for spine in ax.spines.values():
            spine.set_linewidth(0.5)
        ax.tick_params(axis='both', which='major', labelsize=6, width=0.4, direction='in')
        ax.xaxis.set_major_locator(MultipleLocator(2))

        # --- Grid ---
        ax.grid(True, which='major', linestyle='--', linewidth=0.5, alpha=0.3)

        # --- Limits ---
        ax.set_xlim([-5, 13])
        ax.set_ylim([0, 1300000])
        formatter = matplotlib.ticker.ScalarFormatter(useMathText=True)
        formatter.set_scientific(True)
        formatter.set_powerlimits((5, 5))
        ax.yaxis.set_major_formatter(formatter)
        ax.set_yticks([0, 2e5, 4e5, 6e5, 8e5, 1e6])
        ax.yaxis.get_offset_text().set_fontsize(6)
        ax.yaxis.get_offset_text().set_y(1.02)

        # --- Legend ---
        legend = ax.legend(fontsize=6, fancybox=False,edgecolor='black', loc='upper right', markerscale=2.0)
        legend.get_frame().set_linewidth(0.5)

        # --- Save ---
        plt.tight_layout(pad=0.6)
        plt.savefig(f'plot_parameter_space/{output_file}_single.png', dpi=600)
        plt.close()
        print(f'\nPlot saved as plot_parameter_space/{output_file}_single.png.')

    
    # ===================================================================================================
    # 2. DOUBLE-COLUMN VISUALIZATION
    # ===================================================================================================
    # Setting plot parameters:
    with plt.rc_context({'font.family': 'serif', 'font.serif': ['Times New Roman'], 
        'mathtext.fontset': 'stix'}):

        # --- Figure size: single-column ---
        fig, ax = plt.subplots(figsize=(6.7, 4.8))

        # --- Plot ---
        ax.scatter(dataset['AoA'], dataset['Re'], alpha=0.8, s=6, 
            facecolors=marker_color[0], edgecolors='#000000', linewidth=0.1, label=data_label[0])
        
        if (two_fidelities or three_fidelities) == True:
            ax.scatter(dataset['AoA'], dataset['Re'], alpha=0.8, s=6, 
                facecolors=marker_color[1], edgecolors='#000000', linewidth=0.1, label=data_label[1])
            
        if three_fidelities == True:
            ax.scatter(dataset['AoA'], dataset['Re'], alpha=0.8, s=6, 
                facecolors=marker_color[2], edgecolors='#000000', linewidth=0.1, label=data_label[2])

        # --- Labels ---
        ax.set_xlabel(r'Angle of attack ($\alpha$) [$^\circ$]', fontsize=18, labelpad=10, fontname='Times New Roman')
        ax.set_ylabel(r'Reynolds number ($Re$)', fontsize=18, labelpad=10, fontname='Times New Roman')

        # --- Ticks ---
        ax.tick_params(axis='both', which='major', labelsize=14, width=0.8, direction='in')
        ax.xaxis.set_major_locator(MultipleLocator(2))

        # --- Grid ---
        ax.grid(True, which='major', linestyle='--', linewidth=0.6, alpha=0.3)

        # --- Limits ---
        ax.set_xlim([-5, 13])
        ax.set_ylim([0, 1300000])
        formatter = matplotlib.ticker.ScalarFormatter(useMathText=True)
        formatter.set_scientific(True)
        formatter.set_powerlimits((5, 5))
        ax.yaxis.set_major_formatter(formatter)
        ax.set_yticks([0, 2e5, 4e5, 6e5, 8e5, 1e6])
        ax.yaxis.get_offset_text().set_fontsize(14)
        ax.yaxis.get_offset_text().set_y(1.02)

        # --- Legend ---
        ax.legend(fontsize=16, fancybox=False,edgecolor='black', loc='upper right', markerscale=2.0, handletextpad=0.1)

        # --- Save ---
        plt.tight_layout(pad=0.6)
        plt.savefig(f'plot_parameter_space/{output_file}_double.png', dpi=600)
        plt.close()
        print(f'\nPlot saved as plot_parameter_space/{output_file}_double.png.\n')
    
    return

def main():
    """
    Execute the parameter space visualization workflow.

    This function coordinates the complete aerodynamic post-processing pipeline:
        1) Prompt the user to select one or more aerodynamic datasets (.csv) based on the defined 
        fidelity levels.
        2) Load and validate the file paths for low, medium and high fidelity data.
        3) Generate both single-column and double-column parameter space figures (AoA vs. Re).
        4) Save the resulting high-resolution figures in .png format.

    Returns
    -------
    None

    Side Effects
    ------------
    - Opens one or more graphical file selection dialogs depending on fidelity settings.
    - Reads aerodynamic datasets from disk.
    - Saves two images (.png) to disk.
    - Prints execution status and file paths to the console.

    Notes
    -----
    - The number of files requested depends on 'two_fidelitites' and 'three_fidelities' flags
    in the USER INPUT SECTION.
    - Output files are saved inside:
        plot_parameter_space/
    - Intended for aerodynamic sampling analysis and manuscript preparation.
    """

    # --- Prompt user to select dataset(s) ---
    data_path_list = select_data_file()
    print('\nData successfully loaded.')

    # --- Generate lift distribution figures ---
    plot_parameter_space(data_path_list)

    return

if __name__ == "__main__":
    main()