# -*- coding: utf-8 -*-
"""
========================================================================================================
Module: boxplot_hyperparameter_importance
Author: Caio Dias Filho
Creation date: 2026-02-11
Last modification: 2026-02-11
Version: 1.0
========================================================================================================

OVERVIEW
--------
This module generates publication-quality .pdf boxplot visualizations to evaluate the impact of individual
hyperparameters on model performance during hyperparameter optimization.

For each selected hyperparameter, the script produces:

    1) A double-column paper layout figure
    2) A single-column paper layout figure

Both figures follow scientific journal formatting standards and are exported in vector-based .pdf format for
direct inclusion in academic manuscripts.

The workflow consists of:

    - Loading an exported Optuna trials dataset
    - Filtering completed trials
    - Extracting the selected hyperparameter and objective value
    - Generating statistical boxplot visualizations
    - Exporting single- and double-column publication-ready figures


DEPENDENCIES
-----------
Python libraries:
    - matplotlib
    - tkinter
    - seaborn
    - pandas


INPUT FILES
-----------
- .csv file exported from an Optuna study containing:
    - Trial state information
    - Hyperparameter values (prefixed with 'params_')
    - Objective function value ('value')


OUTPUT FILES
-----------
For each analyzed hyperparameter:
    - Single-column paper layout .pdf figure
    - Double-column paper layout .pdf figure

Saved inside:
    boxplot_hyperparameter_importance/

Figures are exported in .pdf format (600 dpi), preserving vector graphics quality suitable for publication.
    

NOTES
------
- Only trials with state == 'COMPLETE' are considered.
- The y-axis is formatted in scientific notation.
- Figure dimensions follow journal single- and double-column standards.
- The script does not perform optimization; it analyzes existing results.

========================================================================================================
"""

# =======================================================================================================
#                                            USER INPUT SECTION
# =======================================================================================================

# --- Parameter to Plot ---
# Parameter options are: 'activation', 'n_layers', 'reg_type', 'optimizer', 'batch_size , 'learning_rate', 
# 'units_layer1', 'units_layer2', 'units_layer3', 'units_layer4', 'units_layer5', 'units_layer6', 
# 'units_layer7', 'dropout_rate', 'weight_initializer'
parameter = 'n_layers'

# --- Upper bound for y-axis ---
upper_bound = 0.5e-2


# =======================================================================================================
#                                              CORE FUNCTION
# =======================================================================================================

from matplotlib.ticker import FormatStrFormatter
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog
import seaborn as sns
import pandas as pd

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

def plot_parameter_impact(data_path: str, param_name: str, x_label: str, save_path:str, upper_bound: float):
    """
    Generate and export publication-quality .pdf boxplots (single- and double-column) showing the impact of
    a selected hyperparameter on validation loss.

    This function reads the Optuna trials dataset, filters completed trials, extracts the specified
    hyperparameter and objective values, and generates two scientific-style boxplot figures:

        1) Single-column format
        2) Double-column format

    Each visualization includes:

        - Median loss per hyperparameter value
        - Mean loss (diamond marker)
        - Interquartile range (IQR)
        - Whiskers and outliers
        - Scientific notation on the loss axis

    Parameters
    ----------
    data_path : str
        Path to the .csv file containing Optuna trials data.

    param_name : str
        Name of the hyperparameter to be analyzed.
        Must match the suffix used in 'params_<param_name>'.

    x_label : str
        Label displayed on the x-axis of the figure.

    save_path : str
        Base file path where the generated .pdf figures will be saved.

    upper_bound : float
        Upper limit of the y-axis (loss value).

    Returns
    -------
    None
        The function saves two .pdf figures (single- and double-column).

    Side Effects
    ------------
    - Reads trials data from disk.
    - Filters incomplete trials (state == 'COMPLETE').
    - Generates two high-resolution .pdf figures (600 dpi).
    - Prints confirmation message to console.

    Computational Cost
    ------------------
    Approximately proportional to:
        N_trials

    Notes
    -----
    - If the selected parameter is not found in the dataset, execution is safely terminated.
    - Single-column width typically follows ~3.35 in standard.
    - Double-column width typically follows ~7 in standard.
    - .pdf output preserves vector graphics quality.
    """

    # ===================================================================================================
    # 1. SINGLE-COLUMN VISUALIZATION
    # ===================================================================================================
    # Setting plot parameters:
    with plt.rc_context({'font.family': 'serif', 'font.serif': ['Times New Roman'], 
        'mathtext.fontset': 'stix'}):

        # --- Data ---
        data = pd.read_csv(data_path, sep=',')
        data = data[data['state'] == 'COMPLETE'].copy()
        col_param = f'params_{param_name}'
        if col_param not in data.columns:
            print(f'\nParameter {param_name} not found in the study.\n')
            return
        data_plot = data[[col_param, 'value']].dropna()

        # --- Figure size: single-column ---
        fig, ax = plt.subplots(figsize=(3.35, 2.55))

        # --- Ticks ---
        if param_name in ['n_layers', 'units_layer1', 'units_layer2', 'units_layer3', 'units_layer4',
                            'units_layer5', 'units_layer6', 'units_layer7']:
            data_plot[col_param] = data_plot[col_param].astype(int)
        ax.tick_params(axis='both', which='major', labelsize=7, width=0.8, direction='in')
        ax.tick_params(axis='both', which='minor', labelsize=7, width=0.8, direction='in')
        ax.ticklabel_format(axis='y', style='sci', scilimits=(0, 0), useMathText=True)
        ax.yaxis.get_offset_text().set_fontsize(7)

        # --- Plot ---
        sns.boxplot(data=data_plot, x=col_param, y='value', width=0.4, linewidth=0.6, showmeans=True,
            boxprops={'facecolor': 'lightgray', 'edgecolor': 'black', 'alpha': 0.7},
            medianprops={'color': 'black', 'linewidth': 1.0},
            whiskerprops={'color': 'black', 'linewidth': 0.6},
            capprops={'color': 'black', 'linewidth': 0.6},
            flierprops={'marker': 'o', 'mec': 'black', 'markersize': 1, 'mew': 0.5},
            meanprops={'marker': 's', 'mfc': 'white', 'mec': 'black', 'markersize': 3, 'mew': 0.5},)
    
        # --- Labels ---
        ax.set_xlabel(x_label, fontsize=9, fontname='Times New Roman')
        ax.set_ylabel('Mean validation loss (MSE)', fontsize=9, fontname='Times New Roman')

        # --- Grid ---
        ax.grid(True, which='major', axis='y', linestyle='--', linewidth=0.5, alpha=0.2)

        # --- Legend ---
        legend_handles = [mlines.Line2D([], [], color='black', linewidth=1.0, label='Median'),
            mlines.Line2D([], [], color='white', marker='s', mfc='white', mec='black', markersize=3, label='Mean')]
        plt.legend(handles=legend_handles, loc='upper right', fontsize=7, frameon=True, framealpha=1, edgecolor='black')

        # --- Limits ---
        ax.set_ylim(0, upper_bound)

        # --- Save ---
        plt.tight_layout(pad=0.6)
        plt.savefig(f'{save_path}_single.pdf', dpi=600)
        plt.close()
        print(f'\nPlot saved as {save_path}_single.pdf.')


    # ===================================================================================================
    # 2. DOUBLE-COLUMN VISUALIZATION
    # ===================================================================================================
    # Setting plot parameters:
    with plt.rc_context({'font.family': 'serif', 'font.serif': ['Times New Roman'], 
        'mathtext.fontset': 'stix'}):

        # --- Figure size: double-column ---
        fig, ax = plt.subplots(figsize=(6.7, 4.8))

        # --- Ticks ---
        if param_name in ['n_layers', 'units_layer1', 'units_layer2', 'units_layer3', 'units_layer4',
                            'units_layer5', 'units_layer6', 'units_layer7']:
            data_plot[col_param] = data_plot[col_param].astype(int)
        ax.tick_params(axis='both', which='major', labelsize=10, width=0.8, direction='in')
        ax.tick_params(axis='both', which='minor', labelsize=10, width=0.8, direction='in')
        ax.ticklabel_format(axis='y', style='sci', scilimits=(0, 0), useMathText=True)
        ax.yaxis.get_offset_text().set_fontsize(10)

        # --- Plot ---
        sns.boxplot(data=data_plot, x=col_param, y='value', width=0.4, linewidth=0.8, showmeans=True,
            boxprops={'facecolor': 'lightgray', 'edgecolor': 'black', 'alpha': 0.7},
            medianprops={'color': 'black', 'linewidth': 1.4},
            whiskerprops={'color': 'black', 'linewidth': 0.8},
            capprops={'color': 'black', 'linewidth': 0.8},
            flierprops={'marker': 'o', 'mec': 'black', 'markersize': 2, 'mew': 0.5},
            meanprops={'marker': 's', 'mfc': 'white', 'mec': 'black', 'markersize': 6, 'mew': 0.5},)
    
        # --- Labels ---
        ax.set_xlabel(x_label, fontsize=11, fontname='Times New Roman')
        ax.set_ylabel('Mean validation loss (MSE)', fontsize=11, fontname='Times New Roman')

        # --- Grid ---
        ax.grid(True, which='major', axis='y', linestyle='--', linewidth=0.5, alpha=0.2)

        # --- Legend ---
        legend_handles = [mlines.Line2D([], [], color='black', linewidth=1.0, label='Median'),
            mlines.Line2D([], [], color='white', marker='s', mfc='white', mec='black', markersize=6, label='Mean')]
        plt.legend(handles=legend_handles, loc='upper right', fontsize=10, frameon=True, framealpha=1, edgecolor='black')

        # --- Limits ---
        ax.set_ylim(0, upper_bound)

        # --- Save ---
        plt.tight_layout(pad=0.6)
        plt.savefig(f'{save_path}_double.pdf', dpi=600)
        plt.close()
        print(f'Plot saved as {save_path}_double.pdf.\n')

        return

def main(parameter: str, upper_bound: float):
    """
    Execute the hyperparameter impact visualization workflow.

    This function coordinates the complete post-optimization analysis pipeline:

        1) Prompt user to select the Optuna trials dataset.
        2) Identify the selected hyperparameter (defined globally).
        3) Generate both single-column and double-column boxplot figures.
        4) Save the resuting figures in .pdf format.

    Parameters
    ----------
    parameter : str
        Name of the hyperparameter to be analyzed.
        Must match the suffix used in 'params_<parameter>'.

    upper_bound : float
        Maximum value to be displayed on the y-axis.

    Returns
    -------
    None

    Side Effects
    ------------
    - Opens graphical file selection dialog.
    - Reads optimization results from disk.
    - Saves two .pdf figures to disk.
    - Prints execution status messages.

    Notes
    -----
    - The hyperparameter analyzed is defined in the global variable 'parameter'.
    - The y-axis upper limit is controlled by 'upper_bound'.
    - Output files are saved inside:
        boxplot_hyperparameter_importance/
    - Intended for scientific post-processing and manuscript preparation.
    """

    # --- Data ---
    data_path = select_data_file()

    # --- Parameter Selection ---
    parameter_map = {
        'activation': ('Activation function', 'activation_impact'),
        'n_layers': ('Network depth', 'depth_impact'),
        'reg_type': ('Regularization strategy', 'regularization_impact'),
        'optimizer': ('Optimizer', 'optimizer_impact'),
        'batch_size': ('Batch size', 'batch_impact'),
        'learning_rate': ('Learning rate', 'learning_impact'),
        'units_layer1': ('2nd layer width', '2ndlayer_impact'),
        'units_layer2': ('3rd layer width', '3rdlayer_impact'),
        'units_layer3': ('4th layer width', '4thlayer_impact'),
        'units_layer4': ('5th layer width', '5thlayer_impact'),
        'units_layer5': ('6th layer width', '6thlayer_impact'),
        'units_layer6': ('7th layer width', '7thlayer_impact'),
        'units_layer7': ('8th layer width', '8thlayer_impact'),
        'dropout_rate': ('Dropout rate', 'dropout_impact'),
        'weight_initializer': ('Weight initialization', 'initialization_impact')}

    if parameter in parameter_map:
        label, filename = parameter_map[parameter]
        plot_parameter_impact(data_path, parameter, label, 
            f'boxplot_hyperparameter_importance/{filename}', upper_bound)

    return

if __name__ == "__main__":
    main(parameter=parameter, upper_bound=upper_bound)