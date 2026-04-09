# -*- coding: utf-8 -*-
"""
========================================================================================================
Module: plot_parameter_impact
Author: Caio Dias Filho
Creation date: 2026-02-11
Last modification: 2026-03-26
Version: 1.1
========================================================================================================

OVERVIEW
--------
This module generates publication-ready plots to evaluate the impact of individual hyperparameters on the
objective function obtained during the multi-objective optimization of the low-fidelity models.

For a selected hyperparameter, the module loads an Optuna study and produces boxplots showing how the
distribution of each optimization objective varies across the tested values of that parameter.

The analyzed objectives are:
    - Validation Mean Squared Error (validation_MSE)
    - Global lift coefficient reconstruction Mean Squared Error (lift_coefficient_reconstruction_MSE)

The plots are intended to support post-optimization analysis and help identify hyperparameter values
associated with better objective values.


DEPENDENCIES
------------
Python libraries:

    - matplotlib
    - seaborn
    - pandas
    - numpy
    - optuna


INPUT FILES
-----------
Optimization database:
    - SQLite database containing the Optuna study results.

Required inputs:
    - study_name
    - storage
    - param_name

    
OUTPUT FILES
------------
Generated figures:
    - optimization-plots/plot_parameter_impact/validation_MSE/<param_name>_impact_<layout>.png
    - optimization-plots/plot_parameter_impact/lift_coefficient_reconstruction_MSE/
    <param_name>_impact_<layout>.png


NOTES
-----
- Only completed Optuna trials are used in the analysis. 
- The selected hyperparameter must exist in the optimization study.
- The current implementation generates boxplots for two objectives separately.
- The layout configuration supports single-columns and double-columns paper layouts.
- Integer-valued architectural parameters are explicitly converted before plotting.

========================================================================================================
"""

# =======================================================================================================
#                                            USER INPUT SECTION
# =======================================================================================================

# --- Parameters available for MLP-ANN ---
# ['activation', 'learning_rate', 'n_hidden_layers', 'optimizer', 'units_layer1', 'units_layer2',
# 'units_layer3', 'units_layer4', 'units_layer5', 'units_layer6']

# --- Parameters available for MHP-ANN ---
# ['activation_cl', 'activation_cp', 'activation_shared', 'cl_loss_weight', 'cl_units_layer1', 
# 'cl_units_layer2', 'cl_units_layer3', 'cl_units_layer4', 'cp_loss_weight', 'cp_units_layer1',
# 'cp_units_layer2', 'cp_units_layer3', 'cp_units_layer4', 'learning_rate', 'n_cl_hidden_layers',
# 'n_cp_hidden_layers', 'n_shared_hidden_layers', 'optimizer', 'shared_units_layer1', 'shared_units_layer2',
# 'shared_units_layer3', 'shared_units_layer4'] 
#

# --- Parameter to plot ---
param_name = 'units_layer6'
upper_bounds = [1e-4, 1e-6]

# --- Optimization study information ---
study_name = 'mlp_ann_study'
storage = 'sqlite:///optimization-plots/data_optimization_results/mlp_ann_study.db'


# =======================================================================================================
#                                              CORE FUNCTION
# =======================================================================================================

import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import optuna

def plot_parameter_impact_by_objective(study: optuna.study.Study, objective:str, param_name:str, 
    upper_bounds:list, layout:str):
    """
    Plot the impact of a selected hyperparameter on one optimization objective.

    This function extracts completed Optuna trials, retrieves the values of a selected hyperparameter and
    the corresponding objective values, and generates a boxplot to visualize how the objective distribution
    varies across the tested values of that parameter.

    The function supports two objectives:
        - validation_MSE
        - lift_coefficient_reconstruction_MSE

    The plot is formatted according to the selected layout and saved to disk.

    Parameters
    ----------
    study : optuna.study.Study
        Loaded Optuna study containing the optimization trials.

    objective : str
        Objective to be plotted. Accepted values are:
            - 'validation_MSE'
            - 'lift_coefficient_reconstruction_MSE'

    param_name : str
        Name of the hyperparameter to be analyzed.

    upper_bounds : list
        Upper bounds used for the vertical axis limits in the plots, in the format:
            [validation_MSE_upper_bound, lift_coefficient_reconstruction_MSE_upper_bound]

    layout : str
        Plot layout configuration. Accepted values are:
            - 'single_column'
            - 'double_column'

    Returns
    -------
    None
        The function saves the plot to disk and does not return any value.
    """

   # --- Define layout-specific plotting parameters ---
    plot_layout = {'single_column': {'width': 3.35, 'height': 2.55, 'main_lw': 0.8, 'sec_lw': 0.4, 
            'scatter': 5, 'label_fs': 12, 'lp': 5, 'sec_fs': 10, 'tick': 0.4, 'grid_alpha': 0.2,
            'mean_size': 3, 'outlier_size': 1}, 
        'double_column': {'width': 6.7, 'height': 4.8, 'main_lw': 1.2, 'sec_lw': 0.6, 'scatter': 10, 
            'label_fs': 18, 'lp': 10, 'sec_fs': 14, 'tick': 0.5, 'grid_alpha':0.3, 'mean_size': 6, 
            'outlier_size': 2}}
    
    cfg = plot_layout[layout]

    # --- Setting figure size ---
    width, height = cfg['width'], cfg['height']
    # --- Setting plot element size ---
    main_lw, sec_lw, scatter = cfg['main_lw'], cfg['sec_lw'], cfg['scatter']
    mean_size, outlier_size = cfg['mean_size'], cfg['outlier_size']
    # --- Setting text font size ---
    label_fs, sec_fs = cfg['label_fs'], cfg['sec_fs']
    # --- Setting aesthetic parameters ---
    lp, tick, grid_alpha = cfg['lp'], cfg['tick'], cfg['grid_alpha']

    # --- Define readable labels for each hyperparameter name ---
    naming_dict = {'activation': 'Activation function', 'learning_rate': 'Learning rate', 
        'n_hidden_layers': 'Number of hidden layers', 'optimizer': 'Optimizer', 
        'units_layer1': '1st layer width', 'units_layer2': '2nd layer width',
        'units_layer3': '3rd layer width', 'units_layer4': '4th layer width', 
        'units_layer5': '5th layer width', 'units_layer6': '6th layer width',
        'activation_cl': r'$C_L$ head activation function', 'activation_cp': r'$C_P$ head activation function',
        'activation_shared': 'Shared layers activation function', 'cl_loss_weight': r'$C_L$ head loss weight',
        'cl_units_layer1': r'$C_L$ head 1st layer width', 'cl_units_layer2': r'$C_L$ head 2nd layer width',
        'cl_units_layer3': r'$C_L$ head 3rd layer width', 'cl_units_layer4': r'$C_L$ head 4th layer width',
        'cp_loss_weight': r'$C_P$ head loss weight', 'cp_units_layer1': r'$C_P$ head 1st layer width',
        'cp_units_layer2': r'$C_P$ head 2nd layer width', 'cp_units_layer3': r'$C_P$ head 3rd layer width',
        'cp_units_layer4': r'$C_P$ head 4th layer width', 'n_cl_hidden_layers': r'Number of $C_L$ head hidden layers',
        'n_cp_hidden_layers': r'Number of $C_P$ head hidden layers', 'n_shared_hidden_layers': 'Number of shared hidden layers',
        'shared_units_layer1': 'Shared 1st layer width', 'shared_units_layer2': 'Shared 2nd layer width',
        'shared_units_layer3': 'Shared 3rd layer width', 'shared_units_layer4': 'Shared 4th layer width'}

    # --- Process the data ---
    trials = [t for t in study.trials if t.state.name == "COMPLETE"]
    trial_parameter = np.array([t.params.get(param_name) for t in trials])
    if objective == 'validation_MSE':
        obj = np.array([t.values[0] for t in trials])
    elif objective == 'lift_coefficient_reconstruction_MSE':
        obj = np.array([t.values[1] for t in trials])
    trial_data = pd.DataFrame({param_name: trial_parameter, objective: obj}).dropna()

    # Setting plot parameters:
    with plt.rc_context({'font.family': 'serif', 'font.serif': ['Times New Roman'],
        'mathtext.fontset': 'stix'}):

        # --- Figure size: single-column ---
        fig, ax = plt.subplots(figsize=(width, height))

        # --- Ticks ---
        for spine in ax.spines.values():
            spine.set_linewidth(0.5)
        if param_name in ['n_hidden_layers', 'units_layer1', 'units_layer2', 'units_layer3', 'units_layer4',
            'units_layer5', 'units_layer6', 'units_layer7']:
            trial_data[param_name] = trial_data[param_name].astype(int)
        ax.tick_params(axis='both', which='major', labelsize=sec_fs, width=tick, direction='in')
        ax.tick_params(axis='both', which='minor', labelsize=sec_fs, width=tick, direction='in')
        ax.ticklabel_format(axis='both', style='sci', scilimits=(0, 0), useMathText=True)
        ax.xaxis.get_offset_text().set_fontsize(sec_fs)
        ax.yaxis.get_offset_text().set_fontsize(sec_fs)

        # --- Plot ---
        sns.boxplot(data=trial_data, x=param_name, y=objective, width=0.4, linewidth=0.5, showmeans=True,
            boxprops={'facecolor': 'lightgray', 'edgecolor': 'black', 'alpha': 0.7},
            medianprops={'color': 'black', 'linewidth': main_lw},
            whiskerprops={'color': 'black', 'linewidth': sec_lw},
            capprops={'color': 'black', 'linewidth': sec_lw},
            flierprops={'marker': 'o', 'mec': 'black', 'markersize': outlier_size, 'mew': 0.5},
            meanprops={'marker': 's', 'mfc': 'white', 'mec': 'black' , 'markersize': mean_size, 'mew': 0.5})

        # --- Labels ---
        ax.set_xlabel(naming_dict[param_name], fontsize=label_fs, fontname='Times New Roman', labelpad=lp)
        if objective == 'validation_MSE':
            ax.set_ylabel('Model validation MSE', fontsize=label_fs, fontname='Times New Roman', labelpad=lp)
        elif objective == 'lift_coefficient_reconstruction_MSE':
            ax.set_ylabel(r'$C_L$ reconstruction MSE', fontsize=label_fs, fontname='Times New Roman', labelpad=lp)

        # --- Grid ---
        ax.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=grid_alpha)

        # --- Limits ---
        if objective == 'validation_MSE':
            ax.set_ylim(0, upper_bounds[0])
        elif objective == 'lift_coefficient_reconstruction_MSE':
            ax.set_ylim(0, upper_bounds[1])

        # --- Legend ---
        legend_handles = [mlines.Line2D([], [], color='black', linewidth=1.0, label='Median'),
            mlines.Line2D([], [], color='white', marker='s', mfc='white', mec='black', markersize=mean_size, label='Mean')]
        legend = ax.legend(handles=legend_handles, loc='upper right', fontsize=sec_fs, edgecolor='#000000', fancybox=False)

        if layout == 'single_column':
            legend.get_frame().set_linewidth(0.5)

        # --- Save plot ---
        plt.tight_layout(pad=0.6)
        plt.savefig(f'optimization-plots/plot_parameter_impact/{objective}/{param_name}_{objective}_impact_{layout}.png', dpi=600)
        plt.close()
        print(f'Saved: optimization-plots/plot_parameter_impact/{objective}/{param_name}_{objective}_impact_{layout}.png\n')

    return

def main(study_name:str, storage:str, param_name:str):
    """
    Execute the hyperparameter impact visualization workflow.

    This function loads the Optuna study from the specified storage backend and generates boxplots 
    showing the impact of the selected hyperparameter on both optimization objectives.

    Parameters
    ----------
    study_name : str
        Name of the Optuna study.

    storage : str
        Database connection string used to access the Optuna study.

    param_name : str
        Name of the hyperparameter to be analyzed.

    Returns
    -------
    None
        The function does not return any value.
    """

    # --- Inform the start of the visualization process ---
    print(f"\nStarting {param_name} impact visualization process...\n")

    # --- Load the Optuna study from storage --- 
    study = optuna.load_study(study_name=study_name, storage=storage)

    # --- Generate the plots ---
    plot_parameter_impact_by_objective(study, 'validation_MSE', param_name, upper_bounds, 'single_column')
    plot_parameter_impact_by_objective(study, 'lift_coefficient_reconstruction_MSE', param_name, 
        upper_bounds, 'single_column')

    return

if __name__ == "__main__":
    main(study_name=study_name, storage=storage, param_name=param_name)