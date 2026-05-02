# -*- coding: utf-8 -*-
"""
========================================================================================================
Module: plot_tuning_results
Author: Caio Dias Filho
Creation date: 2026-04-06
Last modification: 2026-04-06
Version: 1.0
========================================================================================================

OVERVIEW
--------
This module generates publication-ready visualizations for the analysis of hyperparameter tuning results
obtained through grid search.

The module focuses on the visualization of the Pareto front associated with a multi-objective evaluation
framework, where each trained model is assessed based on:

    - Model validation Mean Squared Error (MSE)
    - Global lift coefficient reconstruction Mean Squared Error (CL MSE)

The resulting plot highlights:

    - All evaluated hyperparameter combinations
    - Pareto-optimal solutions
    - A selected best trade-off solution

This visualization supports the identification of optimal congurations by explicitly showing the trade-off
between predictive accuracy and aerodynamic consistency.


DEPENDENCIES
------------
Python libraries:
    - matplotlib
    - pandas


INPUT FILES
-----------
Required input files:

    - optimization-plots/data_optimization_results/<study_name>/tuning_results.csv
    - optimization-plots/data_optimization_results/<study_name>/tuning_pareto_results.csv

These files must contain:

    - MSE_loss : validation mean squared error
    - CL_loss : global lift coefficient reconstruction mean squared error

    
OUTPUT FILES
------------
Generated figures:

    - optimization-plots/plot_tuning_results/<study_name>_tuning_pareto_front_<layout>.png


NOTES
-----
- Only valid (non-NaN) trials are considered for visualization.
- Pareto-optimal solutions are precomputed and loadad from CSV files.
- The selected best solution is assumed to be the first entry in the Pareto dataset.
- The plot uses a fixed axis range for consistent comparison across studies.
- The module supports single-column and double-column publication layouts.

========================================================================================================
"""

# =======================================================================================================
#                                            USER INPUT SECTION
# =======================================================================================================

# --- Tuning data information ---
study_name = 'mlp_ann_study'


# =======================================================================================================
#                                              CORE FUNCTION
# =======================================================================================================

from matplotlib import pyplot as plt
import pandas as pd

def plot_pareto_front(study_name:str, layout:str):
    """
    Plot the Pareto front of the hyperparameter tuning results.

    This function loads all evaluated hyperparameter tuning results.

    This function loads all evaluated hyperparameter combinations and the subset of Pareto-optimal
    solutions, and generates a scatter plot illustrating the trade-off between validation MSE and CL
    reconstruction MSE.

    The plot distinguishes:

        - All evaluated models
        - Pareto-optimal solutions
        - Selected best trade-off solution

    Parameters
    ----------
    study_name : str
        Name of the tuning study, used to locate the input CSV files.

    layout : str
        Plot layout configuration. Accepted values are:
            - 'single_column'
            - 'double_column'

    Returns
    -------
    None
        Th function saves the generated figure to disk and does not return any value.
    """

    # --- Plot layout configurations ---
    plot_layout = {'single_column': {'width': 3.35, 'height': 2.55, 'main_lw': 0.8, 'sec_lw': 0.4, 
            'scatter': 10, 'label_fs': 8, 'lp': 5, 'sec_fs': 6, 'tick': 0.4, 'grid_alpha': 0.2}, 
        'double_column': {'width': 6.7, 'height': 4.8, 'main_lw': 1.2, 'sec_lw': 0.6, 'scatter': 20, 
            'label_fs': 19, 'lp': 10, 'sec_fs': 15, 'tick': 0.5, 'grid_alpha':0.3}}
    
    cfg = plot_layout[layout]
    # --- Setting figure size:
    width, height = cfg['width'], cfg['height']
    # --- Setting plot element size:
    main_lw, sec_lw, scatter = cfg['main_lw'], cfg['sec_lw'], cfg['scatter']
    # --- Setting text font size:
    label_fs, sec_fs = cfg['label_fs'], cfg['sec_fs']
    # --- Setting aestetic parameters:
    lp, tick, grid_alpha = cfg['lp'], cfg['tick'], cfg['grid_alpha']

    # --- Process the data ---
    all_trials = pd.read_csv(f'optimization-plots/data_optimization_results/{study_name}/tuning_results.csv')
    all_trials = all_trials.dropna(subset=['MSE_loss'])
    pareto_trials = pd.read_csv(f'optimization-plots/data_optimization_results/{study_name}/tuning_pareto_results.csv')
    best_trial = pareto_trials.iloc[0,:]

    # Setting plot parameters:
    with plt.rc_context({'font.family': 'serif', 'font.serif': ['Times New Roman'],
        'mathtext.fontset': 'stix'}):

        # --- Figure size: single-column ---
        fig, ax = plt.subplots(figsize=(width, height))

        # --- Plot ---
        plot = ax.scatter(all_trials['MSE_loss'], all_trials['CL_loss'], facecolors='none', s=scatter,
            alpha=0.8, edgecolors='#000000', linewidth=0.3, zorder=5, label='All evaluated models')
        plot = ax.scatter(pareto_trials['MSE_loss'], pareto_trials['CL_loss'], facecolors='#B22222', s=1.5*scatter,
            alpha=1.0, edgecolors='#000000', linewidth=0.3, zorder=5, label='Pareto-optimal solutions')
        plot = ax.scatter(best_trial['MSE_loss'], best_trial['CL_loss'], facecolors='#FFFFFF', s=1.5*scatter,
            alpha=1.0, edgecolors='#FFFFFF', linewidth=0.3, zorder=6)
        plot = ax.scatter(best_trial['MSE_loss'], best_trial['CL_loss'], facecolors='#B22222', s=4*scatter,
            marker='*', alpha=1.0, edgecolors='#000000', linewidth=0.3, zorder=7, label='Selected solution (best trade-off)')

        # --- Labels ---
        ax.set_xlabel('Model validation MSE', fontsize=label_fs, fontname='Times New Roman', labelpad=lp)
        ax.set_ylabel(r'$C_L$ reconstruction MSE', fontsize=label_fs, fontname='Times New Roman', labelpad=lp)
        
        # --- Ticks ---
        for spine in ax.spines.values():
            spine.set_linewidth(0.5)
        ax.tick_params(axis='both', which='major', labelsize=sec_fs, width=tick, direction='in')
        ax.ticklabel_format(axis='both', style='sci', scilimits=(0,0), useMathText=True)
        ax.xaxis.get_offset_text().set_fontsize(sec_fs)
        ax.yaxis.get_offset_text().set_fontsize(sec_fs)

        # --- Grid ---
        ax.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=grid_alpha)

        # --- Limits ---
        ax.set_xlim(0.0000075, 0.000014)
        ax.set_ylim(0, 0.0000004)

        # --- Legend ---
        legend = ax.legend(fontsize=sec_fs, fancybox=False, edgecolor='#000000', loc='upper right',
            markerscale=1.2)
        if layout == 'single_column':
            legend.get_frame().set_linewidth(0.5)

        # --- Save plot ---
        plt.tight_layout(pad=0.6)
        plt.savefig(f'optimization-plots/plot_tuning_results/{study_name}_tuning_pareto_front_{layout}.png', dpi=600)
        plt.close()
        print(f'Saved: optimization-plots/plot_tuning_results/{study_name}_tuning_pareto_front_{layout}.png\n')
                    
    return

def main(study_name:str):
    """
    Execute the tuning results visualization workflow.

    This function generates the Pareto front visualization for the specified tuning study.

    Parameters
    ----------
    study_name : str
        Name of the tuning study.

    Returns
    -------
    None
        The function does not return any value.
    """
    # --- Inform the start of the visualization process ---
    print(f"\nStarting tuning results visualization process...\n")

    # --- Generate pareto front plot ---
    plot_pareto_front(study_name, 'double_column')

    return

if __name__ == '__main__':
    main(study_name=study_name)