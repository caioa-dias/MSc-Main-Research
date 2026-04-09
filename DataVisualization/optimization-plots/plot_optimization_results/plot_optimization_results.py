# -*- coding: utf-8 -*-
"""
========================================================================================================
Module: plot_optimization_results
Author: Caio Dias Filho
Creation date: 2026-03-25
Last modification: 2026-04-04
Version: 1.0
========================================================================================================

OVERVIEW
--------
This module generates publication-ready plots to analyze the results of the multi-objective hyperparameter
optimization performed for the low-fidelity models.

The module loads an Optuna study from a SQLite database and produces:
    - Hyperparameter importance plots for each optimization objective
    - Optimization history scatter plot in objective space

The analyzed objectives are:
    - Model validation Mean Squared Error (validation_MSE)
    - Global lift coefficient reconstruction Mean Squared Error (lift_coefficient_reconstruction_MSE)

These visualizations are intended to support post-optimization analysis by highliting which hyperparameters
have the greatest influence on each objective and by showing the distribution of completed trials in the bi-
objective space.


DEPENDENCIES
------------
Python libraries:
    - matplotlib
    - pandas
    - seaborn
    - optuna
    - numpy


INPUT FILES
-----------
Required input:
    - SQLite database containing the Optuna study

User-defined inputs:
    - study_name
    - storage

    
OUTPUT FILES
------------
Generated figures:
    - optimization-plots/plot_optimization_results/<study_name>_validation_MSE_<layout>.png
    - optimization-plots/plot_optimization_results/<study_name>_lift_coefficient_reconstruction_MSE
    _<layout>.png
    - optimization-plots/plot_optimization_results/<study_name>_history_<layout>.png


NOTES
-----
- Only completed Optuna trials are considered in the plots.
- Hyperparameter importance is computed using the fANOVA evaluator.
- The optimization history is displayed in the objective space, with trial number represented by the
  color scale.
- The script currently generates figures in single-column format by default.
- The SQLite study database must be available locally before execution.

========================================================================================================
"""

# =======================================================================================================
#                                            USER INPUT SECTION
# =======================================================================================================

# --- Optimization Study Information ---
study_name = 'mlp_ann_study'
storage = 'sqlite:///optimization-plots/data_optimization_results/mlp_ann_study.db'


# =======================================================================================================
#                                              CORE FUNCTION
# =======================================================================================================

import matplotlib
from matplotlib import pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import optuna

def plot_importance_by_objective(study:optuna.study.Study, objective:str, layout:str):
    """
    Plot the hyperparameter importance for a selected optimization objective.

    This function computes the relative importance of the hyperparameters using Optuna's fANOVA 
    importance evaluator and generates a horizontal bar plot showing the contribution of each parameter 
    to the selected objective.

    The function supports the following objectives:
        - validation_MSE
        - lift_coefficient_reconstruction_MSE

    Parameters
    ----------
    study : optuna.study.Study
        Loaded Optuna study containing the completed optimization trials.

    objective : str
        Objective for which the hyperparameter importance will be evaluated. Accepted values are:
            - 'validation_MSE'
            - 'lift_coefficient_reconstruction_MSE'

    layout : str
        Plot layout configuration. Accepted values are:
            - 'single_column'
            - 'double_column'

    Returns
    -------
    None
        The function saves the generated figures to disk and does not return any value.

    """

    # --- Plot layout configurations ---
    plot_layout = {'single_column': {'width': 3.35, 'height': 2.55, 'main_lw': 0.8, 'sec_lw': 0.4, 
            'scatter': 10, 'label_fs': 12, 'lp': 5, 'sec_fs': 10, 'tick': 0.4, 'grid_alpha': 0.2}, 
        'double_column': {'width': 6.7, 'height': 4.8, 'main_lw': 1.2, 'sec_lw': 0.6, 'scatter': 20, 
            'label_fs': 18, 'lp': 10, 'sec_fs': 14, 'tick': 0.5, 'grid_alpha':0.3}}
    
    cfg = plot_layout[layout]
    # --- Setting figure size:
    width, height = cfg['width'], cfg['height']
    # --- Setting plot element size:
    main_lw, sec_lw, scatter = cfg['main_lw'], cfg['sec_lw'], cfg['scatter']
    # --- Setting text font size:
    label_fs, sec_fs = cfg['label_fs'], cfg['sec_fs']
    # --- Setting aestetic parameters:
    lp, tick, grid_alpha = cfg['lp'], cfg['tick'], cfg['grid_alpha']

    # --- Filter completed trials ---
    completed_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    completed_study = optuna.create_study(directions=study.directions)
    for t in completed_trials:
        completed_study.add_trial(t)

    # --- Compute importance ---
    if objective == 'validation_MSE':
        importance = optuna.importance.get_param_importances(completed_study, 
            evaluator=optuna.importance.FanovaImportanceEvaluator(seed=42), target=lambda t: t.values[0])
        
    elif objective == 'lift_coefficient_reconstruction_MSE':
        importance = optuna.importance.get_param_importances(completed_study, 
            evaluator=optuna.importance.FanovaImportanceEvaluator(seed=42), target=lambda t: t.values[1])
    
    # --- Process the data ---
    importance_data = pd.DataFrame(list(importance.items()), columns=['Hyperparameter', 'Importance'])
    rename_dict = {'activation': r'Activation function', 'dropout_rate': r'Dropout rate',
        'learning_rate': r'Learning rate', 'n_hidden_layers': r'Number of hidden layers',
        'optimizer': r'Optimizer', 'units_layer1': r'1st layer width', 'units_layer2': r'2nd layer width',
        'units_layer3': r'3rd layer width', 'units_layer4': r'4th layer width', 
        'units_layer5': r'5th layer width', 'units_layer6': r'6th layer width', 
        'units_layer7': r'7th layer width', 'use_dropout': r'Dropout usage'}
    importance_data['Parameters'] = importance_data['Hyperparameter'].map(rename_dict).fillna(importance_data['Hyperparameter'])
    
    # Setting plot parameters:
    with plt.rc_context({'font.family': 'serif', 'font.serif': ['Times New Roman'],
        'mathtext.fontset': 'stix'}):

        # --- Figure size: single-column ---
        fig, ax = plt.subplots(figsize=(width, height))

        # --- Plot ---
        sns.barplot(data=importance_data, x='Importance', y='Parameters', color='#D3D3D3', 
            edgecolor='#000000')

        # --- Labels ---
        ax.set_xlabel('Relative importance score', fontsize=label_fs, fontname='Times New Roman', labelpad=lp)
        ax.set_ylabel(' ', fontsize=label_fs, fontname='Times New Roman', labelpad=lp)
        if objective == 'validation_MSE':
            ax.set_title('Model Validation MSE', fontsize=label_fs, fontname='Times New Roman', pad=lp)
        elif objective == 'lift_coefficient_reconstruction_MSE':
            ax.set_title(r'$C_L$ Reconstruction MSE', fontsize=label_fs, 
                fontname='Times New Roman', pad=lp)
        
        # --- Ticks ---
        for spine in ax.spines.values():
            spine.set_linewidth(0.5)
        ax.tick_params(axis='both', which='major', labelsize=sec_fs, width=tick, direction='in')

        # --- Grid ---
        ax.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=grid_alpha)

        # --- Limits ---
        #ax.set_xlim(0, importance_data['Importance'].max() * 1.15)
        ax.set_xlim(0, 1)

        # --- Save plot ---
        plt.tight_layout(pad=0.6)
        plt.savefig(f'optimization-plots/plot_optimization_results/{study_name}_{objective}_importance_{layout}.png', dpi=600)
        plt.close()
        print(f'\nSaved: optimization-plots/plot_optimization_results/{study_name}_{objective}_importance_{layout}.png\n')
                    
    return

def plot_history(study:optuna.study.Study, layout:str):
    """
    Plot the optimization history in the bi-objective space.

    This function extracts the completed trials from the Optuna study and generates a scatter plot of the 
    two optimization objectives. Each point corresponds to one completed trial, and the color scale indi-
    cates the trial number.

    Parameters
    ----------
    study : optuna.study.Study
        Loaded Optuna study containing the completed optimization trials.

    layout : str
        Plot layout configuration. Accepted values are:
            - 'single_column'
            - 'double_column'
    Returns
    -------
    None
        The function saves the generated figures to disk and does not return any value.
    """

    # --- Plot layout configurations ---
    plot_layout = {'single_column': {'width': 3.35, 'height': 2.55, 'main_lw': 0.8, 'sec_lw': 0.4, 
            'scatter': 10, 'label_fs': 12, 'lp': 5, 'sec_fs': 10, 'tick': 0.4, 'grid_alpha': 0.2}, 
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
    trials = [t for t in study.trials if t.state.name == "COMPLETE"]
    obj_MSE = np.array([t.values[0] for t in trials])
    obj_CL = np.array([t.values[1] for t in trials])
    trial_number = np.array([t.number for t in trials])

    # Setting plot parameters:
    with plt.rc_context({'font.family': 'serif', 'font.serif': ['Times New Roman'],
        'mathtext.fontset': 'stix'}):

        # --- Figure size: single-column ---
        fig, ax = plt.subplots(figsize=(width, height))

        # --- Plot ---
        plot = ax.scatter(obj_MSE, obj_CL, c=trial_number, cmap='viridis', s=scatter, alpha=0.7, 
            edgecolors='#000000', linewidth=0.3, zorder=5)
            
        # --- Colorbar ---
        plot.set_clim(0, 200)
        cbar = plt.colorbar(plot, ax=ax)
        cbar.set_label('Trial number', fontsize=label_fs, fontname='Times New Roman', labelpad=lp)
        cbar.ax.tick_params(labelsize=sec_fs, width=tick, direction='in')
        cbar.ax.yaxis.get_offset_text().set_fontsize(sec_fs)
        cbar.set_ticks([0, 50, 100, 150, 200])
        for spine in cbar.ax.spines.values():
            spine.set_linewidth(0.5)

        # --- Best region ---
        x_min, x_max = 9e-6, 2e-5
        y_min, y_max = 0.0, 2.5e-7
        rect = matplotlib.patches.Rectangle((x_min, y_min), x_max-x_min, y_max-y_min, linewidth=0.4, 
            edgecolor='#B22222', facecolor='#B22222', alpha=0.2, zorder=10, linestyle='--', label='Best-performing region')
        ax.add_patch(rect)

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
        ax.set_xlim(0.0000075, 0.00005)
        ax.set_ylim(-0.00000005, 0.000003)

        # --- Legend ---
        legend = ax.legend(fontsize=sec_fs, fancybox=False, edgecolor='#000000', loc='upper right')
        if layout == 'single_column':
            legend.get_frame().set_linewidth(0.5)

        # --- Save plot ---
        plt.tight_layout(pad=0.6)
        plt.savefig(f'optimization-plots/plot_optimization_results/{study_name}_history_{layout}.png', dpi=600)
        plt.close()
        print(f'Saved: optimization-plots/plot_optimization_results/{study_name}_history_{layout}.png\n')
                    
    return

def plot_pareto_front(study_name:str, layout:str):
    """
    Plot the Pareto front of the multi-objective optimization.

    This function loads the full set of evaluated trials and the subset of Pareto-optimal trials from
    CSV files and generates a scatter plot highlighting the trade-off between the two objectives.

    The Pareto-optimal solutions represents models for which no objective can be improved without 
    degrading the other.

    Parameters
    ----------
    study_name : str
        Name of the optimization study used to locate the corresponding CSV files.

    layout : str
        Plot layout configuration. Accepted values are:
            - 'single_column'
            - 'double_column'
    
    Returns
    -------
    None
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
    all_trials = pd.read_csv(f'optimization-plots/data_optimization_results/{study_name}/optimization_trials.csv')
    all_trials = all_trials.dropna(subset=['MSE_loss'])
    pareto_trials = pd.read_csv(f'optimization-plots/data_optimization_results/{study_name}/optimization_pareto_trials.csv')
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
        plot = ax.scatter(best_trial['MSE_loss'], best_trial['CL_loss'], facecolors='#B22222', s=3*scatter,
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
        ax.set_xlim(0.000009, 0.00002)
        ax.set_ylim(0, 0.0000003)

        # --- Legend ---
        legend = ax.legend(fontsize=sec_fs, fancybox=False, edgecolor='#000000', loc='upper right', 
            markerscale=1.2)
        if layout == 'single_column':
            legend.get_frame().set_linewidth(0.5)

        # --- Save plot ---
        plt.tight_layout(pad=0.6)
        plt.savefig(f'optimization-plots/plot_optimization_results/{study_name}_pareto_front_{layout}.png', dpi=600)
        plt.close()
        print(f'Saved: optimization-plots/plot_optimization_results/{study_name}_pareto_front_{layout}.png\n')
                    
    return

def main(study_name:str, storage:str):
    """
    Execute the optimization results visualization workflow.

    This function loads the Optuna study from the specified SQLite storage and generates:
        - Hyperparameter importance plot for validation_MSE.
        - Hyperparameter importance plot for lift_coefficient_reconstruction_MSE.
        - Optimization history scatter plot in the objective space.

    Parameters
    ----------
    study_name : str
        Name of the Optuna study.

    storage : str
        Database connection string used to access the Optuna study.

    Returns
    -------
    None
        The function does not return any value.
    """
    # --- Inform the start of the visualization process ---
    print(f"\nStarting optimization results visualization process...\n")

    # --- Load the Optuna study from the SQLite database ---
    study = optuna.load_study(study_name=study_name, storage=storage)

    # --- Generate hyperparameter importance plot ---
    plot_importance_by_objective(study, 'validation_MSE', 'single_column')
    plot_importance_by_objective(study, 'lift_coefficient_reconstruction_MSE', 'single_column')

    # --- Generate optimization history plot in the objective space ---
    plot_history(study, 'double_column')

    # --- Generate pareto front plot ---
    plot_pareto_front(study_name, 'double_column')

    return

if __name__ == '__main__':
    main(study_name=study_name, storage=storage)