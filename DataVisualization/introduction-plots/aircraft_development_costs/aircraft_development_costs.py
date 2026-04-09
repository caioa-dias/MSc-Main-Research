# -*- coding: utf-8 -*-
"""
========================================================================================================
Module: aircraft_development_costs
Author: Caio Dias Filho
Creation date: 2026-04-03
Last modification: 2026-04-03
Version: 1.0
========================================================================================================

OVERVIEW
--------
This module generates a publication-ready bar plot illustrating the evolution of aircraft development
costs over time.

The plot present historical data for representative commercial aircraft programs, highlighting the
significant increase in development costs as aircraft technonological requirements, and certification
constraints have evolved.

The visualization is intended to support industry discussions on:

    - The growing complexity of aircraft design.
    - The increasing financial investment required for development.
    - The motivation for advanced design methodologies and surrogate modeling.


DEPENDENCIES
------------
Python libraries:
    - matplotlib

    
INPUT FILES
-----------
No external input files are required. The data is hardcoded within the function for simplicity and 
reproducibility.

    
OUTPUT FILES
------------
Generated figures:

    - introduction-plots/aircraft_development_costs/aircraft_development_costs_<layout>.png


NOTES
-----
- Development costs are expressed in constant 2004 USD for consistency.
- A logrithmic scale is used on the vertical axis to better represent the wide range of cost magnitudes
  across decades.
- The aircraft are sorted chornologically before plotting.
- The module supports both single-column and double-column publication layouts.

========================================================================================================
"""

# =======================================================================================================
#                                            USER INPUT SECTION
# =======================================================================================================

# --- Layout ---
layout = 'double_column'


# =======================================================================================================
#                                              CORE FUNCTION
# =======================================================================================================

from matplotlib import pyplot as plt

def plot_aircraft_development_costs(layout: str):
    """
    Plot the historical evolution of aircraft development costs.

    This function generates a bar chart showing the development cost of selected commercial aircraft
    programs, ordered choronologically by their year of entry into service.

    The plot uses a logarithmic scale on the vertical axis to highlight the rapid growth in development
    costs over time and improve visualization of values spanning multiple orders of magnitude.

    Parameters
    ----------
    layout : str
        Plot layout configuration. Accepted values are:
            - 'single_column'
            - 'double_column'

    Returns
    -------
    None
         The function saves the generated figure to disk and does not return any value.
    """

    # --- Data ---
    aircraft = ["Douglas DC-6\n(1946)", "Boeing 707\n(1958)", "Boeing 747\n(1970)", "Boeing 777\n(1995)",
    "Airbus A380\n(2007)", "Boeing 787\n(2012)", "Airbus A350\n(2015)"]
    year = [1946, 1958, 1970, 1995, 2007, 2012, 2015]
    cost = [144e6, 1.3e9, 3.7e9, 7.0e9, 14.4e9, 13.4e9, 13.4e9]

    # --- Data processing ---
    data = sorted(zip(year, aircraft, cost))
    year, aircraft, cost = zip(*data)

    # --- Plot layout configurations ---
    plot_layout = {'single_column': {'width': 3.35, 'height': 2.55, 'main_lw': 0.8, 'sec_lw': 0.4, 
            'scatter': 2, 'label_fs': 8, 'lp': 5, 'sec_fs': 6, 'tick': 0.4, 'grid_alpha': 0.2, 
            'm_scale': 1}, 
        'double_column': {'width': 17.5, 'height': 6.3, 'main_lw': 1.2, 'sec_lw': 0.4, 'scatter': 6, 
            'label_fs': 31, 'lp': 25, 'sec_fs': 27, 'tick': 0.5, 'grid_alpha':0.3, 'm_scale': 2}}

    cfg = plot_layout[layout]
    # --- Setting figure size:
    width, height = cfg['width'], cfg['height']
    # --- Setting plot element size:
    main_lw, sec_lw = cfg['main_lw'], cfg['sec_lw']
    # --- Setting text font size:
    label_fs, sec_fs = cfg['label_fs'], cfg['sec_fs']
    # --- Setting aestetic parameters:
    lp, tick, grid_alpha = cfg['lp'], cfg['tick'], cfg['grid_alpha']

    # Setting plot parameters:
    with plt.rc_context({'font.family': 'serif', 'font.serif': ['Times New Roman'],
        'mathtext.fontset': 'stix'}):
        
        # --- Figure size ---
        fig, ax = plt.subplots(figsize=(width, height))

        # --- Plot ---
        bars = ax.bar(aircraft, cost, color='#0072B2', edgecolor='black', linewidth=sec_lw, width=0.5)

        # --- Labels ---
        #ax.set_xlabel("Aircraft (entry into service year)", fontsize=label_fs,
            #fontname='Times New Roman', labelpad=lp)
        ax.set_ylabel("Development cost (USD, 2004)", fontsize=label_fs,
            fontname='Times New Roman', labelpad=lp)

        # --- Ticks ---
        for spine in ax.spines.values():
            spine.set_linewidth(0.5)
        ax.tick_params(axis='both', which='major', labelsize=sec_fs, width=tick, direction='in')
        plt.tick_params(axis='x', pad=10)
        plt.yscale('log')

        # --- Grid ---
        ax.grid(axis='y', which='major', linestyle='--', alpha=grid_alpha)

        # --- Limits ---
        ax.set_ylim(None, max(cost)*3)

        # --- Save plot ---
        plt.tight_layout(pad=0.6)
        plt.savefig(f'introduction-plots/aircraft_development_costs/aircraft_development_costs_{layout}.png', dpi=600)
        plt.close()
        print(f"\nSaved: introduction-plots/aircraft_development_costs/aircraft_development_costs_{layout}.png\n")

def main(layout: str):
    """
    Execute the aircraft development cost visualization workflow.

    This function calls the plotting routine using the user-defined layout configuration and generates
    the final figure.

    Returns
    -------
    None
        The function does not return any valua.
    """

    # --- Generate plot ---
    plot_aircraft_development_costs(layout)

    return

if __name__ == "__main__":
    main(layout)