# -*- coding: utf-8 -*-
"""
--------------------------------------------------------------------------------------------------------
Function:               data_sectional_visualization
Author:                 Caio Dias Filho
Creation date:          2025-11-19
Last modification:      2025-11-19
Version:                1.0

Description:
    This script performs a 3D visualization of the pressure coefficient (Cp) distribution over the wing
    surface for a specific flight condition (Re, AoA). The visualization displays the measurement points
    corresponding to the experimental pressure taps, showing the sectional Cp distribution.
        
Dependencies:
    - matplotlib
    - pandas
    - numpy

Future implementations:
    >>> Create a new function named data_comparison_visualization that employs the experimental and 
    numerical datasets.
--------------------------------------------------------------------------------------------------------
"""

from matplotlib.colors import TwoSlopeNorm
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np



def data_sectional_visualization(data:pd.DataFrame, Re:int, AoA:int, scatter:bool, is_exp:bool, out_path:str):
    """
    Performs a 3D visualization of the sectional pressure coefficient (Cp) distribution on the wing surface
    for a specified Reynolds Number (Re) and Angle of Attack (AoA).

    Args:
        data: DataFrame containing the pressure coefficient (cp) for 31 chordwise points and the corresponding
              sectional lift coefficient (Cl), along with the input conditions ('Re', 'AoA', 'y').
        Re: Reynolds Number for the aimed flight condition.
        AoA: Angle of Attack for the aimed flight condition.
        scatter: If True, plots points corresponding to the experimental pressure taps locations.
        is_exp: If True, assumes that the data is already experimental and does not require tap filtering.
        out_path: Path and name to save the output image (.png).

    Returns:
        None. Saves the 3D visualization in an image file.     
    """

    # Defining nomenclature parameters:
    if is_exp == True:
        name = 'Experimental'
    else:
        name = 'Numerical'

    # 1. Filters the data for the specific input flight condition (Re and AoA):
    filtered_cond = data[(data['Re'] == Re) & (data['AoA'] == AoA)]

    # 2. Filters the experimental taps spanwise location if the data is not already experimental:
    if is_exp == False:
        exp_taps = [5, 11, 22, 34, 45, 57, 68, 74]
        filtered_cond = filtered_cond.iloc[exp_taps].reset_index(drop=True)

    # 3. Defines the three axis for the 3D plot:
    # x -> Chord position in mm.
    chord_pos = [0.197015, 0.179695, 0.15588, 0.136395, 0.119075, 0.099590, 0.080105, 0.06495,
                 0.049795, 0.036805, 0.02598, 0.017320, 0.008660, 0.004330, 0.002165, 0.00000,
                 0.000000, 0.004330, 0.00866, 0.017320, 0.028145, 0.038970, 0.049795, 0.06495,
                 0.080105, 0.099590, 0.11691, 0.136395, 0.155880, 0.179695, 0.197015]
    # y -> Span position in mm.
    span_pos = filtered_cond['y']
    # z -> Pressure coefficient.
    cp_data = filtered_cond.iloc[:,4:].to_numpy()    

    # 4. Setting the data for the 3D plot:
    X_grid, Y_grid = np.meshgrid(chord_pos, span_pos)
    X, Y, Z = X_grid.flatten(), Y_grid.flatten(), cp_data.flatten()

    # 5. Setting the figure properties:
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_box_aspect([5, 10, 4])

    # 6. Setting the axes properties:
    ax.set_xticks([0, 0.043, 0.086, 0.129, 0.173, 0.216])
    ax.set_xlabel('Chordwise Position [mm]', labelpad=10, fontname='Times New Roman', fontsize=12)
    ax.set_xlim(0, 0.2165) 
    ax.set_ylabel('Spanwise Position [mm]', labelpad=20, fontname='Times New Roman', fontsize=12)
    ax.set_ylim(0, 0.766)
    ax.set_zlabel('Pressure Coefficient ($C_p$)', labelpad=10, fontname='Times New Roman', fontsize=12)
    ax.zaxis.set_rotate_label(False)
    ax.zaxis.label.set_rotation(90)
    ax.set_zlim(1, -3.5) 
    fig.suptitle(f'{name} Pressure Coefficient Distribution at Re = {Re} and AoA = {AoA}°',
                  fontname='Times New Roman', fontsize=14, fontweight='bold', y = 0.88)

    # 7. Plots the measurement points as scatter plots if required:
    if scatter == True:
        scatter_plot = ax.scatter(X, Y, Z,
                                  c=Z, cmap='RdYlBu_r', s=20,
                                  edgecolors='black', linewidths=0.5,
                                  norm=TwoSlopeNorm(vmin=-3.5, vcenter=0, vmax=1),
                                  depthshade=True)
        cbar = fig.colorbar(scatter_plot, shrink=0.5, aspect=12, pad=0.1)
        cbar.set_label('Pressure Coefficient ($C_p$)', labelpad=10, fontname='Times New Roman', fontsize=12)

    # 8. Plots the pressure distribution lines for each section:
    for i in range(len(span_pos)):
        y_section = np.full_like(chord_pos, span_pos.iloc[i])
        ax.plot(chord_pos, y_section, cp_data[i, :],
                color='black', linewidth=0.75, alpha=0.75)

    # 9. Saves and shows the figure:
    print(f"\nPlot saved as {out_path}\n")
    ax.view_init(elev=40, azim=-140)
    plt.savefig(out_path, dpi=300)
    plt.show()

    return

def main(data_path: str, Re: int, AoA: int, scatter: bool, is_exp: bool, out_path: str):
    """
    Main execution workflow: Loads the data and calls the 3D visualization function.
    """

    # 1. Loads the data:
    print("\nStarting 3D visualization process...\n")
    print(f"Loading data from: {data_path}")
    data = pd.read_csv(data_path, sep=',')
    print("Data loaded successfully.")

    # 2. Calls the 3D visualization function:
    data_sectional_visualization(data, Re, AoA, scatter, is_exp, out_path)
    
    return



if __name__ == "__main__":

    # 1. Sets the run parameters:
    DATA_FILE = 'Numerical-PressureDistributionData.csv'
    REYNOLDS_NUMBER = 235456
    ANGLE_OF_ATTACK = 7
    ENABLE_SCATTER = True
    IS_EXPERIMENTAL_DATA = False
    OUTPUT_FILE = f"SectionalPressure_Re{REYNOLDS_NUMBER}_AoA{ANGLE_OF_ATTACK}.png"

    # 2. Calls the main function:
    main(DATA_FILE, REYNOLDS_NUMBER, ANGLE_OF_ATTACK, ENABLE_SCATTER, IS_EXPERIMENTAL_DATA, OUTPUT_FILE)