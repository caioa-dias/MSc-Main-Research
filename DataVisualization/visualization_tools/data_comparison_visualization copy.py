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
    >>> montar a função
--------------------------------------------------------------------------------------------------------
"""

from matplotlib import pyplot as plt
import pandas as pd
import numpy as np

def data_comparison_visualization(data_dict: dict, data_labels:list, visual_styles:dict, Re:int, AoA:int, 
    specific_section:bool, section:int, out_path:str):
    """
 
    """
    # Defining lists parameters:
    filtered_conds = []
    span_pos = []
    cp_data = []

    # 1. Filters the data for the specific input flight condition (Re and AoA):
    for i in data_labels:
        filtered_cond = data_dict[i][(data_dict[i]['Re'] == Re) & (data_dict[i]['AoA'] == AoA)]
        # Filters the experimental taps position on computational data:
        if len(data_dict[i]['y']) > 80:
            filtered_cond = filtered_cond.iloc[5, 11, 22, 34, 45, 57, 68, 74].reset_index(drop=True)
        filtered_conds.append(filtered_cond)
        # z -> Pressure coefficient.
        cp_data.append(filtered_cond.iloc[:, 4:35].to_numpy())
        
    # 2. Defines the three axis for the 3D plot:
    # x -> Chord position in mm.
    chord_pos = [0.197015, 0.179695, 0.15588, 0.136395, 0.119075, 0.099590, 0.080105, 0.06495,
                 0.049795, 0.036805, 0.02598, 0.017320, 0.008660, 0.004330, 0.002165, 0.00000,
                 0.000000, 0.004330, 0.00866, 0.017320, 0.028145, 0.038970, 0.049795, 0.06495,
                 0.080105, 0.099590, 0.11691, 0.136395, 0.155880, 0.179695, 0.197015]
    # y -> Span position in mm.
    span_pos = filtered_cond[0]['y'].to_numpy()

    # 4. Setting the plot to a 3D plot:
    if specific_section == False:
        # 4.1. Setting the figure properties:
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')
        ax.set_box_aspect([5, 10, 4])

        # 4.2. Setting the axes properties:
        ax.set_xticks([0, 0.043, 0.086, 0.129, 0.173, 0.216])
        ax.set_xlabel('Chordwise Position [mm]', labelpad=10, fontname='Times New Roman', fontsize=12)
        ax.set_xlim(0, 0.2165) 
        ax.set_ylabel('Spanwise Position [mm]', labelpad=20, fontname='Times New Roman', fontsize=12)
        ax.set_ylim(0, 0.766)
        ax.set_zlabel('Pressure Coefficient ($C_p$)', labelpad=10, fontname='Times New Roman', fontsize=12)
        ax.zaxis.set_rotate_label(False)
        ax.zaxis.label.set_rotation(90)
        ax.set_zlim(1, -3.5)
        fig.suptitle(f'Pressure Coefficient Distribution at Re = {Re} and AoA = {AoA}°',
                        fontname='Times New Roman', fontsize=14, fontweight='bold', y = 0.88)

        # 4.3. Plots the pressure distribution lines for each section:
        for j in range(len(data_labels)):
            for i in range(len(span_pos)):
                label = visual_styles[data_labels[j]]['label'] if i == 0 else "_nolegend_"
                y_section = np.full_like(chord_pos, span_pos[i])
                ax.plot(chord_pos, y_section, cp_data[j][i, :], color=visual_styles[data_labels[j]]['color'], 
                    linestyle=visual_styles[data_labels[j]]['linestyle'], label=label, linewidth=0.9, alpha=1)
        
        # 4.4. Saves and shows the figure:
        print(f"\nPlot saved as {out_path}\n")
        ax.view_init(elev=40, azim=-140)
        plt.savefig(out_path, dpi=300)
        plt.show()

    # 5. Setting the plot to a 2D plot:
    if specific_section == False:
        # 5.1. Setting the axes properties:
        plt.xlabel('Chordwise Position [mm]', fontname='Times New Roman', fontsize=12)
        plt.xlim(0, 0.2165) 
        plt.ylabel('Pressure Coefficient ($C_p$)', fontname='Times New Roman', fontsize=12)
        plt.ylim(1, -3.5)
        plt.suptitle(f'Pressure Coefficient Distribution at y = {span_pos[section]}, Re = {Re} and AoA = {AoA}°',
                     fontname='Times New Roman', fontsize=14, fontweight='bold')
        
        # 5.2. Plots the pressure distribution lines for a specific section:
        for j in range(len(data_labels)):
            label = visual_styles[data_labels[j]]['label']
            plt.plot(chord_pos, cp_data[j][section, :], color=visual_styles[data_labels[j]]['color'], 
                linestyle=visual_styles[data_labels[j]]['linestyle'], label=label, 
                linewidth=0.95, alpha=1)

        # 5.3. Saves and shows the figure:
        print(f"\nPlot saved as {out_path}\n")
        plt.savefig(out_path, dpi=300)
        plt.show()

    return

def main(data_path: list, Re: int, AoA: int, scatter: bool, is_exp: bool, out_path: str):
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
    ANGLE_OF_ATTACK = 10
    ENABLE_SCATTER = True
    IS_EXPERIMENTAL_DATA = False
    OUTPUT_FILE = f"plots/SectionalPressure_Re{REYNOLDS_NUMBER}_AoA{ANGLE_OF_ATTACK}.png"

    # 2. Calls the main function:
    main(DATA_FILE, REYNOLDS_NUMBER, ANGLE_OF_ATTACK, ENABLE_SCATTER, IS_EXPERIMENTAL_DATA, OUTPUT_FILE)