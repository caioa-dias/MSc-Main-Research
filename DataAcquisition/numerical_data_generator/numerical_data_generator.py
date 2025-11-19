# -*- coding: utf-8 -*-
"""
--------------------------------------------------------------------------------------------------------
Function:               numerical_data_generator
Author:                 Caio Dias Filho
Creation date:          2025-10-25
Last modification:      2025-11-18
Version:                1.1

Description:
    This script performs a coupled 3D and 2D aerodynamic analysis. 
    1. An AVL 3D analysis is performed to obtain the flow cases (Re, Cl, AoA) at each section (y) of the 3D wing.
    2. Each case is passed to the 2D analysis, obtaining the pressure coefficient distribution for each section.
        
Dependencies:
    - tqdm
    - pandas
    - numpy
    - subprocess
    - time
    - os

Future implementations:
    >>> Implement a routine for generalize analysis3D.
    >>> Substitute the Reynolds Number for the experimental ones.
    >>> Implement the analysis3D for the experimental cases Reynolds.
--------------------------------------------------------------------------------------------------------
"""

from tqdm import tqdm
import pandas as pd
import numpy as np
import subprocess
import time
import os



def analysis3D(input_file: str, v_range: list):
    """
    Runs a AVL 3D analysis of the wing geometry in different AoA to obtain the flow cases (Re, Cl, AoA) at each section (y)
    of the 3D wing.

    Args:
        input_file: Path to the input .in file containing the AVL routine.
        v_range: List representing the range of velocities to be analysed [Initial, Final, Step].

    Returns:
        flow_cases (pd.DataFrame): DataFrame containing the flow cases (Re, Cl, AoA) at each section (y) of the 3D wing,
                                   inputs for the XFoil 2D analysis.
    """

    # Defining the number of AoA cases present in the input file:
    with open("utils/ExperimentalCases.run", 'rb') as f:
        n_conditions = (sum(1 for _ in f)//22)
    AoA_i = -6
    AoA = []
    Re = []

    # 1. Running AVL analysis using the pre-defined geometry (.avl) and conditions (.run) files:
    print("Starting AVL analysis...\n")
    with open(os.devnull, 'w') as FNULL:
        subprocess.call(f"avl.exe < {input_file}", shell=True, stdout=FNULL, stderr=subprocess.STDOUT, cwd="utils")
    print("AVL analysis completed!\n")

    # 2. Loading and merging the AVL data for each AoA case:
    for i in range(0, n_conditions):
        case = pd.read_csv(f"utils/AoA{i + AoA_i}.csv", header=16, nrows=80, sep=r'\s+', usecols=['Yle', 'cl'])
        case['AoA'] = i + AoA_i
        AoA.append(case)
        os.remove(f"utils/AoA{i + AoA_i}.csv")
    flow_cases = pd.concat(AoA, ignore_index=True)

    # 3. Merging the Reynolds Number information in the dataset for each AoA polar:
    for i in range(v_range[0], v_range[1], v_range[2]):
        case = flow_cases.copy()
        case['Re'] = 1.11 * i * 0.2165 / 0.0000183714
        Re.append(case)
    flow_cases = pd.concat(Re, ignore_index=True)

    # 4. Saving the results in a .csv file:
    flow_cases.to_csv("utils/FlowCases.csv", index=False)

    return flow_cases

def analysis2D(Re:float, cl:float, AoA:float, y:float, airfoil="NACA 23015"):
    """
    Runs a XFoil 2D analysis corresponding to each case (Re, Cl, AoA), obtaining the pressure coefficient distribution 
    data for an specific section (y) of the 3D wing.
    
    Args:
        Re: Reynolds Number for the current case.
        Cl: Lift coefficient for the current case.
        AoA: Angle of attack for the current case.
        y: y coordinate for the current section.
        airfoil: Airfoil name (default: "NACA 23015").

    Returns:
        cp_2d_data (pd.DataFrame): DataFrame containing the 2D pressure coefficient (cp) distribution for 31 chordwise
                                    points and the corresponding sectional lift coefficient (Cl), along with the input
                                    conditions ('Re', 'AoA', 'y').
    """
    
    # Defining standard path variables:
    input_path_2d = "utils/xfoil_input.in"
    output_path_2d = "utils/cp_data.csv"

    # Defining the XFoils panels index that represents the experimental measurements:
    experimental_taps = [8, 14, 21, 27, 33, 39, 45, 50, 55, 60, 65, 71, 77, 84, 93, 103, 111, 119, 126, 
                        132, 137, 141, 145, 150, 155, 161, 167, 173, 179, 186, 192]

    # 1. Create a XFoil Input File
    xfoil_file = open("utils/xfoil_input.in", 'w')
    xfoil_file.write(f"PLOP\nG F\n\n{airfoil}\nPPAR\nN 201\n\n\nOPER\nVPAR\nN 7\n\nVISC {Re:.0f}\nITER 400\nCL {cl:.3f}\ncpwr\ncp_data.csv\n\nQUIT\n")
    xfoil_file.close()

    # 2. Run XFoil supressing the output:
    with open(os.devnull, 'w') as FNULL:
        subprocess.call("xfoil.exe < xfoil_input.in", shell=True, stdout=FNULL, stderr=subprocess.STDOUT, cwd="utils")

    # 3. Wait to the XFoil analysis to be written and remove the input file:
    time.sleep(0.2)
    os.remove(input_path_2d)

    # 4. Read the XFoil results data filtering for the experimental measurements points:
    xfoil_data = pd.read_csv(output_path_2d, skiprows=3, sep=r"\s+", names=['x', 'y', 'cp'], usecols=['x', 'cp'])
    xfoil_data = xfoil_data.iloc[experimental_taps].reset_index(drop=True)

    # 5. Wait for the dataframe to be assembled and remove the output file:
    time.sleep(0.2)
    os.remove(output_path_2d)

    # 6. Assemble the final dataset with a single row:
    cp_data_2d = pd.DataFrame(data=[xfoil_data['cp'].values], columns=(np.round(xfoil_data['x'].values, 2)))
    cp_data_2d.insert(0, 'Re', Re)
    cp_data_2d.insert(1, 'AoA', AoA)
    cp_data_2d.insert(2, 'y', y)
    cp_data_2d.insert(3, 'cl', cl) 

    return cp_data_2d

def main(new_cases: bool, v_range: list, AoA_filter: bool, AoA_range: list):
    """
    Main execution workflow: Generate or uses the flow cases data and perform 2D analysis on each one. The rfinal
    pressure distribution data is saved in the 'PressureDistributionData.csv' file.
    """

    # Defining standard parameters:
    input_file_3d = "ExperimentalCases.in"
    output_path = "Numerical-PressureDistributionData.csv"
    Results = []

    # Deletes the output file if it exists to avoid overwriting:
    if os.path.exists(output_path):
        os.remove(output_path)

    # 1. Checks if it is needed to generate new flow conditions or use a previous set:
    if new_cases == True:
        flow_cases = analysis3D(input_file_3d, v_range)
        print("Flow cases generated!\n")
    elif new_cases == False:
        flow_cases = pd.read_csv("utils/FlowCases.csv", dtype={'Re': float, 'cl': float, 'AoA': float})
        print("Flow cases loaded!\n")

    # 2. Checks if it is needed to filter the data by a defined range of AoA:
    if AoA_filter == True:
        filtered_AoA_cases = flow_cases[flow_cases['AoA'].between(AoA_range[0], AoA_range[1])].reset_index(drop=True)
        flow_cases = filtered_AoA_cases.copy()
        print("Filter successful applied!\n")
    elif AoA_filter == False:
        flow_cases = flow_cases
        print("Using the full data set!\n")

    # 3. Runs the analysis for each flow case:
    for i in tqdm(range(len(flow_cases)), desc='Generating aerodynamic dataset...', unit='case'):
        case = analysis2D(flow_cases['Re'][i], flow_cases['cl'][i], flow_cases['AoA'][i], flow_cases['Yle'][i])
        Results.append(case)
    results = pd.concat(Results, ignore_index=True)

    # 4. Save the results:
    results.to_csv(output_path, index=False)
    print(f"\nProcessing complete! Results saved to {output_path}\n")



if __name__ == "__main__":
    main(new_cases=True, v_range=[8, 31, 1], AoA_filter=True, AoA_range=[-5, 10])