# -*- coding: utf-8 -*-
"""
========================================================================================================
Module: potential_data_generator
Author: Caio Dias Filho
Creation date: 2025-11-25
Last modification: 2026-02-24
Version: 1.5
========================================================================================================

OVERVIEW
--------
This module generates a low-fidelity aerodynamic dataset of wing surface pressure distributions by 
coupling:

    1) 3D aerodynamic analysis using AVL (Vortex Lattice Method)
    2) 2D sectional analysis using XFoil

A critical-section methodology is implemented to ensure physical consistency of the potential-flow
dataset. Only the linear region of the lift curve is retained.

For each Reynolds number:
    - Analyses start at AoA = -4° (approximately the zero-lift angle).
    - The dataset is truncated at the onset of stall.
    - All post-stall angles are automatically removed.

The workflow consists of:
    - Extracting spanwise lift coefficient distributions from AVL.
    - Expanding cases over a Reynolds number range.
    - Computing Cl_max(Re) using XFoil.
    - Filtering the dataset using the critical section criterion
    - Running sectional Cp analysis only in the linear regime.
    - Assembling a structured dataset for machine learning application.

The final output is a .csv file containing:
    - Reynolds number (Re).
    - Angle of attack (AoA).
    - Normalized spanwise position (y/b).
    - Sectional lift coefficient (Cl).
    - 201 chordwise pressure coefficient valyes (Cp).

    
DEPENDENCIES
------------
External software:
    - AVL (avl.exe)
    - XFoil (xfoil.exe)

Python libraries:
    - numpy
    - pandas
    - tqdm
    - subprocess
    - os
    - time

    
OUTPUT FILES
------------
- utils/2D-PotentialCases.csv
- utils/2D-LinearPotentialCases.csv
- Potential-PressureDistributionData.csv
    

NOTES
-----
- Only the pre-stall linear regime of the lift curve is retained.
- Stall detection is based on Cl > Cl_max(Re).
- Paths are configured for Windows (.exe usage).

========================================================================================================
"""

# =======================================================================================================
#                                            USER INPUT SECTION
# =======================================================================================================

# --- Analysis Ranges ---
# Reynolds Number range:
initial_Reynolds = 10**5
final_Reynolds = 10**6
step_Reynolds = 2*(10**4)
# Angle of Attack range:
initial_AoA = -4
final_AoA = 18
# Percentage below the Cl_max for stall margin:
stall_margin = 0.1

# --- Analysis Options ---
# Generating new potential cases:
new_cases = True
# Filtering potential cases by AoA:
AoA_filter = False

# --- Wing Geometry Data ---
# Wing geometry half-span:
b_wing = 0.766


# =======================================================================================================
#                                              CORE FUNCTION
# =======================================================================================================

from tqdm import tqdm
import pandas as pd
import numpy as np
import subprocess
import time
import os

def cl_max_calculator(Re:float):
    """
    Compute the maximum lift coefficient (Cl_max) using XFoil.
     
    This function performs a 2D viscous airfoil analysis using XFoil in angle-of-attack sweep mode (ASEQ)
    to determine the maximum lift coefficient (Cl_max) for a specified Reynolds number.

    The procedure consists of:
        1. Generating a temporary XFoil input file.
        2. Executing XFoil via subprocess.
        3. Reading the generated polar output file.
        4. Extracting the maximum Cl value.

    Parameters
    ----------
    Re : float
        Reynolds number for the viscous airfoil analysis.
    
    Returns
    -------
    cl_max : float
        Maximum lift coefficient (Cl_max) obtained within the defined angle-of-attack sweep range.

    Side Effects
    ------------
    - Creates temporary XFoil input file.
    - Executes xfoil.exe via subprocess.
    - Generates polar temporary output file ('cl_max.csv').
    - Deletes temporary input and outút files.

    Notes
    -----
    - Requires XFoil executable available in the working directory.
    - Airfoil geometry is hard-coded (NACA 23015).
    - AoA sweep range is fexed between -10 and 20 degrees with 0.5 degree step.
    - Polar output formatting must remain consistent with XFoil defaults.
    """

    # Standard path definitions:
    input_path = "utils/xfoil_input.in"
    output_path = "utils/cl_max.csv"

    # 1. Create XFoil input file for AoA sweep (ASEQ):
    with open(input_path, "w") as xfoil_file:
        xfoil_file.write(
            f"PLOP\n"
            f"G F\n\n"
            f"NACA 23015\n"
            f"PPAR\nN 201\n\n\n"
            f"OPER\n"
            f"VPAR\nN 7\n\n"
            f"VISC {Re:.0f}\n"
            f"ITER 400\n"
            f"PACC\n"
            f"cl_max.csv\n\n"
            f"aseq -10 20 0.5\n"
            f"PACC\n\n"
            f"QUIT\n")
    xfoil_file.close()

    # 2. Execute XFoil suppressing console output:
    with open(os.devnull, "w") as FNULL:
        subprocess.call("xfoil.exe < xfoil_input.in", shell=True, stdout=FNULL, 
            stderr=subprocess.STDOUT, cwd="utils")

    # 3. Wait for the file writing completion and remove temporary input file:
    time.sleep(0.2)
    os.remove(input_path)

    # 4. Read XFoil polar results:
    xfoil_data = pd.read_csv(output_path, skiprows=12, sep=r"\s+", 
        names=['AoA', 'Cl', 'Cd', 'Cdp', 'Cm', 'Top_Xtr', 'Bot_Xtr'], usecols=['Cl'])
    
    # 5. Wait for the file reading completion and remove temporary output file:
    time.sleep(0.2)
    os.remove(output_path)

    # 6. Extract the maximum Cl value:
    cl_max = xfoil_data['Cl'].max()

    return cl_max

def analysis3D(input_file:str, Re_range:list):
    """
    Perform 3D aerodynamic analysis using AVL and generate sectional flow cases.

    This function executes AVL to compute spanwise lift coefficient distributions for a predefined angle-of-
    attack sweep starting at AoA = -4° (approximately the zero-lift angle).

    The resulting sectional data are expanded over a specified Reynolds number range, generating a structured
    dataset for subsequent 2D viscous analysis.
    
    Parameters
    ----------
    input_file : str
        Path to the AVL execution input file (.in) containing the predefined geometry and execution commands.

    Re_range : list
        Reynolds number range defined as:
        [Re_initial, Re_final, Re_step]

    Returns
    -------
    pd. DataFrame
        DataFrame containing:
            - Yle : Spanwise corrdinate [m]
            - Cl  : Sectional lift coefficient
            - AoA : Angle of attack [degrees]
            - Re  : Reynolds number

    Side Effects
    ------------
    - Executes avl.exe via subprocess.
    - Reads temporary AVL output files.
    - Deletes intermediate AoA .csv files.
    - Saves 'utils/2D-PotentialCases.csv'

    Notes
    -----
    - The AoA sweep begins at -4°, close to the zero-lift condition.
    - Post-stall filtering is NOT performed in this function.
    """

    # Defining the number of AoA cases present in the input file:
    n_conditions = 23
    AoA_i = -4
    AoA = []
    Re = []

    # Deletes the output file if it exists to avoid overwriting:
    if os.path.exists("utils/2D-PotentialCases.csv"):
        os.remove("utils/2D-PotentialCases.csv")
    if os.path.exists("utils/2D-LinearPotentialCases.csv"):
        os.remove("utils/2D-LinearPotentialCases.csv")

    # 1. Running AVL analysis using the pre-defined geometry (.avl) and conditions (.run) files:
    print("\nStarting AVL analysis...\n")
    with open(os.devnull, 'w') as FNULL:
        subprocess.call(f"avl.exe < {input_file}", shell=True, stdout=FNULL, stderr=subprocess.STDOUT, cwd="utils")
    print("AVL analysis completed!\n")

    # 2. Loading and merging the AVL data (Yle, cl) for each AoA case:
    for i in range(0, n_conditions):
        case = pd.read_csv(f"utils/AoA{i + AoA_i}.csv", header=16, nrows=80, sep=r'\s+', usecols=['Yle', 'cl'])
        case['AoA'] = i + AoA_i
        AoA.append(case)
        os.remove(f"utils/AoA{i + AoA_i}.csv")
    potential_cases = pd.concat(AoA, ignore_index=True)

    # 3. Merging the Reynolds Number information in the dataset for each AoA polar:
    for i in range(Re_range[0], Re_range[1], Re_range[2]):
        case = potential_cases.copy()
        case['Re'] = i
        Re.append(case)
    potential_cases = pd.concat(Re, ignore_index=True)

    # 4. Saving the results in a .csv file:
    potential_cases.to_csv("utils/2D-PotentialCases.csv", index=False)

    return potential_cases

def analysis2D(Re:float, cl:float, AoA:float, y:float, b_wing:float):
    """
    Perform 2D sectional analysis using XFoil.

    For a given sectional flow condition obtained from the 3D analysis, this function executes XFil to 
    compute the pressure coefficient (Cp) distribution along the airfoil chord.

    Parameters
    ----------
    Re : float
        Reynolds number of the current sectional case.

    cl : float
        Sectional lift coefficient of the current sectional case.

    AoA : float
        Angle of attack [degrees]of the current sectional case.

    y : float
        Spanwise coordinate [m] of the current sectional case.

    b_wing : float
        Wing half-span [m] (used for normalization of spanwise position).

    Returns
    -------
    pd. DataFrame
        Single-row DataFrame containing:
            - Re  : Reynolds number
            - AoA : Angle of attack [degrees]
            - y   : Normalized spanwise position (y / b_wing)
            - Cl  : Sectional lift coefficient
            - Cp  : 201 chordwise pressure coefficient (columns correspond to chordwise x locations)

    Side Effects
    ------------
    - Creates temporary XFoil input file.
    - Executes xfoil.exe via subprocess.
    - Reads generated Cp output file.
    - Deletes temporary input/output files.

    Notes
    -----
    - Requres XFOIL executable available in the working directory.
    - Airfoil geometry is hard-coded (NACA 23015).
    - Cp resolution is fixed at 201 chordwise points.
    """
    
    # Defining standard path variables:
    input_path_2d = "utils/xfoil_input.in"
    output_path_2d = "utils/cp_data.csv"

    # 1. Create a XFoil Input File:
    with open("utils/xfoil_input.in", "w") as xfoil_file:
        xfoil_file.write(
            f"PLOP\n"
            f"G F\n\n"
            f"NACA 23015\n"
            f"PPAR\nN 201\n\n\n"
            f"OPER\n"
            f"VPAR\nN 7\n\n"
            f"VISC {Re:.0f}\n"
            f"ITER 400\n"
            f"CL {cl:.3f}\n"
            f"cpwr\n"
            f"cp_data.csv\n\n"
            f"QUIT\n")
    xfoil_file.close()

    # 2. Run XFoil supressing the output:
    with open(os.devnull, 'w') as FNULL:
        subprocess.call("xfoil.exe < xfoil_input.in", shell=True, stdout=FNULL, stderr=subprocess.STDOUT, cwd="utils")

    # 3. Wait to the XFoil analysis to be written and remove the input file:
    time.sleep(0.2)
    os.remove(input_path_2d)

    # 4. Read the XFoil results data:
    xfoil_data = pd.read_csv(output_path_2d, skiprows=3, sep=r"\s+", names=['x', 'y', 'cp'], usecols=['x', 'cp'])

    # 5. Wait for the dataframe to be assembled and remove the output file:
    time.sleep(0.3)
    os.remove(output_path_2d)

    # 6. Assemble the final dataset with a single row:
    cp_data_2d = pd.DataFrame(data=[xfoil_data['cp'].values], columns=(np.round(xfoil_data['x'].values, 5)))
    cp_data_2d.insert(0, 'Re', Re)
    cp_data_2d.insert(1, 'AoA', AoA)
    cp_data_2d.insert(2, 'y', y/b_wing)
    cp_data_2d.insert(3, 'cl', cl) 

    return cp_data_2d

def main(new_cases:bool, Re_range:list, AoA_filter:bool, AoA_range:list, b_wing:float):
    """
    Execute the complete aerodynamic dataset generation workflow.

    This function coordinates the complete pipeline:

        1. Generate or load 3D potential flow cases (AVL).
        2. Optionally filter by AoA range.
        3. Compute Cl_max for each Reynolds number.
        4. Apply the critical section method:
            - Analyses start at AoA = -4°.
            - Dataset is truncated at the onset of stall.
            - All post-stall angles are removed.
        5. Perform 2D sectional Cp analysis only in the linear regime.
        6. Assemble and export the final structured dataset.
    
    Parameters
    ----------
    new_cases : bool
        If True, runs AVL to generate new sectional flow cases.
        If False, loads previously generated cases from:
        'utils/2D-PotentialCases.csv'.

    Re_range : list[int]
        Reynolds number range:
        [Re_initial, Re_final, Re_step]

    AoA_filter : bool
        If True, filters the potential flow cases based on AoA range.
    
    AoA_range : list [int]
        Angle of attack range:
        [AoA_initial, AoA_final]

    b_wing : float
        Wing half-span [m] used to normalize spanwise coordinates).

    Returns
    -------
    None
        This function saves the final dataset to disk.

    Output
    ------
    'Potential-PressureDistributionData.csv'
        Structured dataset containing:
            - Flow conditions (Re, AoA, y/b)
            - Sectional lift coefficient (Cl)
            - 201 chordwise pressure coefficient values per case (Cp)

    Side Effects
    ------------
    - May execute avl.exe
    - Executes xfoil.exe multiple times.
    - Deletes existing output file if present.
    - Writes final dataset to disk.

    Computational Cost
    ------------------
    Total tuntime scales approcimately with:
        N_AoA * N_spanwise_section * N_Re

    Notes
    -----
    - This function can generate large datasets depending on the selected Reynolds range and AoA 
      discretization. Disk space and runtime should be considered before execution.    
    - Only the linear portion of the lift curve is retained.
    - The stall condition is defined by:
            Cl_section > Cl_max(Re)
    - Once detected for a given Reynolds number, that AoA and all subsequent angles are discarded.
    """

    # Defining standard parameters:
    input_file_3d = "avl_input.in"
    output_path = "Potential-PressureDistributionData.csv"
    cl_max_values = []
    Re_values = np.arange(Re_range[0], Re_range[1], Re_range[2])
    Results = []

    # Deletes the output file if it exists to avoid overwriting:
    if os.path.exists(output_path):
        os.remove(output_path)

    # 1. Checks if it is needed to generate new flow conditions or use a previous set:
    if new_cases == True:
        potential_cases = analysis3D(input_file_3d, Re_range)
        print("Potential cases generated!\n")
    elif new_cases == False:
        potential_cases = pd.read_csv("utils/2D-PotentialCases.csv", dtype={'Yle': float, 'cl': float, 'AoA': float, 'Re': float})
        print("\nPotential cases loaded!\n")

    # 2. Checks if it is needed to filter the data by a defined range of AoA:
    if AoA_filter == True:
        filtered_AoA_cases = potential_cases[potential_cases['AoA'].between(AoA_range[0], AoA_range[1])].reset_index(drop=True)
        potential_cases = filtered_AoA_cases.copy()
        print("AoA filter successfully applied!\n")
    elif AoA_filter == False:
        potential_cases = potential_cases
        print("Using the full range of AoA!\n")

    # 3. Creates a dataframe accounting for the maximum lift coefficient for each Reynolds number:
    for Re in tqdm(Re_values, desc="Computing the maximum lift coefficient (Cl_max) values...", unit='Re'):
        cl_max_values.append(cl_max_calculator(Re))
    data_cl_max = pd.DataFrame(data={'Re':Re_values, 'cl_max': cl_max_values})
    print("\nMaximum lift coefficient calculation complete!\n")

    # 4. Apply the critical section method (linear regime filtering):
    data = potential_cases.merge(data_cl_max, how='left', on='Re')
    data = data.sort_values(['Re', 'AoA'])
    
    linear_cases = []
    for Re, group in data.groupby('Re'):
        # Identify first stall occurence:
        stall_limit = group['cl_max'] * (1 - stall_margin)
        stall_condition = group['cl'] > stall_limit
        if stall_condition.any():
            stall_aoa = group.loc[stall_condition, 'AoA'].iloc[0]
            group = group[group['AoA'] < stall_aoa]
        linear_cases.append(group)

    linear_potential_cases = pd.concat(linear_cases).drop(columns=['cl_max']).reset_index(drop=True)
    linear_potential_cases.to_csv("utils/2D-LinearPotentialCases.csv", index=False)
    os.remove("utils/2D-PotentialCases.csv")
    print("Linear region filtering completed successfully!\n")

    # 5. Runs the analysis for each flow case:
    for i in tqdm(range(len(linear_potential_cases)), desc='Generating aerodynamic dataset...', unit='case'):
        case = analysis2D(linear_potential_cases['Re'][i], linear_potential_cases['cl'][i], linear_potential_cases['AoA'][i], linear_potential_cases['Yle'][i], b_wing)
        Results.append(case)
    results = pd.concat(Results, ignore_index=True)

    # 6. Save the results:
    results.to_csv(output_path, index=False, sep=';')
    print(f"\nProcessing complete! Results saved to {output_path}\n")

if __name__ == "__main__":
    main(new_cases=new_cases, Re_range=[initial_Reynolds, final_Reynolds+1, step_Reynolds], AoA_filter=AoA_filter, AoA_range=[initial_AoA, final_AoA], b_wing=b_wing)