# -*- coding: utf-8 -*-
"""
========================================================================================================
Module: potential_data_effective_angle
Author: Caio Dias Filho
Creation date: 2026-03-18
Last modification: 2026-03-19
Version: 1.0
========================================================================================================

OVERVIEW
--------
This module generates an aerodynamic dataset containing the effective angle of attack (AoA_effective)for
wing sections based on coupling between 3D potential flow analysis (AVL) and 2D airfoil analysis (XFoil).

The workflow consists of:

    - Generating 3D aerodynamic conditions (spanwise distributions of Cl) using AVL. 
    - Computing the maximum lift coefficient (Cl_max) using XFoil for each Reynolds number.
    - Filtering out post-stall conditions using a defined stall margin.
    - Running 2D XFoil simulations to compute the effective angle of attack for each section.

The final dataset contains:

    - Reynolds number (Re).
    - Wing geometric angle of attack (AoA).
    - Spanwise position (y/b).
    - Sectional lift coefficient (Cl).
    - Sectional effective angle of attack (AoA_effective).

    
DEPENDENCIES
------------
External tools:

    - XFoil (required in /utils directory)
    - AVL (required in /utils directory)

Pytho libraries:

    - numpy
    - pandas
    - tqdm
    - subprocess
    - time 
    - os

    
OUTPUT FILES
------------
Generated datasets:

    Potential-AoAEffectiveData.csv
        Final dataset containing AoA_effective values.

Intermediate files (temporary):

    utils/2D-PotentialCases.csv
    utils/2D-LinearPotentialCases.csv
    utils/xfoil_input.in
    utils/cl_max.csv
    utils/cp_data.csv


NOTES
-----
- Each aerodynamic case contains 80 spanwise sections.
- XFoil is used in two distinct modes:
    1. AoA sweep (ASEQ) -> to determine Cl_max.
    2. Fixed Cl -> to compute AoA_effective.
- AVL provides spanwise Cl distributions used as input for XFoil.
- Post-stall conditions are removed using a conservative margin based on Cl_max.

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

    This function runs an XFoil simulation for a given Reynolds number using an angle-of-attack sweep
    (ASEQ). The resulting polar data is used to extract the maximum lift coefficient, which is later used
    for stall detection.

    Parameters
    ----------
    Re : float
        Reynolds number used in the XFoil simulation.

    Returns
    -------
    cl_max : float
        Maximum lift coefficient obtained from the XFoil polar.
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
    Perform 3D aerodynamic analysis using AVL.

    This function executes AVL using predefined geometry and run files, extracts spanwise sectional lift
    coefficients (Cl) for multiple angles of attack, and expands the dataset across a range of Reynolds
    numbers.

    The resulting dataset represents potential flow conditions for the wing.

    Parameters
    ----------
    input_file : str
        Name of the AVL input file containing simulation commands.

    Re_range : list
        List defining the Reynolds number range in the format:
        [initial_Re, final_Re, step].

    Returns
    -------
    potential_cases : pandas.DataFrame
        Dataset containing spanwise Cl distributions for all AoA and Re cases.
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
    Compute the effective angle of attack using XFoil.

    This function runs a 2D airfoil simulation in XFoil at a fixed sectional lift coefficient (Cl) and 
    extracts the corresponding effective angle of attack (AoA_effective).

    The result represents the equivalent 2D angle that produces the same lift as the 3D wing section.

    Parameters
    ----------
    Re : float
        Reynolds number.

    cl : float
        Sectional lift coefficient.

    AoA : float
        Wing geometric angle of attack.

    y : float
        Spanwise position.

    b_wing : float
        Wing half-span used for normalization.

    Returns
    -------
    AoA_effective_2d : pandas.DataFrame
        Single-row DataFrame containing:
            - Re
            - AoA
            - y/b
            - cl
            - AoA_effective
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
    with open('utils/cp_data.csv', 'r') as f:
        line = f.readlines()[1]
    value = line.split('Alfa = ')[1].split()[0]

    # 5. Wait for the dataframe to be assembled and remove the output file:
    time.sleep(0.3)
    os.remove(output_path_2d)

    # 6. Assemble the final dataset with a single row:
    AoA_effctive_2d = pd.DataFrame(data=[float(value)], columns=['AoA_Effective'])
    AoA_effctive_2d.insert(0, 'Re', Re)
    AoA_effctive_2d.insert(1, 'AoA', AoA)
    AoA_effctive_2d.insert(2, 'y', y/b_wing)
    AoA_effctive_2d.insert(3, 'cl', cl) 

    return AoA_effctive_2d

def main(new_cases:bool, Re_range:list, AoA_filter:bool, AoA_range:list, b_wing:float):
    """
    Execture the full dataset generation pipeline.

    This function orchestrates the complete workflow:

        1. Generate or laod 3D aerodynamic cases (AVL).
        2. Optionally filter cases based on angle of attack.
        3. Compute Cl_max for each Reynolds number using XFoil.
        4. Apply linear regime filtering (pre-stall conditions).
        5. Compute effective angle of attack for each wing section.
        6. Assemble and save the final dataset.

    Parameters
    ----------
    new_cases : bool
        If True, new AVL simulations are executed.
        If False, previously generated cases are loaded.

    Re_range : list
        Reynolds number range in the format:
        [initial_Re, final_Re, step].

    AoA_filter : bool
        If True, filters the dataset using the specified AoA range.

    AoA_range : list
        Range of angle of attack [AoA_min, AoA_max] in degrees used when filtering is enabled.

    b_wing : float
        Wing half-span used for normalization of spanwise position.

    Returns
    -------
    None
        The function sabes the generated dataset to disk.
    """

    # Defining standard parameters:
    input_file_3d = "avl_input.in"
    output_path = "Potential-AoAEffectiveData.csv"
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