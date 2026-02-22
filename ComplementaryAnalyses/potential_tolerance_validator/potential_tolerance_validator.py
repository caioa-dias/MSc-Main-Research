# -*- coding: utf-8 -*-
"""
========================================================================================================
Module: potential_tolerance_validator
Author: Caio Dias Filho
Creation date: 2026-02-21
Last modification: 2026-02-22
Version: 1.0
========================================================================================================

OVERVIEW
--------
This module is designed to validate the physical tolerance threshold applied in the 
'potential_data_checker.py' script.

The primary objective is to prove that the adopetd tolerance for the absolute error (|ΔCl|) effectively
identifies and isolates non-converged XFoil solutions. To induce these numerical non-convergences, this
script evaluates deliberately extreme conditions: the first 8 spanwise sections near the wing root (where
stall initiates) at angles of attack strictly above the stall angle (post-stall regime) across different
Reynolds numbers.

By applying the Cp-integration method to this specific post-stall dataset, the script demonstrates that
the physical tolerance successfully filters out the inconsistent observations generated when XFoil fails
to converge under separated flow conditions.


The primary objective is to prove that the adopted tolerance for the absolute error (|ΔCl|) effectively 
identifies and isolates non-converged XFoil solutions. To induce these numerical non-convergences, this 
script evaluates deliberately extreme conditions: the first 8 spanwise sections near the wing root (where 
stall typically initiates) at angles of attack strictly above the stall angle (post-stall regime) across 
different Reynolds numbers.

By applying the Cp-integration method to this specific post-stall dataset, the script demonstrates that
the physical tolerance successfully filters out the inconsistent observations generated when XFoil fails 
to converge under separated flow conditions. The results shows that unconverged solutions usually presents
errors above 0.06, which is well within the physical tolerance threshold (0.0095), while all converged
solutions present errors below the tolerance.


DEPENDENCIES
------------
Python libraries:
    - tqdm
    - pandas
    - numpy
    - subprocess
    - time
    - os

    
OUTPUT FILES
------------
- potential_error_report.txt
- Potential-PostStallData.csv

NOTES
-----
- The dataset 'Potential-PostStallData.csv' contains specifically post-stall data to test the limits of 
  the XFoil solver and trigger non-convergence.
- The same integration and error evaluation methodologies from the data checker are applied here.

========================================================================================================
"""

# =======================================================================================================
#                                            USER INPUT SECTION
# =======================================================================================================

# --- Wing geometry data ---
b_wing = 0.766

# --- Validation parameters ---
# Physical tolerance for |ΔCl| to be validated:
tolerance = 0.0095


# =======================================================================================================
#                                              CORE FUNCTION
# =======================================================================================================

from tqdm import tqdm
import pandas as pd
import numpy as np
import subprocess
import time
import os

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

def compute_cl_from_cp(cp_data:np.ndarray, airfoil_data:pd.DataFrame, AoA:float):
    """
    Recompute the sectional lift coefficient (Cl) by integrating the pressure coefficient (Cp).

    This function performs a numerical integration of the discrete Cp distribution over the airfoil surface
    to obtain the normal (Cn) and axial (Ca) force coefficients. These are subsequently projected using the
    Angle of Attack (AoA) to calculate the lift coefficient (Cl).

    Parameters
    ----------
    cp_data : np.ndarray
        Array containing the 201 chordwise pressure coefficient values.

    airfoil_data : pd.DataFrame
        DataFrame containing the gometry of the airfoil with columns ['x', 'y'].

    AoA : float 
        Angle of attack [degrees] of the current sectional case.

    Returns
    -------
    calc_cl : float
        The calculated sectional lift coefficient integrated from the Cp distribution.

    Notes
    -----
    - Integration assumes a standard trapezoidal-like approximation based on panel deltas.
    - Skin friction contribution is ignored; only pressure forces are integrated.
    """

    # 1. Calculte panel geometric deltas (dx, dy):
    panel_dx = np.diff(airfoil_data['x'].values)
    panel_dy = np.diff(airfoil_data['y'].values)

    # 2. Compute the average Cp for each panel:
    panel_cp = (cp_data[:-1] + cp_data[1:])/2

    # 3. Integrate to find normal (Cn) and axial (Ca) force coefficients:
    airfoil_cn = np.sum(panel_cp * panel_dx)
    airfoil_ca = -np.sum(panel_cp * panel_dy)

    # 4. Project Cn and Ca into the lift coefficient (Cl) using the angle of attack:
    calc_cl = airfoil_cn * np.cos(np.radians(AoA)) - airfoil_ca * np.sin(np.radians(AoA))

    return calc_cl

def evaluate_absolute_error(dataset:pd.DataFrame, airfoil_data:pd.DataFrame):
    """
    Evaluate the absolute error between the dataset Cl and the Cp-integrated Cl for all cases.

    Iterates through the entire aerodynamic dataset, recomputing the lift coefficient for each
    observations and calculating the absolute difference against the reference Cl generated by XFoil.

    Parameters
    ----------
    dataset : pd.DataFrame
        The generated potential dataset containing Re, AoA, y, cl, and the 201 Cp values.

    airfoil_data : pd.DataFrame
        DataFrame containing the geometry of the airfoil.

    Returns
    -------
    results_df : pd.DataFrame
        DataFrame containing the validation metrics for each case:
            - Re : Reynolds number
            - AoA : Angle of attack [degrees]
            - y : Normalized spanwise position
            - cl : Original reference lift coefficient
            - absolute_error : Absolute difference between original and integrated Cl
    """

    # 1. Initialize the results list:
    results = []
    
    # 2. Iterate through all the observations in the dataset to compute errors:
    for i in tqdm(range(len(dataset)), desc="Evaluating Cl absolute error...", unit="case"):
        # Extract paramaters for the current case:
        cp_values = dataset.iloc[i, 4:].values
        Re = dataset.iloc[i, 0]
        AoA = dataset.iloc[i, 1]
        y = dataset.iloc[i, 2]
        true_cl = dataset.iloc[i, 3]

        # Recompute Cl and calculate the absolute error:
        calc_cl = compute_cl_from_cp(cp_values, airfoil_data, AoA)
        abs_error = np.abs(true_cl - calc_cl)

        results.append([Re, AoA, y, true_cl, abs_error])

    # 3. Assemble and return the results DataFrame:
    results_df = pd.DataFrame(results, columns=['Re', 'AoA', 'y', 'cl', 'absolute_error'])

    return results_df

def generate_error_report(results:pd.DataFrame, tolerance:float):
    """
    Generate a formatted text report of the validation results and identify problematic cases.

    This function filters the results to find observations where the absolute error exceeds the predefined
    physical tolerance, compiling these cases into a readable text report.

    Parameters
    ----------
    results : pd.DataFrame
        DataFrame containing validation metrics (Re, AoA, y, cl, absolute_error) for all cases.

    tolerance : float
        Maximum acceptable absolute difference between reference Cl and integrated Cl.

    Returns
    -------
    errors.index : pd.Index
        The dataset indices of the observations that exceeded the defined tolerance.

    Side Effects
    ------------
    - Writes a structured text report to 'potential_error_report.txt'.
    """

    # 1. Filter observations that exceed the defined tolerance:
    errors = results[results['absolute_error'] > tolerance]

    # 2. Compute global error statistics:
    n_errors = len(errors)
    total = len(results)
    percentage = 100 * n_errors / total

    # 3. Assemble the report header and statistics:
    content = f"""
ABSOLUTE ERROR ANALYSIS REPORT - Cl CALCULATION
===========================================================
Date: {time.strftime("%Y-%m-%d %H:%M:%S")}

1. GLOBAL ERROR STATISTICS
----------------------------------
Tolerance adopted: {tolerance:.4f}.
Total observations: {total}.
Observations above tolerance: {n_errors}.
Percentage above tolerance: {percentage:.2f}%.

2. OBSERVATIONS ABOVE TOLERANCE
----------------------------------
"""

    # 4. Append detailed information for each problematic observation:
    if n_errors > 0:
        detailed = "\n".join([
            f"Re={row.Re:<10.0f} | AoA={row.AoA:<6.0f} | y={row.y:<8.4f} | "
            f"Cl={row.cl:<10.4f} | AbsError={row.absolute_error:.6f}"
            for _, row in errors.iterrows()])
        content += detailed
    else:
        content += "No observations exceeded the defined tolerance."
    content += "\n===========================================================\n"

    # 5. Write the report to disk:
    with open('potential_error_report.txt', "w", encoding="utf-8") as f:
        f.write(content)

    print(f"\nValidation report saved to: 'potential_error_report.txt'.\n")

    return errors.index

def main():
    """
    Execute the data generation and validation workflow for post-stall conditions.

    This function coordinates the complete pipeline:
        1. Read extreme post-stall flow conditions from a raw file.
        2. Run 2D aerodynamic analysis (XFoil) for each case to extract Cp distributions.
        3. Save the generated dataset to 'Potential-PostStallData.csv'.
        4. Remove hard non-convergences (NaN cases) where XFoil failed to output any Cp.
        5. Compute absolute errors for Cl using Cp numerical integration.
        6. Validate if the predetermined tolerance correctly flags bad observations.
        7. Save the final validation report.

    Returns
    -------
    None
        This function outputs console prints and saves the dataset and report to disk.

    Outputs
    -------
    'Potential-PostStallData.csv'
        The generated dataset containing Re, AoA, y, cl, and the 201 Cp values for the extreme post-stall 
        conditions.
        
    'potential_error_report.txt'
        Text file detailing the global error information and the specific observations that exceeded the 
        defined tolerance.
    """

    print('\nStarting data generation for post-stall aerodynamic dataset...\n')

    # 1. Load the raw post-stall conditions:
    stalled_cases = pd.read_csv("utils/2D-PostStallPotentialCases.csv", sep=';')

    # 2. Generate the dataset by running XFoil for each condition:
    Results = []
    for i in tqdm(range(len(stalled_cases)), desc='Generating aerodynamic dataset...', unit='case'):
        case = analysis2D(stalled_cases['Re'][i], stalled_cases['cl'][i], stalled_cases['AoA'][i], 
            stalled_cases['y'][i], b_wing)
        Results.append(case)

    # 3. Assemble and save the generated dataset:
    dataset = pd.concat(Results, ignore_index=True)
    dataset.to_csv('Potential-PostStallData.csv', index=False, sep=';')
    print(f"\nDataset successfully generated and saved to 'Potential-PostStallData.csv'.\n")

    print('Starting tolerance validation on post-stall dataset...\n')

    # 4. Load reference airfoil geometry for integration:
    airfoil_data = pd.read_csv('utils/NACA23015.csv', sep=',', names=['x','y'])

    # 5. Remove NaN observations (hard non-convergences where XFoil failed to output Cp):
    initial_len = len(dataset)
    dataset = dataset.dropna().reset_index(drop=True)
    nan_removed = initial_len - len(dataset)
    if nan_removed > 0:
        print(f"Removed {nan_removed} NaN observations (missing XFoil output) before analysis.\n")

    # 6. Compute absolute errors for the remaining observations:
    results = evaluate_absolute_error(dataset, airfoil_data)

    # 7. Generate validation report and extracts indices of observations that exceeded the tolerance:
    indices_to_remove = generate_error_report(results, tolerance)

    # 8. Filter the dataset based on the physical tolerance:
    filtered_dataset = dataset.drop(index=indices_to_remove)

    # 9. Print the final validation results:
    print(f'Original post-stall dataset size (without NaNs): {len(dataset)} observations.')
    print(f'Observations identified as non-converged (above tolerance): {len(indices_to_remove)} observations.')
    print(f'Observations physically consistent (below tolerance): {len(filtered_dataset)} observations.')

    print('\nTolerance validation completed successfully!\n')

    return

if __name__ == "__main__":
    main()