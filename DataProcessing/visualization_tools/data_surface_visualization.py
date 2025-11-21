"""
--------------------------------------------------------------------------------------------------------
Function:               data_surface_visualization
Author:                 Caio Dias Filho
Creation date:          2025-11-19
Last modification:      2025-11-20
Version:                1.0

Description:
    This script performs a 3D visualization of the pressure coefficient (Cp) distribution over the wing
    surface for a specific flight condition (Re, AoA). The visualization displays the full wing surface
    coloref by the Cp dstribution.
        
Dependencies:
    - mpl_toolkits
    - matplotlib
    - scipy
    - pandas
    - numpy

Future implementations:
    >>> ALL IMPLEMENTATIONS DONE!
--------------------------------------------------------------------------------------------------------
"""

from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib.colors import TwoSlopeNorm
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import pandas as pd
import numpy as np



def data_surface_visualization(data:pd.DataFrame, Re:int, AoA:int, is_pred:bool, surface:str, out_path:str):
    """
    Performs a 3D visualization of the pressure coefficient (Cp) distribution over the wing surface
    for a specified Reynolds Number (Re) and Angle of Attack (AoA).

    Args:
        data: DataFrame containing the pressure coefficient (cp) for 31 chordwise points and the corresponding
              sectional lift coefficient (Cl), along with the input conditions ('Re', 'AoA', 'y').
        Re: Reynolds Number for the aimed flight condition.
        AoA: Angle of Attack for the aimed flight condition.
        is_pred: If True, assumes that the data is the output of a predictive model and not the real data.
        surface: The surface to be visualized, either 'upper' or 'lower'.
        out_path: Path and name to save the output image (.png).

    Returns:
        None. Saves the 3D visualization in an image file.     
    """
    
    # Defining nomenclature parameters:
    if is_pred == True:
        name = 'Predicted'
    else:
        name = 'Numerical'

    # Defining surface parameters:
    if surface == 'upper':
        title_text = f'{name} Pressure Coefficient Distribution at Re = {Re} and AoA = {AoA}° - Upper Surface'
        view_elev = 30
    else:
        title_text = f'{name} Pressure Coefficient Distribution at Re = {Re} and AoA = {AoA}° - Lower Surface'
        view_elev = -30
    
    # 1. Filters the data for the specific input flight condition (Re and AoA):
    filtered_cond = data[(data['Re'] == Re) & (data['AoA'] == AoA)]

    # 2. Defining the wing geometry parameters:
    airfoil = pd.read_csv('visualization_tools/utils/NACA23015.dat', sep=',', names = ['x', 'y'])
    x_sectional = (airfoil['x'].values)*0.2165
    y_sectional = filtered_cond['y']
    z_sectional = (airfoil['y'].values)*0.15

    # 3. Setting the data for the 3D plot:
    X = np.tile(x_sectional, (len(y_sectional), 1)).T
    Y = np.tile(y_sectional, (len(x_sectional), 1))
    Z = np.tile(z_sectional, (len(y_sectional), 1)).T
    cp_data = filtered_cond.iloc[:,4:].to_numpy()

    # 4. Interpolating the chordwise data to match the airfoil discretization:
    idx_real = np.linspace(0, 1, 31)
    idx_target = np.linspace(0, 1, len(x_sectional))
    f_interp = interp1d(idx_real, cp_data.T, kind='cubic', axis=0)
    cp_data = f_interp(idx_target)

    # 5. Setting the figure properties:
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_box_aspect([0.2165, 0.766, 0.022])

    # 6. Setting the axes properties:
    fig.suptitle(title_text, fontname='Times New Roman', fontsize=14, fontweight='bold', y=0.77)
    ax.set_axis_off()

    # 7. Setting the pressure coefficient colormap:
    norm = TwoSlopeNorm(vmin=-3.5, vcenter=0, vmax=1)
    cmap = cm.RdYlBu_r
    face_colors = cmap(norm(cp_data))

    # 8. Plotting the surface pressure coefficient distribution:
    ax.plot_surface(X, Y, Z, facecolors=face_colors, shade=False, edgecolor='black', linewidth=0.05)

    # 9. Adding the wing's root and tip section airfoil:
    v_root = np.array([X[:,0], Y[:, 0], Z[:, 0]]).T
    ax.add_collection3d(Poly3DCollection([v_root], facecolors='darkgray', edgecolors='black', alpha=1.0))
    v_tip = np.array([X[:,-1], Y[:, -1], Z[:, -1]]).T
    ax.add_collection3d(Poly3DCollection([v_tip], facecolors='darkgray', edgecolors='black', alpha=1.0))

    # 10. Adding a colorbar for reference of the Cp distribution:
    m = (cm.ScalarMappable(cmap=cmap, norm=norm))
    m.set_array([])
    cbar = plt.colorbar(m, ax=ax, shrink=0.5, aspect=12)
    cbar.set_label('Pressure Coefficient ($C_p$)', labelpad=10, fontname='Times New Roman', fontsize=12)

    # 11. Saves and shows the figure:
    print(f"\nPlot saved as {out_path}\n")
    ax.view_init(elev=view_elev, azim=-130)
    plt.savefig(out_path, dpi=300)
    plt.show()

    return

def main(data_path: str, Re: int, AoA: int, is_pred: bool, surface:str, out_path: str):
    """
    Main execution workflow: Loads the data and calls the 3D visualization function.
    """

    # 1. Loads the data:
    print("\nStarting 3D visualization process...\n")
    print(f"Loading data from: {data_path}")
    data = pd.read_csv(data_path, sep=',')
    print("Data loaded successfully.")

    # 2. Calls the 3D visualization function:
    data_surface_visualization(data, Re, AoA, is_pred, surface, out_path)
    
    return



if __name__ == "__main__":

    # 1. Sets the run parameters:
    DATA_FILE = 'Numerical-PressureDistributionData.csv'
    REYNOLDS_NUMBER = 235456
    ANGLE_OF_ATTACK = 10
    IS_PREDICTION_DATA = False
    SURFACE = 'upper'
    OUTPUT_FILE = f"SurfacePressure_Re{REYNOLDS_NUMBER}_AoA{ANGLE_OF_ATTACK}_{SURFACE}.png"

    # 2. Calls the main function:
    main(DATA_FILE, REYNOLDS_NUMBER, ANGLE_OF_ATTACK, IS_PREDICTION_DATA, SURFACE, OUTPUT_FILE)