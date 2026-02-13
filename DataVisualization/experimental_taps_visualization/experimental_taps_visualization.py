"""
--------------------------------------------------------------------------------------------------------
Function:               experimental_taps_visualization
Author:                 Caio Dias Filho
Creation date:          2025-11-20
Last modification:      2025-11-21
Version:                1.0

Description:
    This script performs a 3D visualization of the experimental pressure taps distribution over the wing
    surface. It allows selecting between the 'upper' or 'lower' surface for the visualization.
        
Dependencies:
    - mpl_toolkits
    - matplotlib
    - pandas
    - numpy

Future implementations:
    >>> ALL IMPLEMENTATIONS DONE!
--------------------------------------------------------------------------------------------------------
"""

from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np



def wing_surface_visualization(surface: str, out_path:str):
    """
    Performs a 3D visualization of the experimental pressure taps location on the wing surface.

    Args:
        surface: The surface to be visualized, either 'upper' or 'lower'.
        out_path: Path and name to save the output image (.png).

    Returns:
        None. Saves the 3D visualization in an image file.     
    """

    # Defining path parameters:
    airfoil_path = 'utils/NACA23015.dat'
    upper_taps_path = 'utils/experimental_upper_taps.csv'
    lower_taps_path = 'utils/experimental_lower_taps.csv'

    # 1. Defining the wing geometry parameters:
    airfoil = pd.read_csv(airfoil_path, sep=',', names = ['x', 'y'])
    x_sectional = (airfoil['x'].values) * 0.2165
    y_sectional = [0.008, 0.170, 0.332, 0.477, 0.598, 0.690, 0.746, 0.761]
    z_sectional = (airfoil['y'].values) * 0.15

    # 2. Setting the data for the wing surface plot:
    X = np.tile(x_sectional, (len(y_sectional), 1)).T
    Y = np.tile(y_sectional, (len(x_sectional), 1))
    Z = np.tile(z_sectional, (len(y_sectional), 1)).T

    # 3. Loading the experimental taps data based on the surface selection:
    if surface == 'upper':
        taps_path = upper_taps_path
        x_label_pad, y_label_pad = 20, 60
        x_ticks_pad, y_ticks_pad = 10, 10
        title_y_pos = 0.87
        view_elev = 30
        title_text = 'Experimental Pressure Measurements Distribution - Upper Surface'
    else:
        taps_path = lower_taps_path
        x_label_pad, y_label_pad = 12, 60
        x_ticks_pad, y_ticks_pad = 8, 25
        title_y_pos = 1
        view_elev = -30
        title_text = 'Experimental Pressure Measurements Distribution - Lower Surface'

    # 4. Reading the experimental taps data:
    taps_position = pd.read_csv(taps_path, sep=',', names=['x', 'y', 'z'])
    X_taps = taps_position['x'].values
    Y_taps = taps_position['y'].values
    Z_taps = taps_position['z'].values

    # 5. Setting the figure properties:
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_box_aspect([0.2165, 0.766, 0.022])

    # 6. Setting the axes properties:
    ax.set_xlabel('Chordwise Position [mm]', labelpad=x_label_pad, fontname='Times New Roman', fontsize=12)
    ax.tick_params(axis='x', pad=x_ticks_pad)
    ax.set_xticks([0, 0.072, 0.144, 0.216])
    ax.set_xlim(0, 0.2165)

    ax.set_ylabel('Spanwise Position [mm]', labelpad=y_label_pad, fontname='Times New Roman', fontsize=12)
    ax.tick_params(axis='y', pad=y_ticks_pad)
    ax.set_ylim(0, 0.766)

    ax.zaxis._axinfo["grid"]['color'] =  (1.0, 1.0, 1.0, 0.0)
    ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    ax.set_zticks([])

    ax.set_title(title_text, fontname='Times New Roman', fontsize=14, fontweight='bold', y=title_y_pos)

    # 7. Plotting the wing surface:
    ax.plot_surface(X, Y, Z, color='darkgray', shade=False, edgecolor='black', linewidth=0.05)

    # 8. Adding the wing's root and tip section airfoil:
    v_root = np.array([X[:,0], Y[:, 0], Z[:, 0]]).T
    ax.add_collection3d(Poly3DCollection([v_root], facecolors='darkgray', edgecolors='black', alpha=1.0))
    v_tip = np.array([X[:,-1], Y[:, -1], Z[:, -1]]).T
    ax.add_collection3d(Poly3DCollection([v_tip], facecolors='darkgray', edgecolors='black', alpha=1.0))

    # 9. Plotting the pressure taps:
    ax.scatter(X_taps, Y_taps, Z_taps, color='red', s=5, edgecolors='black', depthshade=False, 
               label=' Pressure Taps')

    # 10. Saves and shows the figure:
    ax.view_init(elev=view_elev, azim=-130)
    print(f"\nPlot saved as {out_path}\n")
    plt.savefig(out_path, dpi=300)
    plt.show()

    return

def main(surface:str, out_path: str):
    """
    Main execution workflow: Calls the 3D visualization function.
    """

    # 1. Calls the 3D visualization function:
    print("\nStarting experimental taps visualization process...\n")
    wing_surface_visualization(surface, out_path)
    print("\nProcess finished successfully.\n")
    
    return



if __name__ == "__main__":

    # 1. Sets the run parameters:
    SURFACE = 'lower'
    OUTPUT_FILE = f"ExperimentalTapsDistribution_{SURFACE}.png"

    # 2. Calls the main function:
    main(SURFACE, OUTPUT_FILE)