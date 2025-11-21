import visualization_tools.data_surface_visualization as surf_vis
import visualization_tools.data_sectional_visualization as sec_vis


data_path = 'Numerical-PressureDistributionData.csv'
Re = 235456
AoA = 10
is_pred = False
scatter = True
is_exp = False
#out_path = f"plots/SurfacePressure_Re{Re}_AoA{AoA}.png"
out_path = f"plots/SectionalPressure_Re{Re}_AoA{AoA}.png"

#surf_vis.main(data_path, Re, AoA, is_pred, out_path)
sec_vis.main(data_path, Re, AoA, scatter, is_exp, out_path)