import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from scipy.interpolate import griddata
import pandas as pd

grid_res_x = 200
grid_res_y = 200
i = 0
j = 4*48000

data = pd.read_csv('data_test.csv', sep=',')
sample = data[0:4800]
true_cp = np.array(sample['cp'])

predic = sample.copy()
predicteddata = pd.read_csv('predictions.csv', sep=',')
predic_cp = np.array(predicteddata[0:4800]).T
predic_cp = predic_cp[0]
sample = predic.copy()
sample_upper = sample[sample['Surf._-1'] == 0].reset_index(drop=True)
sample_lower = sample[sample['Surf._-1'] == 1].reset_index(drop=True)


#data = pd.read_csv('PressureDistributionData.csv', sep=',')
#sample = data[0+(4800*i):4800+(4800*i)]
#sample_upper = sample[sample['Surf.'] == 1].reset_index(drop=True)
#sample_lower = sample[sample['Surf.'] == -1].reset_index(drop=True)

x_scatter_upper = sample_upper['x']*0.2165
y_scatter_upper = sample_upper['y']
cp_scatter_upper = sample_upper['cp']

x_scatter_lower = sample_lower['x']*0.2165
y_scatter_lower = sample_lower['y']
cp_scatter_lower = sample_lower['cp']

grid_x_1d_upper = np.linspace(x_scatter_upper.min(), x_scatter_upper.max(), grid_res_x)
grid_y_1d_upper = np.linspace(y_scatter_upper.min(), y_scatter_upper.max(), grid_res_y)
grid_x_upper, grid_y_upper = np.meshgrid(grid_x_1d_upper, grid_y_1d_upper)
grid_cp_upper = griddata((x_scatter_upper, y_scatter_upper), cp_scatter_upper, (grid_x_upper, grid_y_upper), method='linear')

grid_x_1d_lower = np.linspace(x_scatter_lower.min(), x_scatter_lower.max(), grid_res_x)
grid_y_1d_lower = np.linspace(y_scatter_lower.min(), y_scatter_lower.max(), grid_res_y)
grid_x_lower, grid_y_lower = np.meshgrid(grid_x_1d_lower, grid_y_1d_lower)
grid_cp_lower = griddata((x_scatter_lower, y_scatter_lower), cp_scatter_lower, (grid_x_lower, grid_y_lower), method='linear')


fig, ax = plt.subplots(1, 2, figsize=(12, 4), sharex=True, sharey=True)
levels = 200
cmap = 'RdYlBu_r' #viridis

vmin = grid_cp_lower.min()
vmax = 1

ax[0].set_title('Predicted Upper Surface\n', fontsize=14, fontname = 'Times New Roman', fontweight = 'bold')
cf1 = ax[0].contourf(grid_y_upper, grid_x_upper, grid_cp_upper, levels=levels, cmap=cmap, vmin=vmin, vmax=vmax)
ax[0].set_xlabel('Spanwise Position (y)', fontsize=12, fontname = 'Times New Roman')
ax[0].set_ylabel('Chordwise Position (x)', fontsize=12, fontname = 'Times New Roman')
ax[0].set_aspect('equal', adjustable='box')
ax[0].invert_yaxis() # Invert y-axis (chord)

ax[1].set_title('Predicted Lower Surface\n', fontsize=14, fontname = 'Times New Roman', fontweight = 'bold')
cf2 = ax[1].contourf(grid_y_lower, grid_x_lower, grid_cp_lower, levels=levels, cmap=cmap, vmin=vmin, vmax=vmax)
ax[1].set_xlabel('Spanwise Position (y)', fontsize=12, fontname = 'Times New Roman')
ax[1].set_ylabel('Chordwise Position (x)', fontsize=12, fontname = 'Times New Roman')
ax[1].set_aspect('equal', adjustable='box')
ax[1].invert_yaxis() # Invert y-axis (chord)

mappable = plt.cm.ScalarMappable(cmap=cmap, norm=mcolors.Normalize(vmin=vmin, vmax=vmax))

fig.subplots_adjust(bottom=0.2, top=0.9, wspace=0.2)
cax = fig.add_axes([0.25, 0.15, 0.5, 0.03])

cbar = fig.colorbar(mappable, cax=cax, orientation='horizontal', label='Pressure Coefficient')
fig.savefig('surface_pressure_predicted1.png', dpi=300)

plt.show()