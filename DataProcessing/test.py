import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

data_true = 'Numerical-PressureDistributionData.csv'
data_pred = 'numerical_ann/Numerical-ANN-Predicted-PressureDistribution.csv'

data_true = pd.read_csv(data_true, sep=',')
data_pred = pd.read_csv(data_pred, sep=',')

data_dict = {'true': data_true, 'pred': data_pred}
data_labels = ['true', 'pred']
visual_styles = {'true': {'color': 'teal', 'linestyle': '-', 'label': 'CFD Data'}, 'pred': {'color': 'red', 'linestyle': '--', 'label': 'ANN Data'}}

section = 0

filtered_conds = []
span_pos = []
cp_data = []
taps = [5, 11, 22, 34, 45, 57, 68, 74]

chord_pos = [0.197015, 0.179695, 0.15588, 0.136395, 0.119075, 0.099590, 0.080105, 0.06495,
                 0.049795, 0.036805, 0.02598, 0.017320, 0.008660, 0.004330, 0.002165, 0.00000,
                 0.000000, 0.004330, 0.00866, 0.017320, 0.028145, 0.038970, 0.049795, 0.06495,
                 0.080105, 0.099590, 0.11691, 0.136395, 0.155880, 0.179695, 0.197015]

#if len(data_dict['true']['Re']) :
#print(len(data_dict['true']['y']))


for i in data_labels:
    filtered_cond = data_dict[i][(data_dict[i]['Re'] == 235456) & (data_dict[i]['AoA'] == 10)]
    filtered_cond = filtered_cond.iloc[taps].reset_index(drop=True)
    cp_data.append(filtered_cond.iloc[:,4:35].to_numpy())
    filtered_conds.append(filtered_cond)

span_pos = filtered_conds[0]['y'].to_numpy()

plt.xlabel('Chordwise Position [mm]', fontname='Times New Roman', fontsize=12)
#plt.xlim(0, 0.2165) 
plt.ylabel('Pressure Coefficient ($C_p$)', fontname='Times New Roman', fontsize=12)
#plt.ylim(1, -3.5)
plt.suptitle(f'Pressure Coefficient Distribution at y = {span_pos[section]}, Re = {12} and AoA = {10}°',
                fontname='Times New Roman', fontsize=14, fontweight='bold')

# 5.2. Plots the pressure distribution lines for a specific section:
for j in range(len(data_labels)):
    label = visual_styles[data_labels[j]]['label']
    y_section = np.full_like(chord_pos, span_pos[section])
    plt.plot(chord_pos, cp_data[j][section, :], color=visual_styles[data_labels[j]]['color'], 
        linestyle=visual_styles[data_labels[j]]['linestyle'], label=label, 
        linewidth=0.95, alpha=1)

# 5.3. Saves and shows the figure:
#print(f"\nPlot saved as {out_path}\n")
#plt.savefig(out_path, dpi=300)
plt.show()
