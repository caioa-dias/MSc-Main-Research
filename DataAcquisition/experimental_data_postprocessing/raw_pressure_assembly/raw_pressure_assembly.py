



import tkinter as tk
from tkinter import filedialog
import pandas as pd
import os

def select_data_files():
    root = tk.Tk()
    root.withdraw()

    file_paths = filedialog.askopenfilenames(title='Select .dat files for the same test condition',
        filetypes=[('DAT files', '*.dat')])

    return file_paths

def map_files(file_paths: list):
    Alpha = None
    BetaA = None
    BetaB = None

    for path in file_paths:
        name = os.path.basename(path)

        if 'SCAN000' in name:
            Alpha = path
        elif 'SCAN001' in name:
            BetaA = path
        elif 'SCAN002' in name:
            BetaB = path

    return Alpha, BetaA, BetaB

file_paths = select_data_files()
Alpha, BetaA, BetaB = map_files(file_paths)

files = [Alpha, BetaA, BetaB]
data_list = []

for i, file in enumerate(files):
    data = pd.read_csv(file, sep=r'\s+', header=None, names=['Group', 'Module', 'Channel', 'Tap', 'Pressure'])
    data = data.drop(columns=['Group', 'Module', 'Channel'])

    data['Tap'] = data['Tap'] + (i * 64)

    data_list.append(data)

full_data = pd.concat(data_list, axis=0).reset_index(drop=True)
full_data['Tap'] = full_data['Tap'].astype(str)
full_data.loc[full_data.index[-2], 'Tap'] = 'Dynamic pressure'
full_data.loc[full_data.index[-1], 'Tap'] = 'Total pressure'

file_path = file_paths[0]
folder_path = os.path.dirname(file_path)
folder_name = os.path.basename(folder_path)
output_path = os.path.join(folder_path, f"{folder_name}.csv")
file_name = os.path.basename(output_path)

save_path = os.path.join('experimental_data_postprocessing/raw_pressure_assembly/assembled-raw-pressure', file_name)

full_data.to_csv(save_path, index=False)

# até aqui eu gero os dados brutos de pressão
####

# aqui eu realizo a correção dos dados de pressão



### aqui eu calculo o Cp e gero o arquivo final de Cp + plot