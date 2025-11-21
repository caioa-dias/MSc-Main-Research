import pandas as pd
from keras.models import Sequential
from keras.layers import Dropout
from keras.layers import Dense, Input
from sklearn.preprocessing import StandardScaler, OneHotEncoder, MinMaxScaler, MaxAbsScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
import numpy as np
from matplotlib import pyplot as plt
import tensorflow as tf

model = tf.keras.models.load_model('trained_model.keras')

data = pd.read_csv('PressureDistributionData.csv', sep=',')

lines_case = 4800
test_samples = 30

indices = np.arange(len(data))
data['group_id'] = indices // lines_case

all_group_ids = data['group_id'].unique()

data_train_id, data_test_id = train_test_split(all_group_ids, test_size=test_samples, random_state=42)

data_train = data[data['group_id'].isin(data_train_id)]
data_test = data[data['group_id'].isin(data_test_id)]

data_train = data_train.drop(columns=['group_id']).reset_index(drop=True)
data_test = data_test.drop(columns=['group_id']).reset_index(drop=True)

data_train.head()

nome_coluna = 'Surf.'

encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
encoder.fit(data_train[[nome_coluna]])
encoded_cols_treino = encoder.transform(data_train[[nome_coluna]])
encoded_cols_teste = encoder.transform(data_test[[nome_coluna]])
data_treino_encoded = pd.DataFrame(encoded_cols_treino, columns=encoder.get_feature_names_out([nome_coluna]))
data_teste_encoded = pd.DataFrame(encoded_cols_teste, columns=encoder.get_feature_names_out([nome_coluna]))

data_train = pd.concat([data_train, data_treino_encoded], axis=1)
data_test = pd.concat([data_test, data_teste_encoded], axis=1)

data_train = data_train.drop(columns=[nome_coluna])
data_test = data_test.drop(columns=[nome_coluna])

data_train.head()

cols_para_minmax = ['Re', 'AoA', 'y', 'x']
scaler_minmax = MinMaxScaler()
scaler_minmax.fit(data_train[cols_para_minmax])
data_train[cols_para_minmax] = scaler_minmax.transform(data_train[cols_para_minmax])
data_test[cols_para_minmax] = scaler_minmax.transform(data_test[cols_para_minmax])

cols_para_maxabs = ['cp']
scaler_maxabs = MaxAbsScaler()
scaler_maxabs.fit(data_train[cols_para_maxabs])
data_train[cols_para_maxabs] = scaler_maxabs.transform(data_train[cols_para_maxabs])
data_test[cols_para_maxabs] = scaler_maxabs.transform(data_test[cols_para_maxabs])

data_train.head()

coluna_target = 'cp'

Y_train = data_train[coluna_target]
Y_test = data_test[coluna_target]
X_train = data_train.drop(columns=[coluna_target])
X_test = data_test.drop(columns=[coluna_target])

X_train.head()

y_pred = model.predict(X_test)

mae = mean_absolute_error(Y_test, y_pred)
mse = mean_squared_error(Y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(Y_test, y_pred)

print(f"MAE  (Erro Absoluto Médio): {mae:.4f}")
print(f"MSE  (Erro Quadrático Médio): {mse:.4f}")
print(f"RMSE (Raiz do Erro Quadrático Médio): {rmse:.4f}")
print(f"R²   (Coeficiente de Determinação): {r2:.4f}")

y_pred_unscaled = scaler_maxabs.inverse_transform(y_pred)

predictions = pd.DataFrame(y_pred_unscaled, columns=['Predictions'])

predictions.to_csv('predictions.csv', index=False)


data_test_unscaled = data_test.copy()
data_test_unscaled[['Re', 'AoA', 'y', 'x']] = scaler_minmax.inverse_transform(data_test_unscaled[['Re', 'AoA', 'y', 'x']])
data_test_unscaled[['cp']] = scaler_maxabs.inverse_transform(data_test_unscaled[['cp']])
data_test_unscaled.to_csv('data_test.csv', index=False)