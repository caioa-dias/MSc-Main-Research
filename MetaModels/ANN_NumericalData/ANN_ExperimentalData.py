# Importação das bibliotecas:
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.preprocessing import StandardScaler
from keras.layers import Dropout, BatchNormalization
from keras.layers import Dense, Input
from keras.models import Sequential
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np



#------------------------------------------------------------------------------#
#                      LEITURA E TRATAMENTO DOS DADOS                          #
#------------------------------------------------------------------------------#
# Leitura dos dados:
data_train = pd.read_csv('CpData_LinearRegion_Train.csv', sep=';')
data_test = pd.read_csv('CpData_LinearRegion_Test.csv', sep=';')

# Separação dos dados em entrada e saída, treinamento e teste:
X_train = np.array(data_train.iloc[:, 0:2])
Y_train = np.array(data_train.iloc[:, 2:33])
X_test = np.array(data_test.iloc[:, 0:2])
Y_test = np.array(data_test.iloc[:, 2:33])

# Padronização dos dados de entrada:
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)



#------------------------------------------------------------------------------#
#                    DEFINIÇÃO E TREINAMENTO DO MODELO                         #
#------------------------------------------------------------------------------#
# Definição do modelo:
# Dropout:
# Batch Normalization:
model = Sequential([
    Input(shape=(2,)),
    Dense(64, activation='tanh'),
    #BatchNormalization(),
    Dropout(0.2),
    Dense(32, activation='tanh'),
    Dense(31, activation='linear')
])

# Compila o modelo:
model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mae'])

# Early Stopping: para cedo se o val_loss não melhora após 30 epocas e restaura os melhores pesos.
# ReduceLROnPlateau: diminui o learning rate quando o val_loss melhora após 15 epocas.
callbacks = [EarlyStopping(monitor='val_loss', patience=30, restore_best_weights=True),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=15)]

# Treinamento do modelo:
history = model.fit(
    X_train, Y_train,
    validation_split=0.25,
    epochs=500,
    batch_size=5,
    verbose=2,
    callbacks=callbacks)

# Plot do treinamento e validação:
epochs = range(1, len(history.history['loss']) + 1)
plt.plot(epochs, history.history['loss'], 'y', label='Training loss')
plt.plot(epochs, history.history['val_loss'], 'r', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()



#------------------------------------------------------------------------------#
#                         AVALIAÇÃO DO MODELO                                  #
#------------------------------------------------------------------------------#
# Avalia o modelo com base no dataset de teste:
loss, mae = model.evaluate(X_test, Y_test, verbose=2)
print(f"Erro médio absoluto no teste: {mae:.4f}")

# Realiza as predições utilizando o dataset de teste:
y_pred = model.predict(X_test)

# Avalia o modelo com base no dataset de teste:
mae = mean_absolute_error(Y_test, y_pred)
mse = mean_squared_error(Y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(Y_test, y_pred)
print(f"MAE  (Erro Absoluto Médio): {mae:.4f}")
print(f"MSE  (Erro Quadrático Médio): {mse:.4f}")
print(f"RMSE (Raiz do Erro Quadrático Médio): {rmse:.4f}")
print(f"R²   (Coeficiente de Determinação): {r2:.4f}")

# Plot das previsões com os valores reais:
plt.figure(figsize=(7,6))
plt.scatter(Y_test.ravel(), y_pred.ravel(), alpha=0.5)
min_val = min(Y_test.min(), y_pred.min())
max_val = max(Y_test.max(), y_pred.max())
plt.plot([min_val, max_val], [min_val, max_val], 'r--')
plt.xlabel("Valores reais")
plt.ylabel("Previsões RNA")
plt.title("Regressão com RNA")
plt.show()

# Plot das previsões com os valores reais:
X = [0.92, 0.82, 0.73, 0.63, 0.54, 0.46, 0.38, 0.3, 0.24, 0.18, 0.12, 0.08, 0.04, 0.02, 0.01, 0, 0.01, 0.02, 0.04, 0.08,
    0.12, 0.18, 0.24, 0.3, 0.38, 0.46, 0.54, 0.63, 0.73, 0.82, 0.92]

fig, axes = plt.subplots(4, 4, figsize=(15,12), sharex=True, sharey=True)

for idx, ax in enumerate(axes.flat[:16]):
    ax.plot(X, Y_test[idx], label="Real", color="blue")
    ax.plot(X, y_pred[idx], label="Predito", color="red", linestyle="dashed")
    ax.set_title(f"Amostra {idx}")
    ax.grid(True)

axes.flat[-1].legend()
plt.tight_layout()
plt.show()