import os
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score, mean_squared_error

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

file_path = "../yapay_sinir.csv"
df = pd.read_csv(file_path, sep=";", decimal=",")

target = "qe_mgg"
X = df.drop(columns=[target])
y = df[target]

X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.30, random_state=42
)

X_test, X_val, y_test, y_val = train_test_split(
    X_temp, y_temp, test_size=1/3, random_state=42
)

scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()

X_train = scaler_X.fit_transform(X_train)
X_test = scaler_X.transform(X_test)
X_val = scaler_X.transform(X_val)

y_train = scaler_y.fit_transform(y_train.values.reshape(-1,1))
y_test = scaler_y.transform(y_test.values.reshape(-1,1))
y_val = scaler_y.transform(y_val.values.reshape(-1,1))

os.makedirs("models", exist_ok=True)
os.makedirs("results", exist_ok=True)

def calculate_errors(y_true, y_pred):

    y_true = y_true.flatten()
    y_pred = y_pred.flatten()

    N = len(y_true)
    p = X.shape[1]

    SSE = np.sum((y_true - y_pred)**2)
    ARE = (100/N) * np.sum(np.abs((y_true - y_pred) / y_true))
    HYBRID = (100/(N-p)) * np.sum(((y_true - y_pred)**2) / y_true)
    MPSD = 100 * np.sqrt((1/(N-p)) * np.sum(((y_true - y_pred)/y_true)**2))
    MAE = np.mean(np.abs(y_true - y_pred))
    RMSE = np.sqrt(mean_squared_error(y_true, y_pred))
    R2 = r2_score(y_true, y_pred)

    return SSE, ARE, HYBRID, MPSD, MAE, RMSE, R2

results = []

for neuron in range(1, 41):

    print(f"\nModel {neuron} başlıyor...")

    model = Sequential()
    model.add(Dense(neuron, activation='relu', input_dim=X_train.shape[1]))
    model.add(Dense(1))

    model.compile(optimizer='adam', loss='mse')

    model.fit(X_train, y_train, epochs=200, verbose=0)

    y_pred = model.predict(X_test)

    y_pred_real = scaler_y.inverse_transform(y_pred)
    y_test_real = scaler_y.inverse_transform(y_test)

    SSE, ARE, HYBRID, MPSD, MAE, RMSE, R2 = calculate_errors(y_test_real, y_pred_real)

    model.save(f"models/model_{neuron}.h5")

    pred_df = pd.DataFrame({
        "qe_real": y_test_real.flatten(),
        "qe_pred": y_pred_real.flatten(),
        "error": y_test_real.flatten() - y_pred_real.flatten()
    })

    pred_df.to_csv(f"results/model_{neuron}_predictions.csv", index=False)

    results.append([
        neuron, SSE, ARE, HYBRID, MPSD, MAE, RMSE, R2
    ])

    print(f"Model {neuron} bitti")

columns = ["Neuron", "SSE", "ARE", "HYBRID", "MPSD", "MAE", "RMSE", "R2"]
results_df = pd.DataFrame(results, columns=columns)

results_df.to_csv("results/all_models_results.csv", index=False)

print("\nTÜM MODELLER TAMAMLANDI 🚀")