import os
import json
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score, mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input

# =========================================================
# AYARLAR
# =========================================================
file_path = "../yapay_sinir.csv"
target = "qe_mgg"
random_state = 42

# =========================================================
# VERİYİ OKU
# =========================================================
df = pd.read_csv(file_path, sep=";", decimal=",")

X = df.drop(columns=[target])
y = df[target]

all_indices = np.arange(len(df))

X_train, X_temp, y_train, y_temp, idx_train, idx_temp = train_test_split(
    X, y, all_indices, test_size=0.30, random_state=random_state
)

X_test, X_val, y_test, y_val, idx_test, idx_val = train_test_split(
    X_temp, y_temp, idx_temp, test_size=1/3, random_state=random_state
)

# =========================================================
# NORMALİZASYON
# =========================================================
scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()

X_train_scaled = scaler_X.fit_transform(X_train)
X_test_scaled = scaler_X.transform(X_test)
X_val_scaled = scaler_X.transform(X_val)

y_train_scaled = scaler_y.fit_transform(y_train.values.reshape(-1, 1))
y_test_scaled = scaler_y.transform(y_test.values.reshape(-1, 1))
y_val_scaled = scaler_y.transform(y_val.values.reshape(-1, 1))

# =========================================================
# KLASÖRLER
# =========================================================
os.makedirs("models", exist_ok=True)
os.makedirs("results", exist_ok=True)
os.makedirs("artifacts", exist_ok=True)

# scaler kaydet
joblib.dump(scaler_X, "artifacts/scaler_X.pkl")
joblib.dump(scaler_y, "artifacts/scaler_y.pkl")

split_info = {
    "target": target,
    "feature_columns": X.columns.tolist(),
    "train_indices": idx_train.tolist(),
    "test_indices": idx_test.tolist(),
    "val_indices": idx_val.tolist(),
    "random_state": random_state
}

with open("artifacts/split_info.json", "w", encoding="utf-8") as f:
    json.dump(split_info, f, ensure_ascii=False, indent=2)

# test verisini kaydet
test_reference_df = pd.DataFrame(X_test.copy())
test_reference_df[target] = y_test.values
test_reference_df["original_index"] = idx_test
test_reference_df.to_csv("artifacts/test_reference.csv", index=False)

# =========================================================
# HATA FONKSİYONLARI
# =========================================================
def calculate_errors(y_true, y_pred, p):
    y_true = np.asarray(y_true).flatten()
    y_pred = np.asarray(y_pred).flatten()

    eps = 1e-12
    y_true_safe = np.where(np.abs(y_true) < eps, eps, y_true)

    N = len(y_true)

    SSE = np.sum((y_true - y_pred) ** 2)
    ARE = (100 / N) * np.sum(np.abs((y_true - y_pred) / y_true_safe))

    if N - p > 0:
        HYBRID = (100 / (N - p)) * np.sum(((y_true - y_pred) ** 2) / y_true_safe)
        MPSD = 100 * np.sqrt((1 / (N - p)) * np.sum(((y_true - y_pred) / y_true_safe) ** 2))
    else:
        HYBRID = np.nan
        MPSD = np.nan

    MAE = np.mean(np.abs(y_true - y_pred))
    RMSE = np.sqrt(mean_squared_error(y_true, y_pred))
    R2 = r2_score(y_true, y_pred)

    return SSE, ARE, HYBRID, MPSD, MAE, RMSE, R2

# =========================================================
# MODEL EĞİTİMİ
# =========================================================
p = X.shape[1]
summary_rows = []

for neuron in range(1, 41):
    print(f"\nModel {neuron} eğitiliyor...")

    model = Sequential([
        Input(shape=(X_train_scaled.shape[1],)),
        Dense(neuron, activation="relu"),
        Dense(1)
    ])

    model.compile(optimizer="adam", loss="mse")
    model.fit(X_train_scaled, y_train_scaled, epochs=200, verbose=0)

    y_pred_scaled = model.predict(X_test_scaled, verbose=0)
    y_pred_real = scaler_y.inverse_transform(y_pred_scaled).flatten()
    y_test_real = scaler_y.inverse_transform(y_test_scaled).flatten()

    SSE, ARE, HYBRID, MPSD, MAE, RMSE, R2 = calculate_errors(y_test_real, y_pred_real, p)

    model.save(f"models/model_{neuron}.keras")

    detail_df = pd.DataFrame({
        "original_index": idx_test,
        "qe_experimental": y_test_real,
        "qe_predicted": y_pred_real,
        "error": y_test_real - y_pred_real,
        "absolute_error": np.abs(y_test_real - y_pred_real),
        "Neuron": neuron,
        "R2": R2,
        "RMSE": RMSE,
        "SSE": SSE,
        "ARE": ARE,
        "HYBRID": HYBRID,
        "MPSD": MPSD,
        "MAE": MAE
    })

    detail_df.to_csv(f"results/model_{neuron}_full_results.csv", index=False)

    summary_rows.append([neuron, R2, RMSE, SSE, ARE, HYBRID, MPSD, MAE])

summary_df = pd.DataFrame(
    summary_rows,
    columns=["Neuron", "R2", "RMSE", "SSE", "ARE", "HYBRID", "MPSD", "MAE"]
)
summary_df.to_csv("results/all_models_results.csv", index=False)

print("\nEğitim ve kayıt tamamlandı.")