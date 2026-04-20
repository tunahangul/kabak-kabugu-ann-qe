import json
import joblib
import numpy as np
import pandas as pd

from sklearn.metrics import r2_score, mean_squared_error
from tensorflow.keras.models import load_model

# =========================================================
# KULLANILACAK MODEL NUMARASI
# =========================================================
MODEL_NUMBER = 1

# =========================================================
# DOSYALARI YÜKLE
# =========================================================
with open("artifacts/split_info.json", "r", encoding="utf-8") as f:
    split_info = json.load(f)

feature_columns = split_info["feature_columns"]
target = split_info["target"]

scaler_X = joblib.load("artifacts/scaler_X.pkl")
scaler_y = joblib.load("artifacts/scaler_y.pkl")

test_reference = pd.read_csv("artifacts/test_reference.csv")
model = load_model(f"models/model_{MODEL_NUMBER}.keras")

# =========================================================
# TEST VERİSİNDEN TAHMİN
# =========================================================
X_test = test_reference[feature_columns]
y_test = test_reference[target].values

X_test_scaled = scaler_X.transform(X_test)
y_pred_scaled = model.predict(X_test_scaled, verbose=0)
y_pred = scaler_y.inverse_transform(y_pred_scaled).flatten()

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

p = len(feature_columns)
SSE, ARE, HYBRID, MPSD, MAE, RMSE, R2 = calculate_errors(y_test, y_pred, p)

# =========================================================
# TAM TABLO
# =========================================================
result_df = pd.DataFrame({
    "original_index": test_reference["original_index"],
    "qe_experimental": y_test,
    "qe_predicted": y_pred,
    "error": y_test - y_pred,
    "absolute_error": np.abs(y_test - y_pred),
    "Neuron": MODEL_NUMBER,
    "R2": R2,
    "RMSE": RMSE,
    "SSE": SSE,
    "ARE": ARE,
    "HYBRID": HYBRID,
    "MPSD": MPSD,
    "MAE": MAE
})

output_path = f"results/model_{MODEL_NUMBER}_loaded_from_saved_model.csv"
result_df.to_csv(output_path, index=False)

print(f"Kayıtlı model kullanıldı: model_{MODEL_NUMBER}.keras")
print(f"Dosya oluşturuldu: {output_path}")
print(f"R2 = {R2:.6f}")
print(f"RMSE = {RMSE:.6f}")
print(f"SSE = {SSE:.6f}")
print(f"ARE = {ARE:.6f}")
print(f"HYBRID = {HYBRID:.6f}")
print(f"MPSD = {MPSD:.6f}")
print(f"MAE = {MAE:.6f}")