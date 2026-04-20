import json
import joblib
import numpy as np
import pandas as pd

from sklearn.metrics import r2_score, mean_squared_error
from tensorflow.keras.models import load_model

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

X_test = test_reference[feature_columns]
y_test = test_reference[target].values

X_test_scaled = scaler_X.transform(X_test)

# =========================================================
# EXCEL
# =========================================================
writer = pd.ExcelWriter("tum_modeller.xlsx", engine="openpyxl")

p = len(feature_columns)

# =========================================================
# HER MODEL
# =========================================================
for model_number in range(1, 41):

    model = load_model(f"models/model_{model_number}.keras")

    y_pred_scaled = model.predict(X_test_scaled, verbose=0)
    y_pred = scaler_y.inverse_transform(y_pred_scaled).flatten()

    eps = 1e-12
    y_safe = np.where(np.abs(y_test) < eps, eps, y_test)

    error = y_test - y_pred
    abs_error = np.abs(error)
    sq_error = error**2

    N = len(y_test)

    # 🔹 HER SATIR İÇİN HATA ANALİZLERİ
    ARE_i = np.abs(error / y_safe) * 100
    HYBRID_i = (sq_error / y_safe) * 100 / (N - p)
    MPSD_i = ((error / y_safe)**2)

    MAE_i = abs_error
    RMSE_i = sq_error

    # 🔹 GENEL METRİKLER
    SSE = np.sum(sq_error)
    ARE = np.mean(ARE_i)
    HYBRID = np.sum(HYBRID_i)
    MPSD = np.sqrt(np.sum(MPSD_i)/(N-p)) * 100
    MAE = np.mean(abs_error)
    RMSE = np.sqrt(np.mean(sq_error))
    R2 = r2_score(y_test, y_pred)

    df = pd.DataFrame({
        "Deneysel qe": y_test,
        "Tahmin qe": y_pred,
        "Hata": error,
        "Mutlak Hata": abs_error,
        "Karesel Hata": sq_error,
        "ARE (%)": ARE_i,
        "HYBRID": HYBRID_i,
        "MPSD iç": MPSD_i,
        "MAE katkı": MAE_i,
        "RMSE iç": RMSE_i
    })

    # 🔴 TOPLAM SATIRI
    summary = pd.DataFrame({
        "Deneysel qe": ["TOPLAM"],
        "Tahmin qe": [""],
        "Hata": [""],
        "Mutlak Hata": [MAE],
        "Karesel Hata": [SSE],
        "ARE (%)": [ARE],
        "HYBRID": [HYBRID],
        "MPSD iç": [MPSD],
        "MAE katkı": [MAE],
        "RMSE iç": [RMSE]
    })

    df = pd.concat([df, summary], ignore_index=True)

    # 🔹 Sheet yaz
    df.to_excel(writer, sheet_name=f"Model_{model_number}", index=False)

    # 🔹 Altına R2 ekle
    metrics = pd.DataFrame({
        "Metric": ["R2"],
        "Value": [R2]
    })

    metrics.to_excel(writer, sheet_name=f"Model_{model_number}", startrow=len(df)+2, index=False)

# =========================================================
writer.close()

print("YENİ FORMATTA EXCEL OLUŞTURULDU")