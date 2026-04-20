import json
import joblib
import numpy as np
import pandas as pd

from sklearn.metrics import r2_score, mean_squared_error
from tensorflow.keras.models import load_model

pd.set_option("display.max_rows", None)
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 200)
pd.set_option("display.max_colwidth", None)

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
original_index = test_reference["original_index"].values

X_test_scaled = scaler_X.transform(X_test)

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
# HER MODEL İÇİN TABLO BAS
# =========================================================
p = len(feature_columns)

for model_number in range(1, 41):
    print("\n" + "=" * 120)
    print(f"MODEL {model_number}")
    print("=" * 120)

    model = load_model(f"models/model_{model_number}.keras")

    y_pred_scaled = model.predict(X_test_scaled, verbose=0)
    y_pred = scaler_y.inverse_transform(y_pred_scaled).flatten()

    eps = 1e-12
    y_true_safe = np.where(np.abs(y_test) < eps, eps, y_test)

    error = y_test - y_pred
    absolute_error = np.abs(error)
    squared_error = error ** 2

    N = len(y_test)

    are_percent_contrib = np.abs(error / y_true_safe) * 100
    hybrid_contrib = (squared_error / y_true_safe) * 100 / (N - p) if (N - p) > 0 else np.nan
    mpsd_inner_term = ((error / y_true_safe) ** 2) / (N - p) if (N - p) > 0 else np.nan
    mae_contrib = absolute_error
    rmse_inner_term = squared_error

    SSE, ARE, HYBRID, MPSD, MAE, RMSE, R2 = calculate_errors(y_test, y_pred, p)

    table_df = pd.DataFrame({
        "original_index": original_index,
        "qe_experimental": y_test,
        "qe_predicted": y_pred,
        "error": error,
        "absolute_error": absolute_error,
        "squared_error": squared_error,
        "ARE_percent_contrib": are_percent_contrib,
        "HYBRID_contrib": hybrid_contrib,
        "MPSD_inner_term": mpsd_inner_term,
        "MAE_contrib": mae_contrib,
        "RMSE_inner_term": rmse_inner_term
    })

    # En alta özet satırı
    summary_row = {
        "original_index": "TOTAL",
        "qe_experimental": "",
        "qe_predicted": "",
        "error": "",
        "absolute_error": "",
        "squared_error": SSE,
        "ARE_percent_contrib": ARE,
        "HYBRID_contrib": HYBRID,
        "MPSD_inner_term": MPSD,
        "MAE_contrib": MAE,
        "RMSE_inner_term": RMSE
    }

    final_df = pd.concat([table_df, pd.DataFrame([summary_row])], ignore_index=True)

    print(final_df.to_string(index=False))

    print("\nMODEL ÖZETİ")
    print(f"R2     = {R2:.6f}")
    print(f"RMSE   = {RMSE:.6f}")
    print(f"SSE    = {SSE:.6f}")
    print(f"ARE    = {ARE:.6f}")
    print(f"HYBRID = {HYBRID:.6f}")
    print(f"MPSD   = {MPSD:.6f}")
    print(f"MAE    = {MAE:.6f}")