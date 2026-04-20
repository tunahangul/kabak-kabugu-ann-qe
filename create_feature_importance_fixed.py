import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import load_model

# =========================================================
# AYAR
# =========================================================
MODEL_NUMBER = 35

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

X_test = test_reference[feature_columns].copy()
y_test = test_reference[target].values

X_test_scaled = scaler_X.transform(X_test)

# =========================================================
# TAHMİN FONKSİYONU
# =========================================================
def predict_from_scaled(X_scaled):
    pred_scaled = model.predict(X_scaled, verbose=0)
    return scaler_y.inverse_transform(pred_scaled).flatten()

# =========================================================
# PERMUTATION IMPORTANCE
# =========================================================
baseline_pred = predict_from_scaled(X_test_scaled)
baseline_rmse = np.sqrt(mean_squared_error(y_test, baseline_pred))

importance_scores = []
rng = np.random.default_rng(42)

for i, col in enumerate(feature_columns):
    X_permuted = X_test_scaled.copy()
    X_permuted[:, i] = rng.permutation(X_permuted[:, i])
    permuted_pred = predict_from_scaled(X_permuted)
    permuted_rmse = np.sqrt(mean_squared_error(y_test, permuted_pred))
    importance = permuted_rmse - baseline_rmse
    importance_scores.append(importance)

importance_df = pd.DataFrame({
    "Feature": feature_columns,
    "RawImportance": importance_scores
})

# Negatifleri sıfıra çek
importance_df["RawImportance"] = importance_df["RawImportance"].clip(lower=0)

# Yüzdeye çevir
total_importance = importance_df["RawImportance"].sum()
if total_importance == 0:
    importance_df["ImportancePercent"] = 0.0
else:
    importance_df["ImportancePercent"] = (
        importance_df["RawImportance"] / total_importance * 100
    )

# Etiketler
feature_names_tr = {
    "pH": "pH",
    "C0_mgL": "C0",
    "Doz_mg": "Doz",
    "süre_dk": "Süre"
}

feature_names_en = {
    "pH": "pH",
    "C0_mgL": "C0",
    "Doz_mg": "Dose",
    "süre_dk": "Time"
}

importance_df["Feature_TR"] = importance_df["Feature"].map(feature_names_tr).fillna(importance_df["Feature"])
importance_df["Feature_EN"] = importance_df["Feature"].map(feature_names_en).fillna(importance_df["Feature"])

# Büyükten küçüğe sırala
importance_df = importance_df.sort_values(by="ImportancePercent", ascending=False).reset_index(drop=True)

# CSV kaydet
importance_df.to_csv("results/model35_feature_importance_fixed.csv", index=False)

# =========================================================
# TÜRKÇE GRAFİK
# =========================================================
plt.figure(figsize=(8, 5), facecolor="#f2f2f2")
ax = plt.gca()
ax.set_facecolor("#f2f2f2")

bars = plt.barh(importance_df["Feature_TR"], importance_df["ImportancePercent"])
plt.gca().invert_yaxis()

plt.xlabel("Göreceli Önem (%)", fontsize=11)
plt.ylabel("")
plt.title("Girdi Değişkenlerinin Önemi", fontsize=14)

plt.grid(axis="x", linestyle="-", alpha=0.6)

for bar, value in zip(bars, importance_df["ImportancePercent"]):
    plt.text(
        bar.get_width() + 0.4,
        bar.get_y() + bar.get_height() / 2,
        f"{value:.1f}%",
        va="center",
        fontsize=10
    )

plt.xlim(0, max(importance_df["ImportancePercent"]) * 1.15 if len(importance_df) > 0 else 1)
plt.tight_layout()
plt.savefig("results/model35_feature_importance_TR_fixed.png", dpi=300, bbox_inches="tight")
plt.close()

# =========================================================
# İNGİLİZCE GRAFİK
# =========================================================
plt.figure(figsize=(8, 5), facecolor="#f2f2f2")
ax = plt.gca()
ax.set_facecolor("#f2f2f2")

bars = plt.barh(importance_df["Feature_EN"], importance_df["ImportancePercent"])
plt.gca().invert_yaxis()

plt.xlabel("Relative Importance (%)", fontsize=11)
plt.ylabel("")
plt.title("Importance of Input Variables", fontsize=14)

plt.grid(axis="x", linestyle="-", alpha=0.6)

for bar, value in zip(bars, importance_df["ImportancePercent"]):
    plt.text(
        bar.get_width() + 0.4,
        bar.get_y() + bar.get_height() / 2,
        f"{value:.1f}%",
        va="center",
        fontsize=10
    )

plt.xlim(0, max(importance_df["ImportancePercent"]) * 1.15 if len(importance_df) > 0 else 1)
plt.tight_layout()
plt.savefig("results/model35_feature_importance_EN_fixed.png", dpi=300, bbox_inches="tight")
plt.close()

print("Düzeltilmiş önem grafikleri oluşturuldu.")
print("Türkçe grafik: results/model35_feature_importance_TR_fixed.png")
print("İngilizce grafik: results/model35_feature_importance_EN_fixed.png")
print("CSV dosyası: results/model35_feature_importance_fixed.csv")