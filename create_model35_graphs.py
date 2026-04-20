import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import r2_score, mean_squared_error
from tensorflow.keras.models import load_model

# =========================================================
# AYARLAR
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
y_pred_scaled = model.predict(X_test_scaled, verbose=0)
y_pred = scaler_y.inverse_transform(y_pred_scaled).flatten()

# =========================================================
# METRİKLER
# =========================================================
R2 = r2_score(y_test, y_pred)
RMSE = np.sqrt(mean_squared_error(y_test, y_pred))

# =========================================================
# 1) DENEYSEL vs TAHMİN GRAFİĞİ
# =========================================================
# Türkçe
plt.figure(figsize=(7, 6))
plt.scatter(y_test, y_pred)
min_val = min(y_test.min(), y_pred.min())
max_val = max(y_test.max(), y_pred.max())
plt.plot([min_val, max_val], [min_val, max_val], linestyle="--")
plt.xlabel("Deneysel qe")
plt.ylabel("Tahmin qe")
plt.title(f"35 Nöronlu ANN Modeli için Deneysel qe - Tahmin qe Grafiği\nR² = {R2:.4f}, RMSE = {RMSE:.4f}")
plt.grid(True)
plt.tight_layout()
plt.savefig("results/model35_experimental_vs_predicted_TR.png", dpi=300)
plt.close()

# İngilizce
plt.figure(figsize=(7, 6))
plt.scatter(y_test, y_pred)
plt.plot([min_val, max_val], [min_val, max_val], linestyle="--")
plt.xlabel("Experimental qe")
plt.ylabel("Predicted qe")
plt.title(f"Experimental qe vs Predicted qe for ANN Model with 35 Neurons\nR² = {R2:.4f}, RMSE = {RMSE:.4f}")
plt.grid(True)
plt.tight_layout()
plt.savefig("results/model35_experimental_vs_predicted_EN.png", dpi=300)
plt.close()

# =========================================================
# 2) GİRDİ DEĞİŞKENLERİNİN ÖNEMİ (Permutation Importance)
# =========================================================
def predict_from_scaled(X_scaled):
    pred_scaled = model.predict(X_scaled, verbose=0)
    return scaler_y.inverse_transform(pred_scaled).flatten()

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
    "Importance": importance_scores
}).sort_values(by="Importance", ascending=False)

importance_df.to_csv("results/model35_feature_importance.csv", index=False)

# Türkçe
plt.figure(figsize=(8, 5))
plt.barh(importance_df["Feature"], importance_df["Importance"])
plt.gca().invert_yaxis()
plt.xlabel("Önem Skoru (RMSE artışı)")
plt.ylabel("Girdi Değişkeni")
plt.title("35 Nöronlu ANN Modeli için Girdi Değişkenlerinin Önemi")
plt.tight_layout()
plt.savefig("results/model35_feature_importance_TR.png", dpi=300)
plt.close()

# İngilizce
plt.figure(figsize=(8, 5))
plt.barh(importance_df["Feature"], importance_df["Importance"])
plt.gca().invert_yaxis()
plt.xlabel("Importance Score (Increase in RMSE)")
plt.ylabel("Input Variable")
plt.title("Input Variable Importance for ANN Model with 35 Neurons")
plt.tight_layout()
plt.savefig("results/model35_feature_importance_EN.png", dpi=300)
plt.close()

# =========================================================
# 3) ANN MİMARİSİ ŞEMASI
# 4 giriş - 35 gizli nöron - 1 çıkış
# =========================================================
def draw_ann_architecture(output_path, title_text, input_labels, hidden_count=35, output_label="qe"):
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.axis("off")

    x_input = 0.15
    x_hidden = 0.5
    x_output = 0.85

    # Giriş katmanı konumları
    input_y = np.linspace(0.2, 0.8, len(input_labels))
    # Gizli katman konumları
    hidden_y = np.linspace(0.05, 0.95, hidden_count)
    # Çıkış katmanı konumu
    output_y = [0.5]

    # Bağlantılar: input -> hidden
    for yi in input_y:
        for yh in hidden_y:
            ax.plot([x_input, x_hidden], [yi, yh], linewidth=0.2)

    # Bağlantılar: hidden -> output
    for yh in hidden_y:
        ax.plot([x_hidden, x_output], [yh, output_y[0]], linewidth=0.2)

    # Düğümler
    for yi, label in zip(input_y, input_labels):
        circ = plt.Circle((x_input, yi), 0.02, fill=False)
        ax.add_patch(circ)
        ax.text(x_input - 0.08, yi, label, va="center", ha="right", fontsize=9)

    for yh in hidden_y:
        circ = plt.Circle((x_hidden, yh), 0.012, fill=False)
        ax.add_patch(circ)

    circ = plt.Circle((x_output, output_y[0]), 0.02, fill=False)
    ax.add_patch(circ)
    ax.text(x_output + 0.05, output_y[0], output_label, va="center", ha="left", fontsize=10)

    # Katman başlıkları
    ax.text(x_input, 1.02, title_text["input"], ha="center", fontsize=11, fontweight="bold")
    ax.text(x_hidden, 1.02, title_text["hidden"], ha="center", fontsize=11, fontweight="bold")
    ax.text(x_output, 1.02, title_text["output"], ha="center", fontsize=11, fontweight="bold")

    # Açıklama
    ax.text(0.5, -0.05, title_text["main"], ha="center", fontsize=12, fontweight="bold")

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

# Türkçe şema
draw_ann_architecture(
    output_path="results/model35_ann_architecture_TR.png",
    title_text={
        "input": "Giriş Katmanı",
        "hidden": "Gizli Katman\n(35 nöron)",
        "output": "Çıkış Katmanı",
        "main": "35 Nöronlu Yapay Sinir Ağı Mimarisi"
    },
    input_labels=feature_columns,
    output_label="qe"
)

# İngilizce şema
draw_ann_architecture(
    output_path="results/model35_ann_architecture_EN.png",
    title_text={
        "input": "Input Layer",
        "hidden": "Hidden Layer\n(35 neurons)",
        "output": "Output Layer",
        "main": "Artificial Neural Network Architecture with 35 Neurons"
    },
    input_labels=feature_columns,
    output_label="qe"
)

print("35 nöronlu model için Türkçe ve İngilizce grafikler oluşturuldu.")
print("Oluşan dosyalar:")
print("- results/model35_experimental_vs_predicted_TR.png")
print("- results/model35_experimental_vs_predicted_EN.png")
print("- results/model35_feature_importance_TR.png")
print("- results/model35_feature_importance_EN.png")
print("- results/model35_ann_architecture_TR.png")
print("- results/model35_ann_architecture_EN.png")
print("- results/model35_feature_importance.csv")