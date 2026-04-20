import pandas as pd

# =========================================================
# TÜM MODEL SONUÇLARINI OKU
# =========================================================
df = pd.read_csv("results/all_models_results.csv")

# =========================================================
# NORMALİZASYON (Min-Max)
# =========================================================
def normalize(column):
    return (column - column.min()) / (column.max() - column.min())

# Sadece hata metrikleri normalize edilecek
metrics = ["SSE", "ARE", "HYBRID", "MPSD", "MAE", "RMSE"]

for m in metrics:
    df[m + "_norm"] = normalize(df[m])

# =========================================================
# TOPLAM SKOR
# =========================================================
df["TOTAL_SCORE"] = (
    df["SSE_norm"] +
    df["ARE_norm"] +
    df["HYBRID_norm"] +
    df["MPSD_norm"] +
    df["MAE_norm"] +
    df["RMSE_norm"]
)

# =========================================================
# EN İYİ MODEL
# =========================================================
best_model = df.loc[df["TOTAL_SCORE"].idxmin()]

# =========================================================
# SONUÇLARI YAZDIR
# =========================================================
print("\n==============================")
print("EN İYİ MODEL SONUCU")
print("==============================")

print(f"Model (Neuron): {int(best_model['Neuron'])}")
print(f"Toplam Skor   : {best_model['TOTAL_SCORE']:.6f}")

print("\n--- Hata Metrikleri ---")
print(f"SSE    : {best_model['SSE']:.6f}")
print(f"ARE    : {best_model['ARE']:.6f}")
print(f"HYBRID : {best_model['HYBRID']:.6f}")
print(f"MPSD   : {best_model['MPSD']:.6f}")
print(f"MAE    : {best_model['MAE']:.6f}")
print(f"RMSE   : {best_model['RMSE']:.6f}")

print("\n--- R2 (ayrı değerlendirme) ---")
print(f"R2     : {best_model['R2']:.6f}")

# =========================================================
# TÜM MODELLERİ SIRALA
# =========================================================
sorted_df = df.sort_values(by="TOTAL_SCORE")

print("\n==============================")
print("EN İYİ 5 MODEL")
print("==============================")

print(sorted_df[["Neuron", "TOTAL_SCORE", "R2"]].head())

# =========================================================
# DOSYAYA KAYDET
# =========================================================
df.to_csv("results/normalized_model_scores.csv", index=False)

print("\nNormalize edilmiş skorlar kaydedildi:")
print("results/normalized_model_scores.csv")