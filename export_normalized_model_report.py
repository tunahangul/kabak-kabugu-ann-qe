import pandas as pd
import numpy as np

# =========================================================
# DOSYAYI OKU
# =========================================================
df = pd.read_csv("results/all_models_results.csv")

# =========================================================
# NORMALİZASYON FONKSİYONU
# =========================================================
def min_max_normalize(series):
    s_min = series.min()
    s_max = series.max()

    if s_max == s_min:
        return pd.Series([0.0] * len(series), index=series.index)

    return (series - s_min) / (s_max - s_min)

# =========================================================
# NORMALİZE EDİLECEK HATA METRİKLERİ
# =========================================================
metrics = ["SSE", "ARE", "HYBRID", "MPSD", "MAE", "RMSE"]

for metric in metrics:
    df[f"{metric}_norm"] = min_max_normalize(df[metric])

# =========================================================
# TOPLAM NORMALİZE SKOR
# =========================================================
df["TOTAL_NORM_SCORE"] = df[[f"{m}_norm" for m in metrics]].sum(axis=1)

# =========================================================
# YAZILACAK EXCEL DOSYASI
# =========================================================
output_file = "normalized_model_report.xlsx"

with pd.ExcelWriter(output_file, engine="openpyxl") as writer:

    # -----------------------------------------------------
    # 1) HER MODEL İÇİN AYRI SAYFA
    # -----------------------------------------------------
    for _, row in df.iterrows():
        neuron = int(row["Neuron"])

        model_sheet_df = pd.DataFrame({
            "Metric": [
                "SSE", "ARE", "HYBRID", "MPSD", "MAE", "RMSE",
                "SSE_norm", "ARE_norm", "HYBRID_norm", "MPSD_norm", "MAE_norm", "RMSE_norm",
                "TOTAL_NORM_SCORE", "R2"
            ],
            "Value": [
                row["SSE"], row["ARE"], row["HYBRID"], row["MPSD"], row["MAE"], row["RMSE"],
                row["SSE_norm"], row["ARE_norm"], row["HYBRID_norm"], row["MPSD_norm"], row["MAE_norm"], row["RMSE_norm"],
                row["TOTAL_NORM_SCORE"], row["R2"]
            ]
        })

        model_sheet_df.to_excel(writer, sheet_name=f"Model_{neuron}", index=False)

    # -----------------------------------------------------
    # 2) NORMALİZE TOPLAM SKORA GÖRE SIRALAMA
    # -----------------------------------------------------
    sorted_total = df.sort_values(by="TOTAL_NORM_SCORE", ascending=True).copy()
    sorted_total.to_excel(writer, sheet_name="By_Total_Score", index=False)

    # -----------------------------------------------------
    # 3) RMSE'YE GÖRE SIRALAMA
    # -----------------------------------------------------
    sorted_rmse = df.sort_values(by="RMSE", ascending=True).copy()
    sorted_rmse.to_excel(writer, sheet_name="By_RMSE", index=False)

    # -----------------------------------------------------
    # 4) R2'YE GÖRE SIRALAMA
    # -----------------------------------------------------
    sorted_r2 = df.sort_values(by="R2", ascending=False).copy()
    sorted_r2.to_excel(writer, sheet_name="By_R2", index=False)

    # -----------------------------------------------------
    # 5) EN İYİ MODEL ÖZET SAYFASI
    # -----------------------------------------------------
    best_model = sorted_total.iloc[0]

    best_model_summary = pd.DataFrame({
        "Field": [
            "Best Model (Neuron)",
            "TOTAL_NORM_SCORE",
            "R2",
            "RMSE",
            "SSE",
            "ARE",
            "HYBRID",
            "MPSD",
            "MAE"
        ],
        "Value": [
            int(best_model["Neuron"]),
            best_model["TOTAL_NORM_SCORE"],
            best_model["R2"],
            best_model["RMSE"],
            best_model["SSE"],
            best_model["ARE"],
            best_model["HYBRID"],
            best_model["MPSD"],
            best_model["MAE"]
        ]
    })

    best_model_summary.to_excel(writer, sheet_name="Best_Model_Summary", index=False)

print(f"Excel raporu oluşturuldu: {output_file}")