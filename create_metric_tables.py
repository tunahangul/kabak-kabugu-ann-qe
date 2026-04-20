import pandas as pd

# =========================================================
# VERİYİ OKU
# =========================================================
df = pd.read_csv("results/all_models_results.csv")

# =========================================================
# METRİKLER
# =========================================================
metrics = ["R2", "RMSE", "SSE", "ARE", "HYBRID", "MPSD", "MAE"]

# =========================================================
# TÜRKÇE KARŞILIKLAR
# =========================================================
metric_tr = {
    "R2": "R² (Determinasyon Katsayısı)",
    "RMSE": "RMSE (Kök Ortalama Kare Hata)",
    "SSE": "SSE (Hata Kareleri Toplamı)",
    "ARE": "ARE (Ortalama Bağıl Hata)",
    "HYBRID": "HYBRID Hata",
    "MPSD": "MPSD (Standart Sapma)",
    "MAE": "MAE (Ortalama Mutlak Hata)"
}

# =========================================================
# EXCEL OLUŞTUR
# =========================================================
with pd.ExcelWriter("metric_tables.xlsx", engine="openpyxl") as writer:

    for m in metrics:

        # ==============================
        # İNGİLİZCE TABLO
        # ==============================
        df_en = df[["Neuron", m]].copy()
        df_en.columns = ["Neuron Number", m]

        df_en.to_excel(writer, sheet_name=f"{m}_Table_EN", index=False)

        # ==============================
        # TÜRKÇE TABLO
        # ==============================
        df_tr = df[["Neuron", m]].copy()
        df_tr.columns = ["Nöron Sayısı", metric_tr[m]]

        df_tr.to_excel(writer, sheet_name=f"{m}_Table_TR", index=False)

print("Tüm tablolar oluşturuldu: metric_tables.xlsx")