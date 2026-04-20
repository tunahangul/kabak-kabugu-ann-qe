import pandas as pd
import matplotlib.pyplot as plt

# Veriyi oku
df = pd.read_csv("results/all_models_results.csv")

# Metrikler ve Türkçe karşılıkları
metrics_tr = {
    "R2": "R² (Determinasyon Katsayısı)",
    "RMSE": "RMSE (Kök Ortalama Kare Hata)",
    "SSE": "SSE (Hata Kareleri Toplamı)",
    "ARE": "ARE (Ortalama Bağıl Hata)",
    "HYBRID": "HYBRID Hata",
    "MPSD": "MPSD (Marquardt Yüzde Standart Sapma)",
    "MAE": "MAE (Ortalama Mutlak Hata)"
}

for metric, tr_title in metrics_tr.items():
    plt.figure(figsize=(8, 5))
    plt.plot(df["Neuron"], df[metric], marker="o")
    plt.title(f"{tr_title} - Nöron Sayısı")
    plt.xlabel("Nöron Sayısı")
    plt.ylabel(tr_title)
    plt.grid()
    plt.savefig(f"results/{metric}_graph_TR.png", dpi=300, bbox_inches="tight")
    plt.close()

print("Türkçe grafikler oluşturuldu.")