import json
import joblib
import pandas as pd
import tkinter as tk
from tkinter import ttk, messagebox
from tensorflow.keras.models import load_model

# --------------------------------------------------
# Dosyaları yükle
# --------------------------------------------------
with open("artifacts/split_info.json", "r", encoding="utf-8") as f:
    split_info = json.load(f)

scaler_X = joblib.load("artifacts/scaler_X.pkl")
scaler_y = joblib.load("artifacts/scaler_y.pkl")
model = load_model("models/model_35.keras")
feature_columns = split_info["feature_columns"]

# --------------------------------------------------
# Tahmin fonksiyonu
# --------------------------------------------------
def predict_qe():
    try:
        pH = float(entry_ph.get())
        C0 = float(entry_c0.get())
        doz = float(entry_doz.get())
        sure = float(entry_sure.get())

        input_dict = {
            "pH": pH,
            "C0_mgL": C0,
            "Doz_mg": doz,
            "süre_dk": sure
        }

        input_df = pd.DataFrame([input_dict])
        input_df = input_df[feature_columns]

        input_scaled = scaler_X.transform(input_df)
        pred_scaled = model.predict(input_scaled, verbose=0)
        pred_qe = scaler_y.inverse_transform(pred_scaled)[0][0]

        result_var.set(f"Tahmin edilen qₑ değeri: {pred_qe:.4f} mg/g")

    except ValueError:
        messagebox.showerror("Hata", "Lütfen tüm giriş alanlarına sayısal değer girin.")
    except Exception as e:
        messagebox.showerror("Hata", f"Bir sorun oluştu:\n{e}")

# --------------------------------------------------
# Arayüz
# --------------------------------------------------
root = tk.Tk()
root.title("Kabak Kabuğu Aktif Karbonu ANN ile qₑ Tahmini")
root.geometry("520x420")
root.resizable(False, False)

main_frame = ttk.Frame(root, padding=20)
main_frame.pack(fill="both", expand=True)

title_label = ttk.Label(
    main_frame,
    text="Kabak Kabuğu Aktif Karbonu ANN ile qₑ Tahmini",
    font=("Arial", 16, "bold"),
    anchor="center"
)
title_label.pack(pady=(0, 10))

desc_label = ttk.Label(
    main_frame,
    text="Giriş verilerini girin, model tahmin qₑ değerini hesaplayın.",
    font=("Arial", 11),
    anchor="center"
)
desc_label.pack(pady=(0, 20))

form_frame = ttk.Frame(main_frame)
form_frame.pack(pady=10)

ttk.Label(form_frame, text="pH:", font=("Arial", 11)).grid(row=0, column=0, sticky="w", padx=8, pady=8)
entry_ph = ttk.Entry(form_frame, width=20)
entry_ph.grid(row=0, column=1, padx=8, pady=8)
entry_ph.insert(0, "6.0")

ttk.Label(form_frame, text="C0 (mg/L):", font=("Arial", 11)).grid(row=1, column=0, sticky="w", padx=8, pady=8)
entry_c0 = ttk.Entry(form_frame, width=20)
entry_c0.grid(row=1, column=1, padx=8, pady=8)
entry_c0.insert(0, "50.0")

ttk.Label(form_frame, text="Doz (mg):", font=("Arial", 11)).grid(row=2, column=0, sticky="w", padx=8, pady=8)
entry_doz = ttk.Entry(form_frame, width=20)
entry_doz.grid(row=2, column=1, padx=8, pady=8)
entry_doz.insert(0, "50.0")

ttk.Label(form_frame, text="Süre (dk):", font=("Arial", 11)).grid(row=3, column=0, sticky="w", padx=8, pady=8)
entry_sure = ttk.Entry(form_frame, width=20)
entry_sure.grid(row=3, column=1, padx=8, pady=8)
entry_sure.insert(0, "60.0")

predict_button = ttk.Button(
    main_frame,
    text="qₑ Tahmini Yap",
    command=predict_qe
)
predict_button.pack(pady=20)

result_var = tk.StringVar()
result_var.set("Tahmin sonucu burada görünecek.")

result_label = ttk.Label(
    main_frame,
    textvariable=result_var,
    font=("Arial", 12, "bold"),
    foreground="blue",
    anchor="center"
)
result_label.pack(pady=15)

root.mainloop()