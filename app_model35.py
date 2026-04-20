import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st

from tensorflow.keras.models import load_model

# --------------------------------------------------
# Sayfa ayarları
# --------------------------------------------------
st.set_page_config(page_title="ANN qe Tahmin Uygulaması", page_icon="🧠", layout="centered")

st.title("35 Nöronlu ANN ile qe Tahmini")
st.write("Giriş parametrelerini girin, model tahmini qe değerini hesaplasın.")

# --------------------------------------------------
# Dosyaları yükle
# --------------------------------------------------
@st.cache_resource
def load_artifacts():
    with open("artifacts/split_info.json", "r", encoding="utf-8") as f:
        split_info = json.load(f)

    scaler_X = joblib.load("artifacts/scaler_X.pkl")
    scaler_y = joblib.load("artifacts/scaler_y.pkl")
    model = load_model("models/model_35.keras")

    return split_info, scaler_X, scaler_y, model

split_info, scaler_X, scaler_y, model = load_artifacts()
feature_columns = split_info["feature_columns"]

# --------------------------------------------------
# Giriş alanları
# --------------------------------------------------
st.subheader("Giriş Parametreleri")

col1, col2 = st.columns(2)

with col1:
    pH = st.number_input("pH", min_value=0.0, value=6.0, step=0.1)
    C0 = st.number_input("C0 (mg/L)", min_value=0.0, value=50.0, step=0.1)

with col2:
    Doz = st.number_input("Doz (mg)", min_value=0.0, value=50.0, step=0.1)
    Sure = st.number_input("Süre (dk)", min_value=0.0, value=60.0, step=1.0)

# --------------------------------------------------
# Tahmin
# --------------------------------------------------
if st.button("qe Tahmini Yap"):
    input_dict = {
        "pH": pH,
        "C0_mgL": C0,
        "Doz_mg": Doz,
        "süre_dk": Sure
    }

    input_df = pd.DataFrame([input_dict])

    # Gerekirse kolon sırasını modele göre hizala
    input_df = input_df[feature_columns]

    # Ölçekleme
    input_scaled = scaler_X.transform(input_df)

    # Tahmin
    pred_scaled = model.predict(input_scaled, verbose=0)
    pred_qe = scaler_y.inverse_transform(pred_scaled)[0][0]

    st.success(f"Tahmin edilen qe değeri: {pred_qe:.4f} mg/g")

    with st.expander("Girilen değerleri göster"):
        st.dataframe(input_df)