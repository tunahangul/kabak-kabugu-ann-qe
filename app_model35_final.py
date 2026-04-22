import json
import joblib
import pandas as pd
import streamlit as st
from tensorflow.keras.models import load_model

# --------------------------------------------------
# SAYFA AYARI
# --------------------------------------------------
st.set_page_config(
    page_title="Kabak Kabuğu Aktif Karbonu ANN ile qₑ Tahmini",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --------------------------------------------------
# SESSION STATE
# --------------------------------------------------
if "show_info_panel" not in st.session_state:
    st.session_state.show_info_panel = True

if "history" not in st.session_state:
    st.session_state.history = pd.DataFrame(columns=[
        "pH",
        "C₀ (mg/L)",
        "AC Dozu (mg)",
        "Süre (dk)",
        "Tahmin qₑ (mg/g)",
        "Cₑ (mg/L)",
        "% Adsorpsiyon"
    ])

# --------------------------------------------------
# CSS
# keyboard_double hatasını çıkarmaması için
# font-family tüm alt elemanlara zorla verilmedi
# --------------------------------------------------
st.markdown("""
<style>
.block-container {
    max-width: 1250px;
    padding-top: 1rem;
    padding-bottom: 2rem;
}

html, body {
    font-family: "Segoe UI", sans-serif;
}

[data-testid="stAppViewContainer"] {
    font-family: "Segoe UI", sans-serif;
}

/* HERO */
.hero-box {
    background: linear-gradient(135deg, #0f172a 0%, #1d4ed8 100%);
    padding: 28px 32px;
    border-radius: 22px;
    color: white;
    margin-bottom: 20px;
    box-shadow: 0 10px 28px rgba(15, 23, 42, 0.20);
}
.hero-box h1, .hero-box h2, .hero-box h3, .hero-box p, .hero-box div, .hero-box span {
    color: white !important;
}
.hero-title {
    font-size: 34px;
    font-weight: 800;
    margin-bottom: 6px;
    line-height: 1.2;
}
.hero-subtitle {
    font-size: 17px;
    opacity: 0.92;
}

/* CARDS */
.section-card {
    background: rgba(255,255,255,0.94);
    border: 1px solid rgba(15,23,42,0.08);
    border-radius: 18px;
    padding: 20px;
    box-shadow: 0 6px 18px rgba(15, 23, 42, 0.05);
    margin-bottom: 16px;
}

.info-panel {
    background: rgba(255,255,255,0.96);
    border: 1px solid rgba(37,99,235,0.18);
    border-left: 7px solid #2563eb;
    border-radius: 18px;
    padding: 22px;
    box-shadow: 0 6px 18px rgba(37, 99, 235, 0.07);
    margin-bottom: 20px;
}

.metric-card {
    border-radius: 18px;
    padding: 20px 14px;
    text-align: center;
    min-height: 150px;
    border: 1px solid rgba(15,23,42,0.10);
    transition: transform 0.18s ease, box-shadow 0.18s ease;
    box-shadow: 0 6px 18px rgba(15, 23, 42, 0.08);
}

.metric-card:hover {
    transform: translateY(-4px);
    box-shadow: 0 12px 24px rgba(15, 23, 42, 0.16);
}

.metric-green {
    background: linear-gradient(180deg, #ecfdf5 0%, #d1fae5 100%);
    border-color: #86efac;
}

.metric-yellow {
    background: linear-gradient(180deg, #fffbeb 0%, #fef3c7 100%);
    border-color: #fde68a;
}

.metric-blue {
    background: linear-gradient(180deg, #eff6ff 0%, #dbeafe 100%);
    border-color: #93c5fd;
}

.metric-label {
    font-size: 15px;
    color: #475569;
    margin-bottom: 8px;
    font-weight: 600;
}

.metric-value {
    font-size: 30px;
    font-weight: 800;
    color: #0f172a;
    margin-bottom: 4px;
}

.metric-unit {
    font-size: 14px;
    color: #64748b;
}

.section-title {
    font-size: 22px;
    font-weight: 700;
    margin-bottom: 10px;
}

.small-note {
    font-size: 13px;
    color: #64748b;
    margin-bottom: 10px;
}

.history-title {
    font-size: 22px;
    font-weight: 700;
    margin-bottom: 8px;
}

/* BUTTON */
.stButton > button {
    border-radius: 12px;
    font-weight: 600;
    padding: 0.55rem 1rem;
}

/* DARK MODE FIX */
html[data-theme="dark"] .section-card,
html[data-theme="dark"] .info-panel {
    background: rgba(17,25,40,0.94) !important;
    border: 1px solid rgba(255,255,255,0.10) !important;
}

html[data-theme="dark"] .section-card,
html[data-theme="dark"] .section-card p,
html[data-theme="dark"] .section-card div,
html[data-theme="dark"] .section-card span,
html[data-theme="dark"] .section-card li,
html[data-theme="dark"] .section-card h1,
html[data-theme="dark"] .section-card h2,
html[data-theme="dark"] .section-card h3,
html[data-theme="dark"] .section-card h4,
html[data-theme="dark"] .section-card h5,
html[data-theme="dark"] .section-card h6,
html[data-theme="dark"] .info-panel,
html[data-theme="dark"] .info-panel p,
html[data-theme="dark"] .info-panel div,
html[data-theme="dark"] .info-panel span,
html[data-theme="dark"] .info-panel li,
html[data-theme="dark"] .info-panel h1,
html[data-theme="dark"] .info-panel h2,
html[data-theme="dark"] .info-panel h3,
html[data-theme="dark"] .info-panel h4,
html[data-theme="dark"] .info-panel h5,
html[data-theme="dark"] .info-panel h6 {
    color: #f8fafc !important;
}

html[data-theme="dark"] .small-note {
    color: #cbd5e1 !important;
}

html[data-theme="dark"] .metric-label {
    color: #334155 !important;
}
html[data-theme="dark"] .metric-value {
    color: #0f172a !important;
}
html[data-theme="dark"] .metric-unit {
    color: #475569 !important;
}

html[data-theme="dark"] [data-testid="stSidebar"] * {
    color: #f8fafc !important;
}

html[data-theme="dark"] .hero-subtitle {
    color: #e2e8f0 !important;
}
</style>
""", unsafe_allow_html=True)

# --------------------------------------------------
# MODEL YÜKLE
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
# YARDIMCI FONKSİYONLAR
# --------------------------------------------------
def qe_card_class(qe_value):
    if qe_value >= 30:
        return "metric-green"
    elif qe_value >= 15:
        return "metric-yellow"
    return "metric-blue"

def adsorption_card_class(percent_value):
    if percent_value >= 70:
        return "metric-green"
    elif percent_value >= 40:
        return "metric-yellow"
    return "metric-blue"

def ce_card_class(ce_value, c0_value):
    ratio = ce_value / c0_value if c0_value != 0 else 1
    if ratio <= 0.3:
        return "metric-green"
    elif ratio <= 0.6:
        return "metric-yellow"
    return "metric-blue"

def build_input_dataframe(ph, c0, doz, sure, feature_cols):
    """
    Farklı encoding sorunlarına karşı feature_columns içindeki isimleri
    akıllı şekilde eşleştirir.
    """
    row = {}
    for col in feature_cols:
        col_low = col.lower()

        if "ph" in col_low:
            row[col] = ph
        elif "c0" in col_low:
            row[col] = c0
        elif "doz" in col_low:
            row[col] = doz
        elif ("süre" in col_low) or ("sure" in col_low) or ("dk" in col_low):
            row[col] = sure
        else:
            # Beklenmeyen bir kolon varsa 0 bırakmak yerine None verme
            # ama bu projede 4 giriş olduğu için normalde gerekmez
            row[col] = 0

    return pd.DataFrame([row])

# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------
with st.sidebar:
    st.markdown("## ℹ️ Bilgi Menüsü")
    st.write("İstediğiniz zaman açıklama panelini tekrar açabilirsiniz.")

    if st.button("Bilgilendirme Panelini Aç", use_container_width=True):
        st.session_state.show_info_panel = True

    st.markdown("---")
    st.markdown("### Çalışma Aralıkları")
    st.write("**pH:** 2 – 6")
    st.write("**C₀:** 25 – 200 mg/L")
    st.write("**AC dozu:** 25 – 100 mg")
    st.write("**Süre:** 60 – 1440 dk")

    st.markdown("---")
    st.markdown("### Model Notu")
    st.write(
        "Bu model, **ZnCl₂ ile aktive edilmiş kabak kabuğu bazlı aktif karbon** "
        "kullanılarak elde edilen deneysel veriler ile eğitilmiş "
        "**35 nöronlu yapay sinir ağı (ANN)** modeline dayanmaktadır."
    )

# --------------------------------------------------
# ÜST BAŞLIK
# --------------------------------------------------
st.markdown("""
<div class="hero-box">
    <div class="hero-title">🧪 Kabak Kabuğu Aktif Karbonu ANN ile qₑ Tahmini</div>
    <div class="hero-subtitle">Sakarya Üniversitesi • Kimya / Adsorpsiyon Modelleme Arayüzü</div>
</div>
""", unsafe_allow_html=True)

# --------------------------------------------------
# BİLGİ PANELİ
# --------------------------------------------------
if st.session_state.show_info_panel:
    st.markdown('<div class="info-panel">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Uygulama Hakkında</div>', unsafe_allow_html=True)

    st.write("""
Bu uygulama, **ZnCl₂ ile aktive edilmiş kabak kabuğu bazlı aktif karbon** kullanılarak elde edilen
deneysel veriler ile eğitilmiş **35 nöronlu yapay sinir ağı (ANN)** modeline dayanmaktadır.
    """)

    st.markdown("### Uygulama ne yapar?")
    st.write("""
Girilen deneysel parametrelere göre aşağıdaki çıktıları hesaplar:
- **qₑ:** Adsorpsiyon kapasitesi
- **Cₑ:** Denge derişimi
- **% Adsorpsiyon**
    """)

    st.markdown("### Kullanım sınırları")
    st.write("""
Model yalnızca aşağıdaki aralıklarda güvenilirdir:
- **pH:** 2 – 6
- **C₀:** 25 – 200 mg/L
- **Aktif karbon dozu:** 25 – 100 mg
- **Süre:** 60 – 1440 dakika
    """)

    st.markdown("### Neden sınırlandırma uygulanmıştır?")
    st.write("""
Bu model yalnızca bu çalışma kapsamında üretilen aktif karbon için eğitilmiştir.
Eğitim aralığı dışındaki değerlerde model fiziksel olarak anlamlı olmayan tahminler üretebilir.
Bu nedenle uygulama yalnızca deneysel çalışma aralığında kullanılacak şekilde sınırlandırılmıştır.
    """)

    if st.button("Paneli Kapat"):
        st.session_state.show_info_panel = False
        st.rerun()

    st.markdown("</div>", unsafe_allow_html=True)

# --------------------------------------------------
# ÜST KONTROL
# --------------------------------------------------
top_left, top_right = st.columns([8, 2])

with top_right:
    if st.button("🗑 Geçmişi Temizle", use_container_width=True):
        st.session_state.history = st.session_state.history.iloc[0:0]
        st.success("Geçmiş hesaplamalar temizlendi.")
        st.rerun()

# --------------------------------------------------
# GİRİŞ ALANI
# --------------------------------------------------
st.markdown('<div class="section-card">', unsafe_allow_html=True)
st.markdown('<div class="section-title">Giriş Parametreleri</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="small-note">Lütfen yalnızca deneysel çalışma aralığındaki değerleri giriniz.</div>',
    unsafe_allow_html=True
)

r1c1, r1c2 = st.columns(2)
r2c1, r2c2 = st.columns(2)

with r1c1:
    pH = st.number_input("pH", min_value=2.0, max_value=6.0, value=6.0, step=0.1)

with r1c2:
    Doz = st.number_input("AC Dozu (mg)", min_value=25.0, max_value=100.0, value=50.0, step=1.0)

with r2c1:
    C0 = st.number_input("C₀ (mg/L)", min_value=25.0, max_value=200.0, value=50.0, step=1.0)

with r2c2:
    Sure = st.number_input("Süre (dk)", min_value=60.0, max_value=1440.0, value=960.0, step=1.0)

predict_clicked = st.button("🔍 qₑ Tahmini Yap")
st.markdown("</div>", unsafe_allow_html=True)

# --------------------------------------------------
# HESAPLAMA
# --------------------------------------------------
if predict_clicked:
    input_df = build_input_dataframe(pH, C0, Doz, Sure, feature_columns)

    input_scaled = scaler_X.transform(input_df)
    pred_scaled = model.predict(input_scaled, verbose=0)
    pred_qe = scaler_y.inverse_transform(pred_scaled)[0][0]

    # 50 mL = 0.05 L
    V = 0.05
    m = Doz / 1000.0

    Ce = C0 - (pred_qe * m / V)
    if Ce < 0:
        Ce = 0.0

    adsorption_percent = ((C0 - Ce) / C0) * 100 if C0 != 0 else 0.0

    # --------------------------------------------------
    # SONUÇ KARTLARI
    # --------------------------------------------------
    st.markdown("## Hesaplama Sonuçları")
    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown(f"""
        <div class="metric-card {qe_card_class(pred_qe)}">
            <div class="metric-label">Tahmin qₑ</div>
            <div class="metric-value">{pred_qe:.4f}</div>
            <div class="metric-unit">mg/g</div>
        </div>
        """, unsafe_allow_html=True)

    with c2:
        st.markdown(f"""
        <div class="metric-card {ce_card_class(Ce, C0)}">
            <div class="metric-label">Cₑ</div>
            <div class="metric-value">{Ce:.4f}</div>
            <div class="metric-unit">mg/L</div>
        </div>
        """, unsafe_allow_html=True)

    with c3:
        st.markdown(f"""
        <div class="metric-card {adsorption_card_class(adsorption_percent)}">
            <div class="metric-label">% Adsorpsiyon</div>
            <div class="metric-value">{adsorption_percent:.2f}</div>
            <div class="metric-unit">%</div>
        </div>
        """, unsafe_allow_html=True)

    # --------------------------------------------------
    # SON HESAPLAMA TABLOSU
    # --------------------------------------------------
    result_df = pd.DataFrame({
        "pH": [pH],
        "C₀ (mg/L)": [C0],
        "AC Dozu (mg)": [Doz],
        "Süre (dk)": [Sure],
        "Tahmin qₑ (mg/g)": [round(pred_qe, 4)],
        "Cₑ (mg/L)": [round(Ce, 4)],
        "% Adsorpsiyon": [round(adsorption_percent, 2)]
    })

    st.markdown("<div style='height:14px'></div>", unsafe_allow_html=True)

    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Son Hesaplama</div>', unsafe_allow_html=True)
    st.dataframe(result_df, use_container_width=True, hide_index=True)
    st.markdown("</div>", unsafe_allow_html=True)

    # --------------------------------------------------
    # GEÇMİŞE EKLE
    # --------------------------------------------------
    st.session_state.history = pd.concat(
        [st.session_state.history, result_df],
        ignore_index=True
    )

# --------------------------------------------------
# GEÇMİŞ
# --------------------------------------------------
st.markdown('<div class="section-card">', unsafe_allow_html=True)
st.markdown('<div class="history-title">Geçmiş Hesaplamalar</div>', unsafe_allow_html=True)

if len(st.session_state.history) == 0:
    st.info("Henüz geçmiş hesaplama bulunmuyor.")
else:
    st.dataframe(st.session_state.history, use_container_width=True, hide_index=True)

st.markdown("</div>", unsafe_allow_html=True)