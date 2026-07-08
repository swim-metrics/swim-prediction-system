# ============================================================
# SwimML Pro v3.0
# Yapay Zeka Tabanli Yuzme Performans Tahmin ve Antrenor Karar Destek Sistemi
# Streamlit App
# ============================================================

import json
import uuid
from io import BytesIO
from pathlib import Path
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
import streamlit as st
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
from reportlab.pdfgen import canvas

# ============================================================
# 1. TEMEL DOSYA AYARLARI
# ============================================================

DATA_FILE = "swim_data.xlsx"
METADATA_FILE = "model_metadata.json"
APP_VERSION = "SwimML Pro v3.0"

# ============================================================
# 2. SAYFA AYARI
# ============================================================

st.set_page_config(
    page_title="SwimML Pro",
    page_icon="🏊‍♂️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ============================================================
# 3. RENK VE CSS TASARIMI
# ============================================================

PRIMARY = "#071A2C"
SECONDARY = "#00B8C8"
LIGHT_BG = "#EAFBFF"
CARD_BG = "#FFFFFF"
GRAY_TEXT = "#6B7280"
SUCCESS = "#10B981"
WARNING = "#F59E0B"
DANGER = "#EF4444"

CUSTOM_CSS = f"""
<style>
    .main {{
        background: linear-gradient(180deg, #F7FCFF 0%, #FFFFFF 100%);
    }}
    .block-container {{
        padding-top: 1.6rem;
        padding-bottom: 2rem;
        max-width: 1180px;
    }}
    .hero-card {{
        background: linear-gradient(135deg, {PRIMARY} 0%, #0B2F4A 55%, {SECONDARY} 100%);
        color: white;
        padding: 28px 30px;
        border-radius: 24px;
        box-shadow: 0px 12px 32px rgba(7, 26, 44, 0.18);
        margin-bottom: 22px;
    }}
    .hero-title {{
        font-size: 2.2rem;
        font-weight: 800;
        margin-bottom: 6px;
        letter-spacing: -0.03em;
    }}
    .hero-subtitle {{
        font-size: 1.05rem;
        opacity: 0.92;
        line-height: 1.55;
        max-width: 900px;
    }}
    .section-title {{
        font-size: 1.25rem;
        font-weight: 800;
        color: {PRIMARY};
        margin-top: 18px;
        margin-bottom: 8px;
    }}
    .soft-card {{
        background-color: {CARD_BG};
        padding: 18px 20px;
        border-radius: 18px;
        border: 1px solid #E5E7EB;
        box-shadow: 0px 6px 18px rgba(7, 26, 44, 0.06);
        margin-bottom: 16px;
    }}
    .metric-card {{
        background-color: white;
        border-radius: 18px;
        border: 1px solid #E5E7EB;
        padding: 20px;
        box-shadow: 0px 5px 16px rgba(7, 26, 44, 0.06);
        min-height: 118px;
    }}
    .metric-label {{
        color: {GRAY_TEXT};
        font-size: 0.92rem;
        margin-bottom: 6px;
    }}
    .metric-value {{
        color: {PRIMARY};
        font-size: 1.7rem;
        font-weight: 800;
    }}
    .pill {{
        display: inline-block;
        padding: 7px 12px;
        border-radius: 999px;
        background-color: {LIGHT_BG};
        color: {PRIMARY};
        font-weight: 700;
        margin-right: 6px;
        margin-bottom: 6px;
        font-size: 0.9rem;
    }}
    .small-note {{
        color: {GRAY_TEXT};
        font-size: 0.92rem;
        line-height: 1.5;
    }}
    .warning-box {{
        background-color: #FFF7ED;
        border-left: 5px solid {WARNING};
        padding: 13px 15px;
        border-radius: 12px;
        color: #7C2D12;
        margin-top: 12px;
    }}
    .success-box {{
        background-color: #ECFDF5;
        border-left: 5px solid {SUCCESS};
        padding: 13px 15px;
        border-radius: 12px;
        color: #064E3B;
        margin-top: 12px;
    }}
    .danger-box {{
        background-color: #FEF2F2;
        border-left: 5px solid {DANGER};
        padding: 13px 15px;
        border-radius: 12px;
        color: #7F1D1D;
        margin-top: 12px;
    }}
    div.stButton > button:first-child {{
        background: linear-gradient(135deg, {PRIMARY} 0%, {SECONDARY} 100%);
        color: white;
        border: none;
        border-radius: 14px;
        padding: 0.82rem 1.2rem;
        font-weight: 800;
        font-size: 1.05rem;
        width: 100%;
    }}
    div.stDownloadButton > button:first-child {{
        border-radius: 14px;
        padding: 0.72rem 1rem;
        font-weight: 800;
        width: 100%;
    }}
    @media (max-width: 768px) {{
        .hero-title {{ font-size: 1.55rem; }}
        .hero-subtitle {{ font-size: 0.95rem; }}
        .hero-card {{ padding: 20px 18px; border-radius: 18px; }}
        .metric-value {{ font-size: 1.35rem; }}
    }}
</style>
"""

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# ============================================================
# 4. SÖZLÜKLER VE ETİKETLER
# ============================================================

STYLE_TR = {
    "freestyle": "50 m Serbest Stil",
    "backstroke": "50 m Sırtüstü Stil",
    "breaststroke": "50 m Kurbağalama Stil",
    "butterfly": "50 m Kelebek Stil",
}

STYLE_EN = {
    "freestyle": "50 m Freestyle",
    "backstroke": "50 m Backstroke",
    "breaststroke": "50 m Breaststroke",
    "butterfly": "50 m Butterfly",
}

SEX_TR = {"female": "Kadın", "male": "Erkek"}
SEX_EN = {"female": "Female", "male": "Male"}

AGE_GROUP_LABEL_TR = {
    "12_13": "12–13 yaş",
    "14_15": "14–15 yaş",
    "16_17": "16–17 yaş",
}

AGE_GROUP_LABEL_EN = {
    "12_13": "12–13 years",
    "14_15": "14–15 years",
    "16_17": "16–17 years",
}

FEATURE_LABELS_TR = {
    "age": "Yaş",
    "body_height": "Boy (cm)",
    "body_mass": "Kilo (kg)",
    "training_age": "Spor Yaşı (yıl)",
    "vertical_jump": "Dikey Sıçrama (cm)",
    "standing_long_jump": "Durarak Uzun Atlama (cm)",
    "flexed_arm_hang": "Bükülü Kol Asılı Kalma (sn)",
    "sit_ups_1min": "1 Dakika Mekik (adet)",
    "illinois_agility_test": "Illinois Çeviklik Testi (sn)",
    "sprint_30m": "30 m Sprint (sn)",
    "hand_length": "El Uzunluğu (cm)",
    "arm_span": "Kulaç Uzunluğu (cm)",
    "arm_length": "Kol Uzunluğu (cm)",
    "leg_length": "Bacak Uzunluğu (cm)",
    "sitting_height": "Oturma Boyu (cm)",
    "chest_girth": "Göğüs Çevresi (cm)",
    "biceps_girth": "Pazu Çevresi (cm)",
    "flexed_biceps_girth": "Flekse Pazu Çevresi (cm)",
    "forearm_girth": "Ön Kol Çevresi (cm)",
    "thigh_girth": "Uyluk Çevresi (cm)",
    "calf_girth": "Baldır Çevresi (cm)",
    "waist_girth": "Bel Çevresi (cm)",
    "hip_girth": "Kalça Çevresi (cm)",
    "biacromial_breadth": "Biakromial Genişlik (cm)",
    "biiliac_breadth": "Biiliak Genişlik (cm)",
    "elbow_breadth": "Dirsek Genişliği (cm)",
    "wrist_breadth": "El Bileği Genişliği (cm)",
    "knee_breadth": "Diz Genişliği (cm)",
    "ankle_breadth": "Ayak Bileği Genişliği (cm)",
    "sit_and_reach": "Otur-Uzan Testi (cm)",
    "trunk_extension_height": "Gövde Ekstansiyon Yüksekliği (cm)",
    "shoulder_extension_height": "Omuz Ekstansiyon Yüksekliği (cm)",
    "shoulder_mobility": "Omuz Mobilitesi",
    "flamingo_balance_test": "Flamingo Denge Testi",
    "body_density": "Vücut Yoğunluğu",
    "body_fat_percentage": "Vücut Yağ Yüzdesi (%)",
    "fat_mass": "Yağ Kütlesi (kg)",
    "fat_free_mass": "Yağsız Vücut Kütlesi (kg)",
    "handgrip_strength": "El Kavrama Kuvveti (kg)",
    "ankle_dorsiflexion_rom": "Ayak Bileği Dorsifleksiyon ROM",
    "ankle_plantarflexion_rom": "Ayak Bileği Plantar Fleksiyon ROM",
    "foot_length": "Ayak Uzunluğu (cm)",
}

FEATURE_LABELS_EN = {k: k.replace("_", " ").title() for k in FEATURE_LABELS_TR.keys()}

FEATURE_GROUPS = {
    "Antropometrik Ölçümler": [
        "age", "body_height", "body_mass", "training_age", "hand_length",
        "arm_span", "arm_length", "leg_length", "sitting_height", "foot_length",
        "chest_girth", "biceps_girth", "flexed_biceps_girth", "forearm_girth",
        "thigh_girth", "calf_girth", "waist_girth", "hip_girth",
        "biacromial_breadth", "biiliac_breadth", "elbow_breadth",
        "wrist_breadth", "knee_breadth", "ankle_breadth",
    ],
    "Motorik Performans Testleri": [
        "vertical_jump", "standing_long_jump", "flexed_arm_hang",
        "sit_ups_1min", "illinois_agility_test", "sprint_30m", "handgrip_strength",
    ],
    "Esneklik ve Mobilite": [
        "sit_and_reach", "trunk_extension_height", "shoulder_extension_height",
        "shoulder_mobility", "ankle_dorsiflexion_rom", "ankle_plantarflexion_rom",
    ],
    "Vücut Kompozisyonu": [
        "body_density", "body_fat_percentage", "fat_mass", "fat_free_mass",
    ],
    "Denge": ["flamingo_balance_test"],
}

FEATURE_GROUPS_EN = {
    "Anthropometric Measurements": FEATURE_GROUPS["Antropometrik Ölçümler"],
    "Motor Performance Tests": FEATURE_GROUPS["Motorik Performans Testleri"],
    "Flexibility and Mobility": FEATURE_GROUPS["Esneklik ve Mobilite"],
    "Body Composition": FEATURE_GROUPS["Vücut Kompozisyonu"],
    "Balance": FEATURE_GROUPS["Denge"],
}

LOWER_IS_BETTER = {
    "sprint_30m",
    "illinois_agility_test",
    "flamingo_balance_test",
    "body_fat_percentage",
    "fat_mass",
}

INTEGER_FEATURES = {"age", "sit_ups_1min"}

# ============================================================
# 5. YARDIMCI FONKSİYONLAR
# ============================================================

@st.cache_resource
def load_system():
    data_path = Path(DATA_FILE)
    metadata_path = Path(METADATA_FILE)

    if not data_path.exists():
        st.error(f"Veri dosyası bulunamadı: {DATA_FILE}")
        st.stop()

    if not metadata_path.exists():
        st.error(f"Metadata dosyası bulunamadı: {METADATA_FILE}")
        st.stop()

    df_local = pd.read_excel(data_path)
    with open(metadata_path, "r", encoding="utf-8") as f:
        metadata_local = json.load(f)

    return df_local, metadata_local


@st.cache_resource
def load_prediction_model(model_file):
    model_path = Path(str(model_file).replace("\\", "/"))
    if not model_path.exists():
        st.error(f"Model dosyası bulunamadı: {model_path}")
        st.stop()
    return joblib.load(model_path)


def get_default_value(feature):
    defaults = {
        "age": 15,
        "body_height": 160.0,
        "body_mass": 55.0,
        "training_age": 4.0,
        "vertical_jump": 35.0,
        "standing_long_jump": 180.0,
        "flexed_arm_hang": 30.0,
        "sit_ups_1min": 35,
        "illinois_agility_test": 18.0,
        "sprint_30m": 5.0,
        "hand_length": 17.0,
        "arm_span": 165.0,
        "arm_length": 65.0,
        "leg_length": 80.0,
        "sitting_height": 80.0,
        "chest_girth": 80.0,
        "biceps_girth": 25.0,
        "flexed_biceps_girth": 27.0,
        "forearm_girth": 22.0,
        "thigh_girth": 45.0,
        "calf_girth": 32.0,
        "waist_girth": 70.0,
        "hip_girth": 85.0,
        "biacromial_breadth": 35.0,
        "biiliac_breadth": 28.0,
        "elbow_breadth": 6.0,
        "wrist_breadth": 5.0,
        "knee_breadth": 8.0,
        "ankle_breadth": 6.0,
        "sit_and_reach": 20.0,
        "trunk_extension_height": 35.0,
        "shoulder_extension_height": 40.0,
        "shoulder_mobility": 0.0,
        "flamingo_balance_test": 10.0,
        "body_density": 1.05,
        "body_fat_percentage": 15.0,
        "fat_mass": 8.0,
        "fat_free_mass": 45.0,
        "handgrip_strength": 30.0,
        "ankle_dorsiflexion_rom": 30.0,
        "ankle_plantarflexion_rom": 45.0,
        "foot_length": 24.0,
    }
    return defaults.get(feature, 0.0)


def safe_float(value, default=np.nan):
    try:
        return float(value)
    except Exception:
        return default


def model_metric(model_info, key, default=None):
    value = model_info.get(key, default)
    if value is None:
        return default
    return value


def error_margin(model_info):
    mae = safe_float(model_info.get("mae"), np.nan)
    rmse = safe_float(model_info.get("rmse"), np.nan)
    if not np.isnan(mae) and mae > 0:
        return mae
    if not np.isnan(rmse) and rmse > 0:
        return rmse
    return np.nan


def group_filter(df_local, sex, age_group):
    required = {"sex", "age_group"}
    if not required.issubset(set(df_local.columns)):
        return df_local.copy()
    return df_local[(df_local["sex"] == sex) & (df_local["age_group"] == age_group)].copy()


def performance_percentile(group_data, target_col, prediction):
    if target_col not in group_data.columns or group_data.empty:
        return np.nan
    values = pd.to_numeric(group_data[target_col], errors="coerce").dropna()
    if len(values) == 0:
        return np.nan
    return float((values > prediction).mean() * 100)


def performance_level_tr(percentile):
    if np.isnan(percentile):
        return "☆☆☆☆☆", "Referans yetersiz", "Grup verisi yetersiz olduğu için yüzdelik yorum sınırlıdır."
    if percentile >= 85:
        return "★★★★★", "Çok yüksek performans düzeyi", "Sporcu referans grubunun oldukça üzerinde bir tahmin profili göstermektedir."
    if percentile >= 65:
        return "★★★★☆", "İyi performans düzeyi", "Sporcu referans grubuna göre güçlü bir performans profili göstermektedir."
    if percentile >= 40:
        return "★★★☆☆", "Orta performans düzeyi", "Sporcu referans grubuna yakın bir performans profili göstermektedir."
    return "★★☆☆☆", "Geliştirilebilir performans düzeyi", "Sporcunun performans göstergeleri antrenman planlamasıyla desteklenmelidir."


def performance_level_en(percentile):
    if np.isnan(percentile):
        return "☆☆☆☆☆", "Insufficient reference", "Percentile interpretation is limited because reference data are insufficient."
    if percentile >= 85:
        return "★★★★★", "Very high performance level", "The athlete shows a prediction profile clearly above the reference group."
    if percentile >= 65:
        return "★★★★☆", "Good performance level", "The athlete shows a strong performance profile compared with the reference group."
    if percentile >= 40:
        return "★★★☆☆", "Moderate performance level", "The athlete shows a performance profile close to the reference group."
    return "★★☆☆☆", "Developmental performance level", "The athlete's performance indicators should be supported by training planning."


def feature_percentile(feature, value, group_data):
    if feature not in group_data.columns or group_data.empty:
        return np.nan
    ref = pd.to_numeric(group_data[feature], errors="coerce").dropna()
    if len(ref) < 3:
        return np.nan
    value = safe_float(value, np.nan)
    if np.isnan(value):
        return np.nan
    if feature in LOWER_IS_BETTER:
        return float((ref > value).mean() * 100)
    return float((ref < value).mean() * 100)


def group_profile_scores(inputs, group_data, feature_groups):
    rows = []
    for group_name, feats in feature_groups.items():
        available = [f for f in feats if f in inputs]
        scores = [feature_percentile(f, inputs[f], group_data) for f in available]
        scores = [s for s in scores if not np.isnan(s)]
        score = float(np.mean(scores)) if scores else np.nan
        rows.append({"Alan": group_name, "Profil Puanı": score})
    return pd.DataFrame(rows)


def strong_weak_features(inputs, group_data, labels, top_n=5):
    scored = []
    for feature, value in inputs.items():
        pct = feature_percentile(feature, value, group_data)
        if not np.isnan(pct):
            scored.append((feature, labels.get(feature, feature), pct))
    if not scored:
        return [], []
    scored_sorted = sorted(scored, key=lambda x: x[2], reverse=True)
    strong = scored_sorted[:top_n]
    weak = sorted(scored, key=lambda x: x[2])[:top_n]
    return strong, weak


def model_reliability_text_tr(model_info):
    r2 = safe_float(model_info.get("r2"), np.nan)
    mae = safe_float(model_info.get("mae"), np.nan)
    if not np.isnan(r2):
        if r2 >= 0.75:
            level = "yüksek"
        elif r2 >= 0.50:
            level = "orta-yüksek"
        elif r2 >= 0.30:
            level = "orta"
        else:
            level = "sınırlı"
        return f"Modelin açıklayıcılık düzeyi {level} düzeydedir. Tahminler yaklaşık hata payı dikkate alınarak yorumlanmalıdır."
    if not np.isnan(mae):
        return f"Model için ortalama mutlak hata yaklaşık {mae:.2f} sn olarak raporlanmıştır."
    return "Model güvenilirlik bilgileri metadata dosyasındaki mevcut değerlere göre sınırlı düzeyde gösterilmektedir."


def model_reliability_text_en(model_info):
    r2 = safe_float(model_info.get("r2"), np.nan)
    mae = safe_float(model_info.get("mae"), np.nan)
    if not np.isnan(r2):
        if r2 >= 0.75:
            level = "high"
        elif r2 >= 0.50:
            level = "moderate-high"
        elif r2 >= 0.30:
            level = "moderate"
        else:
            level = "limited"
        return f"The model shows a {level} explanatory level. Predictions should be interpreted with the error margin."
    if not np.isnan(mae):
        return f"The model reports an approximate mean absolute error of {mae:.2f} s."
    return "Model reliability information is displayed based on the available metadata."


def pdf_clean(text):
    replacements = {
        "ç": "c", "Ç": "C", "ğ": "g", "Ğ": "G", "ı": "i", "İ": "I",
        "ö": "o", "Ö": "O", "ş": "s", "Ş": "S", "ü": "u", "Ü": "U",
        "–": "-", "—": "-", "²": "2", "±": "+/-", "★": "*", "☆": "-",
    }
    text = str(text)
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text


def create_pdf_report(
    is_tr,
    report_id,
    athlete_name,
    club_name,
    coach_name,
    style_display,
    sex_display,
    age_group_display,
    prediction,
    group_mean,
    difference,
    percentile,
    error_value,
    comment,
    ai_comment,
    strong,
    weak,
    profile_df,
    model_info,
):
    buffer = BytesIO()
    c = canvas.Canvas(buffer, pagesize=A4)
    width, height = A4

    def draw_header():
        c.setFillColorRGB(0.027, 0.102, 0.173)
        c.rect(0, height - 3.2 * cm, width, 3.2 * cm, fill=True, stroke=False)
        c.setFillColorRGB(1, 1, 1)
        c.setFont("Helvetica-Bold", 16)
        title = "SwimML Pro Performans Raporu" if is_tr else "SwimML Pro Performance Report"
        c.drawString(1.5 * cm, height - 1.25 * cm, pdf_clean(title))
        c.setFont("Helvetica", 9)
        c.drawString(1.5 * cm, height - 1.85 * cm, pdf_clean(f"{APP_VERSION} | Report ID: {report_id}"))
        c.drawString(1.5 * cm, height - 2.35 * cm, pdf_clean(datetime.now().strftime("%d.%m.%Y %H:%M")))

    def new_page():
        c.showPage()
        draw_header()
        return height - 4.2 * cm

    def section(y, text):
        if y < 3 * cm:
            y = new_page()
        c.setFillColorRGB(0.0, 0.55, 0.62)
        c.setFont("Helvetica-Bold", 12)
        c.drawString(1.5 * cm, y, pdf_clean(text))
        return y - 0.55 * cm

    def line(y, text, size=9.5):
        if y < 2.3 * cm:
            y = new_page()
        c.setFillColorRGB(0, 0, 0)
        c.setFont("Helvetica", size)
        c.drawString(1.7 * cm, y, pdf_clean(text)[:115])
        return y - 0.42 * cm

    draw_header()
    y = height - 4.2 * cm

    y = section(y, "1. Sporcu ve Model Bilgileri" if is_tr else "1. Athlete and Model Information")
    y = line(y, f"Ad Soyad / Athlete: {athlete_name or '-'}")
    y = line(y, f"Kulup / Club: {club_name or '-'}")
    y = line(y, f"Antrenor / Coach: {coach_name or '-'}")
    y = line(y, f"Model: {style_display} | {sex_display} | {age_group_display}")

    y = section(y, "2. Tahmin Sonucu" if is_tr else "2. Prediction Result")
    y = line(y, f"Tahmini Derece / Predicted Time: {prediction:.2f} sn")
    if not np.isnan(group_mean):
        y = line(y, f"Grup Ortalamasi / Group Mean: {group_mean:.2f} sn")
    if not np.isnan(difference):
        y = line(y, f"Fark / Difference: {difference:.2f} sn")
    if not np.isnan(percentile):
        y = line(y, f"Performans Yuzdeligi / Performance Percentile: %{percentile:.1f}")
    if not np.isnan(error_value):
        y = line(y, f"Tahmin Hata Araligi / Error Margin: +/- {error_value:.2f} sn")
    y = line(y, f"Yorum / Comment: {comment}")

    y = section(y, "3. Karar Destek Yorumu" if is_tr else "3. Decision Support Comment")
    for chunk in [ai_comment[i:i+105] for i in range(0, len(ai_comment), 105)]:
        y = line(y, chunk)

    y = section(y, "4. Guclu Alanlar" if is_tr else "4. Strong Areas")
    if strong:
        for _, label, pct in strong:
            y = line(y, f"- {label}: %{pct:.1f}")
    else:
        y = line(y, "- Referans veri yetersiz" if is_tr else "- Insufficient reference data")

    y = section(y, "5. Gelistirilebilir Alanlar" if is_tr else "5. Development Areas")
    if weak:
        for _, label, pct in weak:
            y = line(y, f"- {label}: %{pct:.1f}")
    else:
        y = line(y, "- Referans veri yetersiz" if is_tr else "- Insufficient reference data")

    y = section(y, "6. Profil Alanlari" if is_tr else "6. Profile Areas")
    if profile_df is not None and not profile_df.empty:
        for _, row in profile_df.iterrows():
            score = row["Profil Puanı"]
            score_text = "-" if np.isnan(score) else f"%{score:.1f}"
            y = line(y, f"- {row['Alan']}: {score_text}")

    y = section(y, "7. Model Bilgileri" if is_tr else "7. Model Information")
    metric_keys = ["model", "n_total", "n_train", "n_test", "mae", "rmse", "mape", "r2", "pearson_r", "cv_r2", "cv_std"]
    for key in metric_keys:
        if key in model_info:
            y = line(y, f"{key}: {model_info.get(key)}")

    y = section(y, "8. Bilimsel Uyari" if is_tr else "8. Scientific Notice")
    notice = (
        "Bu sistem antrenorluk kararlarini desteklemek amaciyla gelistirilmistir. Tahmin sonuclari tek basina kesin secim veya performans karari olarak kullanilmamalidir."
        if is_tr else
        "This system is designed to support coaching decisions. Prediction results should not be used as the sole basis for selection or performance decisions."
    )
    for chunk in [notice[i:i+105] for i in range(0, len(notice), 105)]:
        y = line(y, chunk)

    c.save()
    buffer.seek(0)
    return buffer


def metric_card(label, value, note=""):
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
            <div class="small-note">{note}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_list(title, items, is_strong=True):
    box_class = "success-box" if is_strong else "warning-box"
    if not items:
        st.markdown(
            f"<div class='{box_class}'><b>{title}</b><br>Referans veri yetersiz.</div>",
            unsafe_allow_html=True,
        )
        return
    lines = "<br>".join([f"• {label}: <b>%{pct:.1f}</b>" for _, label, pct in items])
    st.markdown(
        f"<div class='{box_class}'><b>{title}</b><br>{lines}</div>",
        unsafe_allow_html=True,
    )


def render_model_metrics(model_info, is_tr):
    labels_tr = {
        "model": "Model",
        "n_total": "Toplam Veri",
        "n_train": "Eğitim Verisi",
        "n_test": "Bağımsız Test Verisi",
        "mae": "MAE",
        "rmse": "RMSE",
        "mape": "MAPE (%)",
        "r2": "R²",
        "pearson_r": "Pearson r",
        "cv_r2": "CV R²",
        "cv_std": "CV SS",
        "min_error": "Minimum Hata",
        "max_error": "Maksimum Hata",
    }
    labels_en = {
        "model": "Model",
        "n_total": "Total Sample",
        "n_train": "Training Sample",
        "n_test": "Independent Test Sample",
        "mae": "MAE",
        "rmse": "RMSE",
        "mape": "MAPE (%)",
        "r2": "R²",
        "pearson_r": "Pearson r",
        "cv_r2": "CV R²",
        "cv_std": "CV SD",
        "min_error": "Minimum Error",
        "max_error": "Maximum Error",
    }
    labels = labels_tr if is_tr else labels_en
    keys = ["model", "n_total", "n_train", "n_test", "mae", "rmse", "mape", "r2", "pearson_r", "cv_r2", "cv_std", "min_error", "max_error"]
    rows = []
    for key in keys:
        if key in model_info:
            rows.append({"Gösterge" if is_tr else "Metric": labels[key], "Değer" if is_tr else "Value": model_info[key]})
    if rows:
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    else:
        st.info("Model metadata bilgisi sınırlı." if is_tr else "Limited model metadata information.")

# ============================================================
# 6. VERİ VE METADATA YÜKLEME
# ============================================================

df, metadata = load_system()

# ============================================================
# 7. ANA EKRAN - DİL SEÇİMİ
# ============================================================

language = st.radio(
    "Dil / Language",
    ["🇹🇷 Türkçe", "🇺🇸 English"],
    horizontal=True,
    key="language_selector",
)
is_tr = language.startswith("🇹🇷")

if is_tr:
    hero_title = "🏊‍♂️ SwimML Pro"
    hero_subtitle = "Yapay zekâ tabanlı yüzme performans tahmin ve antrenör karar destek sistemi. 12–17 yaş genç yüzücülerde 50 m stil performansını antropometrik, motorik, esneklik ve vücut kompozisyonu ölçümlerine göre analiz eder."
else:
    hero_title = "🏊‍♂️ SwimML Pro"
    hero_subtitle = "AI-based swimming performance prediction and coaching decision support system. It analyzes 50 m stroke performance in 12–17-year-old swimmers using anthropometric, motor, flexibility, and body composition measurements."

st.markdown(
    f"""
    <div class="hero-card">
        <div class="hero-title">{hero_title}</div>
        <div class="hero-subtitle">{hero_subtitle}</div>
        <div style="margin-top:14px;">
            <span class="pill">{APP_VERSION}</span>
            <span class="pill">50 m Prediction</span>
            <span class="pill">Coach Decision Support</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ============================================================
# 8. MODEL SEÇİMİ
# ============================================================

st.markdown(f"<div class='section-title'>{'Model Seçimi' if is_tr else 'Model Selection'}</div>", unsafe_allow_html=True)

style_dict = STYLE_TR if is_tr else STYLE_EN
sex_dict = SEX_TR if is_tr else SEX_EN
age_dict = AGE_GROUP_LABEL_TR if is_tr else AGE_GROUP_LABEL_EN

col_style, col_sex, col_age = st.columns(3)

with col_style:
    style_display = st.selectbox("Stil" if is_tr else "Stroke", list(style_dict.values()))
    style = {v: k for k, v in style_dict.items()}[style_display]

with col_sex:
    sex_display = st.selectbox("Cinsiyet" if is_tr else "Sex", list(sex_dict.values()))
    sex = {v: k for k, v in sex_dict.items()}[sex_display]

with col_age:
    age_group_display = st.selectbox("Yaş Grubu" if is_tr else "Age Group", list(age_dict.values()))
    age_group = {v: k for k, v in age_dict.items()}[age_group_display]

model_key = f"{style}_{age_group}_{sex}"

st.markdown(
    f"<div class='soft-card'><b>{'Seçilen analiz grubu' if is_tr else 'Selected analysis group'}:</b> {style_display} / {sex_display} / {age_group_display}</div>",
    unsafe_allow_html=True,
)

if model_key not in metadata:
    st.error("Bu seçim için kayıtlı model bulunamadı." if is_tr else "No registered model was found for this selection.")
    st.stop()

model_info = metadata[model_key]
model = load_prediction_model(model_info["model_file"])
features = model_info.get("features_en", model_info.get("features", []))
target_col = model_info.get("target")

if not features:
    st.error("Metadata içinde model değişkenleri bulunamadı." if is_tr else "Model features were not found in metadata.")
    st.stop()

if not target_col:
    st.error("Metadata içinde hedef değişken adı bulunamadı." if is_tr else "Target variable was not found in metadata.")
    st.stop()

# ============================================================
# 9. SPORCU BİLGİLERİ - SADECE 3 ALAN
# ============================================================

st.markdown(f"<div class='section-title'>{'Sporcu Bilgileri' if is_tr else 'Athlete Information'}</div>", unsafe_allow_html=True)

col_a, col_b, col_c = st.columns(3)
with col_a:
    athlete_name = st.text_input("Ad Soyad" if is_tr else "Athlete Full Name", value="")
with col_b:
    club_name = st.text_input("Kulüp" if is_tr else "Club", value="")
with col_c:
    coach_name = st.text_input("Antrenör Ad Soyad" if is_tr else "Coach Full Name", value="")

# ============================================================
# 10. ÖLÇÜMLER
# ============================================================

st.markdown(f"<div class='section-title'>{'Ölçüm Girişi' if is_tr else 'Measurement Entry'}</div>", unsafe_allow_html=True)

st.markdown(
    f"<div class='small-note'>{'Bu model için kullanılan değişken sayısı' if is_tr else 'Number of variables used in this model'}: <b>{len(features)}</b></div>",
    unsafe_allow_html=True,
)

inputs = {}
labels = FEATURE_LABELS_TR if is_tr else FEATURE_LABELS_EN
feature_groups_current = FEATURE_GROUPS if is_tr else FEATURE_GROUPS_EN

used_features_set = set(features)
shown_features = set()

for group_name, group_features in feature_groups_current.items():
    group_used = [f for f in group_features if f in used_features_set]
    if not group_used:
        continue

    with st.expander(group_name, expanded=(group_name in ["Motorik Performans Testleri", "Motor Performance Tests"])):
        cols = st.columns(3)
        for idx, feature in enumerate(group_used):
            shown_features.add(feature)
            label = labels.get(feature, feature)
            default_value = get_default_value(feature)
            with cols[idx % 3]:
                if feature in INTEGER_FEATURES:
                    inputs[feature] = st.number_input(
                        label,
                        min_value=0,
                        max_value=300,
                        value=int(default_value),
                        step=1,
                        key=f"input_{feature}",
                    )
                else:
                    inputs[feature] = st.number_input(
                        label,
                        min_value=0.0,
                        max_value=500.0,
                        value=float(default_value),
                        step=0.1,
                        key=f"input_{feature}",
                    )

remaining_features = [f for f in features if f not in shown_features]
if remaining_features:
    with st.expander("Diğer Ölçümler" if is_tr else "Other Measurements", expanded=False):
        cols = st.columns(3)
        for idx, feature in enumerate(remaining_features):
            label = labels.get(feature, feature)
            default_value = get_default_value(feature)
            with cols[idx % 3]:
                if feature in INTEGER_FEATURES:
                    inputs[feature] = st.number_input(label, min_value=0, max_value=300, value=int(default_value), step=1, key=f"input_{feature}")
                else:
                    inputs[feature] = st.number_input(label, min_value=0.0, max_value=500.0, value=float(default_value), step=0.1, key=f"input_{feature}")

# Modelin beklediği sırayı koru
inputs = {feature: inputs[feature] for feature in features if feature in inputs}

# ============================================================
# 11. TAHMİN VE RAPOR
# ============================================================

st.markdown("---")
button_label = "🤖 Tahmin Et ve Karar Destek Raporu Oluştur" if is_tr else "🤖 Predict and Generate Decision Support Report"

if st.button(button_label):
    report_id = f"SWIM-{datetime.now().strftime('%Y%m%d')}-{str(uuid.uuid4())[:8].upper()}"
    input_df = pd.DataFrame([inputs], columns=features)

    try:
        prediction = float(model.predict(input_df)[0])
    except Exception as exc:
        st.error(f"Tahmin sırasında hata oluştu: {exc}" if is_tr else f"Prediction error: {exc}")
        st.stop()

    group_data = group_filter(df, sex, age_group)

    if target_col in group_data.columns and not group_data.empty:
        target_values = pd.to_numeric(group_data[target_col], errors="coerce").dropna()
        group_mean = float(target_values.mean()) if len(target_values) else np.nan
    else:
        group_mean = np.nan

    difference = prediction - group_mean if not np.isnan(group_mean) else np.nan
    percentile = performance_percentile(group_data, target_col, prediction)
    err = error_margin(model_info)

    if is_tr:
        stars, level_text, level_explanation = performance_level_tr(percentile)
        reliability_text = model_reliability_text_tr(model_info)
        if not np.isnan(difference):
            if difference < 0:
                comparison_text = f"Sporcu seçilen referans grubundan yaklaşık {abs(difference):.2f} sn daha hızlı görünmektedir."
            else:
                comparison_text = f"Sporcu seçilen referans grubundan yaklaşık {difference:.2f} sn daha yavaş görünmektedir."
        else:
            comparison_text = "Grup ortalaması hesaplanamadığı için fark yorumu sınırlıdır."
        ai_comment = f"{level_explanation} {comparison_text} {reliability_text} Bu sonuç, antrenör gözlemi ve teknik değerlendirme ile birlikte yorumlanmalıdır."
    else:
        stars, level_text, level_explanation = performance_level_en(percentile)
        reliability_text = model_reliability_text_en(model_info)
        if not np.isnan(difference):
            if difference < 0:
                comparison_text = f"The athlete appears approximately {abs(difference):.2f} s faster than the selected reference group."
            else:
                comparison_text = f"The athlete appears approximately {difference:.2f} s slower than the selected reference group."
        else:
            comparison_text = "Difference interpretation is limited because the group mean could not be calculated."
        ai_comment = f"{level_explanation} {comparison_text} {reliability_text} This result should be interpreted together with coaching observation and technical assessment."

    profile_df = group_profile_scores(inputs, group_data, feature_groups_current)
    strong, weak = strong_weak_features(inputs, group_data, labels, top_n=5)

    # --------------------------------------------------------
    # SONUÇ KARTLARI
    # --------------------------------------------------------
    st.markdown(f"<div class='section-title'>{'Tahmin Sonucu' if is_tr else 'Prediction Result'}</div>", unsafe_allow_html=True)

    m1, m2, m3, m4 = st.columns(4)
    with m1:
        metric_card("Tahmini Derece" if is_tr else "Predicted Time", f"{prediction:.2f} sn", style_display)
    with m2:
        metric_card("Grup Ortalaması" if is_tr else "Group Mean", "-" if np.isnan(group_mean) else f"{group_mean:.2f} sn", age_group_display)
    with m3:
        diff_value = "-" if np.isnan(difference) else f"{difference:.2f} sn"
        metric_card("Fark" if is_tr else "Difference", diff_value, "Negatif değer daha hızlıdır" if is_tr else "Negative value is faster")
    with m4:
        pct_value = "-" if np.isnan(percentile) else f"%{percentile:.1f}"
        metric_card("Yüzdelik" if is_tr else "Percentile", pct_value, level_text)

    st.markdown(
        f"<div class='success-box'><b>{stars} {level_text}</b><br>{ai_comment}</div>",
        unsafe_allow_html=True,
    )

    if not np.isnan(err):
        st.markdown(
            f"<div class='warning-box'><b>{'Tahmin hata aralığı' if is_tr else 'Prediction error margin'}:</b> ±{err:.2f} sn. {'Bu değer modelin ortalama hata göstergesine göre yorumlanmalıdır.' if is_tr else 'This value should be interpreted according to the model average error metric.'}</div>",
            unsafe_allow_html=True,
        )

    # --------------------------------------------------------
    # GÜÇLÜ VE GELİŞTİRİLEBİLİR ALANLAR
    # --------------------------------------------------------
    st.markdown(f"<div class='section-title'>{'Karar Destek Analizi' if is_tr else 'Decision Support Analysis'}</div>", unsafe_allow_html=True)
    s_col, w_col = st.columns(2)
    with s_col:
        render_list("Güçlü Alanlar" if is_tr else "Strong Areas", strong, is_strong=True)
    with w_col:
        render_list("Geliştirilebilir Alanlar" if is_tr else "Development Areas", weak, is_strong=False)

    # --------------------------------------------------------
    # PROFİL GRAFİĞİ
    # --------------------------------------------------------
    st.markdown(f"<div class='section-title'>{'Profil Grafiği' if is_tr else 'Profile Chart'}</div>", unsafe_allow_html=True)
    profile_chart_df = profile_df.dropna().copy()
    if not profile_chart_df.empty:
        st.bar_chart(profile_chart_df.set_index("Alan"), use_container_width=True)
        st.caption("Profil puanları, sporcunun seçilen referans grubuna göre yaklaşık yüzdelik konumunu gösterir." if is_tr else "Profile scores show the athlete's approximate percentile position relative to the selected reference group.")
    else:
        st.info("Profil grafiği için yeterli referans veri bulunamadı." if is_tr else "Insufficient reference data for the profile chart.")

    # --------------------------------------------------------
    # MODEL BİLGİLERİ
    # --------------------------------------------------------
    with st.expander("Model Doğruluk ve Metadata Bilgileri" if is_tr else "Model Accuracy and Metadata Information", expanded=False):
        render_model_metrics(model_info, is_tr)

    # --------------------------------------------------------
    # BİLİMSEL UYARI
    # --------------------------------------------------------
    st.markdown(
        f"<div class='danger-box'><b>{'Bilimsel Uyarı' if is_tr else 'Scientific Notice'}:</b> {'Bu yazılım antrenörlük kararlarını desteklemek için geliştirilmiştir. Tahmin sonuçları sporcu seçimi, performans değerlendirmesi veya sağlıkla ilgili kararlar için tek başına kesin karar aracı olarak kullanılmamalıdır.' if is_tr else 'This software is designed to support coaching decisions. Prediction results should not be used as the sole decision-making tool for athlete selection, performance evaluation, or health-related decisions.'}</div>",
        unsafe_allow_html=True,
    )

    # --------------------------------------------------------
    # PDF RAPOR
    # --------------------------------------------------------
    pdf = create_pdf_report(
        is_tr=is_tr,
        report_id=report_id,
        athlete_name=athlete_name,
        club_name=club_name,
        coach_name=coach_name,
        style_display=style_display,
        sex_display=sex_display,
        age_group_display=age_group_display,
        prediction=prediction,
        group_mean=group_mean,
        difference=difference,
        percentile=percentile,
        error_value=err,
        comment=level_text,
        ai_comment=ai_comment,
        strong=strong,
        weak=weak,
        profile_df=profile_df,
        model_info=model_info,
    )

    file_name = f"{model_key}_{report_id}_swimml_pro_report.pdf"
    st.download_button(
        label="📄 PDF Raporu İndir" if is_tr else "📄 Download PDF Report",
        data=pdf,
        file_name=file_name,
        mime="application/pdf",
    )

# ============================================================
# 12. ALT BİLGİ
# ============================================================

st.markdown("---")
if is_tr:
    st.markdown(
        f"""
        <div class="small-note">
        <b>{APP_VERSION}</b><br>
        Geliştiren: <b>Tuğrul Özkadı & Araştırma Ekibi</b><br>
        Hitit Üniversitesi Spor Bilimleri Fakültesi<br><br>
        Bu uygulama bilimsel araştırma, eğitim ve antrenörlük karar destek amacıyla geliştirilmiştir.
        </div>
        """,
        unsafe_allow_html=True,
    )
else:
    st.markdown(
        f"""
        <div class="small-note">
        <b>{APP_VERSION}</b><br>
        Developed by: <b>Tuğrul Özkadı & Research Team</b><br>
        Hitit University Faculty of Sport Sciences<br><br>
        This application was developed for scientific research, education, and coaching decision support purposes.
        </div>
        """,
        unsafe_allow_html=True,
    )
