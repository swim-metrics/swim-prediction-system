# ============================================================
# SwimML Pro v1.0
# 50 m Yüzme Performans Tahmin ve Antrenör Karar Destek Sistemi
# 50 m Prediction Coach Decision Support System
#
# Geliştirici: Öğr. Gör. Tuğrul Özkadı ve Ekibi
# Hitit Üniversitesi Spor Bilimleri Fakültesi
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
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle


# ============================================================
# 1. TEMEL AYARLAR
# ============================================================

APP_VERSION = "v1.0"
APP_NAME = "SwimML Pro"
DATA_FILE = "swim_data.xlsx"
METADATA_FILE = "model_metadata.json"

PAGE_TITLE_TR = "50 m Yüzme Performans Tahmin ve Antrenör Karar Destek Sistemi"
PAGE_TITLE_EN = "50 m Prediction Coach Decision Support System"

DEVELOPER_TR = "Öğr. Gör. Tuğrul Özkadı ve Ekibi"
DEVELOPER_EN = "Lect. Tuğrul Özkadı and Team"
UNIVERSITY_TR = "Hitit Üniversitesi Spor Bilimleri Fakültesi"
UNIVERSITY_EN = "Hitit University Faculty of Sport Sciences"

PRIMARY_DARK = "#071A2C"
SECOND_DARK = "#0B2338"
TURQUOISE = "#00B8C8"
LIGHT_BLUE = "#EAFBFF"
CARD_WHITE = "#FFFFFF"
TEXT_DARK = "#071A2C"
TEXT_MUTED = "#64748B"
SUCCESS = "#10B981"
WARNING = "#F59E0B"
DANGER = "#EF4444"

st.set_page_config(
    page_title=f"{APP_NAME} {APP_VERSION}",
    page_icon="🏊‍♂️",
    layout="wide",
    initial_sidebar_state="collapsed"
)


# ============================================================
# 2. CSS TASARIM - MOBİL UYUMLU VE OKUNABİLİR
# ============================================================

def inject_css():
    st.markdown(
        f"""
        <style>
        /* Genel zemin */
        .stApp {{
            background: linear-gradient(180deg, #F8FEFF 0%, #EAFBFF 45%, #FFFFFF 100%);
            color: {TEXT_DARK};
        }}

        /* Ana container */
        .block-container {{
            padding-top: 1.2rem;
            padding-bottom: 2.5rem;
            max-width: 1180px;
        }}

        /* Üst menü/sidebar etkisini azalt */
        [data-testid="stSidebar"] {{
            display: none;
        }}

        /* Başlık kartı */
        .hero-card {{
            background: linear-gradient(135deg, {PRIMARY_DARK} 0%, {SECOND_DARK} 55%, #003F54 100%);
            color: white;
            border-radius: 28px;
            padding: 34px 36px;
            box-shadow: 0 22px 55px rgba(7, 26, 44, 0.25);
            margin-bottom: 28px;
            border: 1px solid rgba(255,255,255,0.10);
        }}
        .hero-title {{
            font-size: clamp(34px, 6vw, 58px);
            font-weight: 900;
            letter-spacing: -1.5px;
            margin: 0 0 8px 0;
            line-height: 1.05;
        }}
        .hero-subtitle {{
            font-size: clamp(18px, 3vw, 28px);
            font-weight: 700;
            color: #DFFBFF;
            margin: 0 0 16px 0;
            line-height: 1.22;
        }}
        .hero-desc {{
            font-size: 17px;
            color: #BDEFF5;
            line-height: 1.55;
            max-width: 920px;
            margin-bottom: 18px;
        }}
        .hero-footer {{
            font-size: 14px;
            color: #DFFBFF;
            background: rgba(255,255,255,0.08);
            border-radius: 999px;
            display: inline-block;
            padding: 9px 16px;
        }}

        /* Bölüm başlığı */
        .section-card {{
            background: rgba(255,255,255,0.92);
            border: 1px solid rgba(0,184,200,0.22);
            border-radius: 24px;
            padding: 24px 28px;
            box-shadow: 0 14px 35px rgba(0,184,200,0.09);
            margin: 22px 0 14px 0;
        }}
        .section-title {{
            font-size: clamp(25px, 4vw, 40px);
            color: {TEXT_DARK};
            font-weight: 900;
            margin: 0;
            line-height: 1.1;
        }}

        .soft-note {{
            color: {TEXT_MUTED};
            font-size: 16px;
            font-weight: 600;
            margin: 6px 0 14px 0;
        }}

        .selected-group {{
            background: #DFF7FB;
            border-left: 6px solid {TURQUOISE};
            color: {TEXT_DARK};
            border-radius: 16px;
            padding: 14px 18px;
            font-weight: 800;
            margin: 12px 0 20px 0;
        }}

        /* Kritik: mobilde label'lar görünür olsun */
        label, .stNumberInput label, .stTextInput label, .stSelectbox label, .stRadio label {{
            color: {TEXT_DARK} !important;
            font-weight: 800 !important;
            font-size: 16px !important;
            opacity: 1 !important;
        }}

        /* Input alanları */
        .stTextInput input, .stNumberInput input, .stSelectbox div[data-baseweb="select"] > div {{
            background-color: #F8FAFC !important;
            color: {TEXT_DARK} !important;
            border: 1px solid rgba(7,26,44,0.13) !important;
            border-radius: 14px !important;
            min-height: 48px !important;
            font-weight: 700 !important;
            font-size: 16px !important;
        }}

        .stTextInput input:focus, .stNumberInput input:focus {{
            border: 2px solid {TURQUOISE} !important;
            box-shadow: 0 0 0 3px rgba(0,184,200,0.14) !important;
        }}

        /* Expander */
        [data-testid="stExpander"] {{
            border: 1px solid rgba(0,184,200,0.25) !important;
            border-radius: 18px !important;
            background: rgba(255,255,255,0.95) !important;
            box-shadow: 0 12px 28px rgba(0,184,200,0.07);
            overflow: hidden;
            margin-bottom: 16px;
        }}

        [data-testid="stExpander"] summary {{
            background: #FFFFFF !important;
            color: {TEXT_DARK} !important;
            font-size: 19px !important;
            font-weight: 900 !important;
            padding: 16px 18px !important;
            border-radius: 18px !important;
        }}

        [data-testid="stExpander"] summary p {{
            color: {TEXT_DARK} !important;
            font-size: 19px !important;
            font-weight: 900 !important;
        }}

        [data-testid="stExpander"] label {{
            color: {TEXT_DARK} !important;
            font-weight: 900 !important;
            opacity: 1 !important;
        }}

        /* Buton */
        .stButton > button {{
            background: linear-gradient(135deg, {TURQUOISE} 0%, #005B75 100%) !important;
            color: #FFFFFF !important;
            border: none !important;
            border-radius: 999px !important;
            padding: 16px 30px !important;
            font-size: 19px !important;
            font-weight: 900 !important;
            box-shadow: 0 16px 30px rgba(0,184,200,0.24) !important;
            min-height: 58px !important;
        }}
        .stButton > button:hover {{
            transform: translateY(-1px);
            box-shadow: 0 20px 36px rgba(0,184,200,0.30) !important;
        }}

        /* Metrik kartları */
        .metric-card {{
            background: #FFFFFF;
            border: 1px solid rgba(0,184,200,0.22);
            border-radius: 22px;
            padding: 22px 20px;
            min-height: 138px;
            box-shadow: 0 16px 38px rgba(7,26,44,0.08);
            margin-bottom: 14px;
        }}
        .metric-label {{
            color: {TEXT_MUTED};
            font-size: 14px;
            font-weight: 800;
            margin-bottom: 7px;
        }}
        .metric-value {{
            color: {TEXT_DARK};
            font-size: clamp(26px, 4vw, 42px);
            font-weight: 950;
            line-height: 1.05;
        }}
        .metric-small {{
            color: {TEXT_MUTED};
            font-size: 14px;
            font-weight: 700;
            margin-top: 9px;
        }}

        /* Dairesel gösterge */
        .gauge-wrap {{
            background: #FFFFFF;
            border: 1px solid rgba(0,184,200,0.22);
            border-radius: 24px;
            padding: 24px 18px;
            text-align: center;
            box-shadow: 0 16px 38px rgba(7,26,44,0.08);
            margin-bottom: 14px;
        }}
        .gauge {{
            --p: 70;
            --c: {TURQUOISE};
            width: 152px;
            height: 152px;
            border-radius: 50%;
            margin: 0 auto 14px auto;
            background: conic-gradient(var(--c) calc(var(--p) * 1%), #E5E7EB 0);
            display: flex;
            align-items: center;
            justify-content: center;
            position: relative;
        }}
        .gauge::before {{
            content: "";
            position: absolute;
            width: 112px;
            height: 112px;
            border-radius: 50%;
            background: #FFFFFF;
        }}
        .gauge span {{
            position: relative;
            z-index: 2;
            color: {TEXT_DARK};
            font-size: 28px;
            font-weight: 950;
        }}
        .gauge-title {{
            color: {TEXT_DARK};
            font-size: 18px;
            font-weight: 950;
            margin-top: 6px;
        }}
        .gauge-desc {{
            color: {TEXT_MUTED};
            font-size: 14px;
            font-weight: 700;
            margin-top: 4px;
        }}

        /* Yorum kartı */
        .coach-card {{
            background: linear-gradient(135deg, #FFFFFF 0%, #F0FDFF 100%);
            border: 1px solid rgba(0,184,200,0.24);
            border-radius: 24px;
            padding: 24px 26px;
            box-shadow: 0 16px 38px rgba(0,184,200,0.09);
            margin: 18px 0;
        }}
        .coach-title {{
            color: {TEXT_DARK};
            font-size: 24px;
            font-weight: 950;
            margin-bottom: 10px;
        }}
        .coach-text {{
            color: {TEXT_DARK};
            font-size: 16px;
            line-height: 1.65;
            font-weight: 650;
        }}

        /* Liste kartları */
        .list-card {{
            background: #FFFFFF;
            border-radius: 20px;
            border: 1px solid rgba(7,26,44,0.08);
            padding: 20px 22px;
            min-height: 170px;
            box-shadow: 0 10px 28px rgba(7,26,44,0.06);
        }}
        .list-title {{
            color: {TEXT_DARK};
            font-size: 20px;
            font-weight: 950;
            margin-bottom: 10px;
        }}
        .list-item {{
            color: {TEXT_DARK};
            font-size: 15px;
            font-weight: 700;
            margin: 8px 0;
        }}

        /* Footer */
        .footer-card {{
            background: linear-gradient(135deg, {PRIMARY_DARK} 0%, #111827 100%);
            color: #EAFBFF;
            border-radius: 26px;
            padding: 28px 32px;
            margin-top: 34px;
            box-shadow: 0 18px 44px rgba(7,26,44,0.20);
        }}
        .footer-title {{
            font-size: 26px;
            font-weight: 950;
            margin-bottom: 10px;
            color: #FFFFFF;
        }}
        .footer-text {{
            font-size: 14px;
            line-height: 1.65;
            color: #DFFBFF;
        }}

        /* Uyarı */
        .warning-box {{
            background: #FFF7ED;
            border-left: 6px solid {WARNING};
            color: {TEXT_DARK};
            border-radius: 16px;
            padding: 16px 18px;
            font-weight: 700;
            line-height: 1.55;
            margin: 14px 0;
        }}



        /* PDF indirme butonu - mobil ve masaüstü görünürlük */
        .stDownloadButton > button {{
            background: linear-gradient(135deg, {TURQUOISE} 0%, {PRIMARY_DARK} 100%) !important;
            color: #FFFFFF !important;
            border: none !important;
            border-radius: 18px !important;
            padding: 14px 22px !important;
            font-size: 17px !important;
            font-weight: 950 !important;
            min-height: 54px !important;
            box-shadow: 0 14px 28px rgba(0,184,200,0.24) !important;
        }}
        .stDownloadButton > button * {{
            color: #FFFFFF !important;
            opacity: 1 !important;
            font-weight: 950 !important;
        }}
        .stDownloadButton > button:hover {{
            background: linear-gradient(135deg, {PRIMARY_DARK} 0%, {TURQUOISE} 100%) !important;
            color: #FFFFFF !important;
        }}

        /* MOBİL ÖZEL */
        @media (max-width: 768px) {{
            .block-container {{
                padding-left: 1rem;
                padding-right: 1rem;
                padding-top: 0.8rem;
            }}
            .hero-card {{
                padding: 26px 22px;
                border-radius: 24px;
            }}
            .hero-desc {{
                font-size: 15px;
            }}
            .hero-footer {{
                border-radius: 18px;
                line-height: 1.45;
            }}
            .section-card {{
                padding: 22px 20px;
                border-radius: 22px;
            }}
            label, .stNumberInput label, .stTextInput label, .stSelectbox label, .stRadio label {{
                color: {TEXT_DARK} !important;
                font-size: 17px !important;
                font-weight: 950 !important;
                opacity: 1 !important;
            }}
            .stTextInput input, .stNumberInput input, .stSelectbox div[data-baseweb="select"] > div {{
                background-color: #FFFFFF !important;
                color: {TEXT_DARK} !important;
                border: 1.5px solid rgba(7,26,44,0.18) !important;
                font-size: 17px !important;
                min-height: 54px !important;
            }}
            [data-testid="stExpander"] summary p {{
                color: {TEXT_DARK} !important;
                font-size: 18px !important;
                font-weight: 950 !important;
            }}
            .gauge {{
                width: 138px;
                height: 138px;
            }}
            .gauge::before {{
                width: 102px;
                height: 102px;
            }}
        }}
        </style>
        """,
        unsafe_allow_html=True
    )


inject_css()


# ============================================================
# 3. METİN VE ETİKET SÖZLÜKLERİ
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

FEATURE_LABELS_EN = {k: k.replace("_", " ").title() for k in FEATURE_LABELS_TR}

ANTHROPOMETRY = {
    "age", "body_height", "body_mass", "training_age", "hand_length", "arm_span", "arm_length",
    "leg_length", "sitting_height", "chest_girth", "biceps_girth", "flexed_biceps_girth",
    "forearm_girth", "thigh_girth", "calf_girth", "waist_girth", "hip_girth",
    "biacromial_breadth", "biiliac_breadth", "elbow_breadth", "wrist_breadth",
    "knee_breadth", "ankle_breadth", "foot_length"
}

MOTOR = {
    "vertical_jump", "standing_long_jump", "flexed_arm_hang", "sit_ups_1min",
    "illinois_agility_test", "sprint_30m", "handgrip_strength", "flamingo_balance_test"
}

MOBILITY = {
    "sit_and_reach", "trunk_extension_height", "shoulder_extension_height", "shoulder_mobility",
    "ankle_dorsiflexion_rom", "ankle_plantarflexion_rom"
}

BODY_COMP = {
    "body_density", "body_fat_percentage", "fat_mass", "fat_free_mass"
}

LOWER_IS_BETTER = {
    "illinois_agility_test", "sprint_30m", "body_fat_percentage", "fat_mass", "body_mass",
    "flamingo_balance_test"
}


# ============================================================
# 4. YARDIMCI FONKSİYONLAR
# ============================================================

@st.cache_resource
def load_system():
    data_path = Path(DATA_FILE)
    meta_path = Path(METADATA_FILE)

    if not data_path.exists():
        st.error(f"Veri dosyası bulunamadı: {DATA_FILE}")
        st.stop()

    if not meta_path.exists():
        st.error(f"Model metadata dosyası bulunamadı: {METADATA_FILE}")
        st.stop()

    df_loaded = pd.read_excel(data_path)
    with open(meta_path, "r", encoding="utf-8") as f:
        metadata_loaded = json.load(f)

    return df_loaded, metadata_loaded


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


def label_for(feature, is_tr=True):
    return FEATURE_LABELS_TR.get(feature, feature) if is_tr else FEATURE_LABELS_EN.get(feature, feature)


def safe_float(value, default=0.0):
    try:
        if pd.isna(value):
            return default
        return float(value)
    except Exception:
        return default


def get_model_metric(model_info, key, default=0.0):
    return safe_float(model_info.get(key, default), default)


def calculate_percentile(group_data, target_col, prediction):
    if group_data.empty or target_col not in group_data.columns:
        return 50.0
    valid = group_data[target_col].dropna()
    if len(valid) == 0:
        return 50.0
    # Yüzme süresinde düşük değer daha iyi olduğu için prediction'dan büyük olanların oranı performans yüzdeliğidir.
    return float((valid > prediction).mean() * 100)


def performance_level(percentile, is_tr=True):
    if percentile >= 85:
        return ("Çok yüksek performans düzeyi", SUCCESS) if is_tr else ("Very high performance level", SUCCESS)
    if percentile >= 65:
        return ("İyi performans düzeyi", SUCCESS) if is_tr else ("Good performance level", SUCCESS)
    if percentile >= 40:
        return ("Orta performans düzeyi", WARNING) if is_tr else ("Moderate performance level", WARNING)
    return ("Geliştirilmesi gereken performans düzeyi", DANGER) if is_tr else ("Performance needs improvement", DANGER)


def confidence_score(model_info):
    r2 = get_model_metric(model_info, "r2", 0.50)
    mape = get_model_metric(model_info, "mape", 10.0)
    mae = get_model_metric(model_info, "mae", 1.0)

    # Basit ve anlaşılır bir güven skoru: R² yüksek, MAPE ve MAE düşükse artar.
    score = 50 + (r2 * 40) - (mape * 1.2) - (mae * 2.0)
    return int(np.clip(score, 35, 98))


def physical_profile_score(inputs, group_data, features):
    if group_data.empty:
        return 70

    scores = []
    for f in features:
        if f not in group_data.columns:
            continue
        series = pd.to_numeric(group_data[f], errors="coerce").dropna()
        if len(series) < 5:
            continue
        val = safe_float(inputs.get(f, np.nan), np.nan)
        if pd.isna(val):
            continue
        if f in LOWER_IS_BETTER:
            score = (series > val).mean() * 100
        else:
            score = (series < val).mean() * 100
        scores.append(score)

    if not scores:
        return 70
    return int(np.clip(np.nanmean(scores), 0, 100))


def potential_score(percentile, physical_score, confidence):
    # Performans orta fakat fiziksel profil yüksekse gelişim potansiyeli yükselir.
    score = (0.35 * physical_score) + (0.25 * confidence) + (0.40 * (100 - abs(70 - percentile)))
    return int(np.clip(score, 20, 98))


def identify_strengths_and_needs(inputs, group_data, features, is_tr=True, max_items=5):
    strengths = []
    needs = []

    if group_data.empty:
        return (["Referans veri yetersiz"], ["Referans veri yetersiz"]) if is_tr else (["Insufficient reference data"], ["Insufficient reference data"])

    for f in features:
        if f not in group_data.columns:
            continue
        series = pd.to_numeric(group_data[f], errors="coerce").dropna()
        if len(series) < 5:
            continue

        val = safe_float(inputs.get(f, np.nan), np.nan)
        if pd.isna(val):
            continue

        if f in LOWER_IS_BETTER:
            perc = (series > val).mean() * 100
        else:
            perc = (series < val).mean() * 100

        label = label_for(f, is_tr)
        if perc >= 70:
            strengths.append((label, perc))
        elif perc <= 35:
            needs.append((label, perc))

    strengths = sorted(strengths, key=lambda x: x[1], reverse=True)[:max_items]
    needs = sorted(needs, key=lambda x: x[1])[:max_items]

    if not strengths:
        strengths = [("Belirgin güçlü alan saptanmadı" if is_tr else "No clear strength detected", 0)]
    if not needs:
        needs = [("Belirgin geliştirme alanı saptanmadı" if is_tr else "No clear development area detected", 0)]

    return strengths, needs


def coach_comment(prediction, group_mean, difference, percentile, conf, is_tr=True):
    level_text, _ = performance_level(percentile, is_tr)
    if is_tr:
        if difference < 0:
            diff_sentence = f"Sporcunun tahmini derecesi referans grubun ortalamasından {abs(difference):.2f} sn daha hızlı görünmektedir."
        else:
            diff_sentence = f"Sporcunun tahmini derecesi referans grubun ortalamasından {difference:.2f} sn daha yavaş görünmektedir."
        return (
            f"Bu analizde sporcunun tahmini 50 m derecesi {prediction:.2f} sn olarak hesaplanmıştır. "
            f"{diff_sentence} Performans yüzdeliği %{percentile:.1f} olup sistem bu sonucu '{level_text}' olarak sınıflandırmaktadır. "
            f"Tahmin güven göstergesi %{conf} düzeyindedir. Bu çıktı, antrenörün teknik gözlemi ve saha değerlendirmesiyle birlikte yorumlanmalıdır."
        )
    else:
        if difference < 0:
            diff_sentence = f"The predicted time appears {abs(difference):.2f} s faster than the reference group mean."
        else:
            diff_sentence = f"The predicted time appears {difference:.2f} s slower than the reference group mean."
        return (
            f"The predicted 50 m time is {prediction:.2f} s. "
            f"{diff_sentence} The performance percentile is {percentile:.1f}%, classified as '{level_text}'. "
            f"The prediction confidence indicator is {conf}%. This output should be interpreted together with the coach's technical observation and field assessment."
        )


def pdf_safe(text):
    # ReportLab Helvetica Türkçe karakterlerde sorun çıkarabildiği için sadeleştiriyoruz.
    replacements = {
        "ç": "c", "Ç": "C", "ğ": "g", "Ğ": "G", "ı": "i", "İ": "I",
        "ö": "o", "Ö": "O", "ş": "s", "Ş": "S", "ü": "u", "Ü": "U",
        "²": "2", "±": "+/-", "–": "-", "—": "-"
    }
    text = str(text)
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text


def make_pdf_report(is_tr, report_data):
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4, rightMargin=36, leftMargin=36, topMargin=36, bottomMargin=36)
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="TitleBlue", parent=styles["Title"], textColor=colors.HexColor("#071A2C"), fontSize=18, leading=22))
    styles.add(ParagraphStyle(name="SubTitle", parent=styles["Normal"], textColor=colors.HexColor("#0B2338"), fontSize=11, leading=15))
    styles.add(ParagraphStyle(name="BodyNice", parent=styles["Normal"], fontSize=10, leading=14))

    story = []
    title = f"{APP_NAME} {APP_VERSION}"
    subtitle = PAGE_TITLE_TR if is_tr else PAGE_TITLE_EN
    story.append(Paragraph(pdf_safe(title), styles["TitleBlue"]))
    story.append(Paragraph(pdf_safe(subtitle), styles["SubTitle"]))
    story.append(Spacer(1, 12))

    rows = []
    if is_tr:
        rows = [
            ["Rapor No", report_data["report_id"]],
            ["Tarih", report_data["date"]],
            ["Ad Soyad", report_data["athlete_name"]],
            ["Kulup", report_data["club_name"]],
            ["Antrenor", report_data["coach_name"]],
            ["Analiz Grubu", report_data["analysis_group"]],
            ["Tahmini Derece", f"{report_data['prediction']:.2f} sn"],
            ["Grup Ortalamasi", f"{report_data['group_mean']:.2f} sn"],
            ["Fark", f"{report_data['difference']:.2f} sn"],
            ["Performans Yuzdeligi", f"%{report_data['percentile']:.1f}"],
            ["Tahmin Guveni", f"%{report_data['confidence']}"],
            ["Model", report_data["model_name"]],
            ["MAE", f"{report_data['mae']} sn"],
            ["RMSE", f"{report_data['rmse']} sn"],
            ["R2", str(report_data['r2'])],
        ]
    else:
        rows = [
            ["Report ID", report_data["report_id"]],
            ["Date", report_data["date"]],
            ["Athlete", report_data["athlete_name"]],
            ["Club", report_data["club_name"]],
            ["Coach", report_data["coach_name"]],
            ["Analysis Group", report_data["analysis_group"]],
            ["Predicted Time", f"{report_data['prediction']:.2f} s"],
            ["Group Mean", f"{report_data['group_mean']:.2f} s"],
            ["Difference", f"{report_data['difference']:.2f} s"],
            ["Performance Percentile", f"{report_data['percentile']:.1f}%"],
            ["Prediction Confidence", f"{report_data['confidence']}%"],
            ["Model", report_data["model_name"]],
            ["MAE", f"{report_data['mae']} s"],
            ["RMSE", f"{report_data['rmse']} s"],
            ["R2", str(report_data['r2'])],
        ]

    table = Table([[pdf_safe(a), pdf_safe(b)] for a, b in rows], colWidths=[150, 340])
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#EAFBFF")),
        ("TEXTCOLOR", (0, 0), (-1, -1), colors.HexColor("#071A2C")),
        ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#A7F3F7")),
        ("FONTNAME", (0, 0), (-1, -1), "Helvetica"),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 8),
        ("RIGHTPADDING", (0, 0), (-1, -1), 8),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
    ]))
    story.append(table)
    story.append(Spacer(1, 14))

    section_title = "Antrenor Karar Destek Yorumu" if is_tr else "Coach Decision Support Comment"
    story.append(Paragraph(pdf_safe(section_title), styles["Heading2"]))
    story.append(Paragraph(pdf_safe(report_data["comment"]), styles["BodyNice"]))
    story.append(Spacer(1, 12))

    strengths_title = "Guclu Alanlar" if is_tr else "Strengths"
    needs_title = "Gelistirilebilir Alanlar" if is_tr else "Development Areas"
    story.append(Paragraph(pdf_safe(strengths_title), styles["Heading3"]))
    for item, score in report_data["strengths"]:
        suffix = f" (%{score:.1f})" if score else ""
        story.append(Paragraph(pdf_safe(f"- {item}{suffix}"), styles["BodyNice"]))
    story.append(Spacer(1, 8))
    story.append(Paragraph(pdf_safe(needs_title), styles["Heading3"]))
    for item, score in report_data["needs"]:
        suffix = f" (%{score:.1f})" if score else ""
        story.append(Paragraph(pdf_safe(f"- {item}{suffix}"), styles["BodyNice"]))

    story.append(Spacer(1, 16))
    disclaimer = (
        "Bu yazilim egitim, bilimsel arastirma ve performans analizine destek amaciyla gelistirilmistir. "
        "Tahmin sonuclari tek basina kesin karar araci olarak kullanilmamalidir."
        if is_tr else
        "This software has been developed for education, scientific research, and performance analysis support. "
        "Prediction results must not be used as the sole decision-making tool."
    )
    story.append(Paragraph(pdf_safe(disclaimer), styles["BodyNice"]))
    story.append(Spacer(1, 12))
    story.append(Paragraph(pdf_safe(f"© 2026 {DEVELOPER_TR}. Tum haklari saklidir."), styles["BodyNice"]))

    doc.build(story)
    buffer.seek(0)
    return buffer


def render_hero(is_tr):
    title = PAGE_TITLE_TR if is_tr else PAGE_TITLE_EN
    desc = (
        "12–17 yaş genç yüzücülerde 50 m stil performansını; antropometrik, motorik, esneklik ve vücut kompozisyonu ölçümlerine göre analiz eden yapay zekâ destekli karar destek sistemi."
        if is_tr else
        "An AI-supported decision support system that analyzes 50 m stroke performance in 12–17-year-old swimmers based on anthropometric, motor, flexibility, and body composition measurements."
    )
    dev = f"Geliştirici: {DEVELOPER_TR} | {UNIVERSITY_TR}" if is_tr else f"Developer: {DEVELOPER_EN} | {UNIVERSITY_EN}"
    st.markdown(
        f"""
        <div class="hero-card">
            <div class="hero-title">🏊‍♂️ {APP_NAME} {APP_VERSION}</div>
            <div class="hero-subtitle">{title}</div>
            <div class="hero-desc">{desc}</div>
            <div class="hero-footer">{dev}</div>
        </div>
        """,
        unsafe_allow_html=True
    )


def render_section(title):
    st.markdown(
        f"""
        <div class="section-card">
            <div class="section-title">{title}</div>
        </div>
        """,
        unsafe_allow_html=True
    )


def render_metric(label, value, small=""):
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
            <div class="metric-small">{small}</div>
        </div>
        """,
        unsafe_allow_html=True
    )


def render_gauge(title, value, desc, color=TURQUOISE):
    value_int = int(np.clip(value, 0, 100))
    st.markdown(
        f"""
        <div class="gauge-wrap">
            <div class="gauge" style="--p:{value_int}; --c:{color};"><span>{value_int}%</span></div>
            <div class="gauge-title">{title}</div>
            <div class="gauge-desc">{desc}</div>
        </div>
        """,
        unsafe_allow_html=True
    )


def render_list_card(title, items, icon="✅"):
    lis = "".join([f"<div class='list-item'>{icon} {item[0]}" + (f" <span style='color:{TEXT_MUTED};'>%{item[1]:.1f}</span>" if item[1] else "") + "</div>" for item in items])
    st.markdown(
        f"""
        <div class="list-card">
            <div class="list-title">{title}</div>
            {lis}
        </div>
        """,
        unsafe_allow_html=True
    )


def render_footer(is_tr):
    if is_tr:
        title = f"ℹ️ {APP_NAME} {APP_VERSION} Hakkında"
        body = f"""
        <b>Geliştirici:</b> {DEVELOPER_TR}<br>
        <b>Kurum:</b> {UNIVERSITY_TR}<br><br>
        Bu yazılım eğitim, bilimsel araştırma ve performans analizine destek amacıyla geliştirilmiştir. 
        Üretilen tahminler, antrenörün mesleki değerlendirmesinin yerine geçmez; karar sürecini desteklemek amacıyla kullanılmalıdır.<br><br>
        <b>Telif ve Kullanım:</b> © 2026 {DEVELOPER_TR}. Tüm hakları saklıdır. Bu yazılımın kaynak kodlarının, algoritmalarının, yapay zekâ modellerinin, kullanıcı arayüzünün, raporlarının ve veri yapısının yazılı izin alınmaksızın çoğaltılması, kopyalanması, değiştirilmesi, tersine mühendislik uygulanması, yeniden dağıtılması, ticari amaçla kullanılması veya başka yazılımlara entegre edilmesi yasaktır.
        """
    else:
        title = f"ℹ️ About {APP_NAME} {APP_VERSION}"
        body = f"""
        <b>Developer:</b> {DEVELOPER_EN}<br>
        <b>Institution:</b> {UNIVERSITY_EN}<br><br>
        This software has been developed for education, scientific research, and performance analysis support. 
        Predictions do not replace professional coaching judgment; they should support the decision-making process.<br><br>
        <b>Copyright and Use:</b> © 2026 {DEVELOPER_EN}. All rights reserved. Unauthorized reproduction, copying, modification, reverse engineering, redistribution, commercial use, integration into another software system, or use of the source code, algorithms, AI models, user interface, reports, and data structure is prohibited without written permission.
        """
    st.markdown(
        f"""
        <div class="footer-card">
            <div class="footer-title">{title}</div>
            <div class="footer-text">{body}</div>
        </div>
        """,
        unsafe_allow_html=True
    )


def input_grid(features, inputs, is_tr=True):
    # Özellikle mobilde labels net görünsün diye Streamlit label kullanıyoruz.
    for i in range(0, len(features), 2):
        col1, col2 = st.columns(2)
        pair = features[i:i+2]
        for col, f in zip([col1, col2], pair):
            with col:
                default_value = get_default_value(f)
                label = label_for(f, is_tr)
                if f in ["age", "sit_ups_1min"]:
                    inputs[f] = st.number_input(label, min_value=0, max_value=300, value=int(default_value), key=f"input_{f}")
                else:
                    inputs[f] = st.number_input(label, min_value=0.0, max_value=500.0, value=float(default_value), key=f"input_{f}")


def categorize_features(features):
    fset = list(features)
    ant = [f for f in fset if f in ANTHROPOMETRY]
    mot = [f for f in fset if f in MOTOR]
    mob = [f for f in fset if f in MOBILITY]
    comp = [f for f in fset if f in BODY_COMP]
    other = [f for f in fset if f not in set(ant + mot + mob + comp)]
    return ant, mot, mob, comp, other


# ============================================================
# 5. ANA UYGULAMA
# ============================================================

df, metadata = load_system()

# Dil seçimi artık ana ekranda, analiz grubu seçiminden hemen önce görünür bir kart olarak yer alır.
LANGUAGE_OPTIONS = ["🇹🇷 Türkçe", "🇺🇸 English"]
if "language_choice" not in st.session_state:
    st.session_state["language_choice"] = LANGUAGE_OPTIONS[0]

is_tr = st.session_state["language_choice"].startswith("🇹🇷")

render_hero(is_tr)

# Görünür dil seçimi bölümü
render_section("🌐 Dil Seçimi / Language")
language = st.selectbox(
    "Uygulama dili / Application language",
    LANGUAGE_OPTIONS,
    index=LANGUAGE_OPTIONS.index(st.session_state["language_choice"]),
    key="language_choice",
)
is_tr = language.startswith("🇹🇷")

st.markdown(
    f"<div class='selected-group'>{'Seçilen dil' if is_tr else 'Selected language'}: {language}</div>",
    unsafe_allow_html=True
)

# Analiz grubu seçimi
render_section("📌 Analiz Grubu Seçimi" if is_tr else "📌 Analysis Group Selection")

col1, col2, col3 = st.columns(3)
with col1:
    if is_tr:
        style_display = st.selectbox("Stil", list(STYLE_TR.values()))
        style = {v: k for k, v in STYLE_TR.items()}[style_display]
    else:
        style_display = st.selectbox("Stroke", list(STYLE_EN.values()))
        style = {v: k for k, v in STYLE_EN.items()}[style_display]

with col2:
    if is_tr:
        sex_display = st.selectbox("Cinsiyet", ["Kadın", "Erkek"])
        sex = "female" if sex_display == "Kadın" else "male"
    else:
        sex_display = st.selectbox("Sex", ["Female", "Male"])
        sex = "female" if sex_display == "Female" else "male"

with col3:
    age_group = st.selectbox("Yaş Grubu" if is_tr else "Age Group", ["12_13", "14_15", "16_17"])

model_key = f"{style}_{age_group}_{sex}"
analysis_group = f"{style_display} / {sex_display} / {age_group.replace('_', '-')} " + ("yaş" if is_tr else "years")
st.markdown(f"<div class='selected-group'>{'Seçilen analiz grubu' if is_tr else 'Selected analysis group'}: {analysis_group}</div>", unsafe_allow_html=True)

if model_key not in metadata:
    st.error("Bu seçim için kayıtlı model bulunamadı." if is_tr else "No registered model was found for this selection.")
    st.stop()

model_info = metadata[model_key]
model = load_prediction_model(model_info["model_file"])
features = model_info.get("features_en", model_info.get("features", []))
if not features:
    st.error("Model değişken listesi bulunamadı." if is_tr else "Model feature list was not found.")
    st.stop()

# Sporcu bilgileri
render_section("👤 Sporcu Bilgileri" if is_tr else "👤 Athlete Information")
col1, col2, col3 = st.columns(3)
with col1:
    athlete_name = st.text_input("Ad Soyad" if is_tr else "Full Name", value="")
with col2:
    club_name = st.text_input("Kulüp" if is_tr else "Club", value="")
with col3:
    coach_name = st.text_input("Antrenör Ad Soyad" if is_tr else "Coach Full Name", value="")

# Ölçüm girişi
render_section("📏 Ölçüm Girişi" if is_tr else "📏 Measurement Entry")
st.markdown(
    f"<div class='soft-note'>{'👇 Aç / Kapat işaretli başlıklara basarak ölçüm alanlarını açabilirsiniz.' if is_tr else '👇 Tap the Open / Close headings to display measurement fields.'}</div>",
    unsafe_allow_html=True
)

ant, mot, mob, comp, other = categorize_features(features)
inputs = {}

if ant:
    with st.expander("👇 Aç / Kapat: Antropometrik Ölçümler" if is_tr else "👇 Open / Close: Anthropometric Measurements", expanded=True):
        input_grid(ant, inputs, is_tr)
if mot:
    with st.expander("👇 Aç / Kapat: Motorik Performans Testleri" if is_tr else "👇 Open / Close: Motor Performance Tests", expanded=False):
        input_grid(mot, inputs, is_tr)
if mob:
    with st.expander("👇 Aç / Kapat: Esneklik ve Mobilite" if is_tr else "👇 Open / Close: Flexibility and Mobility", expanded=False):
        input_grid(mob, inputs, is_tr)
if comp:
    with st.expander("👇 Aç / Kapat: Vücut Kompozisyonu" if is_tr else "👇 Open / Close: Body Composition", expanded=False):
        input_grid(comp, inputs, is_tr)
if other:
    with st.expander("👇 Aç / Kapat: Diğer Ölçümler" if is_tr else "👇 Open / Close: Other Measurements", expanded=False):
        input_grid(other, inputs, is_tr)

st.info((f"Modelde kullanılan değişken sayısı: {len(features)}") if is_tr else (f"Number of variables used in the model: {len(features)}"))

button_text = "🧠 Tahmin Et ve Karar Destek Raporu Oluştur" if is_tr else "🧠 Predict and Generate Decision Support Report"
run_prediction = st.button(button_text, use_container_width=False)

if run_prediction:
    report_id = f"SWIM-{datetime.now().strftime('%Y%m%d')}-{str(uuid.uuid4())[:8].upper()}"

    # Modelin beklediği sırayla input dataframe
    input_df = pd.DataFrame([{f: inputs.get(f, get_default_value(f)) for f in features}])

    try:
        prediction = float(model.predict(input_df)[0])
    except Exception as e:
        st.error(("Tahmin sırasında hata oluştu: " if is_tr else "Prediction error: ") + str(e))
        st.stop()

    target_col = model_info.get("target")
    if not target_col or target_col not in df.columns:
        st.error("Hedef değişken veri setinde bulunamadı." if is_tr else "Target variable was not found in the dataset.")
        st.stop()

    group_data = df[(df["sex"] == sex) & (df["age_group"] == age_group)].copy()
    group_mean = safe_float(group_data[target_col].mean(), prediction)
    difference = prediction - group_mean
    percentile = calculate_percentile(group_data, target_col, prediction)
    level_text, level_color = performance_level(percentile, is_tr)
    conf = confidence_score(model_info)
    physical_score = physical_profile_score(inputs, group_data, features)
    potential = potential_score(percentile, physical_score, conf)
    strengths, needs = identify_strengths_and_needs(inputs, group_data, features, is_tr)
    comment = coach_comment(prediction, group_mean, difference, percentile, conf, is_tr)

    mae = get_model_metric(model_info, "mae", 0.0)
    rmse = get_model_metric(model_info, "rmse", 0.0)
    r2 = get_model_metric(model_info, "r2", 0.0)
    error_margin = mae if mae > 0 else rmse

    render_section("🏁 Tahmin Sonucu" if is_tr else "🏁 Prediction Result")

    m1, m2, m3, m4 = st.columns(4)
    with m1:
        render_metric("Tahmini Derece" if is_tr else "Predicted Time", f"{prediction:.2f} sn" if is_tr else f"{prediction:.2f} s", style_display)
    with m2:
        render_metric("Grup Ortalaması" if is_tr else "Group Mean", f"{group_mean:.2f} sn" if is_tr else f"{group_mean:.2f} s", sex_display)
    with m3:
        diff_text = f"{difference:+.2f} sn" if is_tr else f"{difference:+.2f} s"
        render_metric("Fark" if is_tr else "Difference", diff_text, "Negatif değer daha hızlıdır" if is_tr else "Negative means faster")
    with m4:
        render_metric("Hata Aralığı" if is_tr else "Error Range", f"±{error_margin:.2f} sn" if is_tr else f"±{error_margin:.2f} s", "Model MAE/RMSE temelli" if is_tr else "Based on model MAE/RMSE")

    render_section("⭕ Dairesel Performans Göstergeleri" if is_tr else "⭕ Circular Performance Indicators")
    g1, g2, g3, g4 = st.columns(4)
    with g1:
        render_gauge("Performans" if is_tr else "Performance", percentile, level_text, level_color)
    with g2:
        render_gauge("Tahmin Güveni" if is_tr else "Prediction Confidence", conf, "Model doğruluk göstergesi" if is_tr else "Model reliability indicator", TURQUOISE)
    with g3:
        render_gauge("Fiziksel Profil" if is_tr else "Physical Profile", physical_score, "Referans gruba göre" if is_tr else "Compared with reference group", SUCCESS if physical_score >= 65 else WARNING)
    with g4:
        render_gauge("Gelişim Potansiyeli" if is_tr else "Development Potential", potential, "Karar destek puanı" if is_tr else "Decision support score", SUCCESS if potential >= 65 else WARNING)

    st.markdown(
        f"""
        <div class="coach-card">
            <div class="coach-title">🤖 {'AI Coach Karar Destek Yorumu' if is_tr else 'AI Coach Decision Support Comment'}</div>
            <div class="coach-text">{comment}</div>
        </div>
        """,
        unsafe_allow_html=True
    )

    c1, c2 = st.columns(2)
    with c1:
        render_list_card("✅ Güçlü Alanlar" if is_tr else "✅ Strengths", strengths, "✅")
    with c2:
        render_list_card("🔧 Geliştirilebilir Alanlar" if is_tr else "🔧 Development Areas", needs, "🔧")

    render_section("🧠 Model Doğruluk Bilgileri" if is_tr else "🧠 Model Accuracy Information")
    info_cols = st.columns(4)
    with info_cols[0]:
        render_metric("Model", str(model_info.get("model", "Model")), "")
    with info_cols[1]:
        render_metric("MAE", f"{mae:.2f} sn" if is_tr else f"{mae:.2f} s", "")
    with info_cols[2]:
        render_metric("RMSE", f"{rmse:.2f} sn" if is_tr else f"{rmse:.2f} s", "")
    with info_cols[3]:
        render_metric("R²", f"{r2:.3f}", "")

    st.markdown(
        f"""
        <div class="warning-box">
        {'Bu sistem karar destek amacıyla geliştirilmiştir. Tahmin sonucu tek başına kesin sporcu seçimi, performans değerlendirmesi veya sağlık kararı için kullanılmamalıdır.' if is_tr else 'This system is designed for decision support. Prediction results must not be used alone for definitive athlete selection, performance evaluation, or health-related decisions.'}
        </div>
        """,
        unsafe_allow_html=True
    )

    report_data = {
        "report_id": report_id,
        "date": datetime.now().strftime("%d.%m.%Y %H:%M"),
        "athlete_name": athlete_name,
        "club_name": club_name,
        "coach_name": coach_name,
        "analysis_group": analysis_group,
        "prediction": prediction,
        "group_mean": group_mean,
        "difference": difference,
        "percentile": percentile,
        "confidence": conf,
        "model_name": str(model_info.get("model", "Model")),
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
        "comment": comment,
        "strengths": strengths,
        "needs": needs,
    }
    pdf = make_pdf_report(is_tr, report_data)
    pdf_name = f"swimml_pro_{model_key}_{report_id}.pdf"
    st.download_button(
        label="📄 PDF Raporunu İndir" if is_tr else "📄 Download PDF Report",
        data=pdf,
        file_name=pdf_name,
        mime="application/pdf"
    )

render_footer(is_tr)
