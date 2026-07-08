# -*- coding: utf-8 -*-
"""
SwimML Pro v1.0
50 m Yüzme Performans Tahmin ve Antrenör Karar Destek Sistemi
50 m Prediction Coach Decision Support System

Geliştirici / Developer:
Öğr. Gör. Tuğrul Özkadı ve Ekibi
Hitit Üniversitesi Spor Bilimleri Fakültesi

Gerekli dosyalar:
- swim_data.xlsx
- model_metadata.json
- model_metadata.json içinde tanımlı .joblib model dosyaları

Çalıştırma:
streamlit run swimml_pro_v1_nirvana.py
"""

import json
import uuid
import math
from io import BytesIO
from pathlib import Path
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
import streamlit as st
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle


# ============================================================
# CONFIG
# ============================================================

APP_VERSION = "v1.0"
DATA_FILE = "swim_data.xlsx"
METADATA_FILE = "model_metadata.json"

DEVELOPER_TR = "Öğr. Gör. Tuğrul Özkadı ve Ekibi"
DEVELOPER_EN = "Lect. Tuğrul Özkadı and Team"
UNIVERSITY_TR = "Hitit Üniversitesi Spor Bilimleri Fakültesi"
UNIVERSITY_EN = "Hitit University Faculty of Sport Sciences"

THEME = {
    "navy": "#071A2C",
    "navy2": "#0B253F",
    "turquoise": "#00B8C8",
    "turquoise2": "#22D3EE",
    "ice": "#EAFBFF",
    "white": "#FFFFFF",
    "gray": "#64748B",
    "dark_gray": "#334155",
    "green": "#10B981",
    "orange": "#F59E0B",
    "red": "#EF4444",
}

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
AGE_GROUPS = ["12_13", "14_15", "16_17"]

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
    "anthropometry": [
        "age", "body_height", "body_mass", "training_age", "hand_length", "foot_length",
        "arm_span", "arm_length", "leg_length", "sitting_height", "chest_girth",
        "biceps_girth", "flexed_biceps_girth", "forearm_girth", "thigh_girth", "calf_girth",
        "waist_girth", "hip_girth", "biacromial_breadth", "biiliac_breadth",
        "elbow_breadth", "wrist_breadth", "knee_breadth", "ankle_breadth",
    ],
    "motor": [
        "vertical_jump", "standing_long_jump", "flexed_arm_hang", "sit_ups_1min",
        "illinois_agility_test", "sprint_30m", "handgrip_strength", "flamingo_balance_test",
    ],
    "mobility": [
        "sit_and_reach", "trunk_extension_height", "shoulder_extension_height", "shoulder_mobility",
        "ankle_dorsiflexion_rom", "ankle_plantarflexion_rom",
    ],
    "composition": [
        "body_density", "body_fat_percentage", "fat_mass", "fat_free_mass",
    ],
}

GROUP_TITLES_TR = {
    "anthropometry": "▶ Antropometrik Ölçümler",
    "motor": "▶ Motorik Performans Testleri",
    "mobility": "▶ Esneklik ve Mobilite",
    "composition": "▶ Vücut Kompozisyonu",
    "other": "▶ Diğer Model Değişkenleri",
}

GROUP_TITLES_EN = {
    "anthropometry": "▶ Anthropometric Measurements",
    "motor": "▶ Motor Performance Tests",
    "mobility": "▶ Flexibility and Mobility",
    "composition": "▶ Body Composition",
    "other": "▶ Other Model Variables",
}

DEFAULTS = {
    "age": 15, "body_height": 160.0, "body_mass": 55.0, "training_age": 4.0,
    "vertical_jump": 35.0, "standing_long_jump": 180.0, "flexed_arm_hang": 30.0,
    "sit_ups_1min": 35, "illinois_agility_test": 18.0, "sprint_30m": 5.0,
    "hand_length": 17.0, "arm_span": 165.0, "arm_length": 65.0, "leg_length": 80.0,
    "sitting_height": 80.0, "chest_girth": 80.0, "biceps_girth": 25.0,
    "flexed_biceps_girth": 27.0, "forearm_girth": 22.0, "thigh_girth": 45.0,
    "calf_girth": 32.0, "waist_girth": 70.0, "hip_girth": 85.0,
    "biacromial_breadth": 35.0, "biiliac_breadth": 28.0, "elbow_breadth": 6.0,
    "wrist_breadth": 5.0, "knee_breadth": 8.0, "ankle_breadth": 6.0,
    "sit_and_reach": 20.0, "trunk_extension_height": 35.0,
    "shoulder_extension_height": 40.0, "shoulder_mobility": 0.0,
    "flamingo_balance_test": 10.0, "body_density": 1.05, "body_fat_percentage": 15.0,
    "fat_mass": 8.0, "fat_free_mass": 45.0, "handgrip_strength": 30.0,
    "ankle_dorsiflexion_rom": 30.0, "ankle_plantarflexion_rom": 45.0,
    "foot_length": 24.0,
}

LOWER_IS_BETTER = {
    "sprint_30m", "illinois_agility_test", "flamingo_balance_test",
    "body_fat_percentage", "fat_mass", "waist_girth",
}


# ============================================================
# PAGE STYLE
# ============================================================

st.set_page_config(
    page_title="SwimML Pro v1.0",
    page_icon="🏊‍♂️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

CUSTOM_CSS = f"""
<style>
    .stApp {{
        background: linear-gradient(135deg, #F8FCFF 0%, #EAFBFF 42%, #FFFFFF 100%);
    }}
    div[data-testid="stSidebar"] {{display: none;}}
    .block-container {{
        padding-top: 1.2rem;
        padding-bottom: 2.5rem;
        max-width: 1200px;
    }}
    .hero {{
        background: radial-gradient(circle at top left, {THEME['turquoise']} 0%, {THEME['navy']} 38%, #03111F 100%);
        color: white;
        padding: 28px 28px;
        border-radius: 28px;
        box-shadow: 0 18px 45px rgba(7, 26, 44, 0.26);
        margin-bottom: 20px;
        border: 1px solid rgba(255,255,255,0.18);
    }}
    .hero h1 {{
        font-size: clamp(30px, 5vw, 56px);
        line-height: 1.03;
        margin: 0 0 10px 0;
        font-weight: 900;
        letter-spacing: -1.2px;
    }}
    .hero .subtitle {{
        font-size: clamp(16px, 2vw, 23px);
        opacity: 0.96;
        margin-bottom: 14px;
        font-weight: 650;
    }}
    .hero .developer {{
        font-size: 14px;
        opacity: 0.86;
        margin-top: 10px;
    }}
    .pill {{
        display: inline-block;
        padding: 7px 12px;
        border-radius: 999px;
        background: rgba(255,255,255,0.13);
        border: 1px solid rgba(255,255,255,0.22);
        margin-right: 8px;
        margin-top: 8px;
        font-size: 13px;
    }}
    .card {{
        background: rgba(255,255,255,0.92);
        border: 1px solid rgba(0,184,200,0.18);
        border-radius: 22px;
        padding: 20px;
        box-shadow: 0 12px 30px rgba(7, 26, 44, 0.08);
        margin-bottom: 18px;
    }}
    .small-caption {{
        color: {THEME['gray']};
        font-size: 13px;
    }}
    .section-title {{
        color: {THEME['navy']};
        font-weight: 850;
        font-size: 23px;
        margin: 6px 0 12px 0;
    }}
    .metric-card {{
        background: #FFFFFF;
        border-radius: 20px;
        padding: 18px;
        border: 1px solid rgba(0,184,200,0.24);
        box-shadow: 0 8px 26px rgba(7, 26, 44, 0.08);
        height: 100%;
    }}
    .metric-title {{
        color: {THEME['gray']};
        font-size: 13px;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.04em;
    }}
    .metric-value {{
        color: {THEME['navy']};
        font-size: 34px;
        font-weight: 900;
        margin-top: 6px;
        letter-spacing: -0.8px;
    }}
    .metric-note {{
        color: {THEME['dark_gray']};
        font-size: 13px;
        margin-top: 4px;
    }}
    .ai-box {{
        background: linear-gradient(135deg, #FFFFFF 0%, #EAFBFF 100%);
        border: 1px solid rgba(0,184,200,0.32);
        border-left: 6px solid {THEME['turquoise']};
        border-radius: 20px;
        padding: 18px 20px;
        color: {THEME['navy']};
        box-shadow: 0 8px 26px rgba(7, 26, 44, 0.08);
    }}
    .footer-box {{
        background: {THEME['navy']};
        color: white;
        border-radius: 24px;
        padding: 22px;
        margin-top: 26px;
        border: 1px solid rgba(255,255,255,0.12);
    }}
    .footer-box p, .footer-box li {{
        color: rgba(255,255,255,0.88);
        font-size: 13px;
    }}
    .footer-box h3 {{
        color: white;
    }}
    .stButton>button {{
        width: 100%;
        border-radius: 18px;
        background: linear-gradient(135deg, {THEME['turquoise']} 0%, {THEME['navy']} 100%);
        color: white;
        font-weight: 850;
        font-size: 20px;
        padding: 0.85rem 1rem;
        border: 0px;
        box-shadow: 0 12px 28px rgba(0,184,200,0.28);
    }}
    .stButton>button:hover {{
        transform: translateY(-1px);
        box-shadow: 0 16px 34px rgba(0,184,200,0.34);
        color: white;
    }}
    details {{
        background: rgba(255,255,255,0.92) !important;
        border-radius: 18px !important;
        border: 1px solid rgba(0,184,200,0.20) !important;
        box-shadow: 0 7px 20px rgba(7, 26, 44, 0.06) !important;
        margin-bottom: 12px !important;
    }}
    summary {{
        font-weight: 850 !important;
        color: {THEME['navy']} !important;
    }}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# ============================================================
# LOADERS
# ============================================================

@st.cache_data(show_spinner=False)
def load_data_and_metadata():
    data_path = Path(DATA_FILE)
    metadata_path = Path(METADATA_FILE)

    if not data_path.exists():
        st.error(f"Veri dosyası bulunamadı / Data file not found: {DATA_FILE}")
        st.stop()

    if not metadata_path.exists():
        st.error(f"Metadata dosyası bulunamadı / Metadata file not found: {METADATA_FILE}")
        st.stop()

    df = pd.read_excel(data_path)
    with open(metadata_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    return df, metadata


@st.cache_resource(show_spinner=False)
def load_prediction_model(model_file):
    model_path = Path(str(model_file).replace("\\", "/"))
    if not model_path.exists():
        st.error(f"Model dosyası bulunamadı / Model file not found: {model_path}")
        st.stop()
    return joblib.load(model_path)


# ============================================================
# HELPERS
# ============================================================

def safe_float(value, default=np.nan):
    try:
        if pd.isna(value):
            return default
        return float(value)
    except Exception:
        return default


def get_default_value(feature):
    return DEFAULTS.get(feature, 0.0)


def fmt(value, digits=2):
    try:
        if pd.isna(value):
            return "-"
        return f"{float(value):.{digits}f}"
    except Exception:
        return str(value)


def normalize_score(value, mean, std, lower_is_better=False):
    if pd.isna(value) or pd.isna(mean) or pd.isna(std) or std == 0:
        return 50.0
    z = (value - mean) / std
    if lower_is_better:
        z = -z
    score = 50 + (z * 15)
    return max(0.0, min(100.0, score))


def performance_comment(percentile, is_tr=True):
    if percentile >= 85:
        return ("Çok yüksek performans düzeyi", THEME["green"]) if is_tr else ("Very high performance level", THEME["green"])
    if percentile >= 65:
        return ("İyi performans düzeyi", THEME["green"]) if is_tr else ("Good performance level", THEME["green"])
    if percentile >= 40:
        return ("Orta performans düzeyi", THEME["orange"]) if is_tr else ("Moderate performance level", THEME["orange"])
    return ("Geliştirilmesi gereken performans düzeyi", THEME["red"]) if is_tr else ("Performance needs improvement", THEME["red"])


def estimate_error_margin(model_info):
    rmse = safe_float(model_info.get("rmse", np.nan))
    mae = safe_float(model_info.get("mae", np.nan))
    if not pd.isna(rmse):
        return rmse
    if not pd.isna(mae):
        return mae
    return 0.75


def confidence_from_model(model_info):
    r2 = safe_float(model_info.get("r2", np.nan))
    mape = safe_float(model_info.get("mape", np.nan))
    if not pd.isna(r2):
        return max(45, min(98, 50 + r2 * 48))
    if not pd.isna(mape):
        return max(45, min(96, 100 - mape))
    return 80


def assign_features_to_groups(features):
    remaining = list(features)
    grouped = {k: [] for k in FEATURE_GROUPS.keys()}
    grouped["other"] = []

    for group, group_features in FEATURE_GROUPS.items():
        for f in group_features:
            if f in remaining:
                grouped[group].append(f)
                remaining.remove(f)
    grouped["other"] = remaining
    return grouped


def feature_label(feature, is_tr=True):
    return (FEATURE_LABELS_TR if is_tr else FEATURE_LABELS_EN).get(feature, feature)


def group_title(group, is_tr=True):
    return (GROUP_TITLES_TR if is_tr else GROUP_TITLES_EN).get(group, group)


def calculate_domain_scores(inputs, group_data):
    scores = {}
    for group, group_features in FEATURE_GROUPS.items():
        vals = []
        for f in group_features:
            if f in inputs and f in group_data.columns:
                mean = group_data[f].mean()
                std = group_data[f].std()
                vals.append(normalize_score(inputs[f], mean, std, f in LOWER_IS_BETTER))
        scores[group] = float(np.mean(vals)) if vals else 50.0
    return scores


def identify_strengths_and_needs(inputs, group_data, is_tr=True, max_items=5):
    rows = []
    for f, v in inputs.items():
        if f not in group_data.columns:
            continue
        mean = group_data[f].mean()
        std = group_data[f].std()
        score = normalize_score(v, mean, std, f in LOWER_IS_BETTER)
        rows.append((f, score))

    rows = sorted(rows, key=lambda x: x[1], reverse=True)
    strengths = rows[:max_items]
    needs = sorted(rows, key=lambda x: x[1])[:max_items]

    strengths_txt = [feature_label(f, is_tr) for f, _ in strengths]
    needs_txt = [feature_label(f, is_tr) for f, _ in needs]
    return strengths_txt, needs_txt


def ai_coach_comment(percentile, difference, error_margin, strengths, needs, is_tr=True):
    if is_tr:
        if percentile >= 65:
            base = "Bu sporcu, seçilen yaş grubu ve cinsiyet referansına göre güçlü bir performans profili göstermektedir."
        elif percentile >= 40:
            base = "Bu sporcu, seçilen referans gruba göre orta düzeyde ve geliştirilebilir bir performans profili göstermektedir."
        else:
            base = "Bu sporcu için performans gelişimini destekleyecek teknik, motorik ve fiziksel çalışmaların planlanması önerilir."

        diff_txt = "ortalamanın üzerinde" if difference < 0 else "ortalamanın gerisinde"
        strong_txt = ", ".join(strengths[:3]) if strengths else "belirgin güçlü alan hesaplanamadı"
        need_txt = ", ".join(needs[:3]) if needs else "belirgin gelişim alanı hesaplanamadı"
        return (
            f"{base} Tahmini derece, grup ortalamasına göre {diff_txt} bir konumu işaret etmektedir. "
            f"Model hata aralığı yaklaşık ±{error_margin:.2f} sn olduğundan sonuç antrenör gözlemiyle birlikte yorumlanmalıdır. "
            f"Öne çıkan güçlü alanlar: {strong_txt}. Geliştirilebilir alanlar: {need_txt}."
        )

    if percentile >= 65:
        base = "The swimmer shows a strong performance profile compared with the selected age-sex reference group."
    elif percentile >= 40:
        base = "The swimmer shows a moderate and improvable performance profile compared with the reference group."
    else:
        base = "Technical, motor and physical development work is recommended for this swimmer."

    diff_txt = "above the group mean" if difference < 0 else "behind the group mean"
    strong_txt = ", ".join(strengths[:3]) if strengths else "no clear strength could be calculated"
    need_txt = ", ".join(needs[:3]) if needs else "no clear development area could be calculated"
    return (
        f"{base} The predicted time indicates a position {diff_txt}. "
        f"Because the model error margin is approximately ±{error_margin:.2f} s, the result should be interpreted together with coach observation. "
        f"Key strengths: {strong_txt}. Development areas: {need_txt}."
    )


def circular_gauge_html(title, value, caption, color):
    value = max(0, min(100, float(value)))
    return f"""
    <div style="background:#fff;border:1px solid rgba(0,184,200,.22);border-radius:24px;padding:18px;text-align:center;box-shadow:0 10px 28px rgba(7,26,44,.08);height:100%;">
      <div style="font-size:13px;font-weight:800;color:#64748B;text-transform:uppercase;letter-spacing:.04em;">{title}</div>
      <div style="width:142px;height:142px;border-radius:50%;margin:14px auto 8px auto;background:conic-gradient({color} {value*3.6}deg,#E6EEF5 0deg);display:flex;align-items:center;justify-content:center;">
        <div style="width:104px;height:104px;border-radius:50%;background:#fff;display:flex;align-items:center;justify-content:center;box-shadow:inset 0 0 0 1px rgba(7,26,44,.06);">
          <div style="font-size:30px;font-weight:950;color:#071A2C;">{value:.0f}%</div>
        </div>
      </div>
      <div style="font-size:14px;font-weight:800;color:#071A2C;">{caption}</div>
    </div>
    """


def metric_card_html(title, value, note=""):
    return f"""
    <div class="metric-card">
      <div class="metric-title">{title}</div>
      <div class="metric-value">{value}</div>
      <div class="metric-note">{note}</div>
    </div>
    """


def fix_pdf_text(text):
    replacements = {
        "ı": "i", "İ": "I", "ğ": "g", "Ğ": "G", "ü": "u", "Ü": "U",
        "ş": "s", "Ş": "S", "ö": "o", "Ö": "O", "ç": "c", "Ç": "C",
        "²": "2", "±": "+/-", "–": "-", "—": "-",
    }
    text = str(text)
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text


def create_pdf_report(report):
    buffer = BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=1.4 * cm,
        leftMargin=1.4 * cm,
        topMargin=1.2 * cm,
        bottomMargin=1.2 * cm,
    )

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "TitleStyle", parent=styles["Title"], fontName="Helvetica-Bold", fontSize=18,
        leading=22, textColor=colors.HexColor("#071A2C"), spaceAfter=10
    )
    h_style = ParagraphStyle(
        "HeaderStyle", parent=styles["Heading2"], fontName="Helvetica-Bold", fontSize=12,
        leading=15, textColor=colors.HexColor("#071A2C"), spaceBefore=8, spaceAfter=6
    )
    p_style = ParagraphStyle(
        "PStyle", parent=styles["BodyText"], fontName="Helvetica", fontSize=9,
        leading=12, textColor=colors.HexColor("#334155")
    )

    elements = []
    elements.append(Paragraph(fix_pdf_text(report["title"]), title_style))
    elements.append(Paragraph(fix_pdf_text(report["subtitle"]), p_style))
    elements.append(Spacer(1, 8))

    def add_table(title, rows):
        elements.append(Paragraph(fix_pdf_text(title), h_style))
        data = [[fix_pdf_text(a), fix_pdf_text(b)] for a, b in rows]
        table = Table(data, colWidths=[5.2 * cm, 11.0 * cm])
        table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#EAFBFF")),
            ("TEXTCOLOR", (0, 0), (-1, -1), colors.HexColor("#071A2C")),
            ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
            ("FONTNAME", (1, 0), (1, -1), "Helvetica"),
            ("FONTSIZE", (0, 0), (-1, -1), 8.5),
            ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#BFEFF5")),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("ROWBACKGROUNDS", (0, 0), (-1, -1), [colors.white, colors.HexColor("#F8FCFF")]),
            ("LEFTPADDING", (0, 0), (-1, -1), 6),
            ("RIGHTPADDING", (0, 0), (-1, -1), 6),
            ("TOPPADDING", (0, 0), (-1, -1), 5),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ]))
        elements.append(table)
        elements.append(Spacer(1, 8))

    add_table(report["labels"]["athlete"], report["athlete_rows"])
    add_table(report["labels"]["model"], report["model_rows"])
    add_table(report["labels"]["result"], report["result_rows"])

    elements.append(Paragraph(fix_pdf_text(report["labels"]["coach"]), h_style))
    elements.append(Paragraph(fix_pdf_text(report["coach_comment"]), p_style))
    elements.append(Spacer(1, 8))

    add_table(report["labels"]["strengths"], [(str(i + 1), s) for i, s in enumerate(report["strengths"])])
    add_table(report["labels"]["needs"], [(str(i + 1), s) for i, s in enumerate(report["needs"])])

    elements.append(Paragraph(fix_pdf_text(report["labels"]["copyright"]), h_style))
    elements.append(Paragraph(fix_pdf_text(report["copyright_text"]), p_style))

    doc.build(elements)
    buffer.seek(0)
    return buffer


# ============================================================
# APP
# ============================================================

df, metadata = load_data_and_metadata()

language = st.radio("Dil / Language", ["🇹🇷 Türkçe", "🇺🇸 English"], horizontal=True, label_visibility="collapsed")
is_tr = language.startswith("🇹🇷")

if is_tr:
    hero_title = f"SwimML Pro {APP_VERSION}"
    hero_subtitle = "50 m Yüzme Performans Tahmin ve Antrenör Karar Destek Sistemi"
    hero_desc = "Genç yüzücülerde stil, yaş grubu ve cinsiyete özgü performans tahmini; antrenör karar sürecini destekleyen sade ve bilimsel arayüz."
    dev_line = f"Geliştirici: {DEVELOPER_TR} | {UNIVERSITY_TR}"
else:
    hero_title = f"SwimML Pro {APP_VERSION}"
    hero_subtitle = "50 m Prediction Coach Decision Support System"
    hero_desc = "A style-, age-group- and sex-specific 50 m swimming performance prediction interface designed to support coaching decisions."
    dev_line = f"Developer: {DEVELOPER_EN} | {UNIVERSITY_EN}"

st.markdown(
    f"""
    <div class="hero">
      <h1>🏊‍♂️ {hero_title}</h1>
      <div class="subtitle">{hero_subtitle}</div>
      <div>{hero_desc}</div>
      <div>
        <span class="pill">AI-Based</span>
        <span class="pill">50 m Prediction</span>
        <span class="pill">Coach Decision Support</span>
        <span class="pill">Mobile Ready</span>
      </div>
      <div class="developer">{dev_line}</div>
    </div>
    """,
    unsafe_allow_html=True,
)

# Selection panel
st.markdown(f"<div class='card'><div class='section-title'>{'📌 Analiz Grubu Seçimi' if is_tr else '📌 Analysis Group Selection'}</div>", unsafe_allow_html=True)

c1, c2, c3 = st.columns(3)
with c1:
    if is_tr:
        style_display = st.selectbox("Stil", list(STYLE_TR.values()))
        style = {v: k for k, v in STYLE_TR.items()}[style_display]
    else:
        style_display = st.selectbox("Stroke", list(STYLE_EN.values()))
        style = {v: k for k, v in STYLE_EN.items()}[style_display]

with c2:
    if is_tr:
        sex_display = st.selectbox("Cinsiyet", list(SEX_TR.values()))
        sex = {v: k for k, v in SEX_TR.items()}[sex_display]
    else:
        sex_display = st.selectbox("Sex", list(SEX_EN.values()))
        sex = {v: k for k, v in SEX_EN.items()}[sex_display]

with c3:
    age_group = st.selectbox("Yaş Grubu" if is_tr else "Age Group", AGE_GROUPS)

selected_group_text = (
    f"Seçilen analiz grubu: {style_display} / {sex_display} / {age_group.replace('_', '-')} yaş"
    if is_tr else
    f"Selected analysis group: {style_display} / {sex_display} / age {age_group.replace('_', '-')}"
)
st.markdown(f"<div class='small-caption'>{selected_group_text}</div></div>", unsafe_allow_html=True)

# Athlete info
st.markdown(f"<div class='card'><div class='section-title'>{'👤 Sporcu Bilgileri' if is_tr else '👤 Athlete Information'}</div>", unsafe_allow_html=True)
a1, a2, a3 = st.columns(3)
with a1:
    athlete_name = st.text_input("Ad Soyad" if is_tr else "Full Name", value="")
with a2:
    club_name = st.text_input("Kulüp" if is_tr else "Club", value="")
with a3:
    coach_name = st.text_input("Antrenör Ad Soyad" if is_tr else "Coach Full Name", value="")
st.markdown("</div>", unsafe_allow_html=True)

# Model check
model_key = f"{style}_{age_group}_{sex}"
if model_key not in metadata:
    st.error("Bu seçim için kayıtlı model bulunamadı." if is_tr else "No registered model was found for this selection.")
    st.stop()

model_info = metadata[model_key]
model = load_prediction_model(model_info["model_file"])
features = model_info.get("features_en", model_info.get("features", []))
if not features:
    st.error("Metadata içinde model değişkenleri bulunamadı." if is_tr else "Model features were not found in metadata.")
    st.stop()

target_col = model_info.get("target")
if not target_col or target_col not in df.columns:
    st.error("Hedef değişken veri setinde bulunamadı." if is_tr else "Target variable was not found in the dataset.")
    st.stop()

# Inputs
st.markdown(f"<div class='card'><div class='section-title'>{'📏 Ölçüm Girişi' if is_tr else '📏 Measurement Entry'}</div>", unsafe_allow_html=True)
st.markdown(
    "<div class='small-caption'>▶ işaretli başlıklara basarak ölçüm alanlarını açabilirsiniz.</div>" if is_tr else
    "<div class='small-caption'>Tap the ▶ sections to expand measurement fields.</div>",
    unsafe_allow_html=True,
)

grouped_features = assign_features_to_groups(features)
inputs = {}

for group_name, group_features in grouped_features.items():
    if not group_features:
        continue
    with st.expander(group_title(group_name, is_tr), expanded=(group_name == "anthropometry")):
        cols = st.columns(2)
        for idx, feature in enumerate(group_features):
            label = feature_label(feature, is_tr)
            default_value = get_default_value(feature)
            with cols[idx % 2]:
                if feature in ["age", "sit_ups_1min"]:
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

st.info((f"Modelde kullanılan değişken sayısı: {len(features)}" if is_tr else f"Number of variables used in the model: {len(features)}"))
st.markdown("</div>", unsafe_allow_html=True)

# Prediction button
button_label = "🧠 Tahmin Et ve Karar Destek Raporu Oluştur" if is_tr else "🧠 Predict and Generate Coach Decision Report"
predict_clicked = st.button(button_label)

if predict_clicked:
    input_df = pd.DataFrame([inputs])
    input_df = input_df.reindex(columns=features)

    try:
        prediction = float(model.predict(input_df)[0])
    except Exception as e:
        st.error(("Tahmin sırasında hata oluştu: " if is_tr else "Prediction error: ") + str(e))
        st.stop()

    group_data = df[(df["sex"] == sex) & (df["age_group"] == age_group)].copy()
    if group_data.empty:
        group_data = df.copy()
        st.warning("Seçilen gruba ait veri bulunamadı; genel veri seti referans alındı." if is_tr else "No data for selected group; full dataset was used as reference.")

    group_mean = safe_float(group_data[target_col].mean())
    group_std = safe_float(group_data[target_col].std())
    difference = prediction - group_mean
    percentile = float((group_data[target_col] > prediction).mean() * 100) if target_col in group_data else 50.0
    percentile = max(0, min(100, percentile))
    error_margin = estimate_error_margin(model_info)
    confidence = confidence_from_model(model_info)

    comment, comment_color = performance_comment(percentile, is_tr)
    domain_scores = calculate_domain_scores(inputs, group_data)
    physical_profile = float(np.mean(list(domain_scores.values()))) if domain_scores else 50.0
    development_potential = max(35, min(98, 100 - abs(percentile - 70) * 0.7 + (100 - physical_profile) * 0.15))

    strengths, needs = identify_strengths_and_needs(inputs, group_data, is_tr)
    coach_comment = ai_coach_comment(percentile, difference, error_margin, strengths, needs, is_tr)

    st.markdown(f"<div class='card'><div class='section-title'>{'🚀 Tahmin Sonucu' if is_tr else '🚀 Prediction Result'}</div>", unsafe_allow_html=True)

    r1, r2, r3, r4 = st.columns(4)
    with r1:
        st.markdown(metric_card_html("Tahmini Derece" if is_tr else "Predicted Time", f"{prediction:.2f} sn" if is_tr else f"{prediction:.2f} s", style_display), unsafe_allow_html=True)
    with r2:
        st.markdown(metric_card_html("Grup Ortalaması" if is_tr else "Group Mean", f"{group_mean:.2f} sn" if is_tr else f"{group_mean:.2f} s", selected_group_text), unsafe_allow_html=True)
    with r3:
        diff_note = "daha hızlı" if difference < 0 and is_tr else "daha yavaş" if is_tr else "faster" if difference < 0 else "slower"
        st.markdown(metric_card_html("Fark" if is_tr else "Difference", f"{difference:+.2f} sn" if is_tr else f"{difference:+.2f} s", diff_note), unsafe_allow_html=True)
    with r4:
        st.markdown(metric_card_html("Hata Aralığı" if is_tr else "Error Margin", f"±{error_margin:.2f} sn" if is_tr else f"±{error_margin:.2f} s", "RMSE/MAE"), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    g1, g2, g3, g4 = st.columns(4)
    with g1:
        st.markdown(circular_gauge_html("Performans" if is_tr else "Performance", percentile, comment, comment_color), unsafe_allow_html=True)
    with g2:
        st.markdown(circular_gauge_html("Tahmin Güveni" if is_tr else "Prediction Confidence", confidence, "Model güven düzeyi" if is_tr else "Model confidence", THEME["turquoise"]), unsafe_allow_html=True)
    with g3:
        st.markdown(circular_gauge_html("Fiziksel Profil" if is_tr else "Physical Profile", physical_profile, "Ölçüm profili" if is_tr else "Measurement profile", THEME["green"] if physical_profile >= 65 else THEME["orange"]), unsafe_allow_html=True)
    with g4:
        st.markdown(circular_gauge_html("Gelişim Potansiyeli" if is_tr else "Development Potential", development_potential, "Antrenman potansiyeli" if is_tr else "Training potential", THEME["turquoise2"]), unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown(f"""
    <div class="ai-box">
      <div style="font-size:18px;font-weight:900;margin-bottom:8px;">{'🤖 AI Coach Karar Destek Yorumu' if is_tr else '🤖 AI Coach Decision Support Comment'}</div>
      <div style="font-size:15px;line-height:1.55;">{coach_comment}</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    s_col, n_col = st.columns(2)
    with s_col:
        st.markdown(f"<div class='card'><div class='section-title'>{'✅ Güçlü Alanlar' if is_tr else '✅ Strengths'}</div>", unsafe_allow_html=True)
        for item in strengths:
            st.markdown(f"- {item}")
        st.markdown("</div>", unsafe_allow_html=True)
    with n_col:
        st.markdown(f"<div class='card'><div class='section-title'>{'🔧 Geliştirilebilir Alanlar' if is_tr else '🔧 Development Areas'}</div>", unsafe_allow_html=True)
        for item in needs:
            st.markdown(f"- {item}")
        st.markdown("</div>", unsafe_allow_html=True)

    with st.expander("▶ Model Doğruluk Bilgileri" if is_tr else "▶ Model Accuracy Information", expanded=False):
        m1, m2, m3 = st.columns(3)
        m1.metric("Model", str(model_info.get("model", "-")))
        m2.metric("R²", str(model_info.get("r2", "-")))
        m3.metric("Pearson r", str(model_info.get("pearson_r", "-")))
        m4, m5, m6 = st.columns(3)
        m4.metric("MAE", str(model_info.get("mae", "-")))
        m5.metric("RMSE", str(model_info.get("rmse", "-")))
        m6.metric("MAPE", str(model_info.get("mape", "-")))
        st.write(("Toplam veri:" if is_tr else "Total sample:"), model_info.get("n_total", "-"))
        st.write(("Eğitim verisi:" if is_tr else "Training sample:"), model_info.get("n_train", "-"))
        st.write(("Bağımsız test verisi:" if is_tr else "Independent test sample:"), model_info.get("n_test", "-"))

    report_id = f"SWIMML-{datetime.now().strftime('%Y%m%d')}-{str(uuid.uuid4())[:8].upper()}"
    report_title = "SwimML Pro v1.0 Performans Raporu" if is_tr else "SwimML Pro v1.0 Performance Report"
    report_subtitle = hero_subtitle

    report = {
        "title": report_title,
        "subtitle": report_subtitle,
        "labels": {
            "athlete": "Sporcu Bilgileri" if is_tr else "Athlete Information",
            "model": "Model Bilgileri" if is_tr else "Model Information",
            "result": "Tahmin Sonucu" if is_tr else "Prediction Result",
            "coach": "AI Coach Karar Destek Yorumu" if is_tr else "AI Coach Decision Support Comment",
            "strengths": "Güçlü Alanlar" if is_tr else "Strengths",
            "needs": "Geliştirilebilir Alanlar" if is_tr else "Development Areas",
            "copyright": "Telif ve Kullanım Koşulları" if is_tr else "Copyright and Terms of Use",
        },
        "athlete_rows": [
            ("Rapor No" if is_tr else "Report ID", report_id),
            ("Ad Soyad" if is_tr else "Full Name", athlete_name),
            ("Kulüp" if is_tr else "Club", club_name),
            ("Antrenör Ad Soyad" if is_tr else "Coach Full Name", coach_name),
            ("Rapor Tarihi" if is_tr else "Report Date", datetime.now().strftime("%d.%m.%Y %H:%M")),
        ],
        "model_rows": [
            ("Analiz Grubu" if is_tr else "Analysis Group", selected_group_text),
            ("Model Anahtarı" if is_tr else "Model Key", model_key),
            ("Model" if is_tr else "Model", str(model_info.get("model", "-"))),
            ("Değişken Sayısı" if is_tr else "Number of Variables", str(len(features))),
            ("MAE", str(model_info.get("mae", "-"))),
            ("RMSE", str(model_info.get("rmse", "-"))),
            ("R2", str(model_info.get("r2", "-"))),
        ],
        "result_rows": [
            ("Tahmini Derece" if is_tr else "Predicted Time", f"{prediction:.2f} sn" if is_tr else f"{prediction:.2f} s"),
            ("Grup Ortalaması" if is_tr else "Group Mean", f"{group_mean:.2f} sn" if is_tr else f"{group_mean:.2f} s"),
            ("Fark" if is_tr else "Difference", f"{difference:+.2f} sn" if is_tr else f"{difference:+.2f} s"),
            ("Performans Yüzdeliği" if is_tr else "Performance Percentile", f"%{percentile:.1f}"),
            ("Tahmin Güveni" if is_tr else "Prediction Confidence", f"%{confidence:.1f}"),
            ("Fiziksel Profil" if is_tr else "Physical Profile", f"%{physical_profile:.1f}"),
            ("Gelişim Potansiyeli" if is_tr else "Development Potential", f"%{development_potential:.1f}"),
            ("Hata Aralığı" if is_tr else "Error Margin", f"±{error_margin:.2f} sn" if is_tr else f"±{error_margin:.2f} s"),
            ("Performans Yorumu" if is_tr else "Performance Comment", comment),
        ],
        "coach_comment": coach_comment,
        "strengths": strengths,
        "needs": needs,
        "copyright_text": (
            f"© 2026 {DEVELOPER_TR}. Tüm hakları saklıdır. Bu yazılım eğitim, bilimsel araştırma ve antrenör karar destek amacıyla geliştirilmiştir. "
            "Yazılımın kaynak kodlarının, algoritmalarının, yapay zekâ modellerinin, kullanıcı arayüzünün, veri yapısının ve rapor çıktılarının yazılı izin alınmaksızın çoğaltılması, kopyalanması, değiştirilmesi, tersine mühendislik uygulanması, yeniden dağıtılması, ticari amaçla kullanılması veya başka sistemlere entegre edilmesi yasaktır. "
            "Tahmin sonuçları tek başına kesin karar aracı değildir; uzman antrenör değerlendirmesiyle birlikte yorumlanmalıdır."
            if is_tr else
            f"© 2026 {DEVELOPER_EN}. All rights reserved. This software was developed for education, scientific research and coaching decision support. "
            "Unauthorized reproduction, copying, modification, reverse engineering, redistribution, commercial use, integration into other systems, or reuse of its source code, algorithms, AI models, user interface, data structure and report outputs is prohibited without written permission. "
            "Prediction results are not a standalone final decision tool and must be interpreted together with expert coaching evaluation."
        ),
    }

    pdf = create_pdf_report(report)
    st.download_button(
        label="📄 PDF Raporu İndir" if is_tr else "📄 Download PDF Report",
        data=pdf,
        file_name=f"swimml_pro_v1_{model_key}_{report_id}.pdf",
        mime="application/pdf",
    )

# Footer
if is_tr:
    footer_html = f"""
    <div class="footer-box">
      <h3>ℹ️ SwimML Pro {APP_VERSION} Hakkında</h3>
      <p><b>50 m Yüzme Performans Tahmin ve Antrenör Karar Destek Sistemi</b></p>
      <p><b>Geliştirici:</b> {DEVELOPER_TR}<br><b>Kurum:</b> {UNIVERSITY_TR}</p>
      <p>Bu yazılım eğitim, bilimsel araştırma ve antrenör karar destek amacıyla geliştirilmiştir. Tahmin sonuçları uzman antrenör değerlendirmesiyle birlikte yorumlanmalıdır.</p>
      <p><b>Telif Hakkı:</b> © 2026 {DEVELOPER_TR}. Tüm hakları saklıdır.</p>
      <p>Bu yazılımın kaynak kodlarının, algoritmalarının, yapay zekâ modellerinin, kullanıcı arayüzünün, grafiklerinin, veri yapısının ve rapor çıktılarının yazılı izin alınmaksızın çoğaltılması, kopyalanması, değiştirilmesi, tersine mühendislik uygulanması, yeniden dağıtılması, ticari amaçla kullanılması veya başka sistemlere entegre edilmesi yasaktır.</p>
    </div>
    """
else:
    footer_html = f"""
    <div class="footer-box">
      <h3>ℹ️ About SwimML Pro {APP_VERSION}</h3>
      <p><b>50 m Prediction Coach Decision Support System</b></p>
      <p><b>Developer:</b> {DEVELOPER_EN}<br><b>Institution:</b> {UNIVERSITY_EN}</p>
      <p>This software was developed for education, scientific research and coaching decision support. Prediction results must be interpreted together with expert coaching evaluation.</p>
      <p><b>Copyright:</b> © 2026 {DEVELOPER_EN}. All rights reserved.</p>
      <p>Unauthorized reproduction, copying, modification, reverse engineering, redistribution, commercial use, integration into other systems, or reuse of its source code, algorithms, AI models, user interface, graphics, data structure and report outputs is prohibited without written permission.</p>
    </div>
    """

st.markdown(footer_html, unsafe_allow_html=True)
