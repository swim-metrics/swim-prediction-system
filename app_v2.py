# ============================================================
# SwimML Predictor v2.0
# Young Swimmer Performance Prediction & Decision Support System
# Developed for: Tuğrul Özkadı & Research Team
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

# Optional visualization
import matplotlib.pyplot as plt

# ============================================================
# 1. FILE PATHS
# ============================================================

DATA_FILE = "swim_data.xlsx"
METADATA_FILE = "model_metadata.json"
APP_VERSION = "SwimML Predictor v2.0"

# ============================================================
# 2. STROKE / STYLE DEFINITIONS
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

SEX_MAP_TR = {"Kadın": "female", "Erkek": "male"}
SEX_MAP_EN = {"Female": "female", "Male": "male"}

AGE_GROUPS = ["12_13", "14_15", "16_17"]

# ============================================================
# 3. FEATURE LABELS
# ============================================================

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

# ============================================================
# 4. FEATURE GROUPS FOR COACHING INTERPRETATION
# ============================================================

FEATURE_GROUPS = {
    "Antropometrik Profil": [
        "body_height", "body_mass", "arm_span", "arm_length", "leg_length",
        "sitting_height", "hand_length", "foot_length", "chest_girth",
        "waist_girth", "hip_girth", "thigh_girth", "calf_girth",
        "biacromial_breadth", "biiliac_breadth"
    ],
    "Kuvvet ve Patlayıcı Güç": [
        "vertical_jump", "standing_long_jump", "handgrip_strength",
        "flexed_arm_hang", "sit_ups_1min"
    ],
    "Sürat ve Çeviklik": [
        "sprint_30m", "illinois_agility_test"
    ],
    "Esneklik ve Mobilite": [
        "sit_and_reach", "trunk_extension_height", "shoulder_extension_height",
        "shoulder_mobility", "ankle_dorsiflexion_rom", "ankle_plantarflexion_rom"
    ],
    "Vücut Kompozisyonu": [
        "body_density", "body_fat_percentage", "fat_mass", "fat_free_mass"
    ],
    "Denge": [
        "flamingo_balance_test"
    ]
}

LOWER_IS_BETTER = {
    "sprint_30m", "illinois_agility_test", "body_fat_percentage",
    "fat_mass", "flamingo_balance_test"
}

# ============================================================
# 5. DEFAULT VALUES
# ============================================================

def get_default_value(feature: str):
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

# ============================================================
# 6. DATA AND MODEL LOADING
# ============================================================

@st.cache_data
def load_data():
    data_path = Path(DATA_FILE)
    metadata_path = Path(METADATA_FILE)

    if not data_path.exists():
        st.error(f"Veri dosyası bulunamadı: {DATA_FILE}")
        st.stop()

    if not metadata_path.exists():
        st.error(f"Model metadata dosyası bulunamadı: {METADATA_FILE}")
        st.stop()

    df = pd.read_excel(data_path)

    with open(metadata_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    return df, metadata

@st.cache_resource
def load_prediction_model(model_file):
    model_path = Path(str(model_file).replace("\\", "/"))

    if not model_path.exists():
        st.error(f"Model dosyası bulunamadı: {model_path}")
        st.stop()

    return joblib.load(model_path)

# ============================================================
# 7. HELPER FUNCTIONS
# ============================================================

def safe_float(value, default=np.nan):
    try:
        return float(value)
    except Exception:
        return default


def get_metric(model_info, key, default=None):
    return model_info.get(key, default)


def calculate_percentile(group_data, target_col, prediction):
    """
    Swimming time: lower is better.
    Percentile indicates the percentage of swimmers slower than predicted athlete.
    """
    if group_data.empty or target_col not in group_data.columns:
        return np.nan
    return (group_data[target_col] > prediction).mean() * 100


def performance_comment_tr(percentile):
    if pd.isna(percentile):
        return "—", "Yüzdelik hesaplanamadı"
    if percentile >= 85:
        return "★★★★★", "Çok yüksek performans düzeyi"
    if percentile >= 65:
        return "★★★★☆", "İyi performans düzeyi"
    if percentile >= 40:
        return "★★★☆☆", "Orta performans düzeyi"
    return "★★☆☆☆", "Geliştirilmesi gereken performans düzeyi"


def performance_comment_en(percentile):
    if pd.isna(percentile):
        return "—", "Percentile could not be calculated"
    if percentile >= 85:
        return "★★★★★", "Very high performance level"
    if percentile >= 65:
        return "★★★★☆", "Good performance level"
    if percentile >= 40:
        return "★★★☆☆", "Moderate performance level"
    return "★★☆☆☆", "Performance needs improvement"


def error_band_from_metadata(model_info):
    """
    Uses MAE as practical expected error band if available.
    """
    mae = safe_float(model_info.get("mae", np.nan))
    rmse = safe_float(model_info.get("rmse", np.nan))

    if not np.isnan(mae):
        return mae, "MAE"
    if not np.isnan(rmse):
        return rmse, "RMSE"
    return np.nan, ""


def group_reference_stats(df, group_filter, features):
    group_df = df[group_filter].copy()
    stats = {}
    for f in features:
        if f in group_df.columns:
            vals = pd.to_numeric(group_df[f], errors="coerce").dropna()
            if len(vals) > 1:
                stats[f] = {
                    "mean": vals.mean(),
                    "std": vals.std(ddof=1),
                    "min": vals.min(),
                    "max": vals.max(),
                }
    return stats


def z_score(value, mean, std):
    if pd.isna(value) or pd.isna(mean) or pd.isna(std) or std == 0:
        return np.nan
    return (value - mean) / std


def feature_profile(inputs, ref_stats):
    rows = []
    for f, value in inputs.items():
        if f not in ref_stats:
            continue
        mean = ref_stats[f]["mean"]
        std = ref_stats[f]["std"]
        z = z_score(value, mean, std)
        adjusted_z = -z if f in LOWER_IS_BETTER else z
        rows.append({
            "feature": f,
            "value": value,
            "group_mean": mean,
            "z": z,
            "adjusted_z": adjusted_z,
        })
    return pd.DataFrame(rows)


def classify_feature_tr(adjusted_z):
    if pd.isna(adjusted_z):
        return "Yorumlanamadı"
    if adjusted_z >= 0.75:
        return "Güçlü yön"
    if adjusted_z <= -0.75:
        return "Gelişim alanı"
    return "Grup ortalamasına yakın"


def classify_feature_en(adjusted_z):
    if pd.isna(adjusted_z):
        return "Not interpreted"
    if adjusted_z >= 0.75:
        return "Strength"
    if adjusted_z <= -0.75:
        return "Development area"
    return "Close to group mean"


def group_scores(profile_df):
    scores = {}
    for group_name, group_features in FEATURE_GROUPS.items():
        sub = profile_df[profile_df["feature"].isin(group_features)]
        if not sub.empty:
            scores[group_name] = sub["adjusted_z"].mean()
    return scores


def coaching_summary_tr(percentile, difference, error_band, strongest, weakest):
    lines = []

    if not pd.isna(percentile):
        if percentile >= 85:
            lines.append("Sporcu, kendi yaş grubu ve cinsiyet grubuna göre üst düzey performans potansiyeli göstermektedir.")
        elif percentile >= 65:
            lines.append("Sporcu, grup ortalamasının üzerinde değerlendirilebilecek olumlu bir performans profili göstermektedir.")
        elif percentile >= 40:
            lines.append("Sporcu, grup ortalamasına yakın bir performans profili göstermektedir; gelişim için teknik ve fiziksel destek önerilir.")
        else:
            lines.append("Sporcunun performans göstergeleri geliştirmeye açıktır; antrenman planlamasında temel fiziksel ve teknik bileşenler birlikte ele alınmalıdır.")

    if not pd.isna(difference):
        if difference < 0:
            lines.append(f"Tahmini derece grup ortalamasından yaklaşık {abs(difference):.2f} sn daha hızlıdır.")
        else:
            lines.append(f"Tahmini derece grup ortalamasından yaklaşık {difference:.2f} sn daha yavaştır.")

    if not pd.isna(error_band):
        lines.append(f"Tahmin sonucu yaklaşık ±{error_band:.2f} sn model hata payı ile yorumlanmalıdır.")

    if strongest:
        lines.append(f"Öne çıkan güçlü alan: {strongest}.")
    if weakest:
        lines.append(f"Öncelikli gelişim alanı: {weakest}.")

    lines.append("Bu çıktı tek başına seçme veya kulvar tahsisi kararı değildir; antrenör gözlemi, teknik analiz ve yarış performansı ile birlikte değerlendirilmelidir.")
    return lines


def coaching_summary_en(percentile, difference, error_band, strongest, weakest):
    lines = []

    if not pd.isna(percentile):
        if percentile >= 85:
            lines.append("The swimmer demonstrates a high-level performance potential within the age-sex group.")
        elif percentile >= 65:
            lines.append("The swimmer shows a positive performance profile above the group average.")
        elif percentile >= 40:
            lines.append("The swimmer is close to the group average; technical and physical development support is recommended.")
        else:
            lines.append("The swimmer's performance indicators are open to development; technical and physical components should be addressed together.")

    if not pd.isna(difference):
        if difference < 0:
            lines.append(f"The predicted time is approximately {abs(difference):.2f} s faster than the group mean.")
        else:
            lines.append(f"The predicted time is approximately {difference:.2f} s slower than the group mean.")

    if not pd.isna(error_band):
        lines.append(f"The prediction should be interpreted with an approximate ±{error_band:.2f} s model error band.")

    if strongest:
        lines.append(f"Main strength area: {strongest}.")
    if weakest:
        lines.append(f"Priority development area: {weakest}.")

    lines.append("This output is not a stand-alone selection or lane allocation decision; it should be evaluated together with coach observation, technical analysis and race performance.")
    return lines

# ============================================================
# 8. VISUALIZATION
# ============================================================

def create_radar_chart(group_score_dict):
    if not group_score_dict:
        return None

    labels = list(group_score_dict.keys())
    values = [group_score_dict[k] for k in labels]

    # Convert z-score style values to 0-100 practical display scale
    display_values = []
    for v in values:
        if pd.isna(v):
            display_values.append(50)
        else:
            display_values.append(float(np.clip(50 + v * 15, 0, 100)))

    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
    display_values += display_values[:1]
    angles += angles[:1]

    fig = plt.figure(figsize=(7, 7))
    ax = plt.subplot(111, polar=True)
    ax.plot(angles, display_values, linewidth=2)
    ax.fill(angles, display_values, alpha=0.20)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_yticks([20, 40, 60, 80, 100])
    ax.set_ylim(0, 100)
    ax.set_title("Sporcu Performans Profili", fontsize=13, pad=20)
    fig.tight_layout()

    buffer = BytesIO()
    fig.savefig(buffer, format="png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    buffer.seek(0)
    return buffer

# ============================================================
# 9. PDF REPORT
# ============================================================

def pdf_safe(text):
    """
    ReportLab default Helvetica does not fully support Turkish.
    This keeps the PDF stable by converting Turkish characters.
    """
    replacements = {
        "ç": "c", "Ç": "C", "ğ": "g", "Ğ": "G",
        "ı": "i", "İ": "I", "ö": "o", "Ö": "O",
        "ş": "s", "Ş": "S", "ü": "u", "Ü": "U",
        "²": "2", "±": "+/-",
    }
    text = str(text)
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text


def create_pdf_report(title, sections, report_id):
    buffer = BytesIO()
    c = canvas.Canvas(buffer, pagesize=A4)
    width, height = A4

    x = 2 * cm
    y = height - 2 * cm

    def new_page():
        nonlocal y
        c.showPage()
        y = height - 2 * cm
        c.setFont("Helvetica", 9)

    c.setFont("Helvetica-Bold", 15)
    c.drawString(x, y, pdf_safe(title))
    y -= 0.7 * cm

    c.setFont("Helvetica", 9)
    c.drawString(x, y, pdf_safe(f"Rapor No / Report ID: {report_id}"))
    y -= 0.5 * cm
    c.drawString(x, y, pdf_safe(f"Olusturma Tarihi: {datetime.now().strftime('%d.%m.%Y %H:%M')}"))
    y -= 0.8 * cm

    for section_title, lines in sections:
        if y < 3 * cm:
            new_page()

        c.setFont("Helvetica-Bold", 11)
        c.drawString(x, y, pdf_safe(section_title))
        y -= 0.45 * cm

        c.setFont("Helvetica", 9)
        for line in lines:
            if y < 2 * cm:
                new_page()
                c.setFont("Helvetica", 9)

            text = pdf_safe(line)
            max_chars = 105
            chunks = [text[i:i + max_chars] for i in range(0, len(text), max_chars)] or [""]
            for chunk in chunks:
                c.drawString(x, y, chunk)
                y -= 0.38 * cm
        y -= 0.25 * cm

    c.setFont("Helvetica-Oblique", 8)
    footer = f"{APP_VERSION} | Bilimsel arastirma ve antrenorluk karar destek amaciyla gelistirilmistir."
    c.drawString(x, 1.2 * cm, pdf_safe(footer))

    c.save()
    buffer.seek(0)
    return buffer

# ============================================================
# 10. STREAMLIT APP CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="SwimML Predictor v2.0",
    page_icon="🏊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# 11. MAIN APP
# ============================================================

df, metadata = load_data()

with st.sidebar:
    st.title("🏊 SwimML")
    language = st.radio("Dil / Language", ["🇹🇷 Türkçe", "🇺🇸 English"], index=0)
    is_tr = language.startswith("🇹🇷")

    st.markdown("---")

    if is_tr:
        st.subheader("Model Seçimi")
        style_display = st.selectbox("Stil", list(STYLE_TR.values()))
        style = {v: k for k, v in STYLE_TR.items()}[style_display]
        sex_display = st.selectbox("Cinsiyet", list(SEX_MAP_TR.keys()))
        sex = SEX_MAP_TR[sex_display]
        age_group = st.selectbox("Yaş Grubu", AGE_GROUPS)
    else:
        st.subheader("Model Selection")
        style_display = st.selectbox("Stroke", list(STYLE_EN.values()))
        style = {v: k for k, v in STYLE_EN.items()}[style_display]
        sex_display = st.selectbox("Sex", list(SEX_MAP_EN.keys()))
        sex = SEX_MAP_EN[sex_display]
        age_group = st.selectbox("Age Group", AGE_GROUPS)

    model_key = f"{style}_{age_group}_{sex}"
    st.caption(("Model anahtarı: " if is_tr else "Model key: ") + model_key)

# Header
if is_tr:
    st.title("🏊 SwimML Predictor v2.0")
    st.markdown("### Genç Yüzücülerde Stil, Yaş ve Cinsiyete Özgü Performans Tahmin ve Karar Destek Sistemi")
else:
    st.title("🏊 SwimML Predictor v2.0")
    st.markdown("### Stroke-, Age- and Sex-Specific Performance Prediction and Decision Support System for Young Swimmers")

if model_key not in metadata:
    st.error("Bu seçim için kayıtlı model bulunamadı." if is_tr else "No registered model was found for this selection.")
    st.stop()

model_info = metadata[model_key]
model = load_prediction_model(model_info["model_file"])
features = model_info.get("features_en", [])
target_col = model_info.get("target")

if not features:
    st.error("Metadata içinde modele ait değişken listesi bulunamadı." if is_tr else "Feature list could not be found in metadata.")
    st.stop()

# ============================================================
# 12. ATHLETE FORM
# ============================================================

st.markdown("---")

if is_tr:
    st.subheader("👤 Sporcu Bilgileri")
else:
    st.subheader("👤 Athlete Information")

col_a, col_b, col_c = st.columns(3)
with col_a:
    athlete_name = st.text_input("Ad Soyad" if is_tr else "Athlete Name", value="")
with col_b:
    club_name = st.text_input("Kulüp" if is_tr else "Club", value="")
with col_c:
    coach_name = st.text_input("Antrenör" if is_tr else "Coach", value="")

st.markdown("---")

if is_tr:
    st.subheader("📌 Modele Girilecek Ölçümler")
    st.info(f"Bu modelde kullanılan değişken sayısı: {len(features)}")
else:
    st.subheader("📌 Required Measurements")
    st.info(f"Number of variables used in this model: {len(features)}")

inputs = {}

# Dynamic input layout
cols = st.columns(3)
for i, feature in enumerate(features):
    label = FEATURE_LABELS_TR.get(feature, feature) if is_tr else FEATURE_LABELS_EN.get(feature, feature)
    default = get_default_value(feature)

    with cols[i % 3]:
        if feature in ["age", "sit_ups_1min"]:
            inputs[feature] = st.number_input(label, min_value=0, max_value=300, value=int(default), step=1)
        else:
            inputs[feature] = st.number_input(label, min_value=0.0, max_value=500.0, value=float(default), step=0.1)

# ============================================================
# 13. PREDICTION
# ============================================================

st.markdown("---")
button_text = "🏁 Tahmin Et ve Raporla" if is_tr else "🏁 Predict and Report"

if st.button(button_text, use_container_width=True):
    report_id = f"SWIM-{datetime.now().strftime('%Y%m%d')}-{str(uuid.uuid4())[:8].upper()}"

    input_df = pd.DataFrame([inputs])
    input_df = input_df[features]

    prediction = float(model.predict(input_df)[0])

    group_filter = (df["sex"] == sex) & (df["age_group"] == age_group)
    group_data = df[group_filter].copy()

    if target_col in group_data.columns and not group_data.empty:
        group_mean = pd.to_numeric(group_data[target_col], errors="coerce").mean()
        difference = prediction - group_mean
        percentile = calculate_percentile(group_data, target_col, prediction)
    else:
        group_mean = np.nan
        difference = np.nan
        percentile = np.nan

    error_band, error_source = error_band_from_metadata(model_info)

    if is_tr:
        stars, comment = performance_comment_tr(percentile)
    else:
        stars, comment = performance_comment_en(percentile)

    # Reference-based athlete profile
    ref_stats = group_reference_stats(df, group_filter, features)
    profile_df = feature_profile(inputs, ref_stats)
    profile_df["interpretation"] = profile_df["adjusted_z"].apply(classify_feature_tr if is_tr else classify_feature_en)

    score_dict = group_scores(profile_df)
    strongest = None
    weakest = None
    if score_dict:
        strongest = max(score_dict, key=score_dict.get)
        weakest = min(score_dict, key=score_dict.get)

    # ========================================================
    # 14. RESULT CARDS
    # ========================================================

    st.success(
        f"🏁 Tahmini {style_display} Derecesi: {prediction:.2f} sn"
        if is_tr else
        f"🏁 Predicted {style_display} Time: {prediction:.2f} s"
    )

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Tahmin" if is_tr else "Prediction", f"{prediction:.2f} sn")
    col2.metric("Grup Ortalaması" if is_tr else "Group Mean", "—" if pd.isna(group_mean) else f"{group_mean:.2f} sn")
    col3.metric("Fark" if is_tr else "Difference", "—" if pd.isna(difference) else f"{difference:.2f} sn")
    col4.metric("Yüzdelik" if is_tr else "Percentile", "—" if pd.isna(percentile) else f"%{percentile:.1f}")

    st.subheader("📈 Performans Yorumu" if is_tr else "📈 Performance Interpretation")
    st.write(f"**{stars} {comment}**")

    if not pd.isna(error_band):
        st.warning(
            f"Bu tahmin yaklaşık ±{error_band:.2f} sn hata payı ile yorumlanmalıdır. Hata kaynağı: {error_source}."
            if is_tr else
            f"This prediction should be interpreted with an approximate ±{error_band:.2f} s error band. Error source: {error_source}."
        )

    # ========================================================
    # 15. COACHING DECISION SUPPORT
    # ========================================================

    st.subheader("🧠 Antrenör Karar Destek Yorumu" if is_tr else "🧠 Coaching Decision Support")

    summary_lines = coaching_summary_tr(percentile, difference, error_band, strongest, weakest) if is_tr else coaching_summary_en(percentile, difference, error_band, strongest, weakest)
    for line in summary_lines:
        st.write(f"- {line}")

    # ========================================================
    # 16. PROFILE TABLE
    # ========================================================

    if not profile_df.empty:
        st.subheader("📊 Sporcu Profil Analizi" if is_tr else "📊 Athlete Profile Analysis")

        display_df = profile_df.copy()
        display_df["Ölçüm" if is_tr else "Feature"] = display_df["feature"].map(FEATURE_LABELS_TR if is_tr else FEATURE_LABELS_EN)
        display_df["Girilen Değer" if is_tr else "Input Value"] = display_df["value"].round(2)
        display_df["Grup Ortalaması" if is_tr else "Group Mean"] = display_df["group_mean"].round(2)
        display_df["Profil Skoru" if is_tr else "Profile Score"] = display_df["adjusted_z"].round(2)
        display_df["Yorum" if is_tr else "Interpretation"] = display_df["interpretation"]

        keep_cols = [
            "Ölçüm" if is_tr else "Feature",
            "Girilen Değer" if is_tr else "Input Value",
            "Grup Ortalaması" if is_tr else "Group Mean",
            "Profil Skoru" if is_tr else "Profile Score",
            "Yorum" if is_tr else "Interpretation",
        ]

        st.dataframe(display_df[keep_cols], use_container_width=True)

        strengths = display_df[display_df["interpretation"].isin(["Güçlü yön", "Strength"])]
        developments = display_df[display_df["interpretation"].isin(["Gelişim alanı", "Development area"])]

        c1, c2 = st.columns(2)
        with c1:
            st.markdown("#### ✅ Güçlü Yönler" if is_tr else "#### ✅ Strengths")
            if strengths.empty:
                st.write("Belirgin güçlü yön saptanmadı." if is_tr else "No clear strength was identified.")
            else:
                for _, row in strengths.head(5).iterrows():
                    st.write(f"- {row['Ölçüm' if is_tr else 'Feature']}")

        with c2:
            st.markdown("#### 🔧 Gelişim Alanları" if is_tr else "#### 🔧 Development Areas")
            if developments.empty:
                st.write("Belirgin gelişim alanı saptanmadı." if is_tr else "No clear development area was identified.")
            else:
                for _, row in developments.head(5).iterrows():
                    st.write(f"- {row['Ölçüm' if is_tr else 'Feature']}")

    # ========================================================
    # 17. RADAR CHART
    # ========================================================

    if score_dict:
        st.subheader("🕸️ Performans Profil Grafiği" if is_tr else "🕸️ Performance Profile Chart")
        radar_buffer = create_radar_chart(score_dict)
        if radar_buffer:
            st.image(radar_buffer, use_container_width=False)

    # ========================================================
    # 18. MODEL INFORMATION
    # ========================================================

    with st.expander("🧪 Model Doğruluk Bilgileri" if is_tr else "🧪 Model Accuracy Information", expanded=True):
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Model", str(model_info.get("model", "—")))
        m2.metric("n", str(model_info.get("n_total", "—")))
        m3.metric("MAE", str(model_info.get("mae", "—")))
        m4.metric("R²", str(model_info.get("r2", "—")))

        st.write(f"**RMSE:** {model_info.get('rmse', '—')}")
        st.write(f"**MAPE:** {model_info.get('mape', '—')}")
        st.write(f"**Pearson r:** {model_info.get('pearson_r', '—')}")
        st.write(f"**CV R²:** {model_info.get('cv_r2', '—')} ± {model_info.get('cv_std', '—')}")
        st.write(f"**Minimum hata / Minimum error:** {model_info.get('min_error', '—')}")
        st.write(f"**Maksimum hata / Maximum error:** {model_info.get('max_error', '—')}")

    # ========================================================
    # 19. PDF REPORT
    # ========================================================

    if is_tr:
        report_sections = [
            ("SPORCU BILGILERI", [
                f"Ad Soyad: {athlete_name}",
                f"Kulup: {club_name}",
                f"Antrenor: {coach_name}",
                f"Stil: {style_display}",
                f"Cinsiyet: {sex_display}",
                f"Yas Grubu: {age_group}",
            ]),
            ("TAHMIN SONUCU", [
                f"Tahmini Derece: {prediction:.2f} sn",
                f"Grup Ortalamasi: {'-' if pd.isna(group_mean) else f'{group_mean:.2f} sn'}",
                f"Fark: {'-' if pd.isna(difference) else f'{difference:.2f} sn'}",
                f"Performans Yuzdeligi: {'-' if pd.isna(percentile) else f'%{percentile:.1f}'}",
                f"Performans Yorumu: {comment}",
                f"Tahmin Hata Payi: {'-' if pd.isna(error_band) else f'+/- {error_band:.2f} sn'}",
            ]),
            ("ANTRENOR KARAR DESTEK YORUMU", summary_lines),
            ("MODEL BILGILERI", [
                f"Model: {model_info.get('model', '-')}",
                f"Toplam Veri: {model_info.get('n_total', '-')}",
                f"Egitim Verisi: {model_info.get('n_train', '-')}",
                f"Bagimsiz Test Verisi: {model_info.get('n_test', '-')}",
                f"MAE: {model_info.get('mae', '-')}",
                f"RMSE: {model_info.get('rmse', '-')}",
                f"MAPE: {model_info.get('mape', '-')}",
                f"R2: {model_info.get('r2', '-')}",
                f"Pearson r: {model_info.get('pearson_r', '-')}",
                f"CV R2: {model_info.get('cv_r2', '-')} +/- {model_info.get('cv_std', '-')}",
            ]),
            ("GIRILEN OLCUMLER", [
                f"{FEATURE_LABELS_TR.get(k, k)}: {v}" for k, v in inputs.items()
            ]),
        ]
        pdf_title = f"{style_display} Tahmin ve Karar Destek Raporu"
        pdf_file_name = f"{style}_{age_group}_{sex}_{report_id}_swimml_v2_rapor.pdf"
        download_label = "📄 PDF Raporu İndir"
    else:
        report_sections = [
            ("ATHLETE INFORMATION", [
                f"Athlete Name: {athlete_name}",
                f"Club: {club_name}",
                f"Coach: {coach_name}",
                f"Stroke: {style_display}",
                f"Sex: {sex_display}",
                f"Age Group: {age_group}",
            ]),
            ("PREDICTION RESULT", [
                f"Predicted Time: {prediction:.2f} s",
                f"Group Mean: {'-' if pd.isna(group_mean) else f'{group_mean:.2f} s'}",
                f"Difference: {'-' if pd.isna(difference) else f'{difference:.2f} s'}",
                f"Performance Percentile: {'-' if pd.isna(percentile) else f'%{percentile:.1f}'}",
                f"Performance Interpretation: {comment}",
                f"Prediction Error Band: {'-' if pd.isna(error_band) else f'+/- {error_band:.2f} s'}",
            ]),
            ("COACHING DECISION SUPPORT", summary_lines),
            ("MODEL INFORMATION", [
                f"Model: {model_info.get('model', '-')}",
                f"Total Sample: {model_info.get('n_total', '-')}",
                f"Training Sample: {model_info.get('n_train', '-')}",
                f"Independent Test Sample: {model_info.get('n_test', '-')}",
                f"MAE: {model_info.get('mae', '-')}",
                f"RMSE: {model_info.get('rmse', '-')}",
                f"MAPE: {model_info.get('mape', '-')}",
                f"R2: {model_info.get('r2', '-')}",
                f"Pearson r: {model_info.get('pearson_r', '-')}",
                f"CV R2: {model_info.get('cv_r2', '-')} +/- {model_info.get('cv_std', '-')}",
            ]),
            ("ENTERED MEASUREMENTS", [
                f"{FEATURE_LABELS_EN.get(k, k)}: {v}" for k, v in inputs.items()
            ]),
        ]
        pdf_title = f"{style_display} Prediction and Decision Support Report"
        pdf_file_name = f"{style}_{age_group}_{sex}_{report_id}_swimml_v2_report.pdf"
        download_label = "📄 Download PDF Report"

    pdf = create_pdf_report(pdf_title, report_sections, report_id)
    st.download_button(
        label=download_label,
        data=pdf,
        file_name=pdf_file_name,
        mime="application/pdf",
        use_container_width=True,
    )

# ============================================================
# 20. FOOTER
# ============================================================

st.markdown("---")

if is_tr:
    st.markdown(
        f"""
        ### ℹ️ {APP_VERSION} Hakkında

        **Geliştiren:** Tuğrul Özkadı & Araştırma Ekibi  
        **Kurum:** Hitit Üniversitesi Spor Bilimleri Fakültesi  
        **Amaç:** Bilimsel araştırma, eğitim ve antrenörlük karar destek süreci.

        Bu uygulama, genç yüzücülerde 50 m performans tahmini yapmak ve antrenörlük kararlarını desteklemek amacıyla geliştirilmiştir. Tahmin sonuçları tek başına kesin karar aracı değildir; teknik gözlem, yarış performansı ve uzman antrenör değerlendirmesiyle birlikte yorumlanmalıdır.

        © 2026 Tuğrul Özkadı & Araştırma Ekibi. Tüm hakları saklıdır.
        """
    )
else:
    st.markdown(
        f"""
        ### ℹ️ About {APP_VERSION}

        **Developed by:** Tuğrul Özkadı & Research Team  
        **Institution:** Hitit University Faculty of Sport Sciences  
        **Purpose:** Scientific research, education and coaching decision support.

        This application has been developed to predict 50 m swimming performance in young swimmers and to support coaching decisions. Prediction results are not stand-alone final decisions; they should be interpreted together with technical observation, race performance and expert coaching evaluation.

        © 2026 Tuğrul Özkadı & Research Team. All rights reserved.
        """
    )
