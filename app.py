import json
from io import BytesIO
from pathlib import Path
from datetime import datetime

import joblib
import pandas as pd
import streamlit as st
from reportlab.pdfgen import canvas
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfbase import pdfmetrics


DATA_FILE = "swim_data.xlsx"
METADATA_FILE = "model_metadata.json"

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

FEATURE_LABELS_EN = {
    k: k.replace("_", " ").title()
    for k in FEATURE_LABELS_TR.keys()
}


def setup_font():
    font_path = Path("C:/Windows/Fonts/arial.ttf")
    if font_path.exists():
        pdfmetrics.registerFont(TTFont("Arial", str(font_path)))
        return "Arial"
    return "Helvetica"


PDF_FONT = setup_font()


@st.cache_resource
def load_system():
    df = pd.read_excel(DATA_FILE)
    with open(METADATA_FILE, "r", encoding="utf-8") as f:
        metadata = json.load(f)
    return df, metadata


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


def performance_comment_tr(percentile):
    if percentile >= 85:
        return "★★★★★", "Çok yüksek performans düzeyi"
    elif percentile >= 65:
        return "★★★★☆", "İyi performans düzeyi"
    elif percentile >= 40:
        return "★★★☆☆", "Orta performans düzeyi"
    return "★★☆☆☆", "Geliştirilmesi gereken performans düzeyi"


def performance_comment_en(percentile):
    if percentile >= 85:
        return "★★★★★", "Very high performance level"
    elif percentile >= 65:
        return "★★★★☆", "Good performance level"
    elif percentile >= 40:
        return "★★★☆☆", "Moderate performance level"
    return "★★☆☆☆", "Performance needs improvement"


def create_pdf_report(title, report_lines):
    buffer = BytesIO()
    c = canvas.Canvas(buffer)
    c.setFont(PDF_FONT, 14)
    c.drawString(40, 800, title)

    c.setFont(PDF_FONT, 10)
    y = 770

    for line in report_lines:
        c.drawString(40, y, str(line))
        y -= 18
        if y < 50:
            c.showPage()
            c.setFont(PDF_FONT, 10)
            y = 800

    c.save()
    buffer.seek(0)
    return buffer


df, metadata = load_system()

st.set_page_config(
    page_title="Swimming Prediction System",
    layout="centered"
)

language = st.selectbox("Dil / Language", ["🇹🇷 Türkçe", "🇺🇸 English"])

is_tr = language.startswith("🇹🇷")

if is_tr:
    st.markdown("## 🇹🇷 Yüzme Performansı Tahmin Sistemi")
    style_display = st.selectbox(
        "Stil Seçiniz",
        list(STYLE_TR.values())
    )
    style = {v: k for k, v in STYLE_TR.items()}[style_display]

    page_title = f"{STYLE_TR[style]} Tahmin Modeli"
    st.markdown(f"### 🏊 {page_title}")

    sex_display = st.selectbox("Cinsiyet", ["Kadın", "Erkek"])
    sex = "female" if sex_display == "Kadın" else "male"

    age_group = st.selectbox("Yaş Grubu", ["12_13", "14_15", "16_17"])

else:
    st.markdown("## 🇺🇸 Swimming Performance Prediction System")
    style_display = st.selectbox(
        "Select Stroke",
        list(STYLE_EN.values())
    )
    style = {v: k for k, v in STYLE_EN.items()}[style_display]

    page_title = f"{STYLE_EN[style]} Prediction Model"
    st.markdown(f"### 🏊 {page_title}")

    sex_display = st.selectbox("Sex", ["Female", "Male"])
    sex = "female" if sex_display == "Female" else "male"

    age_group = st.selectbox("Age Group", ["12_13", "14_15", "16_17"])


model_key = f"{style}_{age_group}_{sex}"

if model_key not in metadata:
    st.error("Bu seçim için model bulunamadı.")
    st.stop()

meta = metadata[model_key]
model = joblib.load(meta["model_file"])

features = meta["features_en"]

st.markdown("---")

if is_tr:
    st.subheader("📌 Gerekli Ölçümler")
else:
    st.subheader("📌 Required Measurements")

inputs = {}

for feature in features:
    label = FEATURE_LABELS_TR.get(feature, feature) if is_tr else FEATURE_LABELS_EN.get(feature, feature)
    default_value = get_default_value(feature)

    if feature in ["age", "sit_ups_1min"]:
        inputs[feature] = st.number_input(
            label,
            min_value=0,
            max_value=300,
            value=int(default_value)
        )
    else:
        inputs[feature] = st.number_input(
            label,
            min_value=0.0,
            max_value=400.0,
            value=float(default_value)
        )

st.markdown("---")

if is_tr:
    st.info("📊 Bu model 4 sabit değişken ve aile kontrollü seçilmiş 6 belirleyici ile çalışmaktadır.")
else:
    st.info("📊 This model uses 4 core variables and 6 family-controlled selected predictors.")

button_text = "Tahmin Et" if is_tr else "Predict"

if st.button(button_text):

    input_df = pd.DataFrame([inputs])

    prediction = model.predict(input_df)[0]

    target_col = meta["target"]

    group_data = df[
        (df["style"] if "style" in df.columns else df["sex"]) == df["sex"]
    ] if False else df[
        (df["sex"] == sex) &
        (df["age_group"] == age_group)
    ]

    group_mean = group_data[target_col].mean()
    group_std = group_data[target_col].std()

    difference = prediction - group_mean
    percentile = (group_data[target_col] > prediction).mean() * 100

    if is_tr:
        stars, comment = performance_comment_tr(percentile)

        st.success(f"🏁 Tahmini {STYLE_TR[style]} Derecesi: {prediction:.2f} sn")
        st.metric("Grup Ortalaması", f"{group_mean:.2f} sn")
        st.metric("Fark", f"{difference:.2f} sn")
        st.metric("Performans Yüzdeliği", f"%{percentile:.1f}")

        st.subheader("📈 Performans Yorumu")
        st.write(f"**{stars} {comment}**")

        st.subheader("🧠 Model Bilgileri")
        st.write(f"Model: **{meta['model']}**")
        st.write(f"n toplam: **{meta['n_total']}**, eğitim: **{meta['n_train']}**, bağımsız test: **{meta['n_test']}**")
        st.write(f"MAE: **{meta['mae']} sn**")
        st.write(f"RMSE: **{meta['rmse']} sn**")
        st.write(f"MAPE: **%{meta['mape']}**")
        st.write(f"R²: **{meta['r2']}**")
        st.write(f"Pearson r: **{meta['pearson_r']}**")
        st.write(f"5-Kat CV R²: **{meta['cv_r2']} ± {meta['cv_std']}**")

        st.subheader("📋 Kullanılan Değişkenler")
        for i, f in enumerate(features, start=1):
            st.write(f"{i}. {FEATURE_LABELS_TR.get(f, f)}")

        report_lines = [
            f"Rapor tarihi: {datetime.now().strftime('%d.%m.%Y %H:%M')}",
            f"Model başlığı: {page_title}",
            f"Cinsiyet: {sex_display}",
            f"Yaş grubu: {age_group}",
            "",
            "Girilen ölçümler:",
        ]

        for f, v in inputs.items():
            report_lines.append(f"{FEATURE_LABELS_TR.get(f, f)}: {v}")

        report_lines += [
            "",
            f"Tahmini derece: {prediction:.2f} sn",
            f"Grup ortalaması: {group_mean:.2f} sn",
            f"Fark: {difference:.2f} sn",
            f"Performans yüzdeliği: %{percentile:.1f}",
            f"Yorum: {comment}",
            "",
            "Model doğrulama bilgileri:",
            f"Model: {meta['model']}",
            f"n toplam: {meta['n_total']}",
            f"n eğitim: {meta['n_train']}",
            f"n bağımsız test: {meta['n_test']}",
            f"MAE: {meta['mae']} sn",
            f"RMSE: {meta['rmse']} sn",
            f"MAPE: %{meta['mape']}",
            f"R²: {meta['r2']}",
            f"Pearson r: {meta['pearson_r']}",
            f"CV R²: {meta['cv_r2']} ± {meta['cv_std']}",
        ]

        pdf = create_pdf_report(page_title, report_lines)

        st.download_button(
            "📄 PDF Raporu İndir",
            data=pdf,
            file_name=f"{style}_{age_group}_{sex}_prediction_report.pdf",
            mime="application/pdf"
        )

    else:
        stars, comment = performance_comment_en(percentile)

        st.success(f"🏁 Predicted {STYLE_EN[style]} Time: {prediction:.2f} s")
        st.metric("Group Mean", f"{group_mean:.2f} s")
        st.metric("Difference", f"{difference:.2f} s")
        st.metric("Performance Percentile", f"%{percentile:.1f}")

        st.subheader("📈 Performance Interpretation")
        st.write(f"**{stars} {comment}**")

        st.subheader("🧠 Model Information")
        st.write(f"Model: **{meta['model']}**")
        st.write(f"Total n: **{meta['n_total']}**, training: **{meta['n_train']}**, independent test: **{meta['n_test']}**")
        st.write(f"MAE: **{meta['mae']} s**")
        st.write(f"RMSE: **{meta['rmse']} s**")
        st.write(f"MAPE: **%{meta['mape']}**")
        st.write(f"R²: **{meta['r2']}**")
        st.write(f"Pearson r: **{meta['pearson_r']}**")
        st.write(f"5-Fold CV R²: **{meta['cv_r2']} ± {meta['cv_std']}**")

        st.subheader("📋 Used Variables")
        for i, f in enumerate(features, start=1):
            st.write(f"{i}. {FEATURE_LABELS_EN.get(f, f)}")

        report_lines = [
            f"Report date: {datetime.now().strftime('%d.%m.%Y %H:%M')}",
            f"Model title: {page_title}",
            f"Sex: {sex_display}",
            f"Age group: {age_group}",
            "",
            "Entered measurements:",
        ]

        for f, v in inputs.items():
            report_lines.append(f"{FEATURE_LABELS_EN.get(f, f)}: {v}")

        report_lines += [
            "",
            f"Predicted time: {prediction:.2f} s",
            f"Group mean: {group_mean:.2f} s",
            f"Difference: {difference:.2f} s",
            f"Performance percentile: %{percentile:.1f}",
            f"Interpretation: {comment}",
            "",
            "Model validation information:",
            f"Model: {meta['model']}",
            f"Total n: {meta['n_total']}",
            f"Training n: {meta['n_train']}",
            f"Independent test n: {meta['n_test']}",
            f"MAE: {meta['mae']} s",
            f"RMSE: {meta['rmse']} s",
            f"MAPE: %{meta['mape']}",
            f"R²: {meta['r2']}",
            f"Pearson r: {meta['pearson_r']}",
            f"CV R²: {meta['cv_r2']} ± {meta['cv_std']}",
        ]

        pdf = create_pdf_report(page_title, report_lines)

        st.download_button(
            "📄 Download PDF Report",
            data=pdf,
            file_name=f"{style}_{age_group}_{sex}_prediction_report.pdf",
            mime="application/pdf"
        )
        
