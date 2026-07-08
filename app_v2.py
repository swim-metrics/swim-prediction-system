import json
import uuid
from io import BytesIO
from pathlib import Path
from datetime import datetime

import joblib
import pandas as pd
import streamlit as st
from reportlab.pdfgen import canvas


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

FEATURE_LABELS_EN = {k: k.replace("_", " ").title() for k in FEATURE_LABELS_TR.keys()}


st.set_page_config(page_title="Swimming Prediction System", layout="centered")


@st.cache_resource
def load_system():
    df = pd.read_excel(DATA_FILE)
    with open(METADATA_FILE, "r", encoding="utf-8") as f:
        metadata = json.load(f)
    return df, metadata


@st.cache_resource
def load_prediction_model(model_file):
    model_path = Path(str(model_file).replace("\\", "/"))

    if not model_path.exists():
        st.error(f"Model dosyası bulunamadı: {model_path}")
        st.stop()

    return joblib.load(model_path)


def get_default_value(feature):
    defaults = {
        "age": 15, "body_height": 160.0, "body_mass": 55.0, "training_age": 4.0,
        "vertical_jump": 35.0, "standing_long_jump": 180.0,
        "flexed_arm_hang": 30.0, "sit_ups_1min": 35,
        "illinois_agility_test": 18.0, "sprint_30m": 5.0,
        "hand_length": 17.0, "arm_span": 165.0, "arm_length": 65.0,
        "leg_length": 80.0, "sitting_height": 80.0, "chest_girth": 80.0,
        "biceps_girth": 25.0, "flexed_biceps_girth": 27.0,
        "forearm_girth": 22.0, "thigh_girth": 45.0, "calf_girth": 32.0,
        "waist_girth": 70.0, "hip_girth": 85.0,
        "biacromial_breadth": 35.0, "biiliac_breadth": 28.0,
        "elbow_breadth": 6.0, "wrist_breadth": 5.0,
        "knee_breadth": 8.0, "ankle_breadth": 6.0,
        "sit_and_reach": 20.0, "trunk_extension_height": 35.0,
        "shoulder_extension_height": 40.0, "shoulder_mobility": 0.0,
        "flamingo_balance_test": 10.0, "body_density": 1.05,
        "body_fat_percentage": 15.0, "fat_mass": 8.0,
        "fat_free_mass": 45.0, "handgrip_strength": 30.0,
        "ankle_dorsiflexion_rom": 30.0,
        "ankle_plantarflexion_rom": 45.0, "foot_length": 24.0,
    }
    return defaults.get(feature, 0.0)


def fix_pdf_text(text):
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


def create_pdf_report(title, lines, report_id):
    buffer = BytesIO()
    c = canvas.Canvas(buffer)


    c.setFont("Helvetica", 14)
    c.drawString(40, 800, fix_pdf_text(title))

    c.setFont("Helvetica", 10)
    c.drawString(40, 782, fix_pdf_text(f"Rapor No / Report ID: {report_id}"))

    y = 750
    for line in lines:
        c.drawString(40, y, fix_pdf_text(line))
        y -= 18

        if y < 50:
            c.showPage()
            c.setFont("Helvetica", 10)
            y = 800

    c.save()
    buffer.seek(0)
    return buffer


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


df, metadata = load_system()

language = st.selectbox("Dil / Language", ["🇹🇷 Türkçe", "🇺🇸 English"])
is_tr = language.startswith("🇹🇷")

if is_tr:
    st.title("🇹🇷 Yüzme Performansı Tahmin Sistemi")

    style_display = st.selectbox("Stil Seçiniz", list(STYLE_TR.values()))
    style = {v: k for k, v in STYLE_TR.items()}[style_display]

    st.header(f"🏊 {style_display} Tahmin Modeli")

    sex_display = st.selectbox("Cinsiyet", ["Kadın", "Erkek"])
    sex = "female" if sex_display == "Kadın" else "male"

    age_group = st.selectbox("Yaş Grubu", ["12_13", "14_15", "16_17"])

    st.markdown("---")
    st.subheader("👤 Sporcu Bilgileri")
    athlete_name = st.text_input("Ad Soyad", value="")
    club_name = st.text_input("Kulüp", value="")
    coach_name = st.text_input("Antrenör", value="")

else:
    st.title("🇺🇸 Swimming Performance Prediction System")

    style_display = st.selectbox("Select Stroke", list(STYLE_EN.values()))
    style = {v: k for k, v in STYLE_EN.items()}[style_display]

    st.header(f"🏊 {style_display} Prediction Model")

    sex_display = st.selectbox("Sex", ["Female", "Male"])
    sex = "female" if sex_display == "Female" else "male"

    age_group = st.selectbox("Age Group", ["12_13", "14_15", "16_17"])

    st.markdown("---")
    st.subheader("👤 Athlete Information")
    athlete_name = st.text_input("Athlete Name", value="")
    club_name = st.text_input("Club", value="")
    coach_name = st.text_input("Coach", value="")


model_key = f"{style}_{age_group}_{sex}"

if is_tr:
    st.caption(f"Seçilen model anahtarı: {model_key}")
else:
    st.caption(f"Selected model key: {model_key}")

if model_key not in metadata:
    st.error("Bu seçim için kayıtlı model bulunamadı.")
    st.stop()

model_info = metadata[model_key]
model = load_prediction_model(model_info["model_file"])
features = model_info["features_en"]

st.markdown("---")
st.subheader("📌 Modele Girilecek Ölçümler" if is_tr else "📌 Required Measurements")

inputs = {}

for feature in features:
    label = FEATURE_LABELS_TR.get(feature, feature) if is_tr else FEATURE_LABELS_EN.get(feature, feature)
    default_value = get_default_value(feature)

    if feature in ["age", "sit_ups_1min"]:
        inputs[feature] = st.number_input(label, min_value=0, max_value=300, value=int(default_value))
    else:
        inputs[feature] = st.number_input(label, min_value=0.0, max_value=500.0, value=float(default_value))

if is_tr:
    st.info(f"Modelde kullanılan değişken sayısı: {len(features)}")
else:
    st.info(f"Number of variables used in the model: {len(features)}")

st.markdown("---")

button_text = "Tahmin Et" if is_tr else "Predict"

if st.button(button_text):

    report_id = f"SWIM-{datetime.now().strftime('%Y%m%d')}-{str(uuid.uuid4())[:8].upper()}"

    input_df = pd.DataFrame([inputs])
    prediction = model.predict(input_df)[0]

    target_col = model_info["target"]

    group_data = df[
        (df["sex"] == sex) &
        (df["age_group"] == age_group)
    ]

    group_mean = group_data[target_col].mean()
    difference = prediction - group_mean
    percentile = (group_data[target_col] > prediction).mean() * 100

    if is_tr:
        stars, comment = performance_comment_tr(percentile)

        st.success(f"🏁 Tahmini {style_display} Derecesi: {prediction:.2f} sn")

        col1, col2, col3 = st.columns(3)
        col1.metric("Grup Ortalaması", f"{group_mean:.2f} sn")
        col2.metric("Fark", f"{difference:.2f} sn")
        col3.metric("Performans Yüzdeliği", f"%{percentile:.1f}")

        st.subheader("📈 Performans Yorumu")
        st.write(f"**{stars} {comment}**")

        if difference < 0:
            st.write(f"Bu sporcu kendi yaş grubu ve cinsiyet ortalamasından **{abs(difference):.2f} sn daha hızlı** görünmektedir.")
        else:
            st.write(f"Bu sporcu kendi yaş grubu ve cinsiyet ortalamasından **{difference:.2f} sn daha yavaş** görünmektedir.")

        st.subheader("🧠 Model Doğruluk Bilgileri")
        st.write(f"Model: **{model_info['model']}**")
        st.write(f"Toplam veri: **{model_info['n_total']}**")
        st.write(f"Eğitim verisi: **{model_info['n_train']}**")
        st.write(f"Bağımsız test verisi: **{model_info['n_test']}**")
        st.write(f"MAE: **{model_info['mae']} sn**")
        st.write(f"RMSE: **{model_info['rmse']} sn**")
        st.write(f"MAPE: **%{model_info['mape']}**")
        st.write(f"R²: **{model_info['r2']}**")
        st.write(f"Pearson r: **{model_info['pearson_r']}**")
        st.write(f"5-Kat CV R²: **{model_info['cv_r2']} ± {model_info['cv_std']}**")
        st.write(f"Minimum hata: **{model_info['min_error']} sn**")
        st.write(f"Maksimum hata: **{model_info['max_error']} sn**")

        st.subheader("📋 Kullanılan Değişkenler")
        for i, f in enumerate(features, start=1):
            st.write(f"{i}. {FEATURE_LABELS_TR.get(f, f)}")

        report_lines = [
            "SPORCU BILGILERI",
            f"Rapor No: {report_id}",
            f"Ad Soyad: {athlete_name}",
            f"Kulup: {club_name}",
            f"Antrenor: {coach_name}",
            f"Rapor Tarihi: {datetime.now().strftime('%d.%m.%Y %H:%M')}",
            "",
            f"Model Basligi: {style_display} Tahmin Modeli",
            f"Cinsiyet: {sex_display}",
            f"Yas Grubu: {age_group}",
            "",
            "Girilen Olcumler:"
        ]

        for f, v in inputs.items():
            report_lines.append(f"{FEATURE_LABELS_TR.get(f, f)}: {v}")

        report_lines += [
            "",
            f"Tahmini Derece: {prediction:.2f} sn",
            f"Grup Ortalamasi: {group_mean:.2f} sn",
            f"Fark: {difference:.2f} sn",
            f"Performans Yuzdeligi: %{percentile:.1f}",
            f"Performans Yorumu: {comment}",
            "",
            "Model Dogruluk Bilgileri:",
            f"Model: {model_info['model']}",
            f"Toplam Veri: {model_info['n_total']}",
            f"Egitim Verisi: {model_info['n_train']}",
            f"Bagimsiz Test Verisi: {model_info['n_test']}",
            f"MAE: {model_info['mae']} sn",
            f"RMSE: {model_info['rmse']} sn",
            f"MAPE: %{model_info['mape']}",
            f"R2: {model_info['r2']}",
            f"Pearson r: {model_info['pearson_r']}",
            f"5-Kat CV R2: {model_info['cv_r2']} +/- {model_info['cv_std']}",
            f"Minimum Hata: {model_info['min_error']} sn",
            f"Maksimum Hata: {model_info['max_error']} sn",
        ]

        pdf = create_pdf_report(f"{style_display} Tahmin Raporu", report_lines, report_id)

        st.download_button(
            label="📄 PDF Raporu İndir",
            data=pdf,
            file_name=f"{style}_{age_group}_{sex}_{report_id}_tahmin_raporu.pdf",
            mime="application/pdf"
        )

    else:
        stars, comment = performance_comment_en(percentile)

        st.success(f"🏁 Predicted {style_display} Time: {prediction:.2f} s")

        col1, col2, col3 = st.columns(3)
        col1.metric("Group Mean", f"{group_mean:.2f} s")
        col2.metric("Difference", f"{difference:.2f} s")
        col3.metric("Performance Percentile", f"%{percentile:.1f}")

        st.subheader("📈 Performance Interpretation")
        st.write(f"**{stars} {comment}**")

        if difference < 0:
            st.write(f"This swimmer appears **{abs(difference):.2f} s faster** than the age-sex group mean.")
        else:
            st.write(f"This swimmer appears **{difference:.2f} s slower** than the age-sex group mean.")

        st.subheader("🧠 Model Accuracy Information")
        st.write(f"Model: **{model_info['model']}**")
        st.write(f"Total sample: **{model_info['n_total']}**")
        st.write(f"Training sample: **{model_info['n_train']}**")
        st.write(f"Independent test sample: **{model_info['n_test']}**")
        st.write(f"MAE: **{model_info['mae']} s**")
        st.write(f"RMSE: **{model_info['rmse']} s**")
        st.write(f"MAPE: **%{model_info['mape']}**")
        st.write(f"R²: **{model_info['r2']}**")
        st.write(f"Pearson r: **{model_info['pearson_r']}**")
        st.write(f"5-Fold CV R²: **{model_info['cv_r2']} ± {model_info['cv_std']}**")
        st.write(f"Minimum error: **{model_info['min_error']} s**")
        st.write(f"Maximum error: **{model_info['max_error']} s**")

        st.subheader("📋 Used Variables")
        for i, f in enumerate(features, start=1):
            st.write(f"{i}. {FEATURE_LABELS_EN.get(f, f)}")

        report_lines = [
            "ATHLETE INFORMATION",
            f"Report ID: {report_id}",
            f"Athlete Name: {athlete_name}",
            f"Club: {club_name}",
            f"Coach: {coach_name}",
            f"Report Date: {datetime.now().strftime('%d.%m.%Y %H:%M')}",
            "",
            f"Model Title: {style_display} Prediction Model",
            f"Sex: {sex_display}",
            f"Age Group: {age_group}",
            "",
            "Entered Measurements:"
        ]

        for f, v in inputs.items():
            report_lines.append(f"{FEATURE_LABELS_EN.get(f, f)}: {v}")

        report_lines += [
            "",
            f"Predicted Time: {prediction:.2f} s",
            f"Group Mean: {group_mean:.2f} s",
            f"Difference: {difference:.2f} s",
            f"Performance Percentile: %{percentile:.1f}",
            f"Performance Interpretation: {comment}",
            "",
            "Model Accuracy Information:",
            f"Model: {model_info['model']}",
            f"Total Sample: {model_info['n_total']}",
            f"Training Sample: {model_info['n_train']}",
            f"Independent Test Sample: {model_info['n_test']}",
            f"MAE: {model_info['mae']} s",
            f"RMSE: {model_info['rmse']} s",
            f"MAPE: %{model_info['mape']}",
            f"R2: {model_info['r2']}",
            f"Pearson r: {model_info['pearson_r']}",
            f"5-Fold CV R2: {model_info['cv_r2']} +/- {model_info['cv_std']}",
            f"Minimum Error: {model_info['min_error']} s",
            f"Maximum Error: {model_info['max_error']} s",
        ]

        pdf = create_pdf_report(f"{style_display} Prediction Report", report_lines, report_id)

        st.download_button(
            label="📄 Download PDF Report",
            data=pdf,
            file_name=f"{style}_{age_group}_{sex}_{report_id}_prediction_report.pdf",
            mime="application/pdf"
        )
# ============================================================
# FOOTER / COPYRIGHT NOTICE
# ============================================================

st.markdown("---")

if is_tr:
    st.markdown(
        """
### ℹ️ SwimML Karar Destek Sistemi Hakkında

**SwimML Predictor v1.0**

**Geliştiren:**  
**Tuğrul Özkadı & Araştırma Ekibi**  
Hitit Üniversitesi  
Spor Bilimleri Fakültesi

---

© 2026 Tuğrul Özkadı & Araştırma Ekibi.  
Tüm hakları saklıdır.

Bu yazılım bilimsel araştırma ve eğitim amacıyla geliştirilmiştir.

Bu uygulamanın veya kaynak kodlarının yazılı izin alınmadan çoğaltılması, değiştirilmesi, yeniden dağıtılması veya ticari amaçla kullanılması yasaktır.

---

Bu uygulama, antrenörlük kararlarını desteklemek amacıyla geliştirilmiştir.

Tahmin sonuçları, uzman antrenör değerlendirmesiyle birlikte yorumlanmalıdır. Sistem, sporcu seçimi, performans değerlendirmesi veya sağlıkla ilgili kararlar için tek başına kesin karar aracı olarak kullanılmamalıdır.
        """
    )

else:
    st.markdown(
        """
### ℹ️ About SwimML Decision Support System

**SwimML Predictor v1.0**

**Developed by:**  
**Tuğrul Özkadı & Research Team**  
Hitit University  
Faculty of Sport Sciences

---

© 2026 Tuğrul Özkadı & Research Team.  
All Rights Reserved.

This software has been developed for scientific research and educational purposes.

Unauthorized reproduction, modification, redistribution, or commercial use of this application or its source code is prohibited without prior written permission from the authors.

---

This application is intended to support coaching decisions.

Prediction results should be interpreted together with expert coaching evaluation. The system should not be used as the sole decision-making tool for athlete selection, performance evaluation, or health-related decisions.
        """
    )
