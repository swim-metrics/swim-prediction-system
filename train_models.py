import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor

DATA_FILE = "swim_data.xlsx"
RESULT_FILE = "swim_final_family_controlled_24_model_results.xlsx"

TARGETS = {
    "freestyle": "time_50m_freestyle",
    "backstroke": "time_50m_backstroke",
    "breaststroke": "time_50m_breaststroke",
    "butterfly": "time_50m_butterfly",
}

print("24 MODEL EĞİTİMİ BAŞLADI")

df = pd.read_excel(DATA_FILE)
selected = pd.read_excel(RESULT_FILE, sheet_name="Selected_Features")
best = pd.read_excel(RESULT_FILE, sheet_name="Best_24_Models")

models_dir = Path("models")
models_dir.mkdir(exist_ok=True)

metadata = {}

for _, row in selected.iterrows():
    style = row["Style"]
    age_group = row["Age_Group"]
    sex = row["Sex"]

    target = TARGETS[style]

    features = [x.strip() for x in row["Final_Features_EN"].split(",")]
    features_tr = [x.strip() for x in row["Final_Features_TR"].split(",")]

    sub = df[
        (df["age_group"] == age_group) &
        (df["sex"] == sex)
    ].copy()

    X = sub[features]
    y = sub[target]

    model = ExtraTreesRegressor(
        n_estimators=500,
        random_state=42
    )

    model.fit(X, y)

    model_name = f"{style}_{age_group}_{sex}.pkl"
    model_path = models_dir / model_name

    joblib.dump(model, model_path)

    metric_row = best[
        (best["Style"] == style) &
        (best["Age_Group"] == age_group) &
        (best["Sex"] == sex)
    ].iloc[0]

    key = f"{style}_{age_group}_{sex}"

    metadata[key] = {
        "model_file": str(model_path),
        "style": style,
        "age_group": age_group,
        "sex": sex,
        "target": target,
        "features_en": features,
        "features_tr": features_tr,
        "n_total": int(metric_row["n_total"]),
        "n_train": int(metric_row["n_train"]),
        "n_test": int(metric_row["n_test"]),
        "model": str(metric_row["Model"]),
        "mae": float(metric_row["MAE"]),
        "rmse": float(metric_row["RMSE"]),
        "mape": float(metric_row["MAPE_%"]),
        "r2": float(metric_row["R2_Test"]),
        "pearson_r": float(metric_row["Pearson_r"]),
        "cv_r2": float(metric_row["CV_R2_Mean"]),
        "cv_std": float(metric_row["CV_R2_Std"]),
        "min_error": float(metric_row["Min_Error"]),
        "max_error": float(metric_row["Max_Error"]),
    }

with open("model_metadata.json", "w", encoding="utf-8") as f:
    json.dump(metadata, f, ensure_ascii=False, indent=4)

print("24 MODEL BAŞARIYLA KAYDEDİLDİ")
print("models klasörü oluşturuldu")
print("model_metadata.json oluşturuldu")
print("Toplam model sayısı:", len(metadata))