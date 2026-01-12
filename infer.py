import joblib
import torch
import pandas as pd
import numpy as np
from PIL import Image
from torchvision import transforms

# ---------------------------------------------------------
# 1. LOAD MODELS & METADATA
# ---------------------------------------------------------
xgb_model = joblib.load("xgboost_env_severity_model.pkl")
label_encoder = joblib.load("label_encoder.pkl")
train_columns = joblib.load("train_columns-2.pkl")

cnn_model = torch.load("cnn_model.pth", map_location="cpu")
cnn_model.eval()

# ---------------------------------------------------------
# 2. IMAGE PREPROCESSING (MATCH YOUR CNN TRAINING)
# ---------------------------------------------------------
image_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

def predict_image(image_path):
    image = Image.open(image_path).convert("RGB")
    image = image_transform(image).unsqueeze(0)

    with torch.no_grad():
        logits = cnn_model(image)
        prob = torch.sigmoid(logits).item()  # binary CNN

    return prob  # P_img_bleached

# ---------------------------------------------------------
# 3. TABULAR INFERENCE
# ---------------------------------------------------------
def predict_environment(input_dict):
    X = pd.DataFrame([input_dict])[train_columns]
    probs = xgb_model.predict_proba(X)

    P_moderate = probs[0, 0]
    P_severe = probs[0, 1]

    return P_moderate, P_severe

# ---------------------------------------------------------
# 4. FUSION LOGIC
# ---------------------------------------------------------
def fuse_severity(P_img_bleached, P_severe):
    if P_img_bleached < 0.5:
        if P_severe >= 0.6:
            return 1, "Moderate (High Risk – Early)"
        return 0, "Mild"

    if P_severe >= 0.6:
        return 2, "Severe"
    elif P_severe >= 0.3:
        return 1, "Moderate"
    else:
        return 0, "Mild"

# ---------------------------------------------------------
# 5. RUN DEMO
# ---------------------------------------------------------
if __name__ == "__main__":

    # ---- Example inputs ----
    image_path = "sample.jpg"

    tabular_input = {
        "Thermal_Stress_Index": 0.62,
        "TSA_DHWMean": 0.45,
        "Bleaching_Duration_weeks": 6,
        "Temperature_Mean": 29.4,
        "Wind_Mitigation": 0.6,
        "Windspeed": 5.2,
        "Turbidity": 0.3,
        "Cyclone_Frequency": 0.1,
        "Abs_Latitude": 12.5,
        "Exposure": 1,      # 0=Sheltered, 1=Exposed
        "Date_Month": 3,
        "Date_Year": 2023
    }

    # ---- Predictions ----
    P_img = predict_image(image_path)
    P_mod, P_sev = predict_environment(tabular_input)

    severity_int, severity_label = fuse_severity(P_img, P_sev)

    print("\n--- FINAL OUTPUT ---")
    print(f"Image bleaching probability : {P_img:.3f}")
    print(f"Environmental severe prob   : {P_sev:.3f}")
    print(f"Final severity              : {severity_label}")
