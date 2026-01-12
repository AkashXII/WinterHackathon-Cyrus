from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
import torch
import torch.nn.functional as F
from torchvision import models, transforms
import pandas as pd
import joblib
from PIL import Image
import io

# ----------------------------------------------------
# FastAPI setup
# ----------------------------------------------------
app = FastAPI(title="DeepReef AI Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ----------------------------------------------------
# Device
# ----------------------------------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ----------------------------------------------------
# Load CNN (PyTorch ResNet50)
# ----------------------------------------------------
CNN_MODEL_PATH = "resnet50_coral.pth"
NUM_CLASSES = 2

loaded = torch.load(
    CNN_MODEL_PATH,
    map_location=DEVICE,
    weights_only=False   # REQUIRED for legacy models
)

if isinstance(loaded, dict):
    cnn_model = models.resnet50(weights=None)
    cnn_model.fc = torch.nn.Linear(cnn_model.fc.in_features, NUM_CLASSES)
    cnn_model.load_state_dict(loaded)
else:
    cnn_model = loaded

cnn_model.to(DEVICE)
cnn_model.eval()

# IMPORTANT: match training order
CLASS_LABELS = ["Bleached Coral", "Healthy Coral"]

# ----------------------------------------------------
# Load XGBoost + metadata
# ----------------------------------------------------
xgb_model = joblib.load("xgboost_env_severity_model.pkl")
train_columns = joblib.load("train_columns-2.pkl")

# ----------------------------------------------------
# Image preprocessing
# ----------------------------------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

def preprocess_image(uploaded_file: UploadFile):
    img_bytes = uploaded_file.file.read()
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    img = transform(img).unsqueeze(0)
    return img.to(DEVICE)

# ----------------------------------------------------
# Fusion logic (FIXED & CONSISTENT)
# ----------------------------------------------------
def fuse_severity(P_img_bleached: float, P_Severe_env: float):
    if P_img_bleached < 0.5:
        if P_Severe_env >= 0.6:
            return 1, "Moderate (High Risk – Early)"
        return 0, "Mild"

    if P_Severe_env >= 0.6:
        return 2, "Severe"
    elif P_Severe_env >= 0.3:
        return 1, "Moderate"
    else:
        return 0, "Mild"

# ----------------------------------------------------
# Main endpoint
# ----------------------------------------------------
@app.post("/analyze_coral")
async def analyze_coral(
    file: UploadFile = File(...),

    Thermal_Stress_Index: float = Form(...),
    TSA_DHWMean: float = Form(...),
    Bleaching_Duration_weeks: int = Form(...),
    Temperature_Mean: float = Form(...),
    Wind_Mitigation: float = Form(...),
    Windspeed: float = Form(...),
    Turbidity: float = Form(...),
    Cyclone_Frequency: float = Form(...),
    Abs_Latitude: float = Form(...),
    Exposure: int = Form(...),
    Date_Month: int = Form(...),
    Date_Year: int = Form(...)
):
    # -------- Image inference --------
    img_tensor = preprocess_image(file)
    with torch.no_grad():
        logits = cnn_model(img_tensor)
        probs = F.softmax(logits, dim=1)

        # Index 0 = Bleached
        P_img_bleached = float(probs[0][0])
        idx = int(torch.argmax(probs, dim=1))

    coral_status = CLASS_LABELS[idx]

    # -------- Tabular inference --------
    input_data = {
        "Thermal_Stress_Index": Thermal_Stress_Index,
        "TSA_DHWMean": TSA_DHWMean,
        "Bleaching_Duration_weeks": Bleaching_Duration_weeks,
        "Temperature_Mean": Temperature_Mean,
        "Wind_Mitigation": Wind_Mitigation,
        "Windspeed": Windspeed,
        "Turbidity": Turbidity,
        "Cyclone_Frequency": Cyclone_Frequency,
        "Abs_Latitude": Abs_Latitude,
        "Exposure": Exposure,
        "Date_Month": Date_Month,
        "Date_Year": Date_Year
    }

    X = pd.DataFrame([input_data])[train_columns]
    probs_env = xgb_model.predict_proba(X)
    P_Severe_env = float(probs_env[0][1])

    # -------- Fusion --------
    severity_int, severity_label = fuse_severity(P_img_bleached, P_Severe_env)

    return {
        "image_prediction": coral_status,
        "image_bleaching_probability": round(P_img_bleached, 4),
        "environment_severe_probability": round(P_Severe_env, 4),
        "final_severity": severity_label,
        "severity_code": severity_int
    }

# ----------------------------------------------------
# Root endpoint
# ----------------------------------------------------
@app.get("/")
def root():
    return {"status": "DeepReef AI backend running"}
