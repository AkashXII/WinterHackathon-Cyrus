from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import os
import io

import torch
import torch.nn.functional as F
from torchvision import models, transforms

import pandas as pd
import joblib
from PIL import Image
from dotenv import load_dotenv

print(">>> BEFORE load_dotenv:", os.getenv("GEMINI_API_KEY"))
load_dotenv()
print(">>> AFTER load_dotenv:", os.getenv("GEMINI_API_KEY"))

# ----------------------------------------------------
# Gemini setup (FIXED)
# ----------------------------------------------------
import google.generativeai as genai

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if GEMINI_API_KEY and GEMINI_API_KEY.strip():
    genai.configure(api_key=GEMINI_API_KEY)
    GEMINI_ENABLED = True
else:
    GEMINI_ENABLED = False

print(f"[STARTUP] Gemini enabled: {GEMINI_ENABLED}")

# ----------------------------------------------------
# FastAPI setup
# ----------------------------------------------------
app = FastAPI(title="DeepReef – Fusion Backend")

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
    weights_only=False
)

if isinstance(loaded, dict):
    cnn_model = models.resnet50(weights=None)
    cnn_model.fc = torch.nn.Linear(cnn_model.fc.in_features, NUM_CLASSES)
    cnn_model.load_state_dict(loaded)
else:
    cnn_model = loaded

cnn_model.to(DEVICE)
cnn_model.eval()

CLASS_LABELS = ["Bleached Coral", "Healthy Coral"]

# ----------------------------------------------------
# Load XGBoost + metadata
# ----------------------------------------------------
xgb_model = joblib.load("xgboost_env_severity_model.pkl")
train_columns = joblib.load("train_columns-2.pkl")

# ----------------------------------------------------
# Image preprocessing
# ----------------------------------------------------
image_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

async def preprocess_image(uploaded_file: UploadFile):
    img_bytes = await uploaded_file.read()
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    img = image_transform(img).unsqueeze(0)
    return img.to(DEVICE)

# ----------------------------------------------------
# Fusion logic
# ----------------------------------------------------
def fuse_severity(P_img_bleached: float, P_Severe_env: float):
    if P_img_bleached < 0.5:
        if P_Severe_env >= 0.6:
            return 1, "Moderate (High Risk – Early Stage)"
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
    img_tensor = await preprocess_image(file)

    with torch.no_grad():
        logits = cnn_model(img_tensor)
        probs = F.softmax(logits, dim=1)
        P_img_bleached = float(probs[0][0])
        idx = int(torch.argmax(probs, dim=1))

    coral_status = CLASS_LABELS[idx]

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

    X = pd.DataFrame([input_data]).reindex(columns=train_columns, fill_value=0)
    probs_env = xgb_model.predict_proba(X)
    P_Severe_env = float(probs_env[0][1])

    severity_int, severity_label = fuse_severity(P_img_bleached, P_Severe_env)

    if GEMINI_ENABLED:
        try:
            model = genai.GenerativeModel("models/gemini-flash-lite-latest")
            prompt = f"""
            Visual bleaching probability: {P_img_bleached:.2f}
            Environmental severe stress probability: {P_Severe_env:.2f}
            Final severity classification: {severity_label}

            Explain the result in 3 short scientific sentences.
            """
            response = model.generate_content(prompt)
            explanation = response.text
            gemini_status = "USED"
        except Exception as e:
            explanation = f"Gemini error: {str(e)}"
            gemini_status = "FAILED"
    else:
        explanation = "Gemini disabled (API key not detected)"
        gemini_status = "DISABLED"

    return {
        "fusion_model_result": {
            "image_prediction": coral_status,
            "image_bleaching_probability": round(P_img_bleached, 4),
            "environment_severe_probability": round(P_Severe_env, 4),
            "final_severity": severity_label,
            "severity_code": severity_int
        },
        "gemini": {
            "status": gemini_status,
            "enabled_at_startup": GEMINI_ENABLED,
            "explanation": explanation
        }
    }
@app.get("/coral_facts")
def coral_facts():
    try:
        prompt = (
            "Provide exactly 3 concise, scientifically accurate facts about coral reefs. "
            "Each fact must be one sentence. Do not number them."
        )

        model = genai.GenerativeModel("models/gemini-flash-lite-latest")
        response = model.generate_content(prompt)

        if hasattr(response, "text") and response.text:
            lines = response.text.strip().split("\n")
        else:
            lines = response.candidates[0].content.parts[0].text.split("\n")

        facts = [l.strip("-• ").strip() for l in lines if l.strip()][:3]

        return {"facts": facts}

    except Exception as e:
        print("❌ Gemini facts error:", repr(e))
        return {
            "facts": [
                "Coral reefs are among the most biodiverse ecosystems on Earth.",
                "Rising ocean temperatures are the leading cause of coral bleaching.",
                "Healthy coral reefs protect coastlines from erosion and storms."
            ]
        }
@app.get("/")
def root():
    return {"status": "DeepReef backend running"}
