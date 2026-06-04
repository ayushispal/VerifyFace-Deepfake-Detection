import os
import sys
import tensorflow as tf
import numpy as np
import cv2
import base64
import shutil
from PIL import Image
from fastapi import FastAPI, File, UploadFile, HTTPException
from pydantic import BaseModel

# Set path roots
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

from models.efficientnet import build_model
from explainability.gradcam import GradCAM, overlay_heatmap

app = FastAPI(title="VerifyFace Backend API", description="FastAPI server for Deepfake Detection & Explainability")

# Ensure temp directory exists
TEMP_DIR = os.path.join(BASE_DIR, "temp_uploads")
os.makedirs(TEMP_DIR, exist_ok=True)

# Load global variables for model at startup
model_path = os.path.join(BASE_DIR, "saved_models", "best_model.keras")
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model file missing at: {model_path}")

print("Loading deep learning model...")
try:
    # Recreate model structure and load weights
    model, base_model = build_model()
    model = tf.keras.models.load_model(model_path)
    base_model = model.get_layer("efficientnetv2-b0")
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {str(e)}")
    raise e

def img_to_base64(img):
    """Converts a BGR image to a base64 encoded string."""
    _, buffer = cv2.imencode('.jpg', img)
    img_base64 = base64.b64encode(buffer).decode('utf-8')
    return img_base64

class PredictionResponse(BaseModel):
    prediction: str
    confidence: float
    real_prob: float
    fake_prob: float
    images: dict

@app.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...)):
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file selected")
        
    try:
        # Save temp file for Grad-CAM path reading
        temp_path = os.path.join(TEMP_DIR, f"temp_{file.filename}")
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
            
        # Load image via PIL
        image = Image.open(temp_path).convert("RGB")
        
        # Preprocessing: 224x224, RGB, preprocess_input
        img_resized = image.resize((224, 224))
        img_array = np.array(img_resized, dtype=np.float32)
        img_batch = np.expand_dims(img_array, axis=0)
        img_preprocessed = tf.keras.applications.efficientnet_v2.preprocess_input(img_batch)
        
        # Predict sigmoid
        pred_prob = float(model(img_preprocessed, training=False).numpy()[0][0])
        
        # Check uncertainty zone (+/-0.08 around threshold 0.65)
        is_uncertain = 0.57 <= pred_prob <= 0.73
        
        if is_uncertain:
            prediction = "Uncertain - Needs Manual Review"
            is_fake = pred_prob >= 0.65
        else:
            is_fake = pred_prob >= 0.65
            prediction = "Fake" if is_fake else "Real"
            
        # Confidence Score (scaled relative to 0.65 threshold to span [0.5, 1.0])
        if is_fake:
            confidence = (pred_prob - 0.65) / (1.0 - 0.65) * 0.5 + 0.5
        else:
            confidence = (0.65 - pred_prob) / 0.65 * 0.5 + 0.5
            
        real_prob = (1.0 - pred_prob) * 100
        fake_prob = pred_prob * 100
        
        # Run Grad-CAM
        gradcam = GradCAM(model, base_model)
        heatmap, _ = gradcam(img_preprocessed, pred_label=(1 if is_fake else 0))
        
        # Apply overlay heatmap on saved temp path
        orig_img, heatmap_colored, overlay = overlay_heatmap(temp_path, heatmap, alpha=0.45)
        
        # Clean up temp file
        if os.path.exists(temp_path):
            os.remove(temp_path)
            
        # Encode outputs as base64
        base64_original = img_to_base64(orig_img)
        base64_heatmap = img_to_base64(heatmap_colored)
        base64_overlay = img_to_base64(overlay)
        
        return {
            "prediction": prediction,
            "confidence": confidence,
            "real_prob": real_prob,
            "fake_prob": fake_prob,
            "images": {
                "original": base64_original,
                "heatmap": base64_heatmap,
                "overlay": base64_overlay
            }
        }
        
    except Exception as e:
        # Clean up temp file in case of failure
        if 'temp_path' in locals() and os.path.exists(temp_path):
            os.remove(temp_path)
        print(f"Error during prediction processing: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")

@app.get("/health")
async def health():
    return {"status": "healthy", "model_path": model_path}
