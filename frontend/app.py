import os
import sys
import streamlit as st
import requests
import base64
from io import BytesIO
from PIL import Image

# Streamlit Page Config
st.set_page_config(
    page_title="VerifyFace | Explainable Deepfake Detection",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Premium Styling (Outfit typography, dark styling, custom components)
st.markdown("""
    <link href="https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;800&display=swap" rel="stylesheet">
    <style>
        * {
            font-family: 'Outfit', sans-serif;
        }
        .main-title {
            background: linear-gradient(135deg, #60a5fa, #c084fc);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            font-size: 2.8rem;
            font-weight: 800;
            margin-bottom: 0.25rem;
        }
        .subtitle {
            font-size: 1.15rem;
            color: #94a3b8;
            margin-bottom: 2rem;
        }
        .panel {
            background: rgba(30, 41, 59, 0.4);
            border: 1px solid rgba(71, 85, 105, 0.3);
            border-radius: 16px;
            padding: 24px;
            margin-bottom: 20px;
        }
        .panel-header {
            font-size: 1.3rem;
            font-weight: 600;
            color: #f8fafc;
            border-bottom: 1px solid rgba(71, 85, 105, 0.3);
            padding-bottom: 10px;
            margin-bottom: 20px;
        }
        .pred-card-real {
            background: linear-gradient(135deg, #064e3b 0%, #022c22 100%);
            border: 1px solid #10b981;
            padding: 24px;
            border-radius: 12px;
            margin-bottom: 20px;
        }
        .pred-card-fake {
            background: linear-gradient(135deg, #9f1239 0%, #4c0519 100%);
            border: 1px solid #f43f5e;
            padding: 24px;
            border-radius: 12px;
            margin-bottom: 20px;
        }
        .pred-card-uncertain {
            background: linear-gradient(135deg, #78350f 0%, #451a03 100%);
            border: 1px solid #ecc94b;
            padding: 24px;
            border-radius: 12px;
            margin-bottom: 20px;
        }
        .pred-label {
            font-size: 0.85rem;
            color: #cbd5e1;
            text-transform: uppercase;
            letter-spacing: 0.1em;
        }
        .pred-value {
            font-size: 2.2rem;
            font-weight: 800;
            margin-top: 5px;
            margin-bottom: 15px;
        }
        .pred-value-real { color: #10b981; }
        .pred-value-fake { color: #f43f5e; }
        .pred-value-uncertain { 
            color: #ecc94b;
            font-size: 1.8rem;
        }
    </style>
""", unsafe_allow_html=True)

# API endpoint URL
API_URL = "http://localhost:8000/predict"

def decode_base64_image(base64_str):
    image_data = base64.b64decode(base64_str)
    return Image.open(BytesIO(image_data))

# Header Area
st.markdown('<div class="main-title">🛡️ VerifyFace</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Explainable Deepfake Detection Platform</div>', unsafe_allow_html=True)

# Tabs Navigation
tab_diagnosis, tab_model_info, tab_about = st.tabs([
    "🔍 Live Diagnosis", 
    "⚙️ Model Information", 
    "📄 About Project"
])

with tab_diagnosis:
    col_input, col_output = st.columns([1, 2], gap="large")
    
    with col_input:
        st.markdown('<div class="panel-header">Inference Controller</div>', unsafe_allow_html=True)
        
        uploaded_file = st.file_uploader(
            "Upload Face Image", 
            type=["jpg", "jpeg", "png"],
            help="Supports JPEG and PNG face images."
        )
        
        if uploaded_file is not None:
            # Show preview
            image = Image.open(uploaded_file).convert("RGB")
            st.image(image, caption="Uploaded Image Preview", use_container_width=True)
            
            # Diagnose button
            run_btn = st.button("Run Deepfake Diagnosis", type="primary", use_container_width=True)
        else:
            run_btn = False
            st.info("💡 Upload a face image on this panel to execute deepfake diagnosis.")
            
    with col_output:
        st.markdown('<div class="panel-header">Diagnosis & Grad-CAM Metrics</div>', unsafe_allow_html=True)
        
        if run_btn and uploaded_file is not None:
            with st.spinner("Uploading image to backend and executing inference..."):
                try:
                    # Prepare file stream for request
                    file_bytes = uploaded_file.getvalue()
                    files = {"file": (uploaded_file.name, file_bytes, uploaded_file.type)}
                    
                    response = requests.post(API_URL, files=files)
                    
                    if response.status_code == 200:
                        data = response.json()
                        prediction = data["prediction"]
                        confidence = data["confidence"]
                        real_prob = data["real_prob"]
                        fake_prob = data["fake_prob"]
                        images_b64 = data["images"]
                        
                        # Select Card styles
                        if prediction == "Uncertain - Needs Manual Review":
                            card_style = "pred-card-uncertain"
                            val_style = "pred-value-uncertain"
                            display_pred = "UNCERTAIN - Manual Review"
                            bar_color = "orange"
                        elif prediction == "Fake":
                            card_style = "pred-card-fake"
                            val_style = "pred-value-fake"
                            display_pred = "FAKE"
                            bar_color = "red"
                        else:
                            card_style = "pred-card-real"
                            val_style = "pred-value-real"
                            display_pred = "REAL"
                            bar_color = "green"
                            
                        # Result Card Display
                        st.markdown(f"""
                            <div class="{card_style}">
                                <div class="pred-label">Classification Output</div>
                                <div class="pred-value {val_style}">{display_pred}</div>
                            </div>
                        """, unsafe_allow_html=True)
                        
                        # Metrics Columns
                        col_m1, col_m2, col_m3 = st.columns(3)
                        with col_m1:
                            st.metric("Real Probability", f"{real_prob:.2f}%")
                        with col_m2:
                            st.metric("Fake Probability", f"{fake_prob:.2f}%")
                        with col_m3:
                            st.metric("Confidence Score", f"{confidence*100:.2f}%")
                            
                        # Progress Bar for Confidence
                        st.markdown("**Confidence Progress:**")
                        st.progress(confidence)
                        
                        # Grad-CAM images decode
                        orig_img = decode_base64_image(images_b64["original"])
                        heatmap_img = decode_base64_image(images_b64["heatmap"])
                        overlay_img = decode_base64_image(images_b64["overlay"])
                        
                        # Visual display columns
                        st.markdown("### Grad-CAM Interpretability Mapping")
                        col_img1, col_img2, col_img3 = st.columns(3)
                        with col_img1:
                            st.image(orig_img, caption="Original Image", use_container_width=True)
                        with col_img2:
                            st.image(heatmap_img, caption="Grad-CAM Heatmap", use_container_width=True)
                        with col_img3:
                            st.image(overlay_img, caption="Overlay Explanation", use_container_width=True)
                            
                        st.info(
                            "💡 **Grad-CAM Interpretation Guide:** The red/orange overlay highlights the "
                            "pixels and regions that contributed most to the model's classification. "
                            "For deepfakes, artifacts around the mouth, nose boundary, or eye structures "
                            "typically trigger strong activations."
                        )
                    else:
                        st.error(f"Backend API Error: {response.text}")
                except Exception as e:
                    st.error(f"Failed to connect to FastAPI backend server: {str(e)}")
                    st.warning("Please ensure the FastAPI backend is running (e.g. `uvicorn backend.main:app --reload`).")
        else:
            st.markdown(
                "<div style='text-align: center; color: #64748b; padding: 5rem 2rem;'>"
                "<h3>Awaiting Inference Request</h3>"
                "<p>Please select a face image on the left controller and execute the diagnosis.</p>"
                "</div>",
                unsafe_allow_html=True
            )

with tab_model_info:
    st.markdown("### Model Configuration Details")
    
    col_c1, col_c2 = st.columns(2)
    with col_c1:
        st.markdown("#### 🛠️ Pipeline Details")
        st.write("**Architecture:** `EfficientNetV2B0` Backbone")
        st.write("**Learning Technique:** Transfer Learning (Fine-tuned classification head & final backbone layers)")
        st.write("**Explainability Method:** Gradient-weighted Class Activation Mapping (Grad-CAM)")
        st.write("**Calibration Method:** Decision Threshold Calibration & Uncertainty Zone Isolation")
        
    with col_c2:
        st.markdown("#### 🎯 Decision Boundary settings")
        st.write("**Configured Optimal Threshold:** `0.65`")
        st.write("**Uncertainty Zone Boundary:** `0.57 to 0.73` (+/-0.08 around threshold)")
        st.write("**Fallback Mode:** Flagged cases in the uncertainty zone default to Manual Review")
        
    st.markdown("---")
    st.markdown("### 📊 Calibrated Performance Metrics (Complete Test Set)")
    
    col_p1, col_p2, col_p3 = st.columns(3)
    with col_p1:
        st.metric("Test Accuracy (Resolved Cases)", "80.58%")
    with col_p2:
        st.metric("Test ROC-AUC", "0.9242")
    with col_p3:
        st.metric("F1-Score (Resolved Cases)", "81.71%")
        
    st.info(
        "📊 *Note: Calibrated performance metrics represent the accuracy of the system on the "
        "88.16% of test images that fall outside of the uncertainty zone. Under this setup, the "
        "False Positive Rate drops from 39.70% to 23.79% on resolved cases.*"
    )

with tab_about:
    st.markdown("### About the Project")
    st.write(
        "VerifyFace is a diagnostic platform designed to detect face-swaps and AI-synthesized faces "
        "using deep learning. In critical verification environments, pure black-box classification "
        "is insufficient. VerifyFace combines high accuracy with visual activation heatmaps to assist "
        "auditors in examining specific face parts for synthetic artifacts."
    )
    
    st.markdown("#### 📂 Training & Evaluation Dataset")
    st.write(
        "The model is trained on a subset of the **Kaggle Deepfake and Real Images Dataset**. The training "
        "pipeline uses a balanced partition of real/fake files, applying data augmentations like random rotation, "
        "flips, zoom, contrast, and brightness shifts to generalize synthetic boundary artifacts."
    )
    
    st.markdown("#### 🔍 Explainable AI (XAI) Purpose")
    st.write(
        "Through Grad-CAM, the dashboard superimposes class-activation gradients on the input image. This shows "
        "the exact spatial context of where the model detected anomalies (such as unnatural borders, blending shadows, "
        "or asymmetric textures) rather than forcing a blind classification decision."
    )
