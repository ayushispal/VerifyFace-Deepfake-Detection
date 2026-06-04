🛡️ VerifyFace: Explainable Deepfake Detection System
AI-powered Deepfake Detection using Computer Vision and Explainable Artificial Intelligence
EfficientNetV2 | FastAPI | Streamlit | Grad-CAM | Deep Learning

📌 Overview

VerifyFace is an end-to-end Explainable AI based Deepfake Detection System that detects whether a facial image is Real or AI-generated using advanced deep learning and computer vision techniques.

The system uses an EfficientNetV2 deep learning architecture trained with transfer learning to classify manipulated facial images. Unlike traditional black-box AI models, VerifyFace integrates Grad-CAM visualization to highlight the important facial regions considered by the model while making predictions.

It provides a complete machine learning pipeline including model training, evaluation, explainable AI visualization, backend API serving, and an interactive frontend dashboard.

🌟 Features

✨ Real vs Fake image classification using Deep Learning

🧠 EfficientNetV2 CNN architecture with Transfer Learning

🔥 Grad-CAM Explainable AI heatmap visualization

⚡ FastAPI backend for high-performance model inference

🎨 Interactive Streamlit web dashboard

📊 Confidence-based prediction system

⚖️ Threshold optimized model calibration

❓ Uncertainty handling for difficult predictions

📈 Complete evaluation and audit workflow

🛠️ Tech Stack
Layer	Technology
Programming Language	Python
Deep Learning	TensorFlow, Keras, EfficientNetV2
Computer Vision	OpenCV
Explainable AI	Grad-CAM
Backend	FastAPI, Uvicorn
Frontend	Streamlit
Data Processing	NumPy, Pandas
Evaluation	Scikit-learn, Matplotlib
Version Control	Git, GitHub
📊 Model Performance
Metric	Result
Accuracy	80.58%
ROC-AUC Score	92.42%
F1 Score	81.71%
Optimized Threshold	0.65
📁 Project Structure

VerifyFace-Deepfake-Detection/

├── backend/
│ └── main.py
│
├── frontend/
│ └── app.py
│
├── models/
│ └── efficientnet.py
│
├── training/
│ ├── train.py
│ └── resume.py
│
├── evaluation/
│ ├── evaluate.py
│ └── audit.py
│
├── explainability/
│ └── gradcam.py
│
├── requirements.txt
└── README.md

🚀 Getting Started
Clone Repository

git clone https://github.com/ayushispal/VerifyFace-Deepfake-Detection.git

Move into project folder:

cd VerifyFace-Deepfake-Detection

Install dependencies:

pip install -r requirements.txt

▶️ Run Application

Start FastAPI backend:

uvicorn backend.main:app --reload

Backend:

http://127.0.0.1:8000

API Documentation:

http://127.0.0.1:8000/docs

Start Streamlit frontend:

streamlit run frontend/app.py

Frontend:

http://localhost:8501

🧠 How It Works

1️⃣ User uploads a facial image through the Streamlit dashboard.

2️⃣ Image is sent to the FastAPI backend.

3️⃣ Image preprocessing is applied.

4️⃣ EfficientNetV2 model analyzes facial features.

5️⃣ Model predicts:

REAL Image
FAKE Image
UNCERTAIN (Manual Review)

6️⃣ Grad-CAM generates explainability heatmaps.

7️⃣ Prediction, confidence score, and visualization are displayed.

🔥 Explainable AI (Grad-CAM)

Deep learning models are often difficult to interpret.

VerifyFace solves this using Gradient-weighted Class Activation Mapping (Grad-CAM).

Grad-CAM highlights the regions of an image that influenced the model's decision, making predictions more transparent and trustworthy.

Outputs:

✔ Original Image
✔ Activation Heatmap
✔ Explainability Overlay

📚 Dataset

Dataset Used:

Deepfake and Real Images Dataset

The dataset contains real and synthetically generated facial images used for training, validation, and testing.

🔬 Research Concepts Implemented

✔ Computer Vision
✔ Deep Learning
✔ CNN Architecture
✔ Transfer Learning
✔ Fine Tuning
✔ Explainable AI (XAI)
✔ Grad-CAM Visualization
✔ Threshold Calibration
✔ Model Evaluation
✔ Error Analysis

🚀 Future Enhancements

🔹 Video Deepfake Detection

🔹 Real-time Webcam Detection

🔹 Vision Transformer Integration

🔹 Ensemble Learning

🔹 Cloud Deployment
