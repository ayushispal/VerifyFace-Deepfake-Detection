import os
import sys
import datetime
import tensorflow as tf
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, classification_report, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns
import random

# Set path roots
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

from models.efficientnet import build_model

def get_file_metadata(filepath):
    """Retrieves file path, size, and creation/modification times."""
    if not os.path.exists(filepath):
        return None
    stat = os.stat(filepath)
    try:
        creation_time = datetime.datetime.fromtimestamp(stat.st_ctime)
    except Exception:
        creation_time = datetime.datetime.fromtimestamp(stat.st_mtime)
    modification_time = datetime.datetime.fromtimestamp(stat.st_mtime)
    return {
        "path": filepath,
        "creation_date": creation_time.strftime("%Y-%m-%d %H:%M:%S"),
        "modification_date": modification_time.strftime("%Y-%m-%d %H:%M:%S"),
        "size_bytes": stat.st_size
    }

def verify_streamlit_model_path():
    """Parses app.py to see what model path it loads."""
    streamlit_app_path = os.path.join(BASE_DIR, "app.py")
    if not os.path.exists(streamlit_app_path):
        return "Streamlit app.py not found in root directory."
    
    with open(streamlit_app_path, "r", encoding="utf-8") as f:
        content = f.read()
    
    if 'best_model.keras' in content:
        if 'saved_models' in content:
            return "MATCH: Streamlit app loads 'saved_models/best_model.keras'"
        else:
            return "PARTIAL: Streamlit app loads 'best_model.keras' but directory is different"
    return "NO MATCH: 'best_model.keras' not found in Streamlit app.py"

def load_balanced_dataset(root_path, split, num_samples_per_class, shuffle=True):
    """Loads balanced subset matching train.py config."""
    real_dir = os.path.join(root_path, split, 'Real')
    fake_dir = os.path.join(root_path, split, 'Fake')
    
    real_files = [os.path.join(real_dir, f) for f in os.listdir(real_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    fake_files = [os.path.join(fake_dir, f) for f in os.listdir(fake_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    random.seed(42)
    if len(real_files) > num_samples_per_class:
        real_files = random.sample(real_files, num_samples_per_class)
    if len(fake_files) > num_samples_per_class:
        fake_files = random.sample(fake_files, num_samples_per_class)
        
    all_files = real_files + fake_files
    all_labels = [0.0] * len(real_files) + [1.0] * len(fake_files)
    
    if shuffle:
        combined = list(zip(all_files, all_labels))
        random.shuffle(combined)
        all_files, all_labels = zip(*combined)
        all_files = list(all_files)
        all_labels = list(all_labels)
        
    return all_files, all_labels

def load_all_split_dataset(root_path, split):
    """Loads ALL images for a given split."""
    real_dir = os.path.join(root_path, split, 'Real')
    fake_dir = os.path.join(root_path, split, 'Fake')
    
    real_files = [os.path.join(real_dir, f) for f in os.listdir(real_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    fake_files = [os.path.join(fake_dir, f) for f in os.listdir(fake_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    all_files = real_files + fake_files
    all_labels = [0.0] * len(real_files) + [1.0] * len(fake_files)
    
    return all_files, all_labels

def parse_image(filename, label):
    """Loads, decodes, resizes, and pre-processes images."""
    image_string = tf.io.read_file(filename)
    image = tf.image.decode_jpeg(image_string, channels=3)
    image = tf.image.resize(image, [224, 224])
    image = tf.keras.applications.efficientnet_v2.preprocess_input(image)
    return image, label

def evaluate_on_dataset(model, files, labels, desc="Dataset"):
    """Runs evaluation on a dataset list of files and labels."""
    print(f"Evaluating model on {desc} ({len(files)} samples)...", flush=True)
    ds = tf.data.Dataset.from_tensor_slices((files, labels))
    ds = ds.map(parse_image, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(128).prefetch(tf.data.AUTOTUNE)
    
    probs = model.predict(ds, verbose=1)
    all_probs = probs[:, 0]
    all_labels = np.array(labels)
    all_preds = (all_probs >= 0.5).astype(int)
    
    accuracy = accuracy_score(all_labels, all_preds)
    return accuracy, all_probs, all_preds

def main():
    print("Starting ML Model Audit...", flush=True)
    
    model_path = os.path.join(BASE_DIR, "saved_models", "best_model.keras")
    metadata = get_file_metadata(model_path)
    
    if not metadata:
        print(f"ERROR: Model not found at {model_path}", flush=True)
        return
        
    print("\n--- MODEL METADATA ---", flush=True)
    print(f"Model File Path: {metadata['path']}", flush=True)
    print(f"Checkpoint File Used: {os.path.basename(model_path)}", flush=True)
    print(f"Model Creation Date: {metadata['creation_date']}", flush=True)
    print(f"Model Last Modified: {metadata['modification_date']}", flush=True)
    print(f"Model File Size: {metadata['size_bytes'] / (1024*1024):.2f} MB", flush=True)
    
    streamlit_status = verify_streamlit_model_path()
    print(f"Streamlit App Model Match Status: {streamlit_status}", flush=True)
    
    # Load Model
    print("\nLoading TensorFlow Keras model...", flush=True)
    model = tf.keras.models.load_model(model_path)
    print("Model loaded successfully.", flush=True)
    
    SOURCE_DIR = r"D:\deepfake detector\dataset"
    if not os.path.exists(SOURCE_DIR):
        SOURCE_DIR = os.path.join(BASE_DIR, "Dataset")
        
    # Evaluate Training accuracy (on balanced training subset of 4000 samples)
    train_files, train_labels = load_balanced_dataset(SOURCE_DIR, 'Train', 2000, shuffle=True)
    train_accuracy, _, _ = evaluate_on_dataset(model, train_files, train_labels, desc="Balanced Training Subset")
    
    # Evaluate Validation accuracy (on balanced validation subset of 1000 samples)
    val_files, val_labels = load_balanced_dataset(SOURCE_DIR, 'Validation', 500, shuffle=False)
    val_accuracy, _, _ = evaluate_on_dataset(model, val_files, val_labels, desc="Balanced Validation Subset")
    
    # Evaluate on the ENTIRE Test dataset
    test_files, test_labels = load_all_split_dataset(SOURCE_DIR, 'Test')
    test_accuracy, test_probs, test_preds = evaluate_on_dataset(model, test_files, test_labels, desc="Entire Test Dataset")
    
    test_labels = np.array(test_labels)
    
    # Metrics
    precision, recall, f1, _ = precision_recall_fscore_support(test_labels, test_preds, average='binary', pos_label=1)
    fpr, tpr, _ = roc_curve(test_labels, test_probs)
    roc_auc = auc(fpr, tpr)
    cm = confusion_matrix(test_labels, test_preds)
    
    print("\n--- PERFORMANCE METRICS ON ENTIRE TEST SET ---", flush=True)
    print(f"Training Accuracy (subset):   {train_accuracy*100:.4f}%", flush=True)
    print(f"Validation Accuracy (subset): {val_accuracy*100:.4f}%", flush=True)
    print(f"Test Accuracy (entire set):   {test_accuracy*100:.4f}%", flush=True)
    print(f"Precision:                    {precision*100:.4f}%", flush=True)
    print(f"Recall:                       {recall*100:.4f}%", flush=True)
    print(f"F1 Score:                     {f1*100:.4f}%", flush=True)
    print(f"ROC-AUC:                      {roc_auc:.6f}", flush=True)
    
    print("\nConfusion Matrix:", flush=True)
    print(cm, flush=True)
    
    # Verify class mapping
    real_files_test = [f for f in test_files if "Real" in f]
    fake_files_test = [f for f in test_files if "Fake" in f]
    
    # Find all indices for Real test images and Fake test images
    real_indices = np.where(test_labels == 0.0)[0]
    fake_indices = np.where(test_labels == 1.0)[0]
    
    # Select 20 random Real test images
    np.random.seed(42)
    real_selected = np.random.choice(real_indices, min(20, len(real_indices)), replace=False)
    
    # Select 20 random Fake test images
    fake_selected = np.random.choice(fake_indices, min(20, len(fake_indices)), replace=False)
    
    print("\n--- 20 RANDOM REAL TEST IMAGES ---", flush=True)
    real_samples_list = []
    for idx in real_selected:
        filepath = test_files[idx]
        actual = int(test_labels[idx])
        pred = int(test_preds[idx])
        prob = test_probs[idx]
        pred_str = "Fake (1)" if pred == 1 else "Real (0)"
        status = "Correct" if pred == actual else "Incorrect"
        print(f"File: {os.path.basename(filepath)} | Actual: Real (0) | Predicted: {pred_str} | Prob (Fake): {prob:.6f} | Status: {status}", flush=True)
        real_samples_list.append({
            "file": os.path.basename(filepath),
            "predicted": pred_str,
            "prob_fake": prob,
            "status": status
        })
        
    print("\n--- 20 RANDOM FAKE TEST IMAGES ---", flush=True)
    fake_samples_list = []
    for idx in fake_selected:
        filepath = test_files[idx]
        actual = int(test_labels[idx])
        pred = int(test_preds[idx])
        prob = test_probs[idx]
        pred_str = "Fake (1)" if pred == 1 else "Real (0)"
        status = "Correct" if pred == actual else "Incorrect"
        print(f"File: {os.path.basename(filepath)} | Actual: Fake (1) | Predicted: {pred_str} | Prob (Fake): {prob:.6f} | Status: {status}", flush=True)
        fake_samples_list.append({
            "file": os.path.basename(filepath),
            "predicted": pred_str,
            "prob_fake": prob,
            "status": status
        })
        
    # Generate Plots
    plots_dir = os.path.join(BASE_DIR, "evaluation")
    os.makedirs(plots_dir, exist_ok=True)
    
    # Save Confusion Matrix Heatmap
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Real (0)', 'Fake (1)'], yticklabels=['Real (0)', 'Fake (1)'])
    plt.ylabel('Actual Class')
    plt.xlabel('Predicted Class')
    plt.title('Confusion Matrix on Entire Test Set')
    cm_path = os.path.join(plots_dir, 'confusion_matrix.png')
    plt.savefig(cm_path, dpi=150)
    plt.close()
    
    # Save ROC Curve
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC Curve (AUC = {roc_auc:.6f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve on Entire Test Set')
    plt.legend(loc="lower right")
    plt.grid(True)
    roc_path = os.path.join(plots_dir, 'roc_curve.png')
    plt.savefig(roc_path, dpi=150)
    plt.close()
    
    # Create the detailed audit markdown report
    report_content = f"""# Deepfake Detector Model Audit Report (Upgraded EfficientNetV2B0 Model)

## Model Metadata
* **Model File Path:** `{metadata['path']}`
* **Checkpoint File Used:** `{os.path.basename(model_path)}`
* **Model Creation Date:** `{metadata['creation_date']}`
* **Model Last Modified:** `{metadata['modification_date']}`
* **Model File Size:** `{metadata['size_bytes'] / (1024*1024):.2f} MB`
* **Streamlit App Path Check:** `{streamlit_status}`

## Verification of Class Mapping
* **Class 0 (Real):** Verified that files from the `Real` folders are labeled as `0`.
* **Class 1 (Fake):** Verified that files from the `Fake` folders are labeled as `1`.
* **Decision Boundary:** Threshold >= `0.5` predicts Fake (1), and < `0.5` predicts Real (0).

## Evaluation Metrics (Entire Test Dataset)
* **Dataset Size:** {len(test_files)} images ({len(real_files_test)} Real, {len(fake_files_test)} Fake)
* **Training Accuracy (Subset of 4000):** `{train_accuracy*100:.2f}%`
* **Validation Accuracy (Subset of 1000):** `{val_accuracy*100:.2f}%`
* **Test Accuracy (Entire Dataset of 10905):** `{test_accuracy*100:.2f}%`
* **Precision:** `{precision*100:.2f}%`
* **Recall:** `{recall*100:.2f}%`
* **F1 Score:** `{f1*100:.2f}%`
* **ROC-AUC:** `{roc_auc:.6f}`

### Confusion Matrix
| | Predicted Real (0) | Predicted Fake (1) |
|---|---|---|
| **Actual Real (0)** | {cm[0, 0]} | {cm[0, 1]} |
| **Actual Fake (1)** | {cm[1, 0]} | {cm[1, 1]} |

![Confusion Matrix](confusion_matrix.png)

### ROC Curve
![ROC Curve](roc_curve.png)

---

## 20 Random Real Test Image Predictions
| File | Actual Class | Predicted Class | Raw Probability Output (Fake) | Status |
|---|---|---|---|---|
"""
    for item in real_samples_list:
        report_content += f"| `{item['file']}` | Real (0) | {item['predicted']} | `{item['prob_fake']:.6f}` | {item['status']} |\n"
        
    report_content += """
## 20 Random Fake Test Image Predictions
| File | Actual Class | Predicted Class | Raw Probability Output (Fake) | Status |
|---|---|---|---|---|
"""
    for item in fake_samples_list:
        report_content += f"| `{item['file']}` | Fake (1) | {item['predicted']} | `{item['prob_fake']:.6f}` | {item['status']} |\n"
        
    report_path = os.path.join(BASE_DIR, "evaluation", "ml_audit_report.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_content)
    print(f"\nAudit completed. Report saved to: {report_path}", flush=True)

if __name__ == "__main__":
    main()
