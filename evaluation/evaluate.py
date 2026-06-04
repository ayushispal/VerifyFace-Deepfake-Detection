import os
import sys
import tensorflow as tf
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, classification_report, roc_curve, auc
import matplotlib.pyplot as plt
import random

# Set path roots
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

def load_test_dataset(root_path, num_samples_per_class=500):
    """Loads a balanced test set of file paths and labels."""
    real_dir = os.path.join(root_path, 'Test', 'Real')
    fake_dir = os.path.join(root_path, 'Test', 'Fake')
    
    # Check if directories exist
    if not os.path.exists(real_dir) or not os.path.exists(fake_dir):
        real_dir = os.path.join(BASE_DIR, 'Dataset', 'Test', 'Real')
        fake_dir = os.path.join(BASE_DIR, 'Dataset', 'Test', 'Fake')
        
    real_files = [os.path.join(real_dir, f) for f in os.listdir(real_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    fake_files = [os.path.join(fake_dir, f) for f in os.listdir(fake_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    random.seed(42)
    if len(real_files) > num_samples_per_class:
        real_files = random.sample(real_files, num_samples_per_class)
    if len(fake_files) > num_samples_per_class:
        fake_files = random.sample(fake_files, num_samples_per_class)
        
    print(f"Loaded balanced Test subset: Real={len(real_files)}, Fake={len(fake_files)}")
    
    all_files = real_files + fake_files
    # Real = 0, Fake = 1
    all_labels = [0] * len(real_files) + [1] * len(fake_files)
    
    return all_files, all_labels

def parse_image(filename, label):
    """Loads, decodes, resizes, and pre-processes test images."""
    image_string = tf.io.read_file(filename)
    image = tf.image.decode_jpeg(image_string, channels=3)
    image = tf.image.resize(image, [224, 224])
    image = tf.keras.applications.efficientnet_v2.preprocess_input(image)
    return image, label

def evaluate():
    print("Evaluating trained model on Test dataset...")
    
    # Check model path
    model_path = os.path.join(BASE_DIR, "saved_models", "best_model.keras")
    if not os.path.exists(model_path):
        print(f"ERROR: Trained model checkpoint not found at {model_path}.")
        return False
        
    # 1. Load Model
    print(f"Loading Keras model from: {model_path}")
    model = tf.keras.models.load_model(model_path)
    
    # 2. Dataset path
    SOURCE_DIR = r"D:\deepfake detector\dataset"
    if not os.path.exists(SOURCE_DIR):
        SOURCE_DIR = os.path.join(BASE_DIR, "Dataset")
        
    test_files, test_labels = load_test_dataset(SOURCE_DIR, num_samples_per_class=500)
    
    # 3. Build Dataset
    test_ds = tf.data.Dataset.from_tensor_slices((test_files, test_labels))
    test_ds = test_ds.map(parse_image, num_parallel_calls=tf.data.AUTOTUNE)
    test_ds = test_ds.batch(32).prefetch(tf.data.AUTOTUNE)
    
    # 4. Predict
    print("Running model inference on test images...")
    all_probs = []
    all_labels = []
    
    for images, labels in test_ds:
        probs = model(images, training=False).numpy()
        all_probs.extend(probs[:, 0])
        all_labels.extend(labels.numpy())
        
    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)
    
    # Threshold predictions (>=0.5 -> 1 (Fake), <0.5 -> 0 (Real))
    all_preds = (all_probs >= 0.5).astype(int)
    
    # 5. Calculate Metrics
    accuracy = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='binary', pos_label=1)
    
    # ROC-AUC
    fpr, tpr, _ = roc_curve(all_labels, all_probs)
    roc_auc = auc(fpr, tpr)
    
    print("\n================ EVALUATION METRICS ================")
    print(f"Accuracy:  {accuracy*100:.2f}%")
    print(f"Precision: {precision*100:.2f}%")
    print(f"Recall:    {recall*100:.2f}%")
    print(f"F1-Score:  {f1*100:.2f}%")
    print(f"ROC-AUC:   {roc_auc:.4f}")
    print("====================================================")
    
    # Confusion Matrix
    cm = confusion_matrix(all_labels, all_preds)
    print("\nConfusion Matrix:")
    print(cm)
    
    # Classification Report
    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, target_names=['Real', 'Fake']))
    
    # Create evaluation outputs folder
    plots_dir = os.path.join(BASE_DIR, "evaluation")
    os.makedirs(plots_dir, exist_ok=True)
    
    # Save ROC Curve Plot
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC Curve (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) - Deepfake Detector')
    plt.legend(loc="lower right")
    plt.grid(True)
    plot_path = os.path.join(plots_dir, 'roc_curve.png')
    plt.savefig(plot_path)
    plt.close()
    print(f"\nSaved ROC Curve plot to: {plot_path}")
    
    # Save Confusion Matrix Plot
    plt.figure(figsize=(6, 5))
    import seaborn as sns
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Real', 'Fake'], yticklabels=['Real', 'Fake'])
    plt.ylabel('Actual Label')
    plt.xlabel('Predicted Label')
    plt.title('Confusion Matrix')
    cm_path = os.path.join(plots_dir, 'confusion_matrix.png')
    plt.savefig(cm_path)
    plt.close()
    print(f"Saved Confusion Matrix plot to: {cm_path}")
    return True

if __name__ == "__main__":
    evaluate()
