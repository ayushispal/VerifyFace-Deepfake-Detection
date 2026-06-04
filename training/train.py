import os
import sys
import tensorflow as tf
import numpy as np
import random
import matplotlib.pyplot as plt

# Disable JIT compiler on CPU to avoid massive compilation overhead
tf.config.optimizer.set_jit(False)

# Set path roots
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

from models.efficientnet import build_model

def load_balanced_dataset(root_path, split, num_samples_per_class, shuffle=True):
    """
    Finds file paths and labels for split.
    Real = 0, Fake = 1
    """
    real_dir = os.path.join(root_path, split, 'Real')
    fake_dir = os.path.join(root_path, split, 'Fake')
    
    # Check if directories exist
    if not os.path.exists(real_dir) or not os.path.exists(fake_dir):
        print(f"Split {split} not fully found. Falling back to workspace Dataset split...")
        real_dir = os.path.join(BASE_DIR, 'Dataset', split, 'Real')
        fake_dir = os.path.join(BASE_DIR, 'Dataset', split, 'Fake')
        
    # If still not found, fallback to Test directory
    if not os.path.exists(real_dir) or len(os.listdir(real_dir)) == 0:
        print(f"Fallback: using Test split directories for {split}...")
        real_dir = os.path.join(root_path, 'Test', 'Real')
        fake_dir = os.path.join(root_path, 'Test', 'Fake')
        
    if not os.path.exists(real_dir):
        # Last fallback workspace Test
        real_dir = os.path.join(BASE_DIR, 'Dataset', 'Test', 'Real')
        fake_dir = os.path.join(BASE_DIR, 'Dataset', 'Test', 'Fake')

    real_files = [os.path.join(real_dir, f) for f in os.listdir(real_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    fake_files = [os.path.join(fake_dir, f) for f in os.listdir(fake_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    # Balance and sample
    random.seed(42)
    if len(real_files) > num_samples_per_class:
        real_files = random.sample(real_files, num_samples_per_class)
    if len(fake_files) > num_samples_per_class:
        fake_files = random.sample(fake_files, num_samples_per_class)
        
    print(f"Loaded balanced subset for {split}: Real={len(real_files)}, Fake={len(fake_files)}")
    
    all_files = real_files + fake_files
    # Real = 0, Fake = 1
    all_labels = [0.0] * len(real_files) + [1.0] * len(fake_files)
    
    if shuffle:
        combined = list(zip(all_files, all_labels))
        random.shuffle(combined)
        all_files, all_labels = zip(*combined)
        all_files = list(all_files)
        all_labels = list(all_labels)
        
    return all_files, all_labels

def parse_image(filename, label):
    """Loads, decodes, resizes, and pre-processes images."""
    image_string = tf.io.read_file(filename)
    image = tf.image.decode_jpeg(image_string, channels=3)
    image = tf.image.resize(image, [224, 224])
    image = tf.keras.applications.efficientnet_v2.preprocess_input(image)
    return image, label

def plot_and_save_curves(history1, history2, output_dir):
    """Plots and saves unified training history curves for accuracy, loss, and AUC."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Combine training history from Stage 1 and Stage 2
    loss = history1.history['loss'] + history2.history['loss']
    val_loss = history1.history['val_loss'] + history2.history['val_loss']
    
    accuracy = history1.history['accuracy'] + history2.history['accuracy']
    val_accuracy = history1.history['val_accuracy'] + history2.history['val_accuracy']
    
    auc_metric = history1.history['auc'] + history2.history['auc']
    val_auc = history1.history['val_auc'] + history2.history['val_auc']
    
    epochs_range = range(1, len(loss) + 1)
    stage1_epochs = len(history1.history['loss'])
    
    # 1. Loss Curve
    plt.figure(figsize=(8, 5))
    plt.plot(epochs_range, loss, label='Training Loss', color='#D32F2F', lw=2)
    plt.plot(epochs_range, val_loss, label='Validation Loss', color='#F44336', linestyle='--', lw=2)
    plt.axvline(x=stage1_epochs, color='gray', linestyle=':', label='Fine-tuning Started')
    plt.title('Model Loss Curve')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend(loc='upper right')
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'loss_curve.png'), dpi=150)
    plt.close()
    print("Saved loss curve to evaluation/loss_curve.png")
    
    # 2. Accuracy Curve
    plt.figure(figsize=(8, 5))
    plt.plot(epochs_range, accuracy, label='Training Accuracy', color='#1976D2', lw=2)
    plt.plot(epochs_range, val_accuracy, label='Validation Accuracy', color='#2196F3', linestyle='--', lw=2)
    plt.axvline(x=stage1_epochs, color='gray', linestyle=':', label='Fine-tuning Started')
    plt.title('Model Accuracy Curve')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'accuracy_curve.png'), dpi=150)
    plt.close()
    print("Saved accuracy curve to evaluation/accuracy_curve.png")
    
    # 3. AUC Curve
    plt.figure(figsize=(8, 5))
    plt.plot(epochs_range, auc_metric, label='Training AUC', color='#388E3C', lw=2)
    plt.plot(epochs_range, val_auc, label='Validation AUC', color='#4CAF50', linestyle='--', lw=2)
    plt.axvline(x=stage1_epochs, color='gray', linestyle=':', label='Fine-tuning Started')
    plt.title('Model AUC Curve')
    plt.xlabel('Epochs')
    plt.ylabel('AUC')
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'auc_curve.png'), dpi=150)
    plt.close()
    print("Saved AUC curve to evaluation/auc_curve.png")

def main():
    # Primary Source Dataset Path
    SOURCE_DIR = r"D:\deepfake detector\dataset"
    if not os.path.exists(SOURCE_DIR):
        SOURCE_DIR = os.path.join(BASE_DIR, "Dataset")
        
    print(f"Reading dataset from: {SOURCE_DIR}")
    
    # 1. Load subsets: 4000 training (2000 real, 2000 fake), 1000 validation (500 real, 500 fake)
    train_files, train_labels = load_balanced_dataset(SOURCE_DIR, 'Train', 2000, shuffle=True)
    val_files, val_labels = load_balanced_dataset(SOURCE_DIR, 'Validation', 500, shuffle=False)
    
    # 2. Build tf.data datasets (no caching to prevent memory crash)
    train_ds = tf.data.Dataset.from_tensor_slices((train_files, train_labels))
    train_ds = train_ds.map(parse_image, num_parallel_calls=tf.data.AUTOTUNE)
    train_ds = train_ds.batch(16).prefetch(tf.data.AUTOTUNE)
    
    val_ds = tf.data.Dataset.from_tensor_slices((val_files, val_labels))
    val_ds = val_ds.map(parse_image, num_parallel_calls=tf.data.AUTOTUNE)
    val_ds = val_ds.batch(16).prefetch(tf.data.AUTOTUNE)
    
    # 3. Create Model
    model, base_model = build_model()
    
    # 4. Stage 1: Freeze backbone, train classification head
    print("\n=== Stage 1: Training classifier head only (backbone frozen) ===")
    base_model.trainable = False
    
    model.compile(
        optimizer=tf.keras.optimizers.AdamW(learning_rate=0.0005),
        loss=tf.keras.losses.BinaryCrossentropy(label_smoothing=0.05),
        metrics=[
            'accuracy',
            tf.keras.metrics.Precision(name='precision'),
            tf.keras.metrics.Recall(name='recall'),
            tf.keras.metrics.AUC(name='auc')
        ]
    )
    
    model.summary()
    
    # Set up checkpoints directory
    save_dir = os.path.join(BASE_DIR, "saved_models")
    os.makedirs(save_dir, exist_ok=True)
    model_checkpoint_path = os.path.join(save_dir, "best_model.keras")
    
    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor='val_auc', mode='max', patience=8, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor='val_auc', mode='max', factor=0.5, patience=4, min_lr=1e-7),
        tf.keras.callbacks.ModelCheckpoint(filepath=model_checkpoint_path, monitor='val_auc', mode='max', save_best_only=True)
    ]
    
    print("\nStarting Stage 1 training...")
    history_stage1 = model.fit(
        train_ds,
        epochs=15,
        validation_data=val_ds,
        callbacks=callbacks
    )
    
    # 5. Stage 2: Fine-tuning - Unfreeze last 50 layers
    print("\n=== Stage 2: Fine-tuning backbone (last 50 layers unfrozen) ===")
    base_model.trainable = True
    
    # Freeze layers before the last 50
    for layer in base_model.layers[:-50]:
        layer.trainable = False
        
    model.compile(
        optimizer=tf.keras.optimizers.AdamW(learning_rate=0.00001),
        loss=tf.keras.losses.BinaryCrossentropy(label_smoothing=0.05),
        metrics=[
            'accuracy',
            tf.keras.metrics.Precision(name='precision'),
            tf.keras.metrics.Recall(name='recall'),
            tf.keras.metrics.AUC(name='auc')
        ]
    )
    
    print("\nStarting Stage 2 training...")
    history_stage2 = model.fit(
        train_ds,
        epochs=25,
        validation_data=val_ds,
        callbacks=callbacks
    )
    
    print(f"\nTraining completed. Best model saved to: {model_checkpoint_path}")
    
    # 6. Save unified training history curves
    plots_dir = os.path.join(BASE_DIR, "evaluation")
    plot_and_save_curves(history_stage1, history_stage2, plots_dir)

if __name__ == "__main__":
    main()
