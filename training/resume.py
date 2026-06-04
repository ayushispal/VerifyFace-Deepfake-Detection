import os
import sys
import tensorflow as tf

# Disable JIT compiler to avoid CPU compile overhead
tf.config.optimizer.set_jit(False)

# Set path roots
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

from training.train import load_balanced_dataset, parse_image

def resume_training():
    model_checkpoint_path = os.path.join(BASE_DIR, "saved_models", "best_model.keras")
    if not os.path.exists(model_checkpoint_path):
        print(f"ERROR: No checkpoint found at {model_checkpoint_path}")
        return
        
    print(f"Loading checkpoint from: {model_checkpoint_path}")
    model = tf.keras.models.load_model(model_checkpoint_path)
    base_model = model.get_layer("efficientnetv2-b0")
    print("Checkpoint loaded successfully.")
    
    # Primary Source Dataset Path
    SOURCE_DIR = r"D:\deepfake detector\dataset"
    if not os.path.exists(SOURCE_DIR):
        SOURCE_DIR = os.path.join(BASE_DIR, "Dataset")
        
    print(f"Reading dataset from: {SOURCE_DIR}")
    train_files, train_labels = load_balanced_dataset(SOURCE_DIR, 'Train', 2000, shuffle=True)
    val_files, val_labels = load_balanced_dataset(SOURCE_DIR, 'Validation', 500, shuffle=False)
    
    # Build tf.data datasets (no caching to prevent memory crash)
    train_ds = tf.data.Dataset.from_tensor_slices((train_files, train_labels))
    train_ds = train_ds.map(parse_image, num_parallel_calls=tf.data.AUTOTUNE)
    train_ds = train_ds.batch(16).prefetch(tf.data.AUTOTUNE)
    
    val_ds = tf.data.Dataset.from_tensor_slices((val_files, val_labels))
    val_ds = val_ds.map(parse_image, num_parallel_calls=tf.data.AUTOTUNE)
    val_ds = val_ds.batch(16).prefetch(tf.data.AUTOTUNE)
    
    # Configure Stage 2 (Fine-tuning)
    print("\n=== Resuming Stage 2: Fine-tuning backbone ===")
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
    
    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor='val_auc', mode='max', patience=8, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor='val_auc', mode='max', factor=0.5, patience=4, min_lr=1e-7),
        tf.keras.callbacks.ModelCheckpoint(filepath=model_checkpoint_path, monitor='val_auc', mode='max', save_best_only=True)
    ]
    
    # Remaining epochs for Stage 2 (out of 25)
    remaining_epochs = 20
    print(f"Resuming Stage 2 fine-tuning for {remaining_epochs} epochs...")
    
    model.fit(
        train_ds,
        epochs=remaining_epochs,
        validation_data=val_ds,
        callbacks=callbacks
    )
    print("Resumed training completed successfully.")

if __name__ == "__main__":
    resume_training()
