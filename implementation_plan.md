# Implementation Plan - Rebuilding Deepfake Training Pipeline for High Accuracy

This plan details the upgrade of the deepfake detection model from `EfficientNetB3` (224x224 input) to `EfficientNetV2B3` (300x300 input) along with associated training and inference pipeline improvements.

## Proposed Changes

### 1. Model Architecture Setup

#### [MODIFY] [efficientnet.py](file:///d:/deepfake_detector/models/efficientnet.py)
* Rebuild model using `tf.keras.applications.EfficientNetV2B3`.
* Change `input_shape` to `(300, 300, 3)`.
* Add native Keras sequential data augmentation layers at the model's entry point:
  - `RandomFlip("horizontal")`
  - `RandomRotation(0.15)`
  - `RandomZoom(0.2)`
  - `RandomContrast(0.2)`
  - `RandomBrightness(0.2)`
* Structure the output classification head:
  - `GlobalAveragePooling2D()`
  - `BatchNormalization()`
  - `Dense(512, activation='relu')`
  - `Dropout(0.6)`
  - `Dense(256, activation='relu')`
  - `Dropout(0.4)`
  - `Dense(1, activation='sigmoid')`

### 2. Training Strategy Setup

#### [MODIFY] [train.py](file:///d:/deepfake_detector/training/train.py)
* Update image resize size in `parse_image` to `[300, 300]`.
* Change preprocessing function to `tf.keras.applications.efficientnet_v2.preprocess_input`.
* Configure model compilation with `AdamW` optimizer and `BinaryCrossentropy(label_smoothing=0.05)`.
* Define two-stage training strategy:
  - **Stage 1**: Freeze backbone (`base_model.trainable = False`), train for 20 epochs at learning rate `0.0005`.
  - **Stage 2 (Fine-tuning)**: Unfreeze last 100 layers of the backbone, train for 40 epochs at learning rate `0.00001`.
* Add callbacks:
  - `EarlyStopping(monitor='val_auc', mode='max', patience=8, restore_best_weights=True)`
  - `ReduceLROnPlateau(monitor='val_auc', mode='max', factor=0.5, patience=4)`
  - `ModelCheckpoint(filepath=model_checkpoint_path, monitor='val_auc', mode='max', save_best_only=True)`
* Compile metrics to track: `accuracy`, `precision`, `recall`, `auc`.

### 3. Application and Test Preprocessing Integration

To prevent input shape mismatch errors at runtime (from changing model input size to `300, 300` and model class to `EfficientNetV2B3`), we must align the preprocessing and image resizing in the backend inference files.

#### [MODIFY] [app.py](file:///d:/deepfake_detector/app.py)
* Update image resize to `(300, 300)`.
* Update preprocessing to `tf.keras.applications.efficientnet_v2.preprocess_input`.

#### [MODIFY] [app.py](file:///d:/deepfake_detector/app/app.py)
* Update image resize to `(300, 300)`.
* Update preprocessing to `tf.keras.applications.efficientnet_v2.preprocess_input`.

#### [MODIFY] [evaluate.py](file:///d:/deepfake_detector/evaluation/evaluate.py)
* Update image resize to `[300, 300]`.
* Update preprocessing to `tf.keras.applications.efficientnet_v2.preprocess_input`.

---

## Verification Plan

### Automated Tests
- Run `python training/train.py` to train and save the model.
- Run `python evaluation/evaluate.py` to calculate accuracy, precision, recall, F1 score, and ROC-AUC on the full Test dataset.
- Confirm both validation accuracy and test accuracy achieve >90%.
- Generate final confusion matrix and ROC curve plots.
