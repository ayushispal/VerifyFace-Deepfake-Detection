import tensorflow as tf

def build_model():
    """
    Creates an EfficientNetV2B3-based binary classifier model with built-in data augmentation.
    Returns:
        model: Compiled tf.keras.Model
        base_model: The pre-trained backbone Model
    """
    # Load pretrained EfficientNetV2B0 backbone without top classification head
    base_model = tf.keras.applications.EfficientNetV2B0(
        weights='imagenet',
        include_top=False,
        input_shape=(224, 224, 3),
        name="efficientnetv2-b0"
    )
    
    # Strong data augmentation sequential block
    data_augmentation = tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(0.15),
        tf.keras.layers.RandomZoom(0.2),
        tf.keras.layers.RandomContrast(0.2),
        tf.keras.layers.RandomBrightness(0.2)
    ], name="data_augmentation")
    
    # Model structure definition
    inputs = tf.keras.Input(shape=(224, 224, 3))
    
    # Pass inputs through data augmentation (automatically disabled during inference/evaluation)
    x = data_augmentation(inputs)
    
    # Pass through base backbone model
    x = base_model(x)
    
    # Classification Head
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.BatchNormalization()(x)
    
    # Dense 512 + Dropout 0.5
    x = tf.keras.layers.Dense(512, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    
    # Dense 128 + Dropout 0.3
    x = tf.keras.layers.Dense(128, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    
    # Dense 1 with sigmoid for binary classification (Real=0, Fake=1)
    outputs = tf.keras.layers.Dense(1, activation='sigmoid')(x)
    
    model = tf.keras.Model(inputs, outputs, name="deepfake_detector")
    return model, base_model
