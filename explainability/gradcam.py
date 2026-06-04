import cv2
import numpy as np
import tensorflow as tf

class GradCAM:
    def __init__(self, model, base_model):
        """
        Inits the Keras-based Grad-CAM solver.
        Args:
            model: Compiled tf.keras.Model
            base_model: The pre-trained backbone layer/Model
        """
        self.model = model
        self.base_model = base_model
        # EfficientNet last conv/activation layer is 'top_activation'
        self.last_conv_layer_name = 'top_activation'

    def __call__(self, img_array, pred_label=None):
        """
        Computes the Grad-CAM activation heatmap.
        Args:
            img_array: Preprocessed image batch of shape (1, 224, 224, 3)
            pred_label: Optional label index. Defaults to predicted label.
        Returns:
            heatmap: Normalised 2D numpy array (224x224)
            pred_label: Predicted or target class label (0=Real, 1=Fake)
        """
        # Retrieve the backbone conv activation layer output
        conv_output = self.base_model.get_layer(self.last_conv_layer_name).output
        sub_base_model = tf.keras.Model(inputs=self.base_model.inputs, outputs=conv_output)
        
        # Convert image array to tensor
        img_tensor = tf.convert_to_tensor(img_array, dtype=tf.float32)
        
        with tf.GradientTape() as tape:
            # 1. Forward pass through base backbone model
            conv_activations = sub_base_model(img_tensor)
            
            # 2. Forward pass through classification head
            x = conv_activations
            backbone_idx = 0
            for i, layer in enumerate(self.model.layers):
                if layer == self.base_model or getattr(layer, 'name', '') == self.base_model.name:
                    backbone_idx = i
                    break
                    
            for layer in self.model.layers[backbone_idx+1:]:
                x = layer(x, training=False)
            preds = x
            
            # For binary classification: outputs are shape (1, 1) sigmoidal
            # Real = 0 (preds closer to 0), Fake = 1 (preds closer to 1)
            if pred_label is None:
                if preds[0][0] >= 0.5:
                    class_channel = preds[:, 0]
                    pred_label = 1
                else:
                    class_channel = 1.0 - preds[:, 0]
                    pred_label = 0
            else:
                if pred_label == 1:
                    class_channel = preds[:, 0]
                else:
                    class_channel = 1.0 - preds[:, 0]
                    
        # Gradient of predicted class w.r.t backbone conv activations
        grads = tape.gradient(class_channel, conv_activations)
        
        # Global Average Pooling of gradients
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        
        # Weight the conv activations by the pooled gradients
        conv_activations = conv_activations[0]
        heatmap = conv_activations @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)
        
        # ReLU to keep only positive contributions and normalize
        heatmap = tf.maximum(heatmap, 0.0)
        max_val = tf.reduce_max(heatmap)
        if max_val == 0.0:
            max_val = 1e-10
        heatmap /= max_val
        
        return heatmap.numpy(), pred_label

def overlay_heatmap(img_path, heatmap, alpha=0.45):
    """
    Superimposes the Grad-CAM activation heatmap onto the original image.
    Returns:
        img_resized: Resize of original (224x224)
        heatmap_colored: Colored 8-bit heatmap
        overlay: Blended image
    """
    img = cv2.imread(img_path)
    if img is None:
        raise ValueError(f"Could not load image at {img_path}")
        
    img_resized = cv2.resize(img, (224, 224))
    
    # Resize and normalize heatmap
    heatmap_resized = cv2.resize(heatmap, (img_resized.shape[1], img_resized.shape[0]))
    heatmap_colored = np.uint8(255 * heatmap_resized)
    heatmap_colored = cv2.applyColorMap(heatmap_colored, cv2.COLORMAP_JET)
    
    # Blended output
    overlay = cv2.addWeighted(img_resized, 1 - alpha, heatmap_colored, alpha, 0)
    return img_resized, heatmap_colored, overlay
