# =====================================================
# inference_finetuned_mobilenetv2.py
# =====================================================

import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os

# =====================================================
# Step 1: Load Fine-Tuned Model
# =====================================================
model_path = "mobilenetv2_finetuned_model.h5"  # make sure this file exists
model = load_model(model_path)
print("✅ Fine-tuned MobileNetV2 model loaded successfully!")

# =====================================================
# Step 2: Define Class Labels
# =====================================================
class_labels = {0: "Drowsy", 1: "Non-Drowsy"}

# =====================================================
# Step 3: Load and Preprocess Image
# =====================================================
# Replace 'test_image.jpg' with the path to your image
img_path = "Driver Drowsiness Dataset (DDD)/Drowsy/M0702.png"  

# Load image with target size 224x224 (MobileNetV2 input)
img = image.load_img(img_path, target_size=(224, 224))
img_array = image.img_to_array(img)
img_array = img_array / 255.0        # Normalize
img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

# =====================================================
# Step 4: Predict
# =====================================================
pred_prob = model.predict(img_array)[0][0]

# Since binary classification with sigmoid, threshold 0.5
if pred_prob >= 0.5:
    pred_class = class_labels[1]
else:
    pred_class = class_labels[0]

print(f"\n🖼️ Image: {os.path.basename(img_path)}")
print(f"🔮 Prediction: {pred_class}")
print(f"📊 Probability: {pred_prob:.4f}")
