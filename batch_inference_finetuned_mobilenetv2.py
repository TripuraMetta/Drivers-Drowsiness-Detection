# =====================================================
# batch_inference_finetuned_mobilenetv2.py
# =====================================================

import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os

# =====================================================
# Step 1: Load Fine-Tuned Model
# =====================================================
model_path = "mobilenetv2_finetuned_model.h5"
model = load_model(model_path)
print("✅ Fine-tuned MobileNetV2 model loaded successfully!")

# =====================================================
# Step 2: Define Class Labels
# =====================================================
class_labels = {0: "Drowsy", 1: "Non-Drowsy"}

# =====================================================
# Step 3: Folder containing test images
# =====================================================
test_folder = "test_images"  # replace with your folder path
image_files = [f for f in os.listdir(test_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

if not image_files:
    print("⚠️ No images found in the folder!")
    exit()

# =====================================================
# Step 4: Loop through images and predict
# =====================================================
for img_file in image_files:
    img_path = os.path.join(test_folder, img_file)
    
    # Load & preprocess image
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    # Predict
    pred_prob = model.predict(img_array)[0][0]
    pred_class = class_labels[1] if pred_prob >= 0.5 else class_labels[0]
    
    # Display result
    print(f"🖼️ {img_file} -> Prediction: {pred_class}, Probability: {pred_prob:.4f}")
