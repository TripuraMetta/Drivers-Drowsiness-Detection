# =====================================================
# 1_preprocessing.py
# =====================================================
# Step 1: Import Libraries
# =====================================================
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# =====================================================
# Step 2: Dataset Parameters
# =====================================================
dataset_path = "Driver Drowsiness Dataset (DDD)"  # ⚠️ update if folder name is different
img_height, img_width = 224, 224
batch_size = 32

# =====================================================
# Step 3: Create ImageDataGenerators
# =====================================================
datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,   # 80% train, 20% validation
    rotation_range=20,
    zoom_range=0.2,
    horizontal_flip=True
)

# =====================================================
# Step 4: Load Training Data
# =====================================================
train_ds = datagen.flow_from_directory(
    dataset_path,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode="binary",
    subset="training",
    shuffle=True
)

# =====================================================
# Step 5: Load Validation Data
# =====================================================
val_ds = datagen.flow_from_directory(
    dataset_path,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode="binary",
    subset="validation",
    shuffle=False
)

# =====================================================
# Step 6: Print Summary
# =====================================================
print("\nClasses found:", train_ds.class_indices)
print("Number of training batches:", len(train_ds))
print("Number of validation batches:", len(val_ds))
print("Sample image shape:", train_ds[0][0].shape)

# =====================================================
# Step 7: Export datasets for training & evaluation
# =====================================================
# (already named correctly as train_ds and val_ds, nothing else needed)
