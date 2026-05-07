# =====================================================
# enhanced_preprocessing.py
# =====================================================
import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Dataset path
dataset_path = "Driver Drowsiness Dataset (DDD)"
img_height, img_width = 224, 224
batch_size = 32

# =====================================================
# Create ImageDataGenerator with enhanced augmentation
# =====================================================
datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,
    rotation_range=30,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.2,
    shear_range=0.1,
    horizontal_flip=True,
    brightness_range=[0.7, 1.3]
)

# Training data
train_ds = datagen.flow_from_directory(
    dataset_path,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode="binary",
    subset="training",
    shuffle=True
)

# Validation data
val_ds = datagen.flow_from_directory(
    dataset_path,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode="binary",
    subset="validation",
    shuffle=False
)

print("\n✅ Enhanced data augmentation applied successfully!")
