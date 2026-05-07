# =====================================================
# fine_tune_mobilenetv2.py
# =====================================================
import tensorflow as tf
from tensorflow.keras import layers, models
from preprocessing import train_ds, val_ds
import pickle

# =====================================================
# Step 1: Load MobileNetV2 Base Model
# =====================================================
base_model = tf.keras.applications.MobileNetV2(
    input_shape=(224, 224, 3),
    include_top=False,
    weights="imagenet"
)

# Allow fine-tuning
base_model.trainable = True

# Freeze all layers except the last 50
for layer in base_model.layers[:-50]:
    layer.trainable = False

print("✅ MobileNetV2 base model loaded and last 50 layers set trainable!")

# =====================================================
# Step 2: Build Model
# =====================================================
model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(128, activation="relu"),
    layers.Dropout(0.3),
    layers.Dense(1, activation="sigmoid")
])

# Compile with smaller learning rate for fine-tuning
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# =====================================================
# Step 3: Train Model
# =====================================================
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=10
)

# =====================================================
# Step 4: Save Fine-Tuned Model & History
# =====================================================
model.save("mobilenetv2_finetuned_model.h5")
with open("mobilenetv2_finetuned_history.pkl", "wb") as f:
    pickle.dump(history.history, f)

print("\n✅ Fine-tuned MobileNetV2 model saved as mobilenetv2_finetuned_model.h5")
print("📂 Training history saved as mobilenetv2_finetuned_history.pkl")
