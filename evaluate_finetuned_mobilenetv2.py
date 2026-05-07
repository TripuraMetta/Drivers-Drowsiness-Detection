# =====================================================
# evaluate_finetuned_mobilenetv2.py
# =====================================================
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from preprocessing import val_ds
import pickle

# =====================================================
# Step 1: Load Fine-Tuned Model
# =====================================================
model = load_model("mobilenetv2_finetuned_model.h5")
print("\n✅ Fine-tuned MobileNetV2 model loaded successfully!")

# =====================================================
# Step 2: Evaluate on Validation Data
# =====================================================
loss, acc = model.evaluate(val_ds, verbose=1)
print(f"\n📊 Validation Accuracy: {acc*100:.2f}%")
print(f"📉 Validation Loss: {loss:.4f}")

# =====================================================
# Step 3: Load Training History
# =====================================================
try:
    with open("mobilenetv2_finetuned_history.pkl", "rb") as f:
        history = pickle.load(f)
except:
    history = None
    print("\n⚠️ Training history not found. Skipping plots.")

# =====================================================
# Step 4: Plot Graphs
# =====================================================
if history:
    plt.figure(figsize=(12,5))

    # Accuracy
    plt.subplot(1,2,1)
    plt.plot(history["accuracy"], label="Train Accuracy")
    plt.plot(history["val_accuracy"], label="Val Accuracy")
    plt.title("Fine-Tuned MobileNetV2 Accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()

    # Loss
    plt.subplot(1,2,2)
    plt.plot(history["loss"], label="Train Loss")
    plt.plot(history["val_loss"], label="Val Loss")
    plt.title("Fine-Tuned MobileNetV2 Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()

    plt.tight_layout()
    plt.show()
