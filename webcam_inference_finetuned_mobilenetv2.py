# =====================================================
# webcam_inference_finetuned_mobilenetv2.py
# =====================================================
import cv2
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np

# =====================================================
# Step 1: Load Fine-Tuned Model
# =====================================================
model_path = "mobilenetv2_finetuned_model.h5"
model = load_model(model_path)
print("✅ Fine-tuned MobileNetV2 model loaded successfully!")

# =====================================================
# Step 2: Class Labels
# =====================================================
class_labels = {0: "Drowsy", 1: "Non-Drowsy"}

# =====================================================
# Step 3: Load Face Detector (Haar Cascade)
# =====================================================
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# =====================================================
# Step 4: Open Webcam
# =====================================================
cap = cv2.VideoCapture(0)  # 0 = default webcam

if not cap.isOpened():
    print("⚠️ Cannot access webcam")
    exit()

print("🎥 Webcam started. Press 'q' to quit.")

# =====================================================
# Step 5: Real-Time Detection Loop
# =====================================================
while True:
    ret, frame = cap.read()
    if not ret:
        print("⚠️ Failed to grab frame")
        break

    # Convert to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect faces
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    for (x, y, w, h) in faces:
        # Extract face region
        face_img = frame[y:y+h, x:x+w]
        
        # Preprocess face for MobileNetV2
        face_resized = cv2.resize(face_img, (224, 224))
        face_array = np.expand_dims(face_resized, axis=0) / 255.0

        # Predict
        pred_prob = model.predict(face_array)[0][0]
        pred_class = class_labels[1] if pred_prob >= 0.5 else class_labels[0]

        # Draw rectangle and prediction
        color = (0, 255, 0) if pred_class == "Non-Drowsy" else (0, 0, 255)
        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
        cv2.putText(frame, f"{pred_class} ({pred_prob:.2f})", (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    # Display the frame
    cv2.imshow("Driver Drowsiness Detection", frame)

    # Exit on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# =====================================================
# Step 6: Release Resources
# =====================================================
cap.release()
cv2.destroyAllWindows()
