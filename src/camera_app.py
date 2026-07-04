import cv2
import numpy as np
import joblib
import os
from pathlib import Path
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling2D

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# ---------- Paths ----------
BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH         = BASE_DIR / "models" / "svm_final_model.pkl"
PCA_PATH           = BASE_DIR / "models" / "svm_pca_cnn.pkl"
SCALER_BUNDLE_PATH = BASE_DIR / "models" / "scaler.pkl"
LABEL_ENCODER_PATH = BASE_DIR / "models" / "label_encoder.pkl"

# ---------- Config ----------
THRESHOLD      = 0.5
CNN_IMG_SIZE   = (224, 224)
ROI_SIZE       = 250
FRAME_SKIP     = 10
UNKNOWN_LABEL  = 6

CLASS_NAMES = ["cardboard", "glass", "metal", "paper", "plastic", "trash", "unknown"]

CLASS_COLORS = {
    "cardboard": (0,   165, 255),
    "glass":     (255, 255,   0),
    "metal":     (180, 180, 180),
    "paper":     (255, 255, 255),
    "plastic":   (0,   255, 128),
    "trash":     (0,     0, 200),
    "unknown":   (100, 100, 100),
}

# ---------- Load models ----------
clf           = joblib.load(MODEL_PATH)
pca           = joblib.load(PCA_PATH)
label_encoder = joblib.load(LABEL_ENCODER_PATH)

bundle     = joblib.load(SCALER_BUNDLE_PATH)
cnn_scaler = bundle["cnn_scaler"]
cnn_weight = bundle["cnn_weight"]

base      = ResNet50(weights="imagenet", include_top=False, input_shape=(224, 224, 3))
cnn_model = Model(inputs=base.input, outputs=GlobalAveragePooling2D()(base.output))

# ---------- Feature extraction  ----------
def extract_features(frame_bgr):
    img = cv2.resize(frame_bgr, CNN_IMG_SIZE)
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    batch = preprocess_input(np.expand_dims(rgb.astype(np.float32), 0))
    cnn_raw    = cnn_model.predict(batch, verbose=0)
    cnn_scaled = cnn_weight * cnn_scaler.transform(cnn_raw)
    return pca.transform(cnn_scaled)

# ---------- Predict ----------
def predict(features):
    probs    = clf.predict_proba(features)[0]
    max_prob = probs.max()
    if max_prob < THRESHOLD:
        return "unknown", max_prob
    label = label_encoder.inverse_transform([np.argmax(probs)])[0]
    return label, max_prob

# ---------- Draw overlay ----------
def draw_overlay(frame, label, conf):
    color = CLASS_COLORS.get(label, (255, 255, 255))
    h, w  = frame.shape[:2]

    cv2.rectangle(frame, (0, h - 70), (w, h), (20, 20, 20), -1)
    cv2.putText(frame, label.upper(), (20, h - 35),
                cv2.FONT_HERSHEY_DUPLEX, 1.4, color, 2, cv2.LINE_AA)
    cv2.putText(frame, f"{conf:.0%}", (20, h - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (180, 180, 180), 1, cv2.LINE_AA)

    bar_w = int((w - 40) * conf)
    cv2.rectangle(frame, (20, h - 6), (w - 20, h - 2), (60, 60, 60), -1)
    cv2.rectangle(frame, (20, h - 6), (20 + bar_w, h - 2), color, -1)

# ---------- Main loop ----------
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_FPS, 30)
print("Place the material inside the box. Press Q to quit.")

frame_count = 0
last_label  = "unknown"
last_conf   = 0.0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    h, w   = frame.shape[:2]
    cx, cy = w // 2, h // 2
    half   = ROI_SIZE // 2
    x1, y1 = cx - half, cy - half
    x2, y2 = cx + half, cy + half

    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.putText(frame, "Place material inside the box",
                (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    roi = frame[y1:y2, x1:x2]
    if roi.size != 0 and frame_count % FRAME_SKIP == 0:
        features           = extract_features(roi)
        last_label, last_conf = predict(features)

    draw_overlay(frame, last_label, last_conf)
    cv2.imshow("MSI - Material Stream Identification", frame)

    frame_count += 1
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()