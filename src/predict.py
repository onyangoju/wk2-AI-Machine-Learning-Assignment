import os
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.abspath(os.path.join(BASE_DIR, "..", "model", "deforestation_detector_model.h5"))
TEST_IMAGE_PATH = os.path.abspath(os.path.join(BASE_DIR, "..", "sample.jpg"))  # <-- Place your image here

# Image settings
IMG_HEIGHT = 128
IMG_WIDTH = 128

# Load model
model = load_model(MODEL_PATH)
print(f"✅ Model loaded from {MODEL_PATH}")

# Load and preprocess image
img = image.load_img(TEST_IMAGE_PATH, target_size=(IMG_HEIGHT, IMG_WIDTH))
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0) / 255.0  # Normalize

# Predict
prediction = model.predict(img_array)[0][0]
label = "Deforested" if prediction > 0.5 else "Not Deforested"

print(f"\n🖼️ Prediction for image: {os.path.basename(TEST_IMAGE_PATH)}")
print(f"🔎 Predicted: {label} (Confidence: {prediction:.4f})")
