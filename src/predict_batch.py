import os
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.abspath(os.path.join(BASE_DIR, "..", "model", "deforestation_detector_model.h5"))
DATASET_DIR = os.path.abspath(os.path.join(BASE_DIR, "..", "dataset"))  # points to dataset folder containing subfolders

IMG_HEIGHT = 128
IMG_WIDTH = 128

model = load_model(MODEL_PATH)
print(f"✅ Model loaded from {MODEL_PATH}")

# Iterate through subfolders and images
for subfolder in os.listdir(DATASET_DIR):
    subfolder_path = os.path.join(DATASET_DIR, subfolder)
    if os.path.isdir(subfolder_path):
        print(f"\n🔎 Predicting images in folder: {subfolder}")
        for img_file in os.listdir(subfolder_path):
            img_path = os.path.join(subfolder_path, img_file)
            if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                img = image.load_img(img_path, target_size=(IMG_HEIGHT, IMG_WIDTH))
                img_array = image.img_to_array(img)
                img_array = np.expand_dims(img_array, axis=0) / 255.0

                prediction = model.predict(img_array)[0][0]
                label = "Deforested" if prediction > 0.5 else "Not Deforested"
                confidence = prediction if prediction > 0.5 else 1 - prediction

                print(f"{img_file}: {label} (Confidence: {confidence:.4f})")
