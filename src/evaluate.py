import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.abspath(os.path.join(BASE_DIR, "..", "dataset"))
MODEL_PATH = os.path.abspath(os.path.join(BASE_DIR, "..", "model", "deforestation_detector_model.h5"))

# Image settings
IMG_HEIGHT = 128
IMG_WIDTH = 128
BATCH_SIZE = 8

# Load the model
model = load_model(MODEL_PATH)
print(f"✅ Loaded model from {MODEL_PATH}")

# Prepare validation data
datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

val_generator = datagen.flow_from_directory(
    DATASET_PATH,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    class_mode='binary',
    subset='validation',
    shuffle=False  # Keep ordering to match predictions
)

# Evaluate model
loss, accuracy = model.evaluate(val_generator)
print(f"\n📊 Evaluation Results:\n   - Accuracy: {accuracy*100:.2f}%\n   - Loss: {loss:.4f}")

# Predict a few images
predictions = model.predict(val_generator)
predicted_classes = (predictions > 0.5).astype("int32").flatten()
true_classes = val_generator.classes

print("\n🔍 Sample Predictions:")
for i in range(min(10, len(true_classes))):
    print(f"  Image {i+1}: Predicted={predicted_classes[i]}, Actual={true_classes[i]}")
