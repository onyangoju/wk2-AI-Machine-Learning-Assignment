import os
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

def evaluate_folder(folder_path, model, img_height=128, img_width=128, threshold=0.5):
    # Determine expected label from folder name
    # Assumes folder name includes either 'deforested' or 'non_deforested'/'forested'
    folder_name = os.path.basename(folder_path).lower()
    if "deforest" in folder_name:
        expected_label = 1  # deforested
    else:
        expected_label = 0  # not deforested

    image_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

    correct = 0
    total = len(image_files)

    for img_file in image_files:
        img_path = os.path.join(folder_path, img_file)
        img = image.load_img(img_path, target_size=(img_height, img_width))
        img_array = image.img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        pred = model.predict(img_array)[0][0]
        pred_label = 1 if pred > threshold else 0

        if pred_label == expected_label:
            correct += 1

        print(f"{img_file}: Predicted {'Deforested' if pred_label else 'Not Deforested'} (Confidence: {pred:.4f})")

    accuracy = correct / total if total > 0 else 0
    print(f"\nAccuracy on folder '{folder_name}': {accuracy:.2%} with threshold={threshold}")

# Example usage:
if __name__ == "__main__":
    import sys

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    MODEL_PATH = os.path.abspath(os.path.join(BASE_DIR, "..", "model", "deforestation_detector_model.h5"))

    model = load_model(MODEL_PATH)
    print(f"Model loaded from {MODEL_PATH}")

    if len(sys.argv) < 2:
        print("Usage: python evaluate_folder.py <folder_path> [threshold]")
        sys.exit(1)

    folder_path = sys.argv[1]
    threshold = float(sys.argv[2]) if len(sys.argv) > 2 else 0.5

    evaluate_folder(folder_path, model, threshold=threshold)
