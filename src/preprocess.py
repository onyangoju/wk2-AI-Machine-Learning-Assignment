import os
import cv2
import numpy as np

def load_images(data_dir, img_size=(128, 128)):
    """
    Load and preprocess images from a dataset folder.

    Args:
        data_dir (str): Path to dataset directory.
        img_size (tuple): Image resize dimensions.

    Returns:
        tuple: Numpy arrays for images (X) and labels (y)
    """
    X, y = [], []
    categories = ['deforested', 'non_deforested']
    for label, category in enumerate(categories):
        folder = os.path.join(data_dir, category)
        for img_file in os.listdir(folder):
            img_path = os.path.join(folder, img_file)
            try:
                img = cv2.imread(img_path)
                img = cv2.resize(img, img_size)
                X.append(img)
                y.append(label)
            except Exception as e:
                print(f"Error loading {img_file}: {e}")
    return np.array(X) / 255.0, np.array(y)
