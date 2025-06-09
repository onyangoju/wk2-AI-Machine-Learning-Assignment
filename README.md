
# AI Deforestation Detector

[![Python](https://img.shields.io/badge/python-3.8%2B-blue)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/tensorflow-2.x-orange)](https://www.tensorflow.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)

A Convolutional Neural Network (CNN) based image classification system to detect deforestation from satellite images. This project classifies images into two categories: **Deforested** and **Not Deforested**.

---

## Features

- Train a CNN model on satellite images
- Predict single images or batches of images
- Evaluate model accuracy on labeled folders
- Easy to use with clear CLI commands
- Adjustable prediction threshold for classification sensitivity

---

## Repository Structure

```

AI-Deforestation-Detector/
│
├── dataset/ # Sample images in class folders
│ ├── deforested/
│ └── non_deforested/
│
├── model/ # Trained model saved here
│ └── deforestation_detector_model.h5
│
├── notebooks/ # Jupyter notebooks for experimentation
│ └── train_model.ipynb
│
├── src/ # Source code
│ ├── preprocess.py # Data preprocessing scripts
│ ├── model.py # Model definition and compilation
│ ├── predict.py # Predict single image
│ ├── predict_batch.py # Batch predictions on image folder
│ ├── evaluate.py # Evaluate model accuracy on a dataset folder
│ └── evaluate_folder.py # (Optional) Alternative evaluation script
│
└── README.md # This documentation

````

---

## Setup Instructions

# 1. Create and activate virtual environment

python -m venv venv

# Windows
venv\Scripts\activate

# macOS/Linux
source venv/bin/activate

# 2. Install dependencies

pip install tensorflow numpy
````
---

## Usage

### 1. Train the Model
- Use the `train_model.ipynb` notebook in the `notebooks/` folder to train your model.
- The model architecture and compilation are defined in `src/model.py`.

### 2. Predict a Single Image
python src/predict.py path/to/image.jpg

### 3. Predict Batch of Images
python src/predict_batch.py path/to/image_folder [threshold]

### 4. Evaluate Model on a Folder
python src/evaluate.py path/to/folder [threshold]


## Notes

* Images must be JPG, JPEG, or PNG format.
* Threshold controls the prediction sensitivity (higher means stricter detection).
* Folder names should reflect the ground truth label for evaluation (`deforested` or `non_deforested`).

---

## Contact

For questions or support, reach out at paulineakoth2002@gmail.com
