# Cats vs Dogs Image Classification

A deep learning project for binary image classification using PyTorch.

The goal of this project is to build a Convolutional Neural Network (CNN) capable of classifying images as either **Cat** or **Dog**.

---

## Project Overview

This project covers an end-to-end image classification workflow, including:

- Real-world image dataset validation
- Data cleaning and corrupted image detection
- Train / validation / test splitting
- Custom PyTorch Dataset
- Data augmentation
- CNN architecture design
- Model training and checkpointing
- Model evaluation
- Confusion matrix analysis
- Error analysis
- Transfer learning
- Fine-tuning
- Single-image inference

---

## Dataset

The project uses the Microsoft Cats and Dogs Dataset.

After validation:

- 24,998 valid images
- 4 corrupted/invalid images
- 12,499 cats
- 12,499 dogs

The dataset was divided into:

- 80% Training
- 10% Validation
- 10% Test

The dataset itself is not included in this repository.

---

## Model

The main model is a custom Convolutional Neural Network built with PyTorch.

### Architecture

The CNN consists of:

- 3 Convolutional layers
- ReLU activation functions
- Max Pooling layers
- Fully connected layers
- Final classification layer

Input images are resized to:

`128 × 128 × 3`

The model produces two output classes:

- `0` → Cat
- `1` → Dog

---

## Data Augmentation

Training images were augmented using:

- Random Horizontal Flip
- Random Rotation

Validation and test images were only resized and converted to tensors.

Data augmentation was applied only to the training set to help improve generalization.

---

## Results

The final CNN achieved:

| Metric | Result |
|---|---:|
| Test Loss | 0.3420 |
| Test Accuracy | 84.92% |
| Test Samples | 2,500 |
| Correct Predictions | 2,123 |
| Incorrect Predictions | 377 |

### Confusion Matrix

| Actual / Predicted | Cat | Dog |
|---|---:|---:|
| Cat | 995 | 255 |
| Dog | 122 | 1,128 |

The model correctly classified 2,123 out of 2,500 test images.

---

## Error Analysis

Several common patterns were observed among misclassified images:

- Blurry or low-quality images
- Unusual poses or viewing angles
- Small subjects within the image
- Complex or cluttered backgrounds
- Visually ambiguous examples

These observations highlight some of the challenges involved in real-world image classification.

---

## Transfer Learning

Transfer learning with MobileNetV3-Small pretrained on ImageNet was also explored.

Two approaches were investigated:

### Feature Extraction

The pretrained feature extractor was frozen and a new classification layer was trained.

### Fine-Tuning

The final feature blocks were unfrozen and trained with a smaller learning rate.

The experiments showed that the custom CNN achieved higher test accuracy than the fine-tuned MobileNet model in this project.

This demonstrates that transfer learning is not automatically superior for every dataset and problem.

---

## Single Image Prediction

The project also includes a prediction pipeline for classifying individual images.

---

predict_image("path/to/image.jpg")

---

## Project Structure

Cats vs Dogs/
│
├── data/
│
├── models/
│
├── notebooks/
│   └── 01_image_classification_final.ipynb
│
├── results/
│   ├── confusion_matrix.png
│   └── results_summary.txt
│
├── src/
│   ├── model.py
│   └── predict.py
│
├── .gitignore
├── requirements.txt
└── README.md

---

## Technologies

- Python
- PyTorch
- Torchvision
- Scikit-learn
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Pillow
- Jupyter Notebook

---

## Key Learning Outcomes

Through this project, I practiced:

- Building CNNs with PyTorch
- Working with RGB image datasets
- Creating custom PyTorch Datasets
- Using DataLoaders
- Applying image augmentation
- Training and validating deep learning models
- Saving and loading model checkpoints
- Evaluating classification models
- Using confusion matrices
- Performing error analysis
- Applying transfer learning
- Fine-tuning pretrained models

---

## Author

mstf-dev

Aspiring Data Analyst | Machine Learning | Data Analysis