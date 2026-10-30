# Breast Ultrasound Image Classification

## Overview

This project explores deep learning for breast ultrasound image classification into three categories: Benign, Malignant, and Normal.

Beyond classification performance, I also examine how the model performs across different classes and visualize the image regions influencing its predictions.

## Dataset

- Total images: 780
- Three classes: Benign, Malignant, and Normal
- Imbalanced class distribution

The dataset is divided into a development set and a held-out test set. Data augmentation is used to address class imbalance during training.

## Model

- **Architecture:** ResNet50 pretrained on ImageNet
- **Fine-tuning:** Last layers
- **Classification head:**
  - Global Average Pooling
  - Dense layer with 512 units
  - Batch Normalization and ReLU
  - Dropout
  - Softmax output with 3 classes

**Loss function:** Sparse Categorical Crossentropy  
**Optimizer:** Adam

## Training and Evaluation

The model is evaluated using 5-fold stratified cross-validation. A new model is initialized and trained for each fold.

Evaluation metrics include:
- Accuracy
- Precision
- Recall
- F1-score
- Balanced accuracy

Class-wise metrics and confusion matrices are used to examine differences in classification performance, particularly for malignant cases.

## Cross-Validation Results

- Average validation accuracy: 0.81
- Best fold accuracy: approximately 0.86

Performance is also examined across folds using precision, recall, and F1-score.

## Final Test Evaluation

The final model is evaluated on a held-out test set.

- **Accuracy:** 85.47%
- **Balanced accuracy:** 85.08%

The results show differences in performance across the three classes. Malignant-case detection receives particular attention because missed malignant predictions are especially important in this application.

## Model Interpretation

Grad-CAM is used to visualize image regions that contribute to the model's predictions.

Heatmaps and overlays provide a way to inspect the model's focus and compare its predictions with the original ultrasound images. These visualizations offer additional context for interpreting model behavior, but do not establish whether the highlighted regions are clinically meaningful.

## Tools

- Python
- TensorFlow and Keras
- NumPy
- Scikit-learn
- Matplotlib