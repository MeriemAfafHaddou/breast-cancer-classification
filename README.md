# Breast Ultrasound Image Classification

## Overview
This project addresses the classification of breast ultrasound images into three classes: Benign, Malignant, and Normal. The objective is to build a robust deep learning model suitable for small and imbalanced medical imaging datasets.

---

## Dataset
- Total images: 780
- Three classes
- Class imbalance present

Data augmentation is applied only to training data. Validation and test data remain untouched.

---

## Model
- Backbone: ResNet50 pretrained on ImageNet
- Fine tuning: last layers only
- Head:
  - Global Average Pooling
  - Dense layer with 512 units
  - Batch Normalization
  - ReLU
  - Dropout
  - Softmax output with 3 classes

Loss function: Sparse Categorical Crossentropy
Optimizer: Adam

---

## Training and Evaluation
- 5 fold stratified cross validation
- New model initialized for each fold
- Metrics:
  - Accuracy
  - Precision
  - Recall
  - F1 score
  - Balanced accuracy

---

## Cross Validation Results
- Average validation accuracy: 0.81
- Best fold accuracy: ~0.86
- Stable precision, recall, and F1 across folds

---

## Final Training on Full Dataset
After validation, the model was trained on all available data and evaluated on a held out test set.

### Classification Report
- Accuracy: 0.889
- Balanced Accuracy: 0.842

Class wise performance:
- Benign: high recall and strong precision
- Malignant: slightly lower recall, clinically important error source
- Normal: high precision with lower recall due to limited samples

---

## Explainability (XAI)
- Grad-CAM was used to visualize which regions of the ultrasound images influenced the model’s predictions.
- Heatmaps and overlays confirm that the model focuses on lesion areas, improving interpretability and clinical trust.

---

## Key Insights
Experiments highlighted strong sensitivity of convolutional neural networks to data imbalance, acquisition variability, and population shift in breast ultrasound classification. These findings exposed limits of scale focused learning and emphasized the importance of inductive bias, structured representations, and self supervised learning approaches, especially for medical imaging tasks with limited data.

---

## Tools
- Python
- TensorFlow and Keras
- NumPy
- Scikit Learn
- Matplotlib
