# Acne Classification using CNNs and Transfer Learning

An image classification project for identifying different types of acne using
Convolutional Neural Networks (CNNs) and transfer learning.

The project focuses on building a baseline CNN from scratch and comparing its
performance with pretrained architectures such as VGG16 and ResNet50. The goal
is to understand how model architecture, regularization, and transfer learning
affect classification performance.

## Project Overview

This project explores acne image classification through a series of
experiments, starting with a custom CNN baseline and progressively evaluating
more advanced pretrained architectures.

### Models

- Custom CNN — Baseline model
- VGG16 — Transfer learning
- ResNet50 — Transfer learning

The baseline model was developed from scratch to establish a reference point
before evaluating pretrained architectures.

## Dataset

This project uses the **Acne Dataset Image** from Kaggle, containing acne
images across 5 classes.

The dataset is not included in this repository. The downloaded dataset is
excluded from version control using `.gitignore` due to its size and to avoid
redistributing the dataset files.

### Dataset Source

- **Dataset:** Acne Dataset Image
- **Author:** tiswan14
- **Platform:** Kaggle
- **Source:** https://www.kaggle.com/datasets/tiswan14/acne-dataset-image

To reproduce the experiments, download the dataset from the original Kaggle
source and place the extracted `Data/` folder in the project root:

```text
acne-classifier/
├── Data/
├── ...
```
## Baseline CNN
The baseline CNN was developed from scratch and consists of two convolutional layers followed by max pooling. The extracted features are passed through a dense layer before the final five-class classifier.

### Architecture:
```text
Input: 224 × 224 × 3
        ↓
Rescaling (1/255)
        ↓
Conv2D (32 filters, 3×3)
        ↓
MaxPooling2D (2×2)
        ↓
Conv2D (64 filters, 3×3)
        ↓
MaxPooling2D (2×2)
        ↓
Flatten
        ↓
Dense (128)
        ↓
Output (5 classes)
```

## VGG16 Transfer Learning
VGG16 was trained without its top dense layers, with all convolutional layers frozen to preserve its pretrained feature extraction capabilities. We used Global Average Pooling (GAP) after the convolutional base, followed by a classification layer for the five acne classes.

### Architecture:
```text
Input: 224 × 224 × 3
        ↓
VGG16 Preprocessing Layer
        ↓
VGG16's Conv Layers (x13)
... 
...
        ↓
GlobalAveragePooling
        ↓
Output (5 classes)
```

## Current Progress:
- [x] Train and Evaluated baseline CNN model.
- [x] Train and Evaluated VGG16 model.
   
## Planned Work
- [ ] Train and evaluate ResNet50
- [ ] Compare all three models
- [ ] Perform hyperparameter tuning to optimize model performance
- [ ] Experiment with unfreezing some convolutional layers of VGG16 and ResNet50
- [ ] Modify the baseline CNN to reduce overfitting
- [ ] Deploy the final model using Flask

