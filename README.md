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

## Baseline CNN

The initial CNN architecture consists of:

```text
Input: 224 × 224 × 3

Conv2D (32 filters, 3×3)
        ↓
MaxPooling2D (2×2)
        ↓
Conv2D (64 filters, 3×3)
        ↓
Flatten
        ↓
Dense (128)
        ↓
Output (5 classes)