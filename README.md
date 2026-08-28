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
## Initial Baseline Scratch CNN 
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


## Optimized Scratch CNN
The baseline CNN was improved upon and now consists of three convolutional blocks each containing two convolutional layers followed by max pooling. The extracted features are passed through a dense(128) layer and a dropout(0.5) layer before the final five-class classifier.

### Architecture:
```text
Input: 224 × 224 × 3
        ↓
Rescaling (1/255)
        ↓
Conv2D (32 filters, 3×3)
        ↓
Conv2D (32 filters, 3×3)
        ↓
MaxPooling2D (2×2)
        ↓
Conv2D (64 filters, 3×3)
        ↓
Conv2D (64 filters, 3×3)
        ↓
MaxPooling2D (2×2)
        ↓
Conv2D (128 filters, 3×3)
        ↓
Conv2D (128 filters, 3×3)
        ↓
MaxPooling2D (2×2)
        ↓
Flatten
        ↓
Dense (128)
        ↓
Dropout (0.5)
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

## ResNet50 Transfer Learning
Similar to VGG16, ResNet50 was trained without its top dense layers, with all convolutional layers frozen to preserve its pretrained feature extraction capabilities. We used Global Average Pooling (GAP) after the convolutional base, followed by a classification layer for the five acne classes.

### Architecture:
```text
Input: 224 × 224 × 3
        ↓
ResNet50 Preprocessing Layer
        ↓
resNet50's Conv Layers + Residual Layers (x50)
... 
...
        ↓
GlobalAveragePooling
        ↓
Output (5 classes)
```

## Results

The current test-set performance of the three main architectures is:

| Model | Test Accuracy | Macro F1 |
|---|---:|---:|
| Custom CNN | 61.98% | 0.62 |
| ResNet50 | 66.88% | 0.69 |
| VGG16 | 69.06% | 0.71 |

VGG16 currently achieves the best overall performance.

The pretrained models substantially outperform the custom CNN trained from
scratch, suggesting that the pretrained visual representations provide a
significant advantage for this dataset.

However, performance is not uniform across classes. Papules and Pustules
remain the most difficult classes to distinguish, while Blackheads and
Whiteheads are generally classified more successfully.

The Whitehead results should be interpreted with some caution because the
test set contains only 57 Whitehead images.

## Experimental Findings

Rather than only comparing final architectures, several controlled
experiments were performed to understand the behavior of the custom CNN.

### Overfitting

The initial CNN reached approximately 95% training accuracy while achieving
only around 60% validation accuracy. This indicated substantial overfitting.

The majority of the model's parameters came from the `Flatten → Dense(128)`
connection, making the dense classifier a major source of model capacity.

### Dropout

`Dropout(0.5)` was added after the `Dense(128)` layer.

Dropout reduced training accuracy and therefore reduced the model's ability to
memorize the training set, but validation accuracy did not improve
substantially.

This demonstrated that reducing overfitting does not necessarily improve
generalization.

### L2 Regularization

L2 regularization was applied to the `Dense(128)` layer:

```python
kernel_regularizer=tf.keras.regularizers.l2(0.0001)
```
The regularized model showed reduced training performance, but validation
accuracy remained approximately in the same range as before.

L2 was therefore not retained in the final scratch CNN.

### Reducing Classifier Capacity
Several experiments were performed to determine whether the large
Dense(128) layer was responsible for the overfitting.
- Reducing Dense(128) to Dense(32) reduced model capacity but also caused a
substantial drop in performance without eliminating the generalization
problem.
- Replacing Flatten with GlobalAveragePooling caused the model to struggle
even with the training data.

These experiments suggested that simply reducing capacity was not sufficient
to solve the problem.

### Increasing Convolutional Depth
Additional convolutional blocks were introduced to test whether the model
needed a more powerful feature extractor.

Increasing the depth improved validation performance. The best scratch CNN
currently uses three convolutional blocks with 32, 64, and 128 filters.

The best observed validation accuracy was approximately 63%.

However, the model still showed a substantial training/validation gap.

### Data Augmentation
Several augmentation techniques were tested independently, including
rotation, horizontal flipping, translation, and brightness changes.

The geometric transformations tested generally reduced performance on the
scratch CNN. Brightness augmentation was less disruptive but did not produce
a convincing improvement.

These experiments suggested that augmentation is not automatically beneficial
for this dataset/model combination and that transformations must be chosen
according to the visual characteristics of the problem.

### Transfer Learning
VGG16 and ResNet50 were used as pretrained feature extractors with their
convolutional bases initially frozen.

Both pretrained models substantially outperformed the scratch CNN.

This suggests that the pretrained models provide more useful visual
representations than the relatively small CNN was able to learn from scratch
on this dataset.

## Current Progress:
- [x] Train and Evaluated baseline CNN model.
- [x] Train and Evaluated VGG16 model.
- [x] Train and evaluate ResNet50
- [x] Compare all three models
- [x] Perform hyperparameter tuning to optimize model performance
- [x] Modify the baseline CNN to reduce overfitting
## Planned Work
- [ ] Experiment with unfreezing some convolutional layers of VGG16 and ResNet50
- [ ] Deploy the final model using Flask

