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

The experiments follow the general process:
```text
Hypothesis
    ↓
Controlled change
    ↓
Training
    ↓
Validation performance
    ↓
Test evaluation
    ↓
Conclusion
    ↓
Next experiment
```
The test set is kept separate from model selection and hyperparameter decisions. Validation performance is used during development, while the test set is used for final evaluation.

### Models

- Custom CNN — Baseline model trained from scratch
- Optimized Scratch CNN — Deeper CNN with dropout
- VGG16 — Transfer learning with frozen convolutional base
- VGG16 — Fine-tuning of the final convolutional block
- ResNet50 — Transfer learning with frozen convolutional base

The baseline model was developed from scratch to establish a reference point
before evaluating pretrained architectures.

## Dataset

This project uses the **Acne Dataset Image** from Kaggle, containing acne
images across 5 classes:
- Blackheads
- Cyst
- Papules
- Pustules
- Whiteheads

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
├── config.py
├── DataLoad.py
├── model.py
├── vgg_model.py
├── resnet_model.py
├── train.py
├── evaluate.py
└── ...
```
Dataset Split
The dataset was already split into training, validation, and test sets.

| Class	| Train | Validation | Test |
|---|---:|---:|---:|
| Blackheads | 735 | 240 | 265 |
| Cyst |645 |206 | 189 |
| Papules | 621 | 209 | 202 |
| Pustules | 584 | 217 | 205 |
| Whiteheads | 193| 49 | 57 |

The test set contains 918 images in total.

Images are resized to:
```text
224 × 224 × 3
```
The test dataset is loaded with shuffle=False so that predictions can be matched reliably with their corresponding true labels when generating confusion matrices and classification reports.
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

### Results
The model contained approximately 23.9 million parameters. Most of these parameters came from the Flatten → Dense(128) connection.

The model achieved approximately 95% training accuracy while validation accuracy remained around 60%, providing strong evidence of overfitting.

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

### Results
Increasing the convolutional depth improved validation performance. The best observed validation accuracy for the scratch CNN was approximately 63%.

Despite this improvement, the scratch model continued to show a substantial training/validation gap.

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

### Results
With the convolutional base completely frozen:
- Test accuracy: 69.06%
- Macro F1: 0.71
  
This substantially outperformed the scratch CNN.

## VGG16 Fine-Tuning
After establishing the performance of frozen VGG16, the next experiment tested whether the pretrained representation could be adapted to the acne dataset.

The entire VGG16 convolutional base was not unfrozen initially because the dataset is relatively small compared with the size of the pretrained network, making large-scale fine-tuning more prone to overfitting and unnecessary disruption of useful pretrained features.

### Fine-Tuning Strategy
The first fine-tuning experiment made all three convolutional layers in Block 5 trainable:
```text
block5_conv1
block5_conv2
block5_conv3
```
Earlier VGG16 layers remained frozen.

The learning rate was reduced compared with the original training configuration to avoid making large updates to the pretrained weights.

### Results
Fine-tuning Block 5 produced a substantial improvement:

- Test accuracy: 90.09%
- Macro F1: 0.91
- Test loss: 0.4740
  
This was a large improvement over the frozen VGG16 model:
```text
Frozen VGG16
69.06% test accuracy
        ↓
Block 5 fine-tuning
90.09% test accuracy
```
This suggests that the pretrained ImageNet representation already provided strong visual features, but adapting several of the highest-level convolutional layers allowed the model to specialize those features for acne classification.

### Fine-Tuning Depth Experiment
A follow-up experiment tested whether only the final convolutional layer of Block 5 needed to be fine-tuned.

Only:
```
block5_conv3
```
was made trainable while the remaining VGG16 layers stayed frozen.

The model achieved:

- Test accuracy: 85.84%
- Macro F1: 0.86
  
This was substantially lower than the 90.09% achieved by fine-tuning all three Block 5 convolutional layers.

This suggests that the additional trainable layers in Block 5 were providing useful task-specific adaptation rather than simply increasing overfitting.

The current best model is therefore:
```
VGG16 pretrained on ImageNet
        ↓
Block 1–4 frozen
        ↓
Block 5 convolutional layers trainable
        ↓
GlobalAveragePooling2D
        ↓
Dense (5 classes)
```

## ResNet50 Transfer Learning
ResNet50 was evaluated using the same general transfer-learning strategy.

The ImageNet pretrained convolutional base was initially frozen, followed by Global Average Pooling and a five-class classification layer.

### Architecture:
```text
Input: 224 × 224 × 3
        ↓
ResNet50 Preprocessing
        ↓
ResNet50 Convolutional + Residual Layers
        ↓
GlobalAveragePooling2D
        ↓
Dense (5 classes)
```
### Result
The frozen ResNet50 achieved:

- Test accuracy: 66.88%
- Macro F1: 0.69

This was slightly below the frozen VGG16 model.

Fine-tuning ResNet50 is the next major experiment.

## Overall Results

The current test-set performance of the three main architectures is:

| Model | Test Accuracy | Macro F1 |
|---|---:|---:|
| Initial Scratch CNN | 58.93% | 0.58 |
| Optimized Scratch CNN | 61.98% | 0.62 |
| ResNet50 | 66.88% | 0.69 |
| VGG16 (Frozen) | 69.06% | 0.71 |
| VGG16 (Block 5, 1 conv) | 85.84% | 0.86 |
| VGG16 (Block 5, 3 convs) | 90.09% | 0.91 |


The current best model is VGG16 with all three convolutional layers in Block 5 fine-tuned.

Performance is not uniform across classes. Papules and Pustules consistently remain the most difficult classes to distinguish, while Blackheads and Whiteheads are generally classified more successfully.

The Whitehead results should be interpreted with some caution because the test set contains only 57 Whitehead images.

## Experimental Findings

Rather than only comparing final architectures, several controlled
experiments were performed to understand the behavior of the custom CNN.

### Overfitting

The initial CNN reached approximately 95% training accuracy while achieving only around 60% validation accuracy.

This indicated substantial overfitting.

The majority of the model's parameters came from the Flatten → Dense(128) connection, making the dense classifier a major source of model capacity.

Later experiments showed that increasing model capacity does not automatically solve the problem, while simply reducing capacity can prevent the model from learning sufficiently useful representations.

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

### Fine-Tuning
Fine-tuning VGG16 produced a much larger improvement than frozen transfer learning.

Unfreezing the final VGG16 convolutional block increased test accuracy from:
```
69.06%
```
to:
```
90.09%
```
Fine-tuning only the final convolutional layer, however, reduced performance to:
```
85.84%
```
This provided evidence that multiple high-level convolutional layers were useful for adapting the pretrained representation to acne-specific visual patterns.

The experiment also demonstrated that a larger trainable parameter count does not necessarily mean worse generalization. In this case, fine-tuning approximately 7 million parameters produced better test performance than restricting fine-tuning to approximately 2.4 million trainable parameters.


## Current Progress:
- [x] Train and Evaluated baseline CNN model.
- [x] Modify baseline CNN to reduce overfitting
- [x] Experiment with dropout
- [x] Experiment with L2 regularization
- [x] Experiment with classifier capacity
- [x] Experiment with Global Average Pooling
- [x] Increase convolutional depth
- [x] Experiment with data augmentation
- [x] Train and evaluate VGG16 model.
- [x] Train and evaluate ResNet50
- [x] Compare scratch CNN, VGG16, and ResNet50
- [x] Fine-tune VGG16
- [x] Compare different VGG16 fine-tuning depths
- [ ] Fine-tune ResNet50
- [ ] Compare fine-tuned VGG16 and ResNet50
- [ ] Deploy the final model using Flask
 
## Current Best Model
The current best-performing model is:
```
VGG16 pretrained on ImageNet
        ↓
Block 1–4 frozen
        ↓
Block 5 convolutional layers trainable
        ↓
GlobalAveragePooling2D
        ↓
Dense (5 classes)
```
Final Test Performance:
- Accuracy : 90.09%
- Macro F1 : 0.91
- Loss     : 0.4740

The strongest remaining classification difficulty is distinguishing Papules from Pustules.

