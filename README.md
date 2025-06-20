# MNIST Digit Recognition with Enhanced MLP

A PyTorch implementation of a multi-layer perceptron (MLP) for recognizing handwritten digits from the MNIST dataset, developed for course SDM366.

## Features

- **Enhanced MLP Architecture**:
  - 4 fully-connected layers with batch normalization
  - Dropout regularization (p=0.2)
  - ReLU activation functions
- **Advanced Training Setup**:
  - AdamW optimizer with weight decay (L2 regularization)
  - Learning rate scheduling with ReduceLROnPlateau
  - Early stopping mechanism
- **Data Augmentation**:
  - Random affine transformations (rotation ±5°, translation ±5%)
  - Separate transforms for training and testing

## The highest success rate should arrive at around 99.4%:

![image](https://github.com/user-attachments/assets/276f81bb-11f2-4c7e-8d16-ccf331e47f90)

## What I learned:
1.how to build an MLP layer

2.how to refine a simple layer to achieve higher performance(however, changing a network would be much better)

3.experience what training a network is really like
