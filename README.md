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

The highest success rate should arrive at around 99.4%

