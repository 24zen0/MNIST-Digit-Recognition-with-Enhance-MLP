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
- **Reproducibility**:
  - Full seed control for random number generators
  - Deterministic CUDA operations

## Requirements

- Python 3.6+
- PyTorch 1.8+
- torchvision
- numpy

## Installation

```bash
git clone https://github.com/yourusername/MNIST_number_recognition_project.git
cd MNIST_number_recognition_project
pip install -r requirements.txt
