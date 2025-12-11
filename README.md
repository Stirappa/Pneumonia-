# Pneumonia vs Normal Classification

This project trains a Convolutional Neural Network (CNN) using
TensorFlow to classify chest X-ray images as **Normal** or
**Pneumonia**. The notebook loads the dataset, preprocesses the images,
builds the model, trains it, evaluates it, and visualizes the results.

## Dataset Structure

The notebook expects the dataset to be organized as follows:

    dataset/
    │
    ├── train/
    │   ├── NORMAL/
    │   └── PNEUMONIA/
    │
    ├── val/
    │   ├── NORMAL/
    │   └── PNEUMONIA/
    │
    └── test/
        ├── NORMAL/
        └── PNEUMONIA/

## Notebook Workflow

### 1. Import Libraries & Setup

Loads TensorFlow, NumPy, Matplotlib, and OS.\
Defines image size, batch size, epochs, and dataset paths.

### 2. Dataset Loading

A custom `make_dataset()` function: - Loads image file paths\
- Reads and preprocesses images\
- Batches, shuffles (for training), and prefetches\
- Returns a `tf.data.Dataset`

Image counts in each folder are printed for verification.

### 3. Model Building

Defines a CNN using: - Conv2D layers\
- BatchNormalization\
- MaxPooling\
- Dense classifier with softmax output

### 4. Training

Trains the model using: - Training dataset\
- Validation dataset\
- Callbacks:\
- `ModelCheckpoint` (saves best model)\
- `ReduceLROnPlateau` (adjusts learning rate)

### 5. Visualization

Plots: - Training vs validation accuracy\
- Training vs validation loss

### 6. Evaluation

Evaluates the final model on the test dataset and prints accuracy.

### 7. Confusion Matrix

Generates predictions on the test set and plots a confusion matrix
comparing: - Normal\
- Pneumonia

## Requirements

The notebook uses:

-   TensorFlow\
-   NumPy\
-   Matplotlib\
-   OS / Python standard library

## How to Run

1.  Place the dataset in the required folder structure.\
2.  Open the notebook in Jupyter or VS Code.\
3.  Run cells sequentially from top to bottom.
