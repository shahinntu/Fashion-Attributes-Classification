
# Fashion Attributes Classification Challenge

This project tackles the problem of **multi-label multi-class classification** in the domain of fashion imagery. The goal is to classify various fashion attributes from images using a Transformer-based model with a Swin Transformer Large (Swin-L) backbone.

## Table of Contents
- [Project Overview](#project-overview)
- [Folder Structure](#folder-structure)
- [Setup and Installation](#setup-and-installation)
- [Data Preparation](#data-preparation)
- [Usage](#usage)
- [Training](#training)
- [Results](#results)

## Project Overview

The project aims to solve the multi-label classification challenge by predicting six attributes for each image from a dataset of 6000 images (5000 for training and 1000 for validation). Each attribute belongs to one of the 26 possible labels, and this setup requires precise identification and classification.

The model integrates a **Swin Transformer Large** for feature extraction and a custom Transformer block for classifying the attributes.

## Folder Structure

```
FASHION ATTRIBUTES CLASSIFICATION/
│
├── configs/                                # Configuration files for training
│   ├── train_config.json                   # Configuration for the main training process
│
├── data/                                   # Directory containing image data and splits
│   ├── img/                                # Image folder
│   ├── results/                            # Output results folder
│   └── split/                              # Split files for training, validation, and test
│       ├── list_attr_cloth.txt             # Attribute descriptions
│       ├── train.txt                       # List of training images
│       ├── train_attr.txt                  # List of corresponding labels for training images
│       ├── val.txt                         # List of validation images
│       ├── val_attr.txt                    # List of corresponding labels for validation images
│
├── model_logs/                             # Directory to save model checkpoints and logs
│
├── notebooks/                              # Jupyter notebooks for analysis and exploration
│   ├── 01-look-at-data.ipynb               # Data exploration
│   ├── 02-look-at-prepared-data.ipynb      # Data preparation and augmentation
│   ├── 03-look-at-model-param-count.ipynb  # Analysis of model parameters
│   ├── 04-look-at-model-and-criterion-output.ipynb  # Check model outputs and loss function results
│   ├── 05-look-at-predictions.ipynb        # Model predictions analysis
│
├── report/                                 # Final project report
│   └── Fashion Attributes Classification.pdf      # PDF version of the report
│
├── scripts/                                # Training and evaluation scripts
│   ├── install_dependencies.sh             # Install necessary dependencies
│   ├── train_model.sh                      # Training script for the model
│
├── src/                                    # Source code for the project
│   ├── data_prep/                          # Data processing functions
│   ├── models/                             # Model definition and utilities
│   │   ├── __init__.py
│   │   ├── arg_parse.py                    # Argument parsing for training
│   │   ├── criterion.py                    # Loss functions
│   │   ├── metrics.py                      # Performance metrics for evaluation
│   │   ├── ml_pipeline.py                  # Main model pipeline
│   │   ├── prediction.py                   # Code for running predictions
│   │   ├── training.py                     # Model training and optimization
│   │   ├── utils.py                        # Utility functions
│
├── requirements.txt                        # Python dependencies
└── train_model.py                          # Main script for model training
```

## Setup and Installation

### Requirements
- Python 3.8+
- PyTorch 1.9+ (with CUDA support)
- NumPy, Pandas, Matplotlib, and other Python dependencies listed in `requirements.txt`

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/shahinntu/Fashion-Attributes-Classification.git
   cd Fashion-Attributes-Classification
   ```

2. Install the required dependencies:
   ```bash
   bash scripts/install_dependencies.sh
   ```

3. Ensure you have access to a GPU for training the model (recommended).

## Data Preparation

### Image Data
Before starting the training, ensure that the **image data** is placed in the correct directory:

- All images should be placed in the `data/img/` directory.
- Training images should be listed in `data/split/train.txt`, for example:
  ```
  img/00000.jpg
  img/00001.jpg
  img/00002.jpg
  ...
  ```

### Labels
The corresponding attribute labels should be listed in `data/split/train_attr.txt` in the following format, where each line represents six attribute values corresponding to a specific image:
```
5 0 2 0 2 2
5 1 2 0 5 1
5 0 2 3 4 2
...
```

For validation, follow the same format with the files `val.txt` (validation images) and `val_attr.txt` (validation labels).

## Usage

### Training

To train the model, use the following command:
```bash
bash scripts/train_model.sh
```

This will use the configuration file from `configs/train_config.json` to start the training process. All checkpoints and logs will be saved in the `model_logs/` directory.

### Configuration

You can modify the training configuration by editing the `configs/train_config.json` file. Some adjustable parameters include:
- Learning rate
- Batch size
- Number of epochs
- Optimizer settings

## Training

The model uses a **Swin Transformer Large (Swin-L)** backbone for feature extraction. The training involves:
- **Adam optimizer** with a learning rate of $1 \times 10^{-4}$
- **Cosine learning rate scheduler** for controlled learning rate decay
- **Categorical Cross-Entropy with Class Weights** as the primary loss function to handle class imbalance.

The model is trained for 20 epochs with a batch size of 64.

## Results

After training, the model achieved:
- A mean class accuracy of **72.66%** on the test dataset, showing good generalization and performance across unseen data.

You can find more details in the final report: `report/Fashion Attributes Classification.pdf`.
