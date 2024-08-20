#Entropy-Guided Multi-Scale Dynamic Vector Quantization for Robust Medical Image Representation
This repository contains the implementation of the method proposed in the paper "Entropy-Guided Multi-Scale Dynamic Vector Quantization for Robust Medical Image Representation"

#Project Overview
This project implements a novel approach for robust medical image representation using multi-scale dynamic vector quantization. The current implementation focuses on training the model using the NIH dataset with a DenseNet121 backbone.

Setup
Prerequisites

Python (version X.X) [Please specify the required Python version]
PyTorch (version X.X) [Please specify the required PyTorch version]
[List any other major dependencies]

Dataset Preparation
Before running the training script, you need to update the dataset paths in the constants.py file located in the data directory:
pythonCopyDATA_BASE_DIR = Path("/path/to/your/NIH/dataset/")
ISIC_BASE_DIR = Path("/path/to/your/ISIC-2018/dataset/")

DATA_BASE_DIR: Path to the NIH dataset
ISIC_BASE_DIR: Path to the ISIC-2018 dataset (not used in the current implementation)

Training
To train the model, use the provided train.sh script:
bashCopy./train.sh
This script runs the following command:
bashCopypython dynamic_chex_main.py -model_name dynamic_dense121 -ne 1024 -ed 64 -cc 0.25 -batch 64
Training Parameters
ParameterDescription-model_nameName of the model (default: dynamic_dense121)-ne[Please explain this parameter]-ed[Please explain this parameter]-cc[Please explain this parameter]-batchBatch size (default: 64)
Current Limitations

The current implementation only supports training on the NIH dataset.
Only the proposed method is implemented; baseline methods and other architectures are not yet available.

Future Updates
We plan to extend this repository in the future to include:

 Baseline methods for comparison
 Support for additional architectures
 Implementation for other datasets (e.g., ISIC-2018)
