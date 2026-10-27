# Traffic Sign Recognition under Image Corruption

This repository contains the ongoing development of a deep learning-based traffic sign classification system using the **GTSRB (German Traffic Sign Recognition Benchmark)** dataset. 

The primary goal of this project is to evaluate model performance and robustness under various image corruption conditions (e.g., weather, blur, noise), comparing standard vs. lightweight architectures.

## Current Progress (Phase 1 Completed)
- [x] Dataset preparation and robust preprocessing pipeline setup.
- [x] Implemented **Track ID-based dataset splitting** to strictly prevent data leakage.
- [x] Initial PyTorch training and evaluation workflow established.
- [x] Baseline model training and validation completed for:
  - **ResNet-18**
  - **MobileNetV3-Large** (Lightweight/Edge AI focused)
  - **ConvNeXt-Tiny** (High-accuracy architecture)
- [ ] Phase 2: Model evaluation under simulated image corruptions (Gaussian noise, motion blur, fog, brightness) across 5 severity levels.
- [ ] Phase 3: Robustness improvement via **AugMix** and mixed-corruption training.

## Tech Stack
- **Framework:** PyTorch, Torchvision
- **Language:** Python
- **Evaluation Metrics:** Accuracy, Macro-F1 Score, Confusion Matrices
- **Hardware:** Local training optimized for NVIDIA RTX 4050 Ti

## Note on Dataset and Model Weights
The GTSRB dataset and the trained model checkpoints (`.pth` files) are not included in this repository due to size constraints. To run the experiments, download the dataset directly from the official source and place it in the local `data/` directory.