# Traffic Sign Recognition Under Image Corruption (Phase 2)
CENG499 - Senior Design Project

## Overview
This repository contains the Phase 2 implementation of the robustness evaluation for three CNN models (ResNet18, MobileNetV3-Large, ConvNeXt-Tiny) against various image corruptions on the GTSRB dataset.

## Phase 2 Updates
* Implemented clean baseline testing on the original GTSRB dataset (`test_clean.py`).
* Evaluated models against 4 corruptions (Gaussian Noise, Motion Blur, Fog, Brightness) at 5 severity levels (`test_corruptions.py`).
* Fixed random seeds (seed=42) to ensure exact reproducibility across model evaluations.
* Generated performance plots and confusion matrices for deep analysis of model failures (`analyze_mobilenet.py`).

## Requirements
Run the following command to install dependencies:
`pip install -r requirements.txt`

## How to Run

1. To test models on clean data:
`python src/test_clean.py`

2. To evaluate robustness against corruptions:
`python src/test_corruptions.py`

3. To generate confusion matrices for MobileNetV3:
`python src/analyze_mobilenet.py`