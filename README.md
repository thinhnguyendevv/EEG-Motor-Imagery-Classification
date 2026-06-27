# 🧠 EEG Motor Imagery Classification: A Deep Learning Pipeline

This repository implements an end-to-end Machine Learning pipeline to classify EEG (Electroencephalography) signals for Brain-Computer Interfaces (BCI). The system predicts 3 specific motor imagery tasks (Rest, Left Hand, Right Hand) using data from the PhysioNet dataset.

## 🚀 Key Features

* **Robust Data Processing:** Automated extraction, filtering (Bandpass 8-30Hz, Notch 50Hz), and conversion of raw `.edf` files into highly optimized memory-mapped formats (`np.memmap`) to handle large datasets without RAM overflow. TFRecord integration for native multithreading.
* **State-of-the-Art Architectures:** Implementation of specialized CNNs for time-series biological data:
    * `EEGNet`: A compact convolutional network.
    * `ShallowConvNet`: A deep learning architecture that simulates Filter Bank Common Spatial Pattern (FBCSP) by directly extracting bandpower features (achieving **>63% Cross-Subject Accuracy**).
* **Strict Evaluation (LOSO-CV):** Implemented Leave-One-Subject-Out Cross-Validation to evaluate the true generalization capability of the model on entirely unseen users, reflecting real-world BCI challenges.
* **Subject-Specific Fine-Tuning:** Addressed inter-subject variability (the biological barrier) by implementing a specialized Transfer Learning module. Used ultra-low learning rates (1e-4) to perform rapid calibration for outlier subjects, successfully mitigating *Catastrophic Forgetting*.

## 📁 Repository Structure

```text
EEG-CLASSIFICATION/
├── configs/                 # Configuration parameters (frequencies, paths)
├── data/                    # Instructions to download the PhysioNet dataset
├── src/                     # Source code directory
│   ├── data_preprocessing.py # EDF parsing, MNE filtering, TFRecord generation
│   ├── fine_tune.py         # Outlier calibration using in-memory datasets
│   ├── models.py            # Keras implementation of EEGNet and ShallowConvNet
│   └── train.py             # LOSO-CV training loops
└── README.md
