# ♻️ Garbage Classification with Deep Learning in MATLAB

A deep-learning-based waste classification system built entirely in **MATLAB**, using transfer learning on pre-trained CNNs (**ResNet-18** and **EfficientNet-B0**) to categorize waste images into **10 classes**. The project also explores a **Progressive Unfreezing** fine-tuning strategy and ships with an interactive **MATLAB GUI** for real-time inference via file upload or webcam.

---

## 📑 Table of Contents

- [Overview](#overview)
- [Classes](#classes)
- [Datasets](#datasets)
- [Project Structure](#project-structure)
- [Models & Training Strategies](#models--training-strategies)
- [Evaluation & Visualization](#evaluation--visualization)
- [GUI Application](#gui-application)
- [Requirements](#requirements)
- [Getting Started](#getting-started)
- [Results](#results)
- [License](#license)

---

## Overview

Proper waste sorting is crucial for recycling efficiency and environmental protection. This project investigates how well modern CNN architectures — fine-tuned through transfer learning — can classify photographs of everyday waste items. Three distinct training pipelines are compared:

| Pipeline | Base Model | Strategy |
|---|---|---|
| **Standard Transfer Learning** | ResNet-18 | Replace final FC layer, train end-to-end |
| **Standard Transfer Learning** | EfficientNet-B0 | Replace final FC layer, train end-to-end |
| **Progressive Unfreezing** | EfficientNet-B0 | Two-phase training (FC-only → full fine-tune) |

All models are trained on a **combined multi-source dataset** with data augmentation and evaluated on a **fixed held-out test set** from the RealWaste dataset.

---

## Classes

The system recognizes the following **10 waste categories**:

| # | Class | Example Items |
|---|---|---|
| 1 | `battery` | AA batteries, button cells |
| 2 | `biological` | Food scraps, organic waste |
| 3 | `cardboard` | Boxes, packaging |
| 4 | `clothes` | Shirts, fabrics |
| 5 | `glass` | Bottles, jars |
| 6 | `metal` | Cans, foil |
| 7 | `paper` | Newspapers, documents |
| 8 | `plastic` | Bottles, bags, containers |
| 9 | `shoes` | Sneakers, boots |
| 10 | `trash` | Mixed / non-recyclable waste |

---

## Datasets

Training data is aggregated from **three sources** to improve diversity and real-world generalization:

| Source | Description | Link |
|---|---|---|
| **Garbage Classification v2** (Kaggle) | Studio-quality images across 10 classes | [Kaggle Dataset](https://www.kaggle.com/datasets/sumn2u/garbage-classification-v2) |
| **RealWaste** (UCI) | Real-world waste images captured at a materials recovery facility | [UCI Dataset](https://archive.ics.uci.edu/dataset/908/realwaste) |
| **real_world** | Supplementary real-world images collected by the team | Included in the Kaggle dataset |

- **50 %** of the RealWaste data is reserved as a **fixed holdout test set** (`testRealWaste_holdout.mat`) to ensure fair, consistent evaluation across all models.
- The remaining 50 % of RealWaste plus the full Kaggle and real_world sets form the **training pool**, which is then split **85 / 15** into training and validation subsets.

---

## Project Structure

```
├── ResNet-18/
│   ├── birlesik_egitim_eski.m       # ResNet-18 transfer learning training script
│   └── trainedNet_combined.mat      # Trained ResNet-18 model weights
│
├── EfficientNet-B0/
│   ├── efficientnet_egitim.m        # EfficientNet-B0 transfer learning training script
│   ├── efficientnet_grafikleri.m    # Model comparison & evaluation plots
│   ├── trainedNet_efficientnet.mat  # Trained EfficientNet-B0 model weights
│   ├── testRealWaste_holdout.mat    # Fixed holdout test set (50% of RealWaste)
│   └── Grafikler/                   # Generated evaluation charts (PNG)
│
├── Progressive Unfreezing/
│   ├── progressive_unfreezing.m     # Two-phase progressive unfreezing script
│   └── trainedNet_progressive.mat   # Trained progressive model weights
│
├── Gui/
│   ├── atik_gui.m                   # Interactive classification GUI (file + webcam)
│   ├── trainedNet_combined.mat      # Model used by the GUI (ResNet-18)
│   ├── trainedNet_efficientnet.mat  # Model used by the GUI (EfficientNet-B0)
│   └── trainedNet_progressive.mat   # Model used by the GUI (Progressive)
│
└── README.md
```

---

## Models & Training Strategies

### 1. ResNet-18 — Standard Transfer Learning

> **Script:** `ResNet-18/birlesik_egitim_eski.m`

- Loads pre-trained **ResNet-18** (ImageNet weights).
- Replaces the final `fc1000` → `softmax` → `classification` layers with a new 10-class head.
- New FC layer uses **10× learn-rate factor** for faster adaptation.

| Hyperparameter | Value |
|---|---|
| Optimizer | Adam |
| Epochs | 20 |
| Initial LR | 1 × 10⁻⁴ |
| LR Schedule | Piecewise (×0.5 every 5 epochs) |
| Batch Size | 32 |
| Validation Patience | 5 |

### 2. EfficientNet-B0 — Standard Transfer Learning

> **Script:** `EfficientNet-B0/efficientnet_egitim.m`

- Loads pre-trained **EfficientNet-B0** (ImageNet weights).
- Replaces the dense classification head with a new 10-class FC layer.
- Same 10× learn-rate factor on the new head.

| Hyperparameter | Value |
|---|---|
| Optimizer | Adam |
| Epochs | 20 |
| Initial LR | 1 × 10⁻⁴ |
| LR Schedule | Piecewise (×0.5 every 5 epochs) |
| Batch Size | 16 |
| Validation Patience | 5 |

### 3. EfficientNet-B0 — Progressive Unfreezing

> **Script:** `Progressive Unfreezing/progressive_unfreezing.m`

A two-phase training approach designed to reduce catastrophic forgetting when fine-tuning on a domain-shifted dataset:

| Phase | What Trains | Epochs | Learning Rate | Rationale |
|---|---|---|---|---|
| **Phase 1** | Primarily the new FC layer (backbone learns very slowly due to low global LR) | 5 | 3 × 10⁻⁵ (global) · 10× (FC) | Warm up the classifier head |
| **Phase 2** | All layers (full fine-tuning) | 10 | 1 × 10⁻⁵ | Gentle end-to-end refinement |

### Data Augmentation (all pipelines)

All training pipelines apply on-the-fly augmentation:

- Random rotation (±10°)
- Random X/Y translation (±10 px)
- Random X/Y scaling (0.95–1.05)
- Random horizontal flip
- Automatic grayscale → RGB conversion

---

## Evaluation & Visualization

> **Script:** `EfficientNet-B0/efficientnet_grafikleri.m`

The evaluation script generates side-by-side comparison charts between **ResNet-18** and **EfficientNet-B0** on the fixed holdout test set:

| Chart | Description |
|---|---|
| **Model Performance** | Grouped bar chart comparing overall Accuracy and Macro F1-Score |
| **Per-Class Metrics** | Precision, Recall, and F1-Score broken down by class |
| **Confusion Matrix** | Side-by-side confusion matrices for both models |
| **Class Samples** | Random example images from each class |
| **Prediction Results** | EfficientNet-B0 predictions with correct/incorrect color coding |
| **Dataset Distribution** | Per-class sample counts for each data source and the combined set |

All charts are saved to the `EfficientNet-B0/Grafikler/` directory.

---

## GUI Application

> **Script:** `Gui/atik_gui.m`

A modern, dark-themed MATLAB GUI for interactive waste classification:

### Features

- 📁 **File Upload** — Load any image (JPG, PNG, BMP, GIF) from disk
- 📸 **Webcam Capture** — Real-time camera preview with one-click capture
- ✂️ **Manual Crop** — Draw a rectangle to crop the region of interest
- 🎯 **Center Crop** — Automatic center crop with configurable ratio (50 %–90 %)
- 📊 **Confidence Visualization** — Horizontal bar chart showing all class probabilities
- 🏷️ **Turkish Labels** — Each class is displayed with a localized Turkish name and a unique color

### How to Run

```matlab
>> cd Gui
>> atik_gui
```

---

## Requirements

- **MATLAB R2021a** or later
- **Deep Learning Toolbox**
- **Deep Learning Toolbox Model for ResNet-18 Network** (Add-On)
- **Deep Learning Toolbox Model for EfficientNet-B0 Network** (Add-On)
- **Image Processing Toolbox** (for `imcrop` / `imrect` in the GUI)
- A CUDA-capable **GPU** is recommended (training falls back to CPU automatically)
- **Webcam Support Package** (optional, for camera features in the GUI)

---

## Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/<your-username>/garbage-classification-matlab.git
cd garbage-classification-matlab
```

### 2. Download the Datasets

1. **Kaggle — Garbage Classification v2**
   → [https://www.kaggle.com/datasets/sumn2u/garbage-classification-v2](https://www.kaggle.com/datasets/sumn2u/garbage-classification-v2)

2. **UCI — RealWaste**
   → [https://archive.ics.uci.edu/dataset/908/realwaste](https://archive.ics.uci.edu/dataset/908/realwaste)

3. Extract and organize the data so that each class has its own subfolder.

### 3. Update Paths

In each training script, update the `basePath` variable to point to your local dataset directory:

```matlab
basePath = "C:\path\to\your\dataset";
```

### 4. Train a Model

```matlab
% Option A — ResNet-18
>> run('ResNet-18/birlesik_egitim_eski.m')

% Option B — EfficientNet-B0
>> run('EfficientNet-B0/efficientnet_egitim.m')

% Option C — Progressive Unfreezing
>> run('Progressive Unfreezing/progressive_unfreezing.m')
```

### 5. Evaluate & Compare

```matlab
>> run('EfficientNet-B0/efficientnet_grafikleri.m')
```

### 6. Launch the GUI

```matlab
>> cd Gui
>> atik_gui
```

---

## Results

Model performance on the **fixed RealWaste holdout test set**:

| Model | Accuracy | Macro F1-Score |
|---|---|---|
| ResNet-18 | %76.9 | %60.9 |
| EfficientNet-B0 | %83.4 | %66.7 |
| EfficientNet-B0 (Progressive) | %84.8 | %75.4 |

> *Run the evaluation script to fill in the results for your trained models.*

---

## License

This project was developed as a university assignment and is shared for educational and portfolio purposes. The datasets are subject to their respective licenses — please refer to the original dataset pages linked above.
