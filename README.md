# Computational Efficiency and Theoretical Energy Analysis of Keyword Spotting

This repository contains the implementation and experimental code for the paper:

**"Computational Efficiency and Theoretical Energy Analysis of Keyword Spotting: A Comparative Study of Convolutional and Spiking Neural Networks"**

## 📋 Overview

This project presents a comprehensive comparison of three neural network architectures for keyword spotting on the Google Speech Commands dataset:
- **CNN** (Convolutional Neural Network)
- **SNN** (Spiking Neural Network)  
- **SCNN** (Spiking Convolutional Neural Network)

The research evaluates these models across multiple dimensions including classification accuracy, computational efficiency, and theoretical energy consumption analysis.

## 🎯 Key Results

- **CNN**: 88.19% accuracy, 22.74M MACs per inference
- **SCNN**: 86.06% accuracy, 74.51M SynOps per inference
- **SNN**: 76.51% accuracy, 22.70M SynOps per inference

## 📁 Repository Structure

```
SCNN/
├── corrected_code_with_data_saving.py  # Main training script for all 3 models
├── generate_real_figures.py             # Script to generate paper figures
├── monitor_training.py                  # Monitor training progress
├── train_colors_scnn.py                 # Training script for custom colors dataset
├── paper.tex                            # LaTeX source for the paper
├── references.bib                       # Bibliography file
├── README.md                            # This file
└── .gitignore                           # Git ignore rules
```

## 🚀 Getting Started

### Prerequisites

- Python 3.7+
- PyTorch 1.8+
- CUDA (optional, for GPU acceleration)
- Required Python packages:
  ```bash
  pip install torch torchaudio snntorch matplotlib seaborn scikit-learn tqdm numpy
  ```

### Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/Eiiz/SCNN.git
   cd SCNN
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Usage

#### 1. Training Models

Run the main training script to train all three models (CNN, SNN, SCNN):

```bash
python corrected_code_with_data_saving.py
```

This script will:
- Download the Google Speech Commands dataset automatically
- Train CNN, SNN, and SCNN models
- Save training data to `training_data.json`
- Save best models as `best_cnn_model.pth`, `best_snn_model.pth`, `best_scnn_model.pth`

**Note**: Training all three models takes several hours. The script uses official Google Speech Commands dataset splits for fair comparison.

#### 2. Generating Paper Figures

After training is complete, generate all figures for the paper:

```bash
python generate_real_figures.py
```

This will create:
- `figure1_training_curves.png` - Training loss and validation accuracy curves
- `figure2_confusion_matrix.png` - Confusion matrix for SCNN model
- `figure3_inference_pipeline.png` - End-to-end inference pipeline visualization
- `figure4_performance_comparison.png` - Comprehensive performance comparison

#### 3. Monitoring Training (Optional)

In a separate terminal, you can monitor training progress:

```bash
python monitor_training.py
```

This script watches for `training_data.json` and generates figures as training progresses.

## 📊 Dataset

The project uses the **Google Speech Commands Dataset v0.02** with 10 keywords:
- `yes`, `no`, `up`, `down`, `left`, `right`, `on`, `off`, `stop`, `go`

The dataset is automatically downloaded when running the training script. Official dataset splits are used:
- Training set: ~85,000 samples
- Validation set: ~10,000 samples  
- Test set: ~11,000 samples

## 🏗️ Model Architectures

### CNN (Convolutional Neural Network)
- 3 convolutional layers (32, 64, 128 filters)
- Batch normalization and dropout regularization
- 3 fully connected layers (512, 256, 10 units)
- ReLU activations

### SNN (Spiking Neural Network)
- 3 fully connected layers with Leaky Integrate-and-Fire (LIF) neurons
- Rate coding for spike generation
- 25 time steps per inference
- β=0.95 decay rate, θ=1.0 threshold

### SCNN (Spiking Convolutional Neural Network)
- 3 convolutional layers with LIF neurons
- 3 fully connected layers with LIF neurons
- Combines spatial feature extraction with spiking dynamics
- Same hyperparameters as SNN

## 📈 Key Metrics

The paper evaluates models using:

1. **Classification Accuracy**: Percentage of correctly classified test samples
2. **Computational Efficiency**: 
   - MACs (Multiply-Accumulate operations) for CNN
   - SynOps (Synaptic Operations) for SNN and SCNN
3. **Theoretical Energy Consumption**: Estimated energy per inference based on hardware models
4. **Spike Efficiency**: Average number of spikes per inference (spiking models only)

## 📝 Scripts Description

### `corrected_code_with_data_saving.py`
Main training script that:
- Implements CNN, SNN, and SCNN architectures
- Trains all three models on Google Speech Commands dataset
- Saves training metrics to `training_data.json`
- Calculates computational efficiency metrics (MACs/SynOps)
- Saves best model checkpoints

### `generate_real_figures.py`
Figure generation script that:
- Loads training data from `training_data.json`
- Loads trained models from checkpoint files
- Generates all 4 figures used in the paper
- Uses real experimental data (not synthetic)

### `monitor_training.py`
Monitoring utility that:
- Watches for `training_data.json` updates
- Generates figures automatically as training progresses
- Useful for long training sessions

### `train_colors_scnn.py`
Alternative training script for a custom Vietnamese colors dataset:
- 4 classes: `do`, `dondep`, `tatca`, `xanh`
- Uses same SCNN architecture
- Not used in the main paper

## 🔬 Experimental Setup

- **Batch Size**: 128
- **Epochs**: 20
- **Learning Rates**: 
  - CNN: 1e-3
  - SNN/SCNN: 5e-4
- **Optimizer**: Adam with weight decay 1e-4
- **Scheduler**: ReduceLROnPlateau
- **Audio Processing**: 
  - Sample rate: 16 kHz
  - Mel spectrogram: 64 bins, 1024-point FFT
  - Time steps: 25 (for spiking models)

## 📄 Paper

The complete paper is available in `paper.tex`. To compile:

```bash
pdflatex paper.tex
bibtex paper
pdflatex paper.tex
pdflatex paper.tex
```

## 🤝 Citation

If you use this code in your research, please cite:

```bibtex
@article{scnn2024,
  title={Computational Efficiency and Theoretical Energy Analysis of Keyword Spotting: A Comparative Study of Convolutional and Spiking Neural Networks},
  author={Nguyen, Khang},
  journal={IEEE Conference},
  year={2024}
}
```

## 📧 Contact

- **Author**: Khang Nguyen
- **Email**: khangvogia070302@gmail.com
- **Institution**: Ho Chi Minh City International University

## 📜 License

This project is provided for research purposes. Please refer to the paper for detailed methodology and results.

## 🙏 Acknowledgments

- PyTorch and snntorch library developers
- Google Speech Commands dataset creators
- All contributors to the open-source deep learning ecosystem
