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
├── main.py                              # Main training script for all 3 models
│   (or corrected_code_with_data_saving.py in older versions)
├── generate_real_figures.py             # Script to generate paper figures
├── figure_plot.py                       # Alternative figure generation script
├── monitor_training.py                  # Monitor training progress
├── train_colors_scnn.py                 # Training script for custom colors dataset
├── paper.tex                            # LaTeX source for the paper
├── references.bib                       # Bibliography file
├── requirements.txt                     # Python dependencies
├── README.md                            # This file
└── .gitignore                           # Git ignore rules
```

## 🚀 Getting Started

### Prerequisites

- **Python**: 3.7 or higher
- **PyTorch**: 1.8.0 or higher (with CUDA support recommended for faster training)
- **CUDA**: Optional but recommended for GPU acceleration (CUDA 10.2+)
- **RAM**: At least 8GB recommended
- **Disk Space**: ~3GB for dataset and models

### Installation

#### Method 1: Using Git Clone (Recommended)

1. **Clone this repository:**
   ```bash
   git clone https://github.com/Eiiz/SCNN.git
   cd SCNN
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
   
   Or install manually:
   ```bash
   pip install torch>=1.8.0 torchaudio>=0.8.0 snntorch>=0.5.0 matplotlib>=3.3.0 seaborn>=0.11.0 scikit-learn>=0.24.0 tqdm>=4.60.0 numpy>=1.19.0
   ```

#### Method 2: Download ZIP

1. **Download the repository:**
   - Go to https://github.com/Eiiz/SCNN
   - Click "Code" → "Download ZIP"
   - Extract the ZIP file

2. **Navigate to the directory:**
   ```bash
   cd SCNN-main
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

### Verify Installation

Test if all dependencies are installed correctly:

```bash
python -c "import torch; import torchaudio; import snntorch; print('All dependencies installed successfully!')"
```

### Usage

#### 1. Training Models

Run the main training script to train all three models (CNN, SNN, SCNN):

```bash
python main.py
```

**Note**: If you see `main.py` not found, the file might be named `corrected_code_with_data_saving.py` in older versions. Use:
```bash
python corrected_code_with_data_saving.py
```

This script will:
- **Automatically download** the Google Speech Commands dataset (if not present)
  - Dataset will be saved to `./data/speech_commands/`
  - Download size: ~1.4 GB
  - First run may take 10-30 minutes depending on internet speed
- Train CNN, SNN, and SCNN models sequentially
- Save training data to `training_data.json`
- Save best models as:
  - `best_cnn_model.pth` (~5-10 MB)
  - `best_snn_model.pth` (~2-5 MB)
  - `best_scnn_model.pth` (~10-15 MB)

**Training Time Estimates:**
- **CNN**: ~2-4 hours (on GPU) or ~8-12 hours (on CPU)
- **SNN**: ~3-5 hours (on GPU) or ~10-15 hours (on CPU)
- **SCNN**: ~4-6 hours (on GPU) or ~12-18 hours (on CPU)
- **Total**: ~9-15 hours (on GPU) or ~30-45 hours (on CPU)

**Note**: The script uses official Google Speech Commands dataset splits for fair comparison with published results.

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

### Google Speech Commands Dataset v0.02

The project uses the **Google Speech Commands Dataset v0.02** with 10 keywords:
- `yes`, `no`, `up`, `down`, `left`, `right`, `on`, `off`, `stop`, `go`

**Dataset Source:**
- **Official Website**: https://ai.googleblog.com/2017/08/launching-speech-commands-dataset.html
- **Direct Download**: http://download.tensorflow.org/data/speech_commands_v0.02.tar.gz
- **Paper**: Warden, P. (2018). "Speech Commands: A Dataset for Limited-Vocabulary Speech Recognition". arXiv:1804.03209
- **License**: Creative Commons BY 4.0 License

**Dataset Statistics:**
- Total samples: ~105,829 one-second audio clips
- Training set: ~85,000 samples
- Validation set: ~10,000 samples  
- Test set: ~11,000 samples
- Sample rate: 16 kHz
- Format: WAV files (1 second duration)

**Automatic Download:**
The dataset is automatically downloaded when running the training script using `torchaudio.datasets.SPEECHCOMMANDS`. The script will:
1. Check if dataset exists in `./data/speech_commands/`
2. Download automatically if not found
3. Extract and organize files according to official splits

**Manual Download (Alternative):**
If you prefer to download manually:

```bash
# Create data directory
mkdir -p data/speech_commands

# Download dataset (1.4 GB)
wget http://download.tensorflow.org/data/speech_commands_v0.02.tar.gz -O data/speech_commands/speech_commands_v0.02.tar.gz

# Extract
tar -xzf data/speech_commands/speech_commands_v0.02.tar.gz -C data/speech_commands/
```

The training script will detect the manually downloaded dataset and use it automatically.

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

### `main.py` (or `corrected_code_with_data_saving.py`)
Main training script that:
- Implements CNN, SNN, and SCNN architectures
- Downloads Google Speech Commands dataset automatically via `torchaudio.datasets.SPEECHCOMMANDS`
- Trains all three models on Google Speech Commands dataset
- Saves training metrics to `training_data.json`
- Calculates computational efficiency metrics (MACs/SynOps)
- Saves best model checkpoints
- **Source**: Original implementation for this paper

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

## 📚 Sources and References

### Datasets

1. **Google Speech Commands Dataset v0.02**
   - **Source**: https://ai.googleblog.com/2017/08/launching-speech-commands-dataset.html
   - **Download**: http://download.tensorflow.org/data/speech_commands_v0.02.tar.gz
   - **Citation**: 
     ```bibtex
     @article{warden2018speech,
       title={Speech Commands: A Dataset for Limited-Vocabulary Speech Recognition},
       author={Warden, Pete},
       journal={arXiv preprint arXiv:1804.03209},
       year={2018}
     }
     ```

### Libraries and Frameworks

1. **PyTorch**
   - **Source**: https://pytorch.org/
   - **License**: BSD-style license
   - **Citation**: 
     ```bibtex
     @article{paszke2019pytorch,
       title={PyTorch: An Imperative Style, High-Performance Deep Learning Library},
       author={Paszke, Adam and others},
       journal={Advances in Neural Information Processing Systems},
       year={2019}
     }
     ```

2. **snntorch**
   - **Source**: https://snntorch.readthedocs.io/
   - **GitHub**: https://github.com/jeshraghian/snntorch
   - **License**: MIT License
   - **Citation**:
     ```bibtex
     @article{eshraghian2021snntorch,
       title={Training Spiking Neural Networks Using Lessons from Deep Learning},
       author={Eshraghian, Jason K. and others},
       journal={arXiv preprint arXiv:2109.12894},
       year={2021}
     }
     ```

3. **torchaudio**
   - **Source**: https://pytorch.org/audio/
   - Part of PyTorch ecosystem
   - Provides `SPEECHCOMMANDS` dataset loader

### Related Papers

1. **Spiking Neural Networks**
   - Maass, W. (1997). "Networks of spiking neurons: the third generation of neural network models". Neural Networks.

2. **Spiking Convolutional Neural Networks**
   - Lee, J. H., et al. (2016). "Training deep spiking neural networks using backpropagation". Frontiers in Neuroscience.

3. **Keyword Spotting**
   - Chen, G., et al. (2014). "Small-footprint keyword spotting using deep neural networks". ICASSP.

## 🙏 Acknowledgments

- **PyTorch Team**: For the excellent deep learning framework
- **snntorch Developers**: For the spiking neural network library
- **Google Research**: For creating and maintaining the Speech Commands dataset
- **All Contributors**: To the open-source deep learning ecosystem

## 📄 License

This project is provided for research purposes. Please refer to the paper for detailed methodology and results.

**Dataset License**: Google Speech Commands Dataset is licensed under Creative Commons BY 4.0 License.

**Code License**: This code is provided as-is for research purposes. Please cite the paper if you use this code in your research.
