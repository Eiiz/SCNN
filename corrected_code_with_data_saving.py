# -*- coding: utf-8 -*-
"""
Energy-Efficient Keyword Spotting: SNN vs. CNN - CORRECTED VERSION WITH DATA SAVING

This script implements and compares a Spiking Neural Network (SNN) and a
standard Convolutional Neural Network (CNN) for a keyword spotting task
on the Google Speech Commands dataset using the OFFICIAL dataset splits.

This version saves all training data to generate real figures.
"""

import torch
import torch.nn as nn
import torchaudio
import snntorch as snn
from snntorch import surrogate
from snntorch import spikegen

from torch.utils.data import DataLoader, Subset
from torchaudio.datasets import SPEECHCOMMANDS
import os
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns
import json

# --- 1. Configuration and Hyperparameters ---

# Data and device configuration
data_path = "./data/speech_commands"
os.makedirs(data_path, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Model and training hyperparameters
batch_size = 128
num_epochs = 20  # Increased for better convergence
learning_rate_cnn = 1e-3
learning_rate_snn = 5e-4
beta = 0.95  # SNN: Leaky neuron decay rate

# SNN specific parameters
num_steps = 25  # Number of time steps to simulate the SNN

# Audio processing parameters
sample_rate = 16000
n_mels = 64  # Number of mel frequency bands
n_fft = 1024
hop_length = 512
target_length_ms = 1000  # All clips will be 1 second

# --- 2. Dataset and Dataloaders (CORRECTED) ---

class SpeechCommandsSubset(SPEECHCOMMANDS):
    """A subset of the Speech Commands dataset using OFFICIAL splits."""
    def __init__(self, subset: str, path: str, download: bool, labels: list):
        super().__init__(path, download=download, subset=subset)

        self.labels = labels
        self.label_to_index = {label: i for i, label in enumerate(labels)}
        self.index_to_label = {i: label for i, label in enumerate(labels)}

        # Filter the dataset to only include the desired labels
        self._walker = [
            w for w in self._walker if self._get_label(w) in self.label_to_index
        ]

    def _get_label(self, filepath):
        return os.path.basename(os.path.dirname(filepath))

    def __getitem__(self, n):
        waveform, sample_rate, label, _, _ = super().__getitem__(n)
        return waveform, self.label_to_index[label]

# Define the keywords we want to classify
labels = ["yes", "no", "up", "down", "left", "right", "on", "off", "stop", "go"]

# Download and create datasets using OFFICIAL splits
print("Loading datasets with OFFICIAL splits...")
train_set = SpeechCommandsSubset("training", data_path, download=True, labels=labels)
validation_set = SpeechCommandsSubset("validation", data_path, download=True, labels=labels)
test_set = SpeechCommandsSubset("testing", data_path, download=True, labels=labels)

print(f"Official dataset splits:")
print(f"  Training samples: {len(train_set)}")
print(f"  Validation samples: {len(validation_set)}")
print(f"  Test samples: {len(test_set)}")

# Define a collate function to process a batch of audio clips
def collate_fn(batch):
    waveforms = []
    labels = []
    target_len = int(sample_rate * target_length_ms / 1000)

    for waveform, label in batch:
        # Pad or truncate to target length
        if waveform.shape[1] > target_len:
            waveform = waveform[:, :target_len]
        else:
            padding = target_len - waveform.shape[1]
            waveform = torch.nn.functional.pad(waveform, (0, padding))

        waveforms.append(waveform)
        labels.append(label)

    return torch.stack(waveforms), torch.tensor(labels)

# Create dataloaders
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, collate_fn=collate_fn, num_workers=0)
validation_loader = DataLoader(validation_set, batch_size=batch_size, shuffle=False, collate_fn=collate_fn, num_workers=0)
test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, collate_fn=collate_fn, num_workers=0)

# --- 3. Data Transformation (Audio -> Spectrogram -> Spikes) ---

# Mel Spectrogram transform
mel_spectrogram = torchaudio.transforms.MelSpectrogram(
    sample_rate=sample_rate,
    n_fft=n_fft,
    hop_length=hop_length,
    n_mels=n_mels
).to(device)

# --- 4. Model Definitions ---

# 4.1. IMPROVED CNN Model
class CNN_Model(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        # Improved architecture with more layers and better regularization
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5, padding=2)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(2)
        self.dropout1 = nn.Dropout2d(0.25)
        
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(2)
        self.dropout2 = nn.Dropout2d(0.25)
        
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(2)
        self.dropout3 = nn.Dropout2d(0.25)

        # Calculate flattened size dynamically
        with torch.no_grad():
            dummy_input = torch.randn(1, 1, n_mels, 32)
            dummy_output = self.pool3(self.dropout3(self.bn3(self.conv3(
                self.pool2(self.dropout2(self.bn2(self.conv2(
                    self.pool1(self.dropout1(self.bn1(self.conv1(dummy_input))))
                )))
            )))))
            self.flattened_size = dummy_output.numel()

        self.fc1 = nn.Linear(self.flattened_size, 512)
        self.dropout4 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, 256)
        self.dropout5 = nn.Dropout(0.5)
        self.fc3 = nn.Linear(256, num_classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.pool1(self.dropout1(self.relu(self.bn1(self.conv1(x)))))
        x = self.pool2(self.dropout2(self.relu(self.bn2(self.conv2(x)))))
        x = self.pool3(self.dropout3(self.relu(self.bn3(self.conv3(x)))))
        x = x.view(-1, self.flattened_size)
        x = self.relu(self.fc1(x))
        x = self.dropout4(x)
        x = self.relu(self.fc2(x))
        x = self.dropout5(x)
        x = self.fc3(x)
        return x

    def calculate_macs(self, input_res):
        """Calculate Multiply-Accumulate operations for a single forward pass."""
        macs = 0
        # Conv1
        macs += self.conv1.kernel_size[0] * self.conv1.kernel_size[1] * self.conv1.in_channels * self.conv1.out_channels * input_res[0] * input_res[1]
        res = (input_res[0] // 2, input_res[1] // 2)
        # Conv2
        macs += self.conv2.kernel_size[0] * self.conv2.kernel_size[1] * self.conv2.in_channels * self.conv2.out_channels * res[0] * res[1]
        res = (res[0] // 2, res[1] // 2)
        # Conv3
        macs += self.conv3.kernel_size[0] * self.conv3.kernel_size[1] * self.conv3.in_channels * self.conv3.out_channels * res[0] * res[1]
        res = (res[0] // 2, res[1] // 2)
        # FC layers
        macs += self.fc1.in_features * self.fc1.out_features
        macs += self.fc2.in_features * self.fc2.out_features
        macs += self.fc3.in_features * self.fc3.out_features
        return macs

# 4.2. IMPROVED SNN Model
n_inputs = n_mels * 32  # n_mels x time_steps from spectrogram
n_hidden = 512  # Increased hidden size
n_outputs = len(labels)
spike_grad = surrogate.atan()

class SNN_Model(nn.Module):
    def __init__(self):
        super().__init__()

        # Hyperparameters
        beta = 0.95
        threshold = 1.0

        # Layer Definition - Improved architecture
        self.fc1 = nn.Linear(n_inputs, n_hidden)
        self.lif1 = snn.Leaky(beta=beta, threshold=threshold, spike_grad=spike_grad)
        self.dropout1 = nn.Dropout(0.3)
        
        self.fc2 = nn.Linear(n_hidden, n_hidden // 2)
        self.lif2 = snn.Leaky(beta=beta, threshold=threshold, spike_grad=spike_grad)
        self.dropout2 = nn.Dropout(0.3)
        
        self.fc3 = nn.Linear(n_hidden // 2, n_outputs)
        self.lif3 = snn.Leaky(beta=beta, threshold=threshold, spike_grad=spike_grad)

        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, a=0, mode='fan_in', nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # Initialize neuron states
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()
        mem3 = self.lif3.init_leaky()
        spk3_rec = []

        # Reshape input
        x_flat = x.view(x.shape[0], x.shape[1], -1)

        # Loop over simulation time steps
        for step in range(x_flat.shape[0]):
            cur1 = self.fc1(x_flat[step])
            spk1, mem1 = self.lif1(cur1, mem1)
            spk1 = self.dropout1(spk1)
            
            cur2 = self.fc2(spk1)
            spk2, mem2 = self.lif2(cur2, mem2)
            spk2 = self.dropout2(spk2)
            
            cur3 = self.fc3(spk2)
            spk3, mem3 = self.lif3(cur3, mem3)
            spk3_rec.append(spk3)

        return torch.stack(spk3_rec, dim=0)

# 4.3. SCNN Model (Spiking Convolutional Neural Network)
class SCNN_Model(nn.Module):
    def __init__(self, num_classes, n_mels, time_frames):
        super().__init__()

        # Hyperparameters
        beta = 0.95
        threshold = 1.0
        spike_grad = surrogate.atan()

        # Store dimensions
        self.n_mels = n_mels
        self.time_frames = time_frames

        # Convolutional layers with spiking dynamics
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5, padding=2)
        self.lif1 = snn.Leaky(beta=beta, spike_grad=spike_grad, threshold=threshold)
        self.pool1 = nn.MaxPool2d(2)
        self.dropout1 = nn.Dropout2d(0.25)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.lif2 = snn.Leaky(beta=beta, spike_grad=spike_grad, threshold=threshold)
        self.pool2 = nn.MaxPool2d(2)
        self.dropout2 = nn.Dropout2d(0.25)

        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.lif3 = snn.Leaky(beta=beta, spike_grad=spike_grad, threshold=threshold)
        self.pool3 = nn.MaxPool2d(2)
        self.dropout3 = nn.Dropout2d(0.25)

        # Calculate flattened size
        with torch.no_grad():
            dummy_input = torch.zeros(1, 1, n_mels, time_frames)
            dummy_output = self.pool3(self.dropout3(self.lif3(self.conv3(
                self.pool2(self.dropout2(self.lif2(self.conv2(
                    self.pool1(self.dropout1(self.lif1(self.conv1(dummy_input))[0]))
                ))[0]))
            ))[0]))
            self.flattened_size = dummy_output.numel()

        # Fully connected layers
        self.fc1 = nn.Linear(self.flattened_size, 512)
        self.lif4 = snn.Leaky(beta=beta, spike_grad=spike_grad, threshold=threshold)
        self.dropout4 = nn.Dropout(0.5)
        
        self.fc2 = nn.Linear(512, 256)
        self.lif5 = snn.Leaky(beta=beta, spike_grad=spike_grad, threshold=threshold)
        self.dropout5 = nn.Dropout(0.5)
        
        self.fc3 = nn.Linear(256, num_classes)
        self.lif6 = snn.Leaky(beta=beta, spike_grad=spike_grad, threshold=threshold)

    def forward(self, x):
        # Initialize membrane potentials
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()
        mem3 = self.lif3.init_leaky()
        mem4 = self.lif4.init_leaky()
        mem5 = self.lif5.init_leaky()
        mem6 = self.lif6.init_leaky()
        spk6_rec = []

        for step in range(x.shape[0]):
            # Take a single time step
            spk_in = x[step].unsqueeze(1)  # Add channel dimension

            # Convolutional layers
            cur1 = self.conv1(spk_in)
            spk1, mem1 = self.lif1(cur1, mem1)
            pool1 = self.pool1(spk1)
            pool1 = self.dropout1(pool1)

            cur2 = self.conv2(pool1)
            spk2, mem2 = self.lif2(cur2, mem2)
            pool2 = self.pool2(spk2)
            pool2 = self.dropout2(pool2)

            cur3 = self.conv3(pool2)
            spk3, mem3 = self.lif3(cur3, mem3)
            pool3 = self.pool3(spk3)
            pool3 = self.dropout3(pool3)

            # Fully connected layers
            flat = pool3.view(pool3.size(0), -1)
            cur4 = self.fc1(flat)
            spk4, mem4 = self.lif4(cur4, mem4)
            spk4 = self.dropout4(spk4)

            cur5 = self.fc2(spk4)
            spk5, mem5 = self.lif5(cur5, mem5)
            spk5 = self.dropout5(spk5)

            cur6 = self.fc3(spk5)
            spk6, mem6 = self.lif6(cur6, mem6)
            spk6_rec.append(spk6)

        return torch.stack(spk6_rec, dim=0)

    def calculate_synops(self, avg_spikes):
        """Calculate Synaptic Operations (SynOps) for SCNN."""
        # Calculate total synapses in the network
        total_synapses = 0
        for module in [self.conv1, self.conv2, self.conv3, self.fc1, self.fc2, self.fc3]:
            if hasattr(module, 'weight'):
                total_synapses += module.weight.numel()
        
        # SynOps = average spikes per inference * total synapses
        return avg_spikes * total_synapses

# --- 5. Training and Evaluation Functions ---

def train_cnn(model, train_loader, validation_loader, optimizer, criterion, scheduler=None):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for waveforms, labels_batch in tqdm(train_loader, desc="Training CNN"):
        waveforms, labels_batch = waveforms.to(device), labels_batch.to(device)
        mel_spec = mel_spectrogram(waveforms)
        
        optimizer.zero_grad()
        outputs = model(mel_spec)
        loss = criterion(outputs, labels_batch)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels_batch.size(0)
        correct += (predicted == labels_batch).sum().item()
    
    train_loss = running_loss / len(train_loader)
    train_acc = 100 * correct / total
    
    # Validation
    val_loss, val_acc = test_cnn(model, validation_loader, criterion)
    
    if scheduler:
        scheduler.step(val_acc)
    
    return train_loss, train_acc, val_loss, val_acc

def test_cnn(model, test_loader, criterion):
    model.eval()
    correct = 0
    total = 0
    running_loss = 0.0
    
    with torch.no_grad():
        for waveforms, labels_batch in tqdm(test_loader, desc="Testing CNN"):
            waveforms, labels_batch = waveforms.to(device), labels_batch.to(device)
            mel_spec = mel_spectrogram(waveforms)
            outputs = model(mel_spec)
            loss = criterion(outputs, labels_batch)
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels_batch.size(0)
            correct += (predicted == labels_batch).sum().item()
    
    test_loss = running_loss / len(test_loader)
    test_acc = 100 * correct / total
    return test_loss, test_acc

def train_snn(model, train_loader, validation_loader, optimizer, criterion, scheduler=None):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for waveforms, labels_batch in tqdm(train_loader, desc="Training SNN"):
        waveforms, labels_batch = waveforms.to(device), labels_batch.to(device)
        mel_spec = mel_spectrogram(waveforms)
        
        # Convert to spikes
        if mel_spec.dim() == 4 and mel_spec.shape[1] == 1:
            mel_spec_3d = mel_spec.squeeze(1)
        else:
            mel_spec_3d = mel_spec
        
        spike_data = spikegen.rate(mel_spec_3d, num_steps=num_steps)
        
        optimizer.zero_grad()
        spk_out = model(spike_data)
        spike_counts = torch.sum(spk_out, dim=0)
        loss = criterion(spike_counts, labels_batch)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = spike_counts.max(1)
        total += labels_batch.size(0)
        correct += (predicted == labels_batch).sum().item()
    
    train_loss = running_loss / len(train_loader)
    train_acc = 100 * correct / total
    
    # Validation
    val_loss, val_acc, _ = test_snn(model, validation_loader, criterion)
    
    if scheduler:
        scheduler.step(val_acc)
    
    return train_loss, train_acc, val_loss, val_acc

def test_snn(model, test_loader, criterion):
    model.eval()
    correct = 0
    total = 0
    running_loss = 0.0
    total_spikes = 0
    num_samples = 0
    
    with torch.no_grad():
        for waveforms, labels_batch in tqdm(test_loader, desc="Testing SNN"):
            waveforms, labels_batch = waveforms.to(device), labels_batch.to(device)
            mel_spec = mel_spectrogram(waveforms)
            
            # Convert to spikes
            if mel_spec.dim() == 4 and mel_spec.shape[1] == 1:
                mel_spec_3d = mel_spec.squeeze(1)
            else:
                mel_spec_3d = mel_spec
            
            spike_data = spikegen.rate(mel_spec_3d, num_steps=num_steps)
            spk_out = model(spike_data)
            spike_counts = torch.sum(spk_out, dim=0)
            loss = criterion(spike_counts, labels_batch)
            
            running_loss += loss.item()
            _, predicted = spike_counts.max(1)
            total += labels_batch.size(0)
            correct += (predicted == labels_batch).sum().item()
            
            total_spikes += torch.sum(spk_out).item()
            num_samples += waveforms.shape[0]
    
    test_loss = running_loss / len(test_loader)
    test_acc = 100 * correct / total
    avg_spikes = total_spikes / num_samples if num_samples > 0 else 0
    return test_loss, test_acc, avg_spikes

def train_scnn(model, train_loader, validation_loader, optimizer, criterion, scheduler=None):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for waveforms, labels_batch in tqdm(train_loader, desc="Training SCNN"):
        waveforms, labels_batch = waveforms.to(device), labels_batch.to(device)
        mel_spec = mel_spectrogram(waveforms)
        
        # Convert to spikes
        if mel_spec.dim() == 4 and mel_spec.shape[1] == 1:
            mel_spec_3d = mel_spec.squeeze(1)
        else:
            mel_spec_3d = mel_spec
        
        spike_data = spikegen.rate(mel_spec_3d, num_steps=num_steps)
        
        optimizer.zero_grad()
        spk_out = model(spike_data)
        spike_counts = torch.sum(spk_out, dim=0)
        loss = criterion(spike_counts, labels_batch)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = spike_counts.max(1)
        total += labels_batch.size(0)
        correct += (predicted == labels_batch).sum().item()
    
    train_loss = running_loss / len(train_loader)
    train_acc = 100 * correct / total
    
    # Validation
    val_loss, val_acc, _ = test_scnn(model, validation_loader, criterion)
    
    if scheduler:
        scheduler.step(val_acc)
    
    return train_loss, train_acc, val_loss, val_acc

def test_scnn(model, test_loader, criterion):
    model.eval()
    correct = 0
    total = 0
    running_loss = 0.0
    total_spikes = 0
    num_samples = 0
    
    with torch.no_grad():
        for waveforms, labels_batch in tqdm(test_loader, desc="Testing SCNN"):
            waveforms, labels_batch = waveforms.to(device), labels_batch.to(device)
            mel_spec = mel_spectrogram(waveforms)
            
            # Convert to spikes
            if mel_spec.dim() == 4 and mel_spec.shape[1] == 1:
                mel_spec_3d = mel_spec.squeeze(1)
            else:
                mel_spec_3d = mel_spec
            
            spike_data = spikegen.rate(mel_spec_3d, num_steps=num_steps)
            spk_out = model(spike_data)
            spike_counts = torch.sum(spk_out, dim=0)
            loss = criterion(spike_counts, labels_batch)
            
            running_loss += loss.item()
            _, predicted = spike_counts.max(1)
            total += labels_batch.size(0)
            correct += (predicted == labels_batch).sum().item()
            
            total_spikes += torch.sum(spk_out).item()
            num_samples += waveforms.shape[0]
    
    test_loss = running_loss / len(test_loader)
    test_acc = 100 * correct / total
    avg_spikes = total_spikes / num_samples if num_samples > 0 else 0
    return test_loss, test_acc, avg_spikes

# --- 6. Main Execution ---
if __name__ == "__main__":
    print("="*60)
    print("SCNN KEYWORD SPOTTING - CORRECTED VERSION WITH DATA SAVING")
    print("Using OFFICIAL Google Speech Commands dataset splits")
    print("="*60)
    
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Initialize data storage
    training_data = {
        'cnn_train_losses': [],
        'cnn_val_accs': [],
        'snn_train_losses': [],
        'snn_val_accs': [],
        'scnn_train_losses': [],
        'scnn_val_accs': [],
        'final_results': {}
    }
    
    # --- CNN Training and Evaluation ---
    print("\n--- Training CNN Model ---")
    cnn_model = CNN_Model(num_classes=len(labels)).to(device)
    cnn_criterion = nn.CrossEntropyLoss()
    cnn_optimizer = torch.optim.Adam(cnn_model.parameters(), lr=learning_rate_cnn, weight_decay=1e-4)
    cnn_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(cnn_optimizer, mode='max', factor=0.5, patience=3)
    
    best_cnn_val_acc = 0
    for epoch in range(num_epochs):
        train_loss, train_acc, val_loss, val_acc = train_cnn(
            cnn_model, train_loader, validation_loader, cnn_optimizer, cnn_criterion, cnn_scheduler
        )
        
        # Store training data
        training_data['cnn_train_losses'].append(train_loss)
        training_data['cnn_val_accs'].append(val_acc)
        
        print(f"CNN Epoch {epoch+1}/{num_epochs}: Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        
        if val_acc > best_cnn_val_acc:
            best_cnn_val_acc = val_acc
            torch.save(cnn_model.state_dict(), "best_cnn_model.pth")
    
    # Test CNN
    cnn_test_loss, cnn_test_acc = test_cnn(cnn_model, test_loader, cnn_criterion)
    print(f"\nCNN Test Results: Loss: {cnn_test_loss:.4f}, Accuracy: {cnn_test_acc:.2f}%")
    
    # Calculate CNN MACs
    sample_wave, _ = next(iter(test_loader))
    sample_spec = mel_spectrogram(sample_wave.to(device))
    cnn_macs = cnn_model.calculate_macs((sample_spec.shape[2], sample_spec.shape[3]))
    print(f"CNN MACs per inference: {cnn_macs/1e6:.2f} Million")
    
    # Store CNN final results
    training_data['final_results']['cnn_test_acc'] = cnn_test_acc
    training_data['final_results']['cnn_macs'] = cnn_macs / 1e6
    
    # --- SNN Training and Evaluation ---
    print("\n--- Training SNN Model ---")
    snn_model = SNN_Model().to(device)
    snn_criterion = nn.CrossEntropyLoss()
    snn_optimizer = torch.optim.Adam(snn_model.parameters(), lr=learning_rate_snn, weight_decay=1e-4)
    snn_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(snn_optimizer, mode='max', factor=0.5, patience=3)
    
    best_snn_val_acc = 0
    for epoch in range(num_epochs):
        train_loss, train_acc, val_loss, val_acc = train_snn(
            snn_model, train_loader, validation_loader, snn_optimizer, snn_criterion, snn_scheduler
        )
        
        # Store training data
        training_data['snn_train_losses'].append(train_loss)
        training_data['snn_val_accs'].append(val_acc)
        
        print(f"SNN Epoch {epoch+1}/{num_epochs}: Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        
        if val_acc > best_snn_val_acc:
            best_snn_val_acc = val_acc
            torch.save(snn_model.state_dict(), "best_snn_model.pth")
    
    # Test SNN
    snn_test_loss, snn_test_acc, snn_avg_spikes = test_snn(snn_model, test_loader, snn_criterion)
    print(f"\nSNN Test Results: Loss: {snn_test_loss:.4f}, Accuracy: {snn_test_acc:.2f}%, Avg Spikes: {snn_avg_spikes:.2f}")
    
    # Calculate SNN SynOps
    snn_synops = snn_avg_spikes * (n_inputs * 512 + 512 * 256 + 256 * len(labels))
    print(f"SNN SynOps per inference: {snn_synops/1e6:.2f} Million")
    
    # Store SNN final results
    training_data['final_results']['snn_test_acc'] = snn_test_acc
    training_data['final_results']['snn_synops'] = snn_synops / 1e6
    training_data['final_results']['snn_avg_spikes'] = snn_avg_spikes
    
    # --- SCNN Training and Evaluation ---
    print("\n--- Training SCNN Model ---")
    time_frames = 32  # This should match your actual spectrogram time dimension
    scnn_model = SCNN_Model(num_classes=len(labels), n_mels=n_mels, time_frames=time_frames).to(device)
    scnn_criterion = nn.CrossEntropyLoss()
    scnn_optimizer = torch.optim.Adam(scnn_model.parameters(), lr=learning_rate_snn, weight_decay=1e-4)
    scnn_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(scnn_optimizer, mode='max', factor=0.5, patience=3)
    
    best_scnn_val_acc = 0
    for epoch in range(num_epochs):
        train_loss, train_acc, val_loss, val_acc = train_scnn(
            scnn_model, train_loader, validation_loader, scnn_optimizer, scnn_criterion, scnn_scheduler
        )
        
        # Store training data
        training_data['scnn_train_losses'].append(train_loss)
        training_data['scnn_val_accs'].append(val_acc)
        
        print(f"SCNN Epoch {epoch+1}/{num_epochs}: Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        
        if val_acc > best_scnn_val_acc:
            best_scnn_val_acc = val_acc
            torch.save(scnn_model.state_dict(), "best_scnn_model.pth")
    
    # Test SCNN
    scnn_test_loss, scnn_test_acc, scnn_avg_spikes = test_scnn(scnn_model, test_loader, scnn_criterion)
    print(f"\nSCNN Test Results: Loss: {scnn_test_loss:.4f}, Accuracy: {scnn_test_acc:.2f}%, Avg Spikes: {scnn_avg_spikes:.2f}")
    
    # Calculate SCNN SynOps
    scnn_synops = scnn_model.calculate_synops(scnn_avg_spikes)
    print(f"SCNN SynOps per inference: {scnn_synops/1e6:.2f} Million")
    
    # Store SCNN final results
    training_data['final_results']['scnn_test_acc'] = scnn_test_acc
    training_data['final_results']['scnn_synops'] = scnn_synops / 1e6
    training_data['final_results']['scnn_avg_spikes'] = scnn_avg_spikes
    
    # --- Save Training Data ---
    with open('training_data.json', 'w') as f:
        json.dump(training_data, f, indent=2)
    print(f"\n✓ Training data saved to training_data.json")
    
    # --- Final Results Summary ---
    print("\n" + "="*60)
    print("FINAL RESULTS SUMMARY (OFFICIAL DATASET SPLITS)")
    print("="*60)
    print(f"CNN Test Accuracy: {cnn_test_acc:.2f}%")
    print(f"CNN MACs per inference: {cnn_macs/1e6:.2f} Million")
    print(f"\nSNN Test Accuracy: {snn_test_acc:.2f}%")
    print(f"SNN SynOps per inference: {snn_synops/1e6:.2f} Million")
    print(f"SNN Avg Spikes per inference: {snn_avg_spikes:.2f}")
    print(f"\nSCNN Test Accuracy: {scnn_test_acc:.2f}%")
    print(f"SCNN SynOps per inference: {scnn_synops/1e6:.2f} Million")
    print(f"SCNN Avg Spikes per inference: {scnn_avg_spikes:.2f}")

    print("\n" + "="*60)
    print("TRAINING COMPLETE! Now you can run generate_real_figures.py to create figures with real data.")
    print("="*60)
