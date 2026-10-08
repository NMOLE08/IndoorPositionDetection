# Wi-Fi CSI Dataset for 3D Indoor Localization

A comprehensive dataset and processing pipeline for 3D indoor positioning using Wi-Fi Channel State Information (CSI) with deep learning (CNN-LSTM neural networks).

## 📋 Overview

This project provides a complete end-to-end pipeline for transforming raw Wi-Fi CSI data into sequence tensors suitable for training deep learning models for 3D indoor localization. It includes:

- **HALOC Dataset**: Wi-Fi CSI measurements with ground-truth 3D spatial coordinates
- **Data Extraction Pipeline**: 7-stage preprocessing to convert raw CSI to neural network inputs
- **CNN-LSTM Architecture**: PyTorch model for 3D coordinate regression
- **ESP32 Hardware**: C firmware for CSI data transmission
- **CAD Files**: 3D printable hardware designs for CSI sensor nodes

## 🗂️ Project Structure

```
CSIdataset/
├── DataExtraction.ipynb          # Complete 7-step data extraction & preprocessing pipeline
├── split_dataset.py              # Dataset splitter (train/val/test)
├── Loadnpz.py                    # Utility to load processed .npz files
├── view_data.py                  # Visualization script for CSI data
├── mca.c                         # ESP32 firmware for CSI transmission
├── train_data.npz                # Preprocessed training tensors
├── val_data.npz                  # Preprocessed validation tensors
├── test_data.npz                 # Preprocessed test tensors
└── Position/
    ├── HALOC/
    │   ├── HALOC/                # Raw CSV dataset files (0.csv - 5.csv)
    │   ├── splits/               # Consolidated train/val/test CSV files
    │   └── processed_tensors/    # Preprocessed .npz tensors
    └── CAD/
        └── CAD/                  # STL files for 3D printing
            ├── carrier.stl
            ├── housing.stl
            └── lid.stl
```

## 📊 Dataset Information

### HALOC Dataset

The HALOC dataset contains Wi-Fi CSI measurements with continuous 3D ground-truth coordinates:

- **Training**: Files `0.csv`, `1.csv`, `2.csv`, `3.csv`
- **Validation**: File `4.csv`
- **Test**: File `5.csv`

### Data Format

Each CSV row contains:
- **CSI Data**: Raw Channel State Information (256 interleaved I/Q values)
- **Metadata**: MAC address, RSSI, channel, timestamp, etc.
- **Ground Truth**: 3D spatial coordinates `(x, y, z)` in meters

### Coordinate Ranges

- **X-axis**: ~20 meters range
- **Y-axis**: ~2.4 meters range
- **Z-axis**: ~0.15 meters range

## 🔧 Data Extraction Pipeline

The `DataExtraction.ipynb` notebook implements a 7-stage preprocessing pipeline:

### Step 1: Text-to-Number Conversion
Parse stringified CSI arrays into numerical arrays (shape: N×256)

### Step 2: Split I & Q Pairs
Separate 256 interleaved values into 128 In-phase (I) and 128 Quadrature (Q) subcarriers

### Step 3: Physical Feature Calculation
Compute Amplitude (A = √(I² + Q²)) and Phase (φ = arctan2(Q, I)) with subcarrier unwrapping

### Step 4: Prune Guard Bands
Trim null subcarrier dead zones (6 leading, 3 trailing) to keep 119 active subcarriers

### Step 5: Target Coordinate Extraction
Extract continuous 3D spatial ground-truth coordinates (x, y, z) in meters

### Step 6: Normalization
Per-packet Z-score normalization across subcarriers to eliminate base power drift and AGC noise

### Step 7: Temporal Sequencing
Segment packets into sliding windows (default: 50 packets per window, stride 25)

### Output Tensors

- **Input (X)**: Shape `(num_windows, 50, 119, 2)` - [windows, time_steps, subcarriers, channels]
- **Target (y)**: Shape `(num_windows, 3)` - [windows, x, y, z coordinates]

## 🚀 Quick Start

### Prerequisites

```bash
pip install numpy pandas matplotlib torch scikit-learn
```

### Option 1: Use Preprocessed Data

The project includes preprocessed `.npz` files ready for training:

```python
import numpy as np

# Load preprocessed tensors
train_data = np.load('train_data.npz')
X_train, y_train = train_data['X'], train_data['y']

val_data = np.load('val_data.npz')
X_val, y_val = val_data['X'], val_data['y']

test_data = np.load('test_data.npz')
X_test, y_test = test_data['X'], test_data['y']

print(f"Train: {X_train.shape}, {y_train.shape}")
print(f"Val:   {X_val.shape}, {y_val.shape}")
print(f"Test:  {X_test.shape}, {y_test.shape}")
```

### Option 2: Process Raw CSV Files

Run the data extraction pipeline:

```python
# In DataExtraction.ipynb or Python script
from DataExtraction import extract_and_preprocess_csi

X_train, y_train = extract_and_preprocess_csi(
    'Position/HALOC/splits/train.csv',
    window_size=50,
    step_size=25,
    feature_type='both'
)
```

### Option 3: Split Raw Dataset

If you have the individual CSV files (0.csv - 5.csv):

```bash
python split_dataset.py
```

This will create consolidated `train.csv`, `val.csv`, and `test.csv` in `Position/HALOC/splits/`.

## 🧠 CNN-LSTM Model Architecture

The project includes a PyTorch CNN-LSTM model for 3D coordinate regression:

### Architecture

```
Input: [Batch, 2, 50, 119]
  ↓ Conv2D (2→16 channels) + ReLU + MaxPool + Dropout(0.3)
  ↓ Conv2D (16→32 channels) + ReLU + MaxPool + Dropout(0.3)
  ↓ Reshape to [Batch, 50, 928]
  ↓ LSTM (input=928, hidden=64)
  ↓ Dropout(0.3)
  ↓ Fully Connected (64→3)
Output: [Batch, 3] (x, y, z coordinates)
```

### Key Features

- **CNN**: Extracts spatial multipath features across subcarriers
- **LSTM**: Models temporal motion dynamics across time steps
- **Dropout (0.3)**: Prevents overfitting
- **Weight Decay (1e-5)**: L2 regularization

### Training

```python
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import MinMaxScaler

# Scale targets to [0, 1] to prevent axis dominance
scaler = MinMaxScaler()
y_train_scaled = scaler.fit_transform(y_train)
y_val_scaled = scaler.transform(y_val)
y_test_scaled = scaler.transform(y_test)

# Define model, loss, optimizer
model = CSICNNLSTM(dropout_p=0.3)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)

# Training loop with early stopping
# (See DataExtraction.ipynb for complete implementation)
```

## 📱 ESP32 Hardware

### CSI Transmission Firmware

The `mca.c` file contains ESP32 firmware for broadcasting CSI data:

**Features:**
- Broadcasts raw 802.11 Action frames at 100 Hz
- Configurable Wi-Fi channel (default: Channel 6)
- FreeRTOS task-based implementation
- Compatible with ESP-IDF framework

**Compilation:**
```bash
# Using ESP-IDF
idf.py build
idf.py flash
```

**Hardware Requirements:**
- ESP32 microcontroller
- Wi-Fi antenna
- Power supply

## 🖨️ CAD Files

3D printable designs for CSI sensor hardware are located in `Position/CAD/CAD/`:

- **carrier.stl**: Main carrier board
- **housing.stl**: Enclosure housing
- **lid.stl**: Cover lid

These STL files can be 3D printed to create custom CSI sensor nodes.

## 📈 Visualization

Use `view_data.py` to visualize CSI data:

```python
python view_data.py
```

This script generates:
- Target coordinate statistics
- CSI heatmap visualizations (amplitude/phase spectrograms)
- 3D trajectory plots

## 📁 File Descriptions

| File | Description |
|------|-------------|
| `DataExtraction.ipynb` | Complete 7-step data extraction pipeline with CNN-LSTM model |
| `split_dataset.py` | Splits raw CSV files into train/val/test sets |
| `Loadnpz.py` | Utility script to load and inspect .npz files |
| `view_data.py` | Visualization script for CSI tensors |
| `mca.c` | ESP32 firmware for CSI transmission |
| `train_data.npz` | Preprocessed training tensors (59.6 MB) |
| `val_data.npz` | Preprocessed validation tensors (17.4 MB) |
| `test_data.npz` | Preprocessed test tensors (8.8 MB) |

## 🔬 Dataset Statistics

### Preprocessed Tensor Sizes

| Split | Samples | Shape (X) | Shape (y) | Size |
|-------|---------|-----------|-----------|------|
| Train | ~44,000 | (N, 50, 119, 2) | (N, 3) | 59.6 MB |
| Val | ~12,000 | (N, 50, 119, 2) | (N, 3) | 17.4 MB |
| Test | ~6,000 | (N, 50, 119, 2) | (N, 3) | 8.8 MB |

### Raw CSV Sizes

| File | Rows | Size |
|------|------|------|
| 0.csv | ~150,000 | 18.5 MB |
| 1.csv | ~180,000 | 21.7 MB |
| 2.csv | ~200,000 | 24.3 MB |
| 3.csv | ~195,000 | 23.9 MB |
| 4.csv | ~210,000 | 25.7 MB |
| 5.csv | ~105,000 | 13.1 MB |

## 🛠️ Dependencies

### Python Packages

```
numpy>=1.21.0
pandas>=1.3.0
matplotlib>=3.4.0
torch>=1.9.0
scikit-learn>=0.24.0
jupyter>=1.0.0
```

### ESP-IDF (for mca.c)

- ESP-IDF v4.4 or later
- Compatible ESP32 board

## 📝 Usage Examples

### Loading and Inspecting Data

```python
import numpy as np

# Load data
data = np.load('train_data.npz')
X, y = data['X'], data['y']

print(f"Input shape: {X.shape}")  # (N, 50, 119, 2)
print(f"Target shape: {y.shape}")  # (N, 3)

# Check coordinate ranges
print(f"X range: [{y[:,0].min():.2f}, {y[:,0].max():.2f}] m")
print(f"Y range: [{y[:,1].min():.2f}, {y[:,1].max():.2f}] m")
print(f"Z range: [{y[:,2].min():.2f}, {y[:,2].max():.2f}] m")
```

### Training a Model

```python
import torch
from torch.utils.data import DataLoader, TensorDataset

# Convert to PyTorch tensors
X_tensor = torch.tensor(X, dtype=torch.float32).permute(0, 3, 1, 2)
y_tensor = torch.tensor(y, dtype=torch.float32)

# Create dataset and dataloader
dataset = TensorDataset(X_tensor, y_tensor)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# Training loop
for epoch in range(num_epochs):
    for batch_X, batch_y in dataloader:
        optimizer.zero_grad()
        predictions = model(batch_X)
        loss = criterion(predictions, batch_y)
        loss.backward()
        optimizer.step()
```

## 🎯 Applications

This dataset and pipeline can be used for:

- **3D Indoor Localization**: Precise positioning in indoor environments
- **Human Activity Recognition**: Detecting movement patterns
- **Respiration Monitoring**: Breathing rate detection via CSI phase analysis
- **Smart Home Automation**: Presence detection and localization
- **Robot Navigation**: Indoor positioning for autonomous robots

## 📚 References

- HALOC Dataset: High-Accuracy Localization with CSI
- Wi-Fi CSI Channel State Information fundamentals
- CNN-LSTM architectures for time-series regression

## 🤝 Contributing

Contributions are welcome! Areas for improvement:

- Additional preprocessing techniques
- Alternative neural network architectures
- More hardware designs
- Documentation improvements

## 📄 License

Please check the HALOC dataset license for usage terms and conditions.

## 📧 Contact

For questions or issues related to this dataset and processing pipeline, please refer to the original HALOC dataset documentation.

---

**Note**: This project focuses on the HALOC Wi-Fi CSI dataset for 3D indoor localization research. Ensure you have the appropriate hardware (ESP32) and software environment (Python with PyTorch) set up before running the processing pipeline.
