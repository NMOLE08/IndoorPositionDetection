import numpy as np
import matplotlib.pyplot as plt

# 1. Load the training data properly
file_path = r'D:\Projects\CSIdataset\val_data.npz'
data = np.load(file_path)
X = data['X']
y = data['y']

print("=========================================")
print("DATASET SHAPES & TYPES")
print("=========================================")
print(f"X shape (Features): {X.shape}")
print(f"y shape (Targets) : {y.shape}")
print(f"X data type: {X.dtype}, y data type: {y.dtype}")

# Check for NaN or Inf values which destroy training stability
if np.isnan(X).any() or np.isnan(y).any():
    print("⚠️ WARNING: NaN values detected in dataset!")
if np.isinf(X).any() or np.isinf(y).any():
    print("⚠️ WARNING: Inf values detected in dataset!")

print("\n=========================================")
print("TARGET COORDINATES (y) INSPECTION")
print("=========================================")
n_coords = y.shape[1] if y.ndim == 2 else 0
if n_coords == 3:
    print("First 15 target coordinates (X, Y, Z):")
else:
    print("First 15 target coordinates (X, Y):")
print(y[:15])

print("\nCoordinate Ranges & Statistics:")
print(f"X-axis range: {np.min(y[:, 0]):.2f} to {np.max(y[:, 0]):.2f} meters | Mean: {np.mean(y[:, 0]):.2f} | Std: {np.std(y[:, 0]):.2f}")
print(f"Y-axis range: {np.min(y[:, 1]):.2f} to {np.max(y[:, 1]):.2f} meters | Mean: {np.mean(y[:, 1]):.2f} | Std: {np.std(y[:, 1]):.2f}")
if n_coords >= 3:
    print(f"Z-axis range: {np.min(y[:, 2]):.2f} to {np.max(y[:, 2]):.2f} meters | Mean: {np.mean(y[:, 2]):.2f} | Std: {np.std(y[:, 2]):.2f}")
else:
    print("Z-axis: dropped (planar x, y targets only)")

print("\n=========================================")
print("CSI SENSOR DATA (X) INSPECTION")
print("=========================================")
print(f"X feature statistics -> Min: {np.min(X):.4f}, Max: {np.max(X):.4f}, Mean: {np.mean(X):.4f}, Std: {np.std(X):.4f}")

# Grab the very first sample in the dataset
sample_idx = 0
sample_csi = X[sample_idx] # Shape will be (50, 119, 2)

# Separate the two channels (Amplitude/Phase or Real/Imaginary)
channel_0 = sample_csi[:, :, 0] # Shape: (50 time steps, 119 subcarriers)
channel_1 = sample_csi[:, :, 1] 

# Plotting the data as a Heatmap
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Plot Channel 0
im1 = axes[0].imshow(channel_0, aspect='auto', cmap='viridis', origin='lower')
axes[0].set_title(f"Sample {sample_idx} - Channel 0")
axes[0].set_xlabel("Subcarriers (119)")
axes[0].set_ylabel("Time Steps (50)")
fig.colorbar(im1, ax=axes[0])

# Plot Channel 1
im2 = axes[1].imshow(channel_1, aspect='auto', cmap='plasma', origin='lower')
axes[1].set_title(f"Sample {sample_idx} - Channel 1")
axes[1].set_xlabel("Subcarriers (119)")
axes[1].set_ylabel("Time Steps (50)")
fig.colorbar(im2, ax=axes[1])

plt.suptitle(f"CSI Heatmap for Target Coordinates: {y[sample_idx]}")
plt.tight_layout()
plt.show()