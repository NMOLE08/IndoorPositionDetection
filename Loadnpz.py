import numpy as np

# Load the .npz file (using raw string r'...' to prevent \t from being read as a tab)
data = np.load(r'D:\Projects\CSIdataset\val_data.npz')

# Check what arrays are stored inside the .npz archive
print(data.files)  # e.g., might output ['X', 'y']

# Access a specific array by its name
X_array = data['X']
y_array = data['y']

# Check the shapes to ensure they match your model's expected input
print("X shape:", X_array.shape)
print("y shape:", y_array.shape)