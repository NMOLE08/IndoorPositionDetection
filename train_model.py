import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    TimeDistributed,
    Conv1D,
    MaxPooling1D,
    Flatten,
    LSTM,
    Dense,
    Dropout,
    BatchNormalization,
)
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

train_data = np.load('train_data.npz')
val_data = np.load('val_data.npz')
test_data = np.load('test_data.npz')

X_train, y_train = train_data['X'], train_data['y']
X_val, y_val = val_data['X'], val_data['y']
X_test, y_test = test_data['X'], test_data['y']

print(f"Loaded X_train shape: {X_train.shape}")
print(f"Loaded y_train shape: {y_train.shape}")

model = Sequential([
    TimeDistributed(Conv1D(32, kernel_size=3, activation='relu', padding='same'), input_shape=(50, 119, 2)),
    TimeDistributed(BatchNormalization()),
    TimeDistributed(MaxPooling1D(pool_size=2)),
    TimeDistributed(Conv1D(64, kernel_size=3, activation='relu', padding='same')),
    TimeDistributed(BatchNormalization()),
    TimeDistributed(MaxPooling1D(pool_size=2)),
    TimeDistributed(Flatten()),
    LSTM(64, return_sequences=False),
    Dropout(0.3),
    Dense(32, activation='relu'),
    Dropout(0.3),
    Dense(2, activation='linear'),
])

model.compile(optimizer='adam', loss='mse', metrics=['mae'])

model.summary()

callbacks = [
    EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
    ModelCheckpoint('best_csi_model.h5', monitor='val_loss', save_best_only=True),
]

history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=50,
    batch_size=32,
    callbacks=callbacks,
    verbose=1,
)

test_loss, test_mae = model.evaluate(X_test, y_test)
print(f"Test Loss (MSE): {test_loss:.6f}")
print(f"Test Mean Absolute Error: {test_mae:.4f} meters")

predictions = model.predict(X_test[:10])
print("\nSample Predictions vs Ground Truth:")
for i in range(10):
    print(f"  Pred: ({predictions[i, 0]:.3f}, {predictions[i, 1]:.3f}) | True: ({y_test[i, 0]:.3f}, {y_test[i, 1]:.3f})")
