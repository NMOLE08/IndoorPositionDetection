import json
import os
import numpy as np
import pandas as pd

BASE_DIR = r"D:\Projects\CSIdataset\Position\HALOC\HALOC"
OUTPUT_DIR = r"D:\Projects\CSIdataset"

TRAIN_FILES = [
    os.path.join(BASE_DIR, "0.csv"),
    os.path.join(BASE_DIR, "1.csv"),
    os.path.join(BASE_DIR, "2.csv"),
    os.path.join(BASE_DIR, "3.csv"),
]
VAL_FILE = os.path.join(BASE_DIR, "4.csv")
TEST_FILE = os.path.join(BASE_DIR, "5.csv")


def text_to_csi_raw(data_series):
    return np.array([json.loads(row) for row in data_series], dtype=np.float32)


def split_iq(raw_csi_array):
    I = raw_csi_array[:, 0::2]
    Q = raw_csi_array[:, 1::2]
    return I, Q


def compute_amplitude_and_phase(I, Q, unwrap_phase=True):
    amplitude = np.sqrt(I**2 + Q**2)
    phase = np.arctan2(Q, I)
    if unwrap_phase:
        phase = np.unwrap(phase, axis=1)
    return amplitude, phase


def prune_guard_bands(amplitude, phase, lead_trim=6, trail_trim=3):
    end_idx = amplitude.shape[1] - trail_trim
    amp_pruned = amplitude[:, lead_trim:end_idx]
    phase_pruned = phase[:, lead_trim:end_idx]
    return amp_pruned, phase_pruned


def extract_targets(df, drop_z=True):
    cols = ['x', 'y'] if drop_z else ['x', 'y', 'z']
    return df[cols].values.astype(np.float32)


def normalize_packetwise_zscore(features, eps=1e-8):
    mean = np.mean(features, axis=1, keepdims=True)
    std = np.std(features, axis=1, keepdims=True)
    return (features - mean) / (std + eps)


def create_sliding_windows(features, targets, window_size=50, step_size=25, target_mode='last'):
    N = len(features)
    X_list = []
    y_list = []
    for start in range(0, N - window_size + 1, step_size):
        end = start + window_size
        window_feat = features[start:end]
        if target_mode == 'last':
            window_target = targets[end - 1]
        elif target_mode == 'mean':
            window_target = np.mean(targets[start:end], axis=0)
        else:
            raise ValueError(f"Unknown target_mode: {target_mode}")
        X_list.append(window_feat)
        y_list.append(window_target)
    X_windows = np.array(X_list, dtype=np.float32)
    y_windows = np.array(y_list, dtype=np.float32)
    return X_windows, y_windows


def extract_and_preprocess_csi(
    data_source,
    window_size=50,
    step_size=25,
    lead_trim=6,
    trail_trim=3,
    target_mode='last',
    feature_type='both'
):
    if isinstance(data_source, str):
        print(f"Reading {data_source}...")
        df_in = pd.read_csv(data_source)
    else:
        df_in = data_source

    if 'local_timestamp' in df_in.columns:
        df_in = df_in.sort_values(by=['local_timestamp']).reset_index(drop=True)
    elif 'id' in df_in.columns:
        df_in = df_in.sort_values(by=['id']).reset_index(drop=True)

    raw = text_to_csi_raw(df_in['data'])
    print(f"  Step 1: Raw CSI shape = {raw.shape}")

    I, Q = split_iq(raw)
    print(f"  Step 2: I shape = {I.shape}, Q shape = {Q.shape}")

    amp, phase = compute_amplitude_and_phase(I, Q, unwrap_phase=True)
    print(f"  Step 3: Amplitude shape = {amp.shape}, Phase shape = {phase.shape}")

    amp_p, phase_p = prune_guard_bands(amp, phase, lead_trim, trail_trim)
    print(f"  Step 4: Pruned shape = {amp_p.shape}")

    targets = extract_targets(df_in, drop_z=True)
    print(f"  Step 5: Targets shape = {targets.shape}")

    amp_norm = normalize_packetwise_zscore(amp_p)
    phase_norm = normalize_packetwise_zscore(phase_p)
    print(f"  Step 6: Normalized amplitude shape = {amp_norm.shape}")

    if feature_type == 'both':
        features = np.stack([amp_norm, phase_norm], axis=-1)
    elif feature_type == 'amplitude':
        features = amp_norm[:, :, np.newaxis]
    elif feature_type == 'phase':
        features = phase_norm[:, :, np.newaxis]
    else:
        raise ValueError(f"Invalid feature_type: {feature_type}")

    X, y = create_sliding_windows(features, targets, window_size, step_size, target_mode)
    print(f"  Step 7: X shape = {X.shape}, y shape = {y.shape}")
    return X, y


def process_split(csv_files, output_name, window_size=50, step_size=25):
    if isinstance(csv_files, str):
        csv_files = [csv_files]

    all_X = []
    all_y = []
    for csv_file in csv_files:
        X, y = extract_and_preprocess_csi(
            csv_file,
            window_size=window_size,
            step_size=step_size,
            feature_type='both'
        )
        all_X.append(X)
        all_y.append(y)

    X_combined = np.concatenate(all_X, axis=0)
    y_combined = np.concatenate(all_y, axis=0)

    out_path = os.path.join(OUTPUT_DIR, f"{output_name}_data.npz")
    np.savez_compressed(out_path, X=X_combined, y=y_combined)
    print(f"\nSaved: {out_path} ({os.path.getsize(out_path) / (1024 * 1024):.2f} MB)")
    print(f"  X: {X_combined.shape}, y: {y_combined.shape}")


if __name__ == "__main__":
    print("=" * 60)
    print("Processing TRAIN split...")
    print("=" * 60)
    process_split(TRAIN_FILES, "train")

    print("\n" + "=" * 60)
    print("Processing VAL split...")
    print("=" * 60)
    process_split(VAL_FILE, "val")

    print("\n" + "=" * 60)
    print("Processing TEST split...")
    print("=" * 60)
    process_split(TEST_FILE, "test")

    print("\n" + "=" * 60)
    print("All splits processed successfully!")
    print("=" * 60)
