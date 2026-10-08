import numpy as np

FILES = [
    r"D:\Projects\CSIdataset\train_data.npz",
    r"D:\Projects\CSIdataset\val_data.npz",
    r"D:\Projects\CSIdataset\test_data.npz",
]


def inspect_npz(file_path):
    print("=" * 70)
    print(f"FILE: {file_path}")
    print("=" * 70)

    data = np.load(file_path)
    print(f"Arrays in file: {data.files}")

    X = data['X']
    y = data['y']

    print(f"\n--- X (Features) ---")
    print(f"  Shape: {X.shape}")
    print(f"  Dtype: {X.dtype}")
    print(f"  Min:   {X.min():.6f}")
    print(f"  Max:   {X.max():.6f}")
    print(f"  Mean:  {X.mean():.6f}")
    print(f"  Std:   {X.std():.6f}")

    print(f"\n--- y (Targets) ---")
    print(f"  Shape: {y.shape}")
    print(f"  Dtype: {y.dtype}")
    print(f"  Min:   {y.min(axis=0)}")
    print(f"  Max:   {y.max(axis=0)}")
    print(f"  Mean:  {y.mean(axis=0)}")
    print(f"  Std:   {y.std(axis=0)}")

    print(f"\n--- Sample Values ---")
    print(f"  X[0, 0, :5, 0] (first 5 subcarriers, amp, window 0, time 0): {X[0, 0, :5, 0]}")
    print(f"  X[0, 0, :5, 1] (first 5 subcarriers, phase, window 0, time 0): {X[0, 0, :5, 1]}")
    print(f"  y[0] (first target): {y[0]}")
    print(f"  y[1] (second target): {y[1]}")
    print(f"  y[-1] (last target): {y[-1]}")

    print(f"\n--- Per-Channel Stats (X) ---")
    for ch, name in enumerate(['Amplitude', 'Phase']):
        ch_data = X[:, :, :, ch]
        print(f"  {name}: min={ch_data.min():.4f}, max={ch_data.max():.4f}, mean={ch_data.mean():.4f}, std={ch_data.std():.4f}")

    print(f"\n--- NaN/Inf Check ---")
    print(f"  X has NaN: {np.isnan(X).any()}")
    print(f"  X has Inf: {np.isinf(X).any()}")
    print(f"  y has NaN: {np.isnan(y).any()}")
    print(f"  y has Inf: {np.isinf(y).any()}")

    print()


if __name__ == "__main__":
    for f in FILES:
        inspect_npz(f)

    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    for f in FILES:
        data = np.load(f)
        X = data['X']
        y = data['y']
        size_mb = (X.nbytes + y.nbytes) / (1024 * 1024)
        print(f"  {f.split(chr(92))[-1]:<20} X: {str(X.shape):<25} y: {str(y.shape):<12} ({size_mb:.1f} MB)")
