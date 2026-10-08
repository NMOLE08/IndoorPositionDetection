"""
Drop the Z-axis from CSI target tensors.

Use this on already-exported train/val/test .npz files (y of shape [N, 3]).
The 7-step pipeline in DataExtraction.ipynb now extracts only (x, y), so new
exports are already 2D. This script is for converting older 3-column tensors.

Usage:
    python drop_z_axis.py
    python drop_z_axis.py --in-place
    python drop_z_axis.py --path D:\\Projects\\CSIdataset\\train_data.npz
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np

DEFAULT_SEARCH_DIRS = [
    Path("."),
    Path("Position") / "HALOC" / "processed_tensors",
    Path(r"D:\Projects\CSIdataset"),
    Path(r"D:\Projects\CSIdataset\Position\HALOC\processed_tensors"),
]

SPLIT_NAMES = ("train_data.npz", "val_data.npz", "test_data.npz")


def drop_z_from_y(y: np.ndarray) -> np.ndarray:
    """Keep only x, y. Accepts (N, 3) or already (N, 2)."""
    y = np.asarray(y)
    if y.ndim != 2:
        raise ValueError(f"Expected y with shape (N, C), got {y.shape}")
    if y.shape[1] == 2:
        return y.astype(np.float32, copy=False)
    if y.shape[1] >= 3:
        return y[:, :2].astype(np.float32, copy=True)
    raise ValueError(f"Expected at least 2 target columns, got {y.shape}")


def convert_npz(npz_path: Path, in_place: bool) -> Path:
    data = np.load(npz_path)
    if "X" not in data.files or "y" not in data.files:
        raise KeyError(f"{npz_path} must contain arrays 'X' and 'y'. Found: {data.files}")

    X = data["X"]
    y_old = data["y"]
    y_new = drop_z_from_y(y_old)

    print(f"{npz_path}")
    print(f"  X: {X.shape}")
    print(f"  y: {y_old.shape} -> {y_new.shape}  (x, y only)")

    if y_old.shape == y_new.shape:
        print("  Already 2D (x, y). No write needed.")
        return npz_path

    out_path = npz_path if in_place else npz_path.with_name(npz_path.stem + "_xy.npz")
    extra = {k: data[k] for k in data.files if k not in ("X", "y")}
    np.savez_compressed(out_path, X=X, y=y_new, **extra)
    print(f"  Saved: {out_path}")
    return out_path


def find_split_files() -> list[Path]:
    found = []
    seen = set()
    for name in SPLIT_NAMES:
        for directory in DEFAULT_SEARCH_DIRS:
            candidate = directory / name
            resolved = candidate.resolve() if candidate.exists() else None
            if resolved is None or str(resolved) in seen:
                continue
            seen.add(str(resolved))
            found.append(resolved)
    return found


def main() -> None:
    parser = argparse.ArgumentParser(description="Remove Z coordinates from CSI .npz targets.")
    parser.add_argument(
        "--path",
        type=Path,
        default=None,
        help="Single .npz file. If omitted, convert train/val/test in known folders.",
    )
    parser.add_argument(
        "--in-place",
        action="store_true",
        help="Overwrite the original .npz. Default writes <name>_xy.npz.",
    )
    args = parser.parse_args()

    if args.path is not None:
        files = [args.path]
    else:
        files = find_split_files()
        if not files:
            raise FileNotFoundError(
                "No train_data.npz / val_data.npz / test_data.npz found. "
                "Pass --path or run DataExtraction.ipynb first."
            )

    print("Dropping Z-axis from target tensors (keeping x, y)\n")
    for f in files:
        if not os.path.exists(f):
            print(f"Skip (missing): {f}")
            continue
        convert_npz(Path(f), in_place=args.in_place)


if __name__ == "__main__":
    main()
