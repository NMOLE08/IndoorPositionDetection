"""
Dataset Splitter for Wi-Fi CSI Positioning Dataset (HALOC)
-----------------------------------------------------------
Split Configuration:
  - Training   : 0.csv, 1.csv, 2.csv, 3.csv
  - Validation : 4.csv
  - Test       : 5.csv

Can be run locally or inside Google Colab.
"""

import os
import sys
import shutil
import pandas as pd

# -------------------------------------------------------------
# 1. Configuration & Path Detection (Google Colab / Local)
# -------------------------------------------------------------
# Check if running in Google Colab
IN_COLAB = 'google.colab' in sys.modules

if IN_COLAB:
    print("Detected Google Colab environment.")
    # Uncomment if using Google Drive:
    # from google.colab import drive
    # drive.mount('/content/drive')
    # DATA_DIR = "/content/drive/MyDrive/HALOC"
    DATA_DIR = "/content/HALOC"
    OUTPUT_DIR = "/content/dataset_splits"
else:
    # Local paths (defaults to dataset location in workspace)
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(BASE_DIR, "Position", "HALOC", "HALOC")
    if not os.path.exists(DATA_DIR):
        DATA_DIR = BASE_DIR  # fallback
    OUTPUT_DIR = os.path.join(BASE_DIR, "Position", "HALOC", "splits")

# Define split mapping
SPLIT_CONFIG = {
    "train": ["0.csv", "1.csv", "2.csv", "3.csv"],
    "val":   ["4.csv"],
    "test":  ["5.csv"]
}


# -------------------------------------------------------------
# 2. Main Splitting Function (Consolidated CSVs)
# -------------------------------------------------------------
def split_and_merge_csv(data_dir=DATA_DIR, output_dir=OUTPUT_DIR):
    """
    Loads individual CSV files according to split specifications,
    merges them into train.csv, val.csv, and test.csv, and saves them.
    """
    os.makedirs(output_dir, exist_ok=True)
    print(f"Reading dataset from : {data_dir}")
    print(f"Saving splits to     : {output_dir}\n")

    summary = {}

    for split_name, file_list in SPLIT_CONFIG.items():
        print(f"[*] Processing '{split_name}' split ({', '.join(file_list)})...")
        dfs = []
        for filename in file_list:
            file_path = os.path.join(data_dir, filename)
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"Error: Could not find {file_path}")
            
            df = pd.read_csv(file_path)
            # Add a source_file column to track origin if needed
            df["source_file"] = filename
            dfs.append(df)
            print(f"    - Loaded {filename}: {len(df):,} rows")

        # Merge DataFrames for this split
        merged_df = pd.concat(dfs, ignore_index=True)
        out_file = os.path.join(output_dir, f"{split_name}.csv")
        merged_df.to_csv(out_file, index=False)

        summary[split_name] = {
            "files": file_list,
            "rows": len(merged_df),
            "output_path": out_file,
            "size_mb": round(os.path.getsize(out_file) / (1024 * 1024), 2)
        }
        print(f"    -> Saved {out_file} ({len(merged_df):,} rows, {summary[split_name]['size_mb']} MB)\n")

    # Display Final Summary
    print("=" * 60)
    print(f"{'Split':<12} | {'Files':<24} | {'Total Rows':<12} | {'Size (MB)'}")
    print("-" * 60)
    total_rows = 0
    for split_name, stats in summary.items():
        files_str = ", ".join(stats["files"])
        print(f"{split_name:<12} | {files_str:<24} | {stats['rows']:<12,} | {stats['size_mb']} MB")
        total_rows += stats["rows"]
    print("-" * 60)
    print(f"{'Total':<12} | {'':<24} | {total_rows:<12,} |")
    print("=" * 60)

    return summary


# -------------------------------------------------------------
# 3. Alternative: Organize into Folder Directories
# -------------------------------------------------------------
def organize_into_folders(data_dir=DATA_DIR, output_dir=OUTPUT_DIR, copy_files=True):
    """
    Copies or moves CSV files into folder structure:
      output_dir/train/
      output_dir/val/
      output_dir/test/
    """
    folder_base = os.path.join(output_dir, "folders")
    os.makedirs(folder_base, exist_ok=True)
    print(f"\n[*] Organizing files into subdirectories at: {folder_base}")

    for split_name, file_list in SPLIT_CONFIG.items():
        split_folder = os.path.join(folder_base, split_name)
        os.makedirs(split_folder, exist_ok=True)
        for filename in file_list:
            src = os.path.join(data_dir, filename)
            dst = os.path.join(split_folder, filename)
            if copy_files:
                shutil.copy2(src, dst)
            else:
                shutil.move(src, dst)
            print(f"    Copied {filename} -> {split_name}/")


if __name__ == "__main__":
    split_and_merge_csv()
    # Uncomment if you also need subfolder organization:
    # organize_into_folders()
