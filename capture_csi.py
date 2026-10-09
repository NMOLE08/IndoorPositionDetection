"""Capture and preprocess ESP32 CSI data from a serial connection."""

import ast
import csv
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import serial
from serial import SerialException


BAUD_RATE = 115200
OUTPUT_FILE = Path("csi_training_data.csv")
CSI_ARRAY_PATTERN = re.compile(r"\[[^\]]*\]")


def extract_amplitudes(line: str) -> list[float]:
    """Extract the CSI integer array and calculate I/Q amplitudes."""
    array_match = CSI_ARRAY_PATTERN.search(line)
    if array_match is None:
        raise ValueError("CSI_DATA line does not contain an integer array")

    try:
        raw_values: Any = ast.literal_eval(array_match.group(0))
    except (SyntaxError, ValueError) as error:
        raise ValueError("CSI array is not a valid integer array") from error

    if (
        not isinstance(raw_values, list)
        or len(raw_values) == 0
        or len(raw_values) % 2 != 0
        or not all(isinstance(value, int) for value in raw_values)
    ):
        raise ValueError("CSI array must contain a non-empty even number of integers")

    values = np.asarray(raw_values, dtype=np.float64)
    imaginary = values[0::2]
    real = values[1::2]
    amplitudes = np.sqrt(np.square(imaginary) + np.square(real))
    return amplitudes.tolist()


def append_sample(label: str, amplitudes: list[float], output_file: Path) -> None:
    """Append one timestamped CSI sample to the CSV output."""
    write_header = not output_file.exists() or output_file.stat().st_size == 0
    with output_file.open("a", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        if write_header:
            writer.writerow(["timestamp", "label", "amplitudes"])
        writer.writerow(
            [
                datetime.now(timezone.utc).isoformat(),
                label,
                json.dumps(amplitudes),
            ]
        )


def capture_csi(port: str, label: str, output_file: Path = OUTPUT_FILE) -> None:
    """Read CSI records from an ESP32 until interrupted."""
    try:
        connection = serial.Serial(port=port, baudrate=BAUD_RATE, timeout=1)
    except SerialException as error:
        raise RuntimeError(f"Could not open serial port {port!r}: {error}") from error

    print(f"Listening for CSI_DATA on {port} at {BAUD_RATE} baud.")
    print(f"Samples will be appended to {output_file}. Press Ctrl+C to stop.")
    try:
        while True:
            line = connection.readline().decode("utf-8", errors="replace").strip()
            if not line.startswith("CSI_DATA"):
                continue

            try:
                amplitudes = extract_amplitudes(line)
            except ValueError as error:
                print(f"Skipping malformed CSI_DATA line: {error}")
                continue

            append_sample(label, amplitudes, output_file)
    except KeyboardInterrupt:
        print("\nCapture stopped.")
    finally:
        connection.close()
        print("Serial port closed.")


def main() -> None:
    """Collect capture settings and start the serial reader."""
    port = input("Enter the ESP32 COM port (for example, COM3): ").strip()
    label = input("Enter Label (for example, location or activity): ").strip()
    if not port:
        raise SystemExit("A COM port is required.")
    if not label:
        raise SystemExit("A non-empty Label is required.")

    try:
        capture_csi(port, label)
    except RuntimeError as error:
        raise SystemExit(str(error)) from error


if __name__ == "__main__":
    main()
