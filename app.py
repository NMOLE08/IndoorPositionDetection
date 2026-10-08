"""Streamlit inference app for the repository's CSI CNN-LSTM model.

Run with:
    streamlit run app.py

The default model is expected at models/csi_cnn_lstm_model.pth. The loader
expects a PyTorch state_dict, which is the format produced by the training
notebook in this repository.
"""

from __future__ import annotations

import io
import json
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
import torch
import torch.nn as nn


BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"
DEFAULT_MODEL_PATH = MODEL_DIR / "csi_cnn_lstm_model.pth"
OPTIONAL_SCALER_PATH = MODEL_DIR / "target_scaler.joblib"
TRAIN_TARGETS_PATH = BASE_DIR / "train_data.npz"
TIME_STEPS = 50
SUBCARRIERS = 119
CHANNELS = 2
FLAT_FEATURES = TIME_STEPS * SUBCARRIERS * CHANNELS


class CSICNNLSTM(nn.Module):
    """Architecture used by DataExtraction.ipynb for CSI regression."""

    def __init__(self, output_dim: int, dropout_p: float = 0.3) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(2, 16, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=(1, 2))
        self.dropout_cnn = nn.Dropout(dropout_p)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=(1, 2))
        self.lstm = nn.LSTM(32 * 29, 64, num_layers=1, batch_first=True)
        self.dropout_lstm = nn.Dropout(dropout_p)
        self.fc = nn.Linear(64, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool1(self.relu(self.conv1(x)))
        x = self.dropout_cnn(x)
        x = self.pool2(self.relu(self.conv2(x)))
        x = self.dropout_cnn(x)
        x = x.permute(0, 2, 1, 3)
        x = x.reshape(x.size(0), x.size(1), -1)
        lstm_out, _ = self.lstm(x)
        return self.fc(self.dropout_lstm(lstm_out[:, -1, :]))


def _state_dict_from_checkpoint(checkpoint: Any) -> dict[str, torch.Tensor]:
    """Accept only tensor state dictionaries, avoiding arbitrary module loads."""
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        checkpoint = checkpoint["state_dict"]
    if not isinstance(checkpoint, dict) or not checkpoint:
        raise ValueError("The checkpoint must contain a non-empty PyTorch state_dict.")

    state_dict = {
        str(key).removeprefix("module."): value
        for key, value in checkpoint.items()
        if isinstance(value, torch.Tensor)
    }
    if not state_dict or "fc.weight" not in state_dict:
        raise ValueError("Checkpoint does not contain the expected CNN-LSTM weights.")
    return state_dict


@st.cache_resource(show_spinner="Loading CNN-LSTM model...")
def load_model(model_path: str) -> tuple[nn.Module, torch.device, int]:
    """Load a state_dict once per model path and select CPU or CUDA safely."""
    path = Path(model_path)
    if not path.is_file():
        raise FileNotFoundError(f"Model file not found: {path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        checkpoint = torch.load(path, map_location="cpu")
    state_dict = _state_dict_from_checkpoint(checkpoint)
    output_dim = int(state_dict["fc.weight"].shape[0])
    model = CSICNNLSTM(output_dim=output_dim)
    try:
        model.load_state_dict(state_dict, strict=True)
    except RuntimeError as exc:
        raise ValueError(f"Checkpoint is incompatible with the bundled architecture: {exc}") from exc
    model.to(device).eval()
    return model, device, output_dim


def _read_uploaded_data(uploaded_file: Any) -> np.ndarray:
    """Read supported file formats and return the contained feature array."""
    suffix = Path(uploaded_file.name).suffix.lower()
    raw = uploaded_file.getvalue()
    if suffix == ".npz":
        with np.load(io.BytesIO(raw), allow_pickle=False) as archive:
            if "X" not in archive.files:
                raise ValueError(f"NPZ must contain an 'X' array; found {archive.files}.")
            return np.asarray(archive["X"])
    if suffix == ".csv":
        return pd.read_csv(io.BytesIO(raw), header=None).to_numpy()
    if suffix in {".pkl", ".pickle"}:
        value = pickle.loads(raw)
        if isinstance(value, dict):
            value = value.get("X", value.get("data"))
        if isinstance(value, pd.DataFrame):
            value = value.to_numpy()
        if value is None:
            raise ValueError("Pickle must contain an array, DataFrame, or an 'X'/'data' value.")
        return np.asarray(value)
    raise ValueError("Unsupported file type. Upload .npz, .csv, .pkl, or .pickle.")


def preprocess_input(values: np.ndarray, normalize: bool = False) -> torch.Tensor:
    """Normalize and convert input to the model layout [batch, 2, 50, 119].

    Accepted arrays are [N, 50, 119, 2], one sample [50, 119, 2], the
    channel-first equivalent [N, 2, 50, 119], or flattened [N, 11900].
    Set normalize=True only for raw values; repository NPZ exports are already
    normalized by packet in the extraction pipeline.
    """
    array = np.asarray(values, dtype=np.float32)
    if array.ndim == 2 and array.shape[1] == FLAT_FEATURES:
        array = array.reshape(-1, TIME_STEPS, SUBCARRIERS, CHANNELS)
    elif array.ndim == 3 and array.shape == (TIME_STEPS, SUBCARRIERS, CHANNELS):
        array = array[None, ...]
    elif array.ndim == 4 and array.shape[1:] == (CHANNELS, TIME_STEPS, SUBCARRIERS):
        array = np.transpose(array, (0, 2, 3, 1))

    expected = (TIME_STEPS, SUBCARRIERS, CHANNELS)
    if array.ndim != 4 or tuple(array.shape[1:]) != expected:
        raise ValueError(
            "Expected input shape [N, 50, 119, 2], [N, 2, 50, 119], "
            f"or [N, {FLAT_FEATURES}]; received {tuple(array.shape)}."
        )
    if not np.isfinite(array).all():
        raise ValueError("Input contains NaN or infinite values.")
    if normalize:
        mean = array.mean(axis=2, keepdims=True)
        std = array.std(axis=2, keepdims=True)
        array = (array - mean) / np.maximum(std, 1e-6)
    return torch.from_numpy(np.transpose(array, (0, 3, 1, 2)).copy())


def inverse_targets(predictions: np.ndarray) -> tuple[np.ndarray, str]:
    """Undo training MinMax scaling using the saved artifact or training targets."""
    if not OPTIONAL_SCALER_PATH.is_file():
        if not TRAIN_TARGETS_PATH.is_file():
            return predictions, "none"
        try:
            with np.load(TRAIN_TARGETS_PATH, allow_pickle=False) as archive:
                if "y" not in archive.files:
                    return predictions, "none"
                training_targets = np.asarray(archive["y"], dtype=np.float32)
            if training_targets.ndim != 2 or training_targets.shape[1] != predictions.shape[1]:
                return predictions, "none"
            data_min = training_targets.min(axis=0)
            data_range = training_targets.max(axis=0) - data_min
            if np.any(data_range <= 0):
                return predictions, "none"
            return predictions * data_range + data_min, "train_data.npz"
        except (OSError, ValueError):
            return predictions, "none"
    try:
        import joblib

        scaler = joblib.load(OPTIONAL_SCALER_PATH)
        return np.asarray(scaler.inverse_transform(predictions)), "target_scaler.joblib"
    except Exception as exc:
        raise ValueError(f"Could not load target scaler: {exc}") from exc


def prepare_trajectory(data: pd.DataFrame) -> pd.DataFrame:
    """Validate and normalize trajectory columns for the location visualizer."""
    required = {"timestamp", "x_coord", "y_coord"}
    missing = required.difference(data.columns)
    if missing:
        raise ValueError(
            "Trajectory data is missing required columns: "
            + ", ".join(sorted(missing))
        )

    trajectory = data.copy()
    trajectory["x_coord"] = pd.to_numeric(trajectory["x_coord"], errors="coerce")
    trajectory["y_coord"] = pd.to_numeric(trajectory["y_coord"], errors="coerce")
    trajectory = trajectory.dropna(subset=["timestamp", "x_coord", "y_coord"])
    if trajectory.empty:
        raise ValueError("Trajectory data has no valid timestamp/x_coord/y_coord rows.")

    trajectory["_time"] = pd.to_datetime(trajectory["timestamp"], errors="coerce")
    if trajectory["_time"].notna().all():
        trajectory = trajectory.sort_values("_time")
        trajectory["frame"] = trajectory["_time"].dt.strftime("%Y-%m-%d %H:%M:%S")
    else:
        trajectory["frame"] = trajectory["timestamp"].astype(str)
        trajectory = trajectory.sort_values("frame")

    group_column = next(
        (column for column in ("person_id", "state") if column in trajectory.columns),
        None,
    )
    if group_column is None:
        trajectory["person_id"] = "person"
        group_column = "person_id"
    trajectory["person"] = trajectory[group_column].astype(str)
    if "state" not in trajectory.columns:
        trajectory["state"] = ""
    return trajectory


def render_location_playback(data: pd.DataFrame) -> None:
    """Render a dark interactive floor-plan scatter plot with timestamp playback."""
    trajectory = prepare_trajectory(data)
    frame_values = trajectory["frame"].drop_duplicates().tolist()
    color_column = "person" if trajectory["person"].nunique() > 1 else "state"
    if not trajectory[color_column].astype(str).str.strip().any():
        color_column = "person"

    figure = px.scatter(
        trajectory,
        x="x_coord",
        y="y_coord",
        color=color_column,
        animation_frame="frame",
        animation_group="person",
        hover_name="person",
        hover_data={"x_coord": ":.3f", "y_coord": ":.3f", "state": True, "frame": False},
        labels={"x_coord": "X position", "y_coord": "Y position", "frame": "Timestamp"},
        title="2D Location Playback",
        template="plotly_dark",
        range_x=[trajectory["x_coord"].min(), trajectory["x_coord"].max()],
        range_y=[trajectory["y_coord"].min(), trajectory["y_coord"].max()],
    )
    figure.update_traces(marker={"size": 13, "line": {"width": 1, "color": "#f8fafc"}})
    figure.update_layout(
        height=620,
        margin={"l": 20, "r": 20, "t": 65, "b": 20},
        paper_bgcolor="#0e1117",
        plot_bgcolor="#151a22",
        xaxis={"constrain": "domain", "showgrid": True, "gridcolor": "#303846", "zeroline": False},
        yaxis={"scaleanchor": "x", "scaleratio": 1, "showgrid": True, "gridcolor": "#303846", "zeroline": False},
        legend={"title": "Person / state", "orientation": "h", "y": 1.02, "x": 0},
    )
    if len(frame_values) > 1:
        figure.layout.sliders[0].currentvalue.prefix = "Time: "
    st.plotly_chart(figure, width="stretch", key="location-playback")
    st.caption(f"{len(trajectory):,} points across {len(frame_values):,} timestamps")


def render_prediction_map(result: pd.DataFrame) -> None:
    """Render a top-down person avatar following the predicted trajectory."""
    if not {"x", "y"}.issubset(result.columns):
        st.warning("A top-down map requires x and y coordinate predictions.")
        return

    path = result[["x", "y"]].dropna().reset_index(drop=True)
    if path.empty:
        st.warning("No valid x/y predictions are available for the spatial map.")
        return

    points = [
        {"x": float(row.x), "y": float(row.y)}
        for row in path.itertuples(index=False)
    ]
    points_json = json.dumps(points, separators=(",", ":"))
    epsilon = max(
        float(path["x"].max() - path["x"].min()),
        float(path["y"].max() - path["y"].min()),
        1e-6,
    ) * 0.01

    canvas_html = """
    <style>
        :root { color-scheme: dark; }
        body { margin: 0; background: #0e1117; font-family: Inter, system-ui, sans-serif; color: #e5e7eb; }
        .shell { border: 1px solid #293241; border-radius: 12px; padding: 14px; background: #111827; }
        .heading { display: flex; justify-content: space-between; align-items: center; gap: 12px; margin-bottom: 10px; }
        .title { font-size: 16px; font-weight: 700; letter-spacing: .01em; }
        .state { color: #93c5fd; font-size: 13px; font-weight: 600; }
        canvas { display: block; width: 100%; height: 520px; border-radius: 8px; background: #151a22; }
        .controls { display: flex; flex-wrap: wrap; align-items: center; gap: 8px 12px; padding-top: 12px; }
        button { border: 1px solid #475569; border-radius: 6px; padding: 7px 14px; color: #f8fafc; background: #1e293b; cursor: pointer; }
        button:hover { background: #334155; }
        label { display: inline-flex; align-items: center; gap: 8px; font-size: 13px; color: #cbd5e1; }
        input[type=range] { accent-color: #60a5fa; width: 140px; }
        .readout { min-width: 150px; margin-left: auto; color: #94a3b8; font-size: 12px; text-align: right; }
    </style>
    <div class="shell">
        <div class="heading">
            <div class="title">2D Spatial Movement</div>
            <div class="state" id="state">Standing</div>
        </div>
        <canvas id="floor"></canvas>
        <div class="controls">
            <button id="play" type="button">Play</button>
            <button id="pause" type="button">Pause</button>
            <button id="reset" type="button">Reset</button>
            <label>Speed <input id="speed" type="range" min="0.25" max="2" step="0.25" value="0.7"><span id="speedValue">0.7x</span></label>
            <div class="readout" id="readout">Position: --</div>
        </div>
    </div>
    <script>
        const points = __POINTS__;
        const canvas = document.getElementById("floor");
        const ctx = canvas.getContext("2d");
        const stateLabel = document.getElementById("state");
        const readout = document.getElementById("readout");
        const speedInput = document.getElementById("speed");
        const speedValue = document.getElementById("speedValue");
        const standingEpsilon = __EPSILON__;
        const animation = { index: 0, progress: 0, playing: false, lastTime: 0 };
        let zoom = 1.0;

        const avatarImage = null;

        function bounds() {
            const xs = points.map(point => point.x);
            const ys = points.map(point => point.y);
            const minX = Math.min(...xs), maxX = Math.max(...xs);
            const minY = Math.min(...ys), maxY = Math.max(...ys);
            const span = Math.max(maxX - minX, maxY - minY, 1e-6);
            const pad = span * 0.18;
            return { minX: minX - pad, maxX: maxX + pad, minY: minY - pad, maxY: maxY + pad };
        }

        function resize() {
            const ratio = window.devicePixelRatio || 1;
            const rect = canvas.getBoundingClientRect();
            canvas.width = Math.round(rect.width * ratio);
            canvas.height = Math.round(rect.height * ratio);
            ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
            draw();
        }

        function project(point, map, width, height) {
            const pad = 32;
            const rawWidth = map.maxX - map.minX;
            const rawHeight = map.maxY - map.minY;
            const scale = Math.min((width - pad * 2) / rawWidth, (height - pad * 2) / rawHeight) * zoom;
            const mapWidth = rawWidth * scale;
            const mapHeight = rawHeight * scale;
            const x = (width - mapWidth) / 2 + (point.x - map.minX) * scale;
            const y = (height + mapHeight) / 2 - (point.y - map.minY) * scale;
            return { x, y };
        }

        function lerp(a, b, t) {
            return a + (b - a) * t;
        }

        function interpolatePosition() {
            const current = points[animation.index];
            if (animation.index >= points.length - 1) return current;
            const next = points[animation.index + 1];
            return {
                x: lerp(current.x, next.x, animation.progress),
                y: lerp(current.y, next.y, animation.progress),
            };
        }

        function drawGrid(width, height) {
            ctx.fillStyle = "#151a22";
            ctx.fillRect(0, 0, width, height);
            ctx.strokeStyle = "#263241";
            ctx.lineWidth = 1;
            for (let x = 24; x < width; x += 32) {
                ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, height); ctx.stroke();
            }
            for (let y = 24; y < height; y += 32) {
                ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(width, y); ctx.stroke();
            }
        }

        function drawTrail(map, width, height) {
            if (animation.index < 1) return;
            for (let i = 0; i <= animation.index; i++) {
                const projected = project(points[i], map, width, height);
                const alpha = Math.max(0.12, (i / Math.max(animation.index, 1)) * 0.9);
                ctx.beginPath();
                ctx.moveTo(projected.x, projected.y);
                const nextProjected = project(points[Math.min(i + 1, animation.index)], map, width, height);
                ctx.lineTo(nextProjected.x, nextProjected.y);
                ctx.strokeStyle = `rgba(125, 211, 252, ${alpha})`;
                ctx.lineWidth = 2.5;
                ctx.stroke();
            }
        }

        function drawPulse(position, radius) {
            const pulse = 12 + Math.sin(performance.now() / 180) * 5;
            ctx.beginPath();
            ctx.arc(position.x, position.y, radius + pulse, 0, Math.PI * 2);
            ctx.strokeStyle = "rgba(96, 165, 250, 0.5)";
            ctx.lineWidth = 2;
            ctx.stroke();
        }

        function drawAvatar(position, angle, standing, scale) {
            const radius = 10 * scale;
            drawPulse(position, radius * 0.8);

            ctx.save();
            ctx.shadowColor = "rgba(14, 165, 233, 0.55)";
            ctx.shadowBlur = 12;
            ctx.fillStyle = "#38bdf8";
            ctx.beginPath();
            ctx.arc(position.x, position.y, radius, 0, Math.PI * 2);
            ctx.fill();
            ctx.restore();

            if (standing) {
                ctx.beginPath();
                ctx.arc(position.x, position.y, radius * 1.5, 0, Math.PI * 2);
                ctx.strokeStyle = "rgba(96, 165, 250, 0.28)";
                ctx.lineWidth = 1.5;
                ctx.stroke();
            }
        }

        function draw() {
            if (!points.length) return;
            const width = canvas.clientWidth;
            const height = canvas.clientHeight;
            const map = bounds();
            drawGrid(width, height);
            drawTrail(map, width, height);

            const current = interpolatePosition();
            const projected = project(current, map, width, height);
            const nextIndex = Math.min(animation.index + 1, points.length - 1);
            const previous = points[Math.max(animation.index - 1, 0)];
            const direction = points[nextIndex];
            const distance = Math.hypot(direction.x - previous.x, direction.y - previous.y);
            const standing = distance <= standingEpsilon;

            let angle = -Math.PI / 2;
            if (animation.index < points.length - 1) {
                const next = points[nextIndex];
                angle = Math.atan2(next.y - current.y, next.x - current.x);
            }

            ctx.fillStyle = "#94a3b8";
            ctx.font = "12px system-ui";
            ctx.fillText("Y (meters)", 18, 22);
            ctx.fillText("X (meters)", width - 82, height - 16);

            const avatarScale = Math.max(0.9, Math.min(width, height) / 320);
            drawAvatar(projected, angle, standing, avatarScale);
            stateLabel.textContent = standing ? "Standing" : "Walking";
            stateLabel.style.color = standing ? "#fbbf24" : "#7dd3fc";
            readout.textContent = `Position: ${current.x.toFixed(3)} m, ${current.y.toFixed(3)} m`;
        }

        function tick(time) {
            if (!animation.playing) return;
            const elapsed = animation.lastTime ? time - animation.lastTime : 0;
            animation.lastTime = time;
            animation.progress += elapsed / (1500 / Number(speedInput.value));
            if (animation.progress >= 1) {
                animation.index += 1;
                animation.progress = 0;
            }
            if (animation.index >= points.length - 1) {
                animation.index = points.length - 1;
                animation.playing = false;
                animation.lastTime = 0;
            }
            draw();
            if (animation.playing) requestAnimationFrame(tick);
        }

        document.getElementById("play").onclick = () => {
            if (animation.index >= points.length - 1) {
                animation.index = 0;
                animation.progress = 0;
            }
            animation.playing = true;
            animation.lastTime = 0;
            requestAnimationFrame(tick);
        };

        document.getElementById("pause").onclick = () => {
            animation.playing = false;
            animation.lastTime = 0;
            draw();
        };

        document.getElementById("reset").onclick = () => {
            animation.playing = false;
            animation.index = 0;
            animation.progress = 0;
            animation.lastTime = 0;
            draw();
        };

        speedInput.oninput = () => {
            speedValue.textContent = `${speedInput.value}x`;
        };

        canvas.addEventListener("wheel", function(e) {
            e.preventDefault();
            const delta = e.deltaY > 0 ? 0.9 : 1.1;
            zoom = Math.max(0.1, Math.min(10, zoom * delta));
            draw();
        }, { passive: false });

        window.addEventListener("resize", resize);
        resize();
    </script>
    """.replace("__POINTS__", points_json).replace("__EPSILON__", str(epsilon))

    st.components.v1.html(canvas_html, height=650, scrolling=False)


def main() -> None:
    st.set_page_config(page_title="CSI Position Inference", page_icon="📡", layout="wide")
    st.title("CSI Position Inference")
    st.caption("CNN-LSTM regression for 50-packet CSI windows")

    with st.sidebar:
        st.header("Model")
        model_path = st.text_input("Checkpoint path", str(DEFAULT_MODEL_PATH))
        normalize = st.checkbox("Normalize raw input per packet", value=False)
        st.info(f"Runtime: {'CUDA' if torch.cuda.is_available() else 'CPU'}")

    try:
        model, device, output_dim = load_model(model_path)
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        st.error(str(exc))
        st.stop()

    st.success(f"Loaded {output_dim}-coordinate model on {device}.")

    st.header("2D Location Playback")
    st.caption("Upload a trajectory CSV with timestamp, x_coord, y_coord, and optionally person_id or state.")
    trajectory_file = st.file_uploader(
        "Upload location data",
        type=["csv"],
        key="trajectory-upload",
    )
    if trajectory_file is not None:
        try:
            render_location_playback(pd.read_csv(trajectory_file))
        except (ValueError, OSError, pd.errors.ParserError) as exc:
            st.error(f"Location data error: {exc}")

    st.header("CSI Model Inference")
    uploaded_file = st.file_uploader("Upload CSI input", type=["npz", "csv", "pkl", "pickle"])
    if uploaded_file is None:
        st.info("Upload preprocessed CSI windows to begin. NPZ files should contain an 'X' array.")
        return

    try:
        raw_values = _read_uploaded_data(uploaded_file)
        input_tensor = preprocess_input(raw_values, normalize=normalize)
    except (ValueError, OSError, pickle.UnpicklingError) as exc:
        st.error(f"Input error: {exc}")
        return

    summary_columns = st.columns(4)
    summary_columns[0].metric("Samples", input_tensor.shape[0])
    summary_columns[1].metric("Time steps", input_tensor.shape[2])
    summary_columns[2].metric("Subcarriers", input_tensor.shape[3])
    summary_columns[3].metric("Channels", input_tensor.shape[1])
    st.dataframe(pd.DataFrame({"shape": [str(tuple(input_tensor.shape))], "dtype": [str(input_tensor.dtype)]}), hide_index=True)

    if not st.button("Run Prediction", type="primary", width="stretch"):
        return

    try:
        with torch.no_grad():
            predictions = model(input_tensor.to(device)).detach().cpu().numpy()
        predictions, scaler_source = inverse_targets(predictions)
    except (RuntimeError, ValueError) as exc:
        st.error(f"Inference error: {exc}")
        return

    labels = ["x", "y", "z"][:predictions.shape[1]]
    result = pd.DataFrame(predictions, columns=labels)
    st.subheader("Predictions")
    if scaler_source == "target_scaler.joblib":
        st.caption("Coordinates were inverse-transformed with models/target_scaler.joblib.")
    elif scaler_source == "train_data.npz":
        st.caption("Coordinates were inverse-transformed using min/max values from train_data.npz.")
    else:
        st.warning("No target scaler or training targets found. Values are in the model's training scale.")
    st.dataframe(result.head(100), width="stretch", hide_index=True)
    render_prediction_map(result)


if __name__ == "__main__":
    main()