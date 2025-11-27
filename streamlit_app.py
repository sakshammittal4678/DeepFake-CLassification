# streamlit_app.py
#
# Streamlit UI for Deepfake Classification
# - Loads Keras .h5 model (stored via Git LFS)
# - Accepts video upload
# - Extracts ALL frames using cv2.VideoCapture
# - Runs inference on all frames and aggregates predictions
# - Saves prediction logs

import os
import time
import tempfile
from pathlib import Path

import cv2
import numpy as np
import streamlit as st
import tensorflow as tf
import pandas as pd

# ==========================
# Configuration
# ==========================

# Adjust this path to where your model .h5 file is stored in the repo
MODEL_PATH = Path("Xception_finetune_86acc.h5")

# Input frame size expected by your model
FRAME_HEIGHT = 224
FRAME_WIDTH = 224

# Where to log predictions
LOG_DIR = Path("results")
LOG_FILE = LOG_DIR / "predictions.csv"


# ==========================
# Utility functions
# ==========================

@st.cache_resource(show_spinner=True)
def load_model(model_path: Path):
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found at: {model_path.resolve()}")
    model = tf.keras.models.load_model(model_path)
    return model


def extract_frames(video_path: str, target_fps: float = 3.0):
    """
    Extract frames at a fixed rate (e.g., 3 FPS) from the video.
    Returns a list of RGB frames.
    """
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise RuntimeError("Failed to open video file.")

    video_fps = cap.get(cv2.CAP_PROP_FPS)
    if video_fps <= 0:
        video_fps = 30  # fallback if metadata is missing

    frame_interval = int(video_fps / target_fps)
    if frame_interval < 1:
        frame_interval = 1

    frames = []
    frame_index = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_index % frame_interval == 0:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame_rgb)

        frame_index += 1

    cap.release()

    if len(frames) == 0:
        raise RuntimeError("No frames extracted from video.")

    return frames


def preprocess_frames(frames):
    """
    Preprocess frames for model input:
    - Resize to FRAME_HEIGHT x FRAME_WIDTH
    - Assume frames are RGB
    - Scale to [0, 1]
    Returns a numpy array of shape (N, H, W, 3)
    """
    processed = []
    for frame in frames:
        # Resize
        frame_resized = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))
        # Normalize
        frame_norm = frame_resized.astype(np.float32) / 255.0
        processed.append(frame_norm)

    return np.stack(processed, axis=0)


def aggregate_predictions(preds):
    """
    Aggregate per-frame predictions to a single score.
    Assumes model outputs either:
    - shape (N, 1): probability of deepfake
    - shape (N, 2): [prob_real, prob_fake]
    Returns: float probability of deepfake (0–1)
    """
    preds = np.array(preds)

    if preds.ndim == 2 and preds.shape[1] == 1:
        # Single probability output
        frame_probs = preds[:, 0]
    elif preds.ndim == 2 and preds.shape[1] == 2:
        # Binary softmax output [real, fake]
        frame_probs = preds[:, 1]
    else:
        raise ValueError(
            f"Unexpected prediction shape {preds.shape}. "
            "Adjust aggregate_predictions() according to your model output."
        )

    # Simple average across frames
    return float(np.mean(frame_probs))


def save_prediction_log(filename: str, prob_fake: float, extra_info: dict | None = None):
    """
    Append prediction result to a CSV file for tracking.
    """
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    record = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "filename": filename,
        "prob_deepfake": prob_fake,
    }
    if extra_info:
        record.update(extra_info)

    if LOG_FILE.exists():
        df = pd.read_csv(LOG_FILE)
        df = pd.concat([df, pd.DataFrame([record])], ignore_index=True)
    else:
        df = pd.DataFrame([record])

    df.to_csv(LOG_FILE, index=False)


# ==========================
# Streamlit UI
# ==========================

def main():
    st.set_page_config(
        page_title="Deepfake Video Classifier",
        layout="wide",
    )

    st.title("Deepfake Video Classification")
    st.write(
        """
        Upload a video file. The app will:
        1. Extract all frames using OpenCV (`cv2.VideoCapture`).
        2. Preprocess the frames.
        3. Run them through the loaded deepfake classification model.
        4. Aggregate predictions and display the final result.
        """
    )

    with st.sidebar:
        st.header("Model & Settings")
        st.text(f"Model path:\n{MODEL_PATH}")
        st.text(f"Input size: {FRAME_WIDTH}x{FRAME_HEIGHT}")

        try:
            model = load_model(MODEL_PATH)
            st.success("Model loaded successfully.")
        except Exception as e:
            st.error(f"Error loading model: {e}")
            st.stop()

    uploaded_file = st.file_uploader(
        "Upload a video file",
        type=["mp4", "avi", "mov", "mkv"],
        accept_multiple_files=False,
    )

    if uploaded_file is not None:
        st.write(f"Uploaded file: `{uploaded_file.name}`")

        # Save uploaded file to a temporary location
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
            tmp.write(uploaded_file.read())
            tmp_path = tmp.name

        st.video(uploaded_file)

        if st.button("Run Deepfake Detection"):
            with st.spinner("Extracting frames and running inference..."):
                try:
                    frames = extract_frames(tmp_path)
                    st.write(f"Extracted {len(frames)} total frames from the video.")

                    # Show a few sample frames
                    st.subheader("Sample Frames")
                    num_preview = min(4, len(frames))
                    cols = st.columns(num_preview)
                    for i in range(num_preview):
                        cols[i].image(
                            frames[i],
                            caption=f"Frame {i + 1}",
                            use_container_width=True,
                        )

                    # Preprocess and predict on ALL frames
                    x = preprocess_frames(frames)
                    preds = model.predict(x, verbose=0)
                    prob_fake = aggregate_predictions(preds)

                    # Log prediction
                    save_prediction_log(uploaded_file.name, prob_fake, {
                        "num_frames": len(frames)
                    })

                    # Display result
                    st.subheader("Prediction Result")
                    col1, col2 = st.columns(2)

                    with col1:
                        st.metric(
                            label="Deepfake Probability",
                            value=f"{prob_fake * 100:.2f} %",
                        )

                    label = "Deepfake" if prob_fake >= 0.5 else "Real"
                    with col2:
                        st.metric(
                            label="Predicted Label",
                            value=label,
                        )

                    st.progress(prob_fake)

                    st.write(
                        """
                        Interpretation:
                        - Values closer to 100% suggest the video is likely a deepfake.
                        - Values closer to 0% suggest the video is likely real.
                        """
                    )

                except Exception as e:
                    st.error(f"Error during processing: {e}")
                finally:
                    # Clean up temp file
                    try:
                        os.remove(tmp_path)
                    except OSError:
                        pass

    st.markdown("---")
    st.subheader("Prediction Logs")
    if LOG_FILE.exists():
        df_logs = pd.read_csv(LOG_FILE)
        st.dataframe(df_logs, use_container_width=True)
    else:
        st.write("No predictions logged yet.")


if __name__ == "__main__":
    main()
