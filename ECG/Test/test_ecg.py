import os
import sys
import json
import argparse
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers
from scipy.signal import butter, filtfilt, iirnotch

# =========================================================
# CONFIG
# =========================================================
BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "..", "Trained Model")

WINDOW_SIZE   = 3600
SAMPLING_RATE = 360

CLASS_NAMES = [
    "Normal",
    "Supraventricular",
    "Ventricular",
    "Conduction Disorder",
    "Myocardial Infarction",
    "Hypertrophy",
    "Ischemia/ST-T",
    "Atrial Fibrillation"
]

WEIGHTS = {
    "resnet": 0.45,
    "inception": 0.35,
    "transformer": 0.20
}

# =========================================================
# CUSTOM LAYER
# =========================================================
@tf.keras.utils.register_keras_serializable()
class PositionalEncoding(layers.Layer):
    def call(self, x):
        seq_len = tf.shape(x)[1]
        d_model = tf.shape(x)[2]

        pos = tf.range(seq_len, dtype=tf.float32)[:, None]
        dim = tf.range(d_model, dtype=tf.float32)[None, :]

        angle = pos / tf.pow(10000.0, (2 * (dim // 2)) / tf.cast(d_model, tf.float32))

        even_mask = tf.cast(tf.math.floormod(dim, 2) == 0, tf.float32)
        odd_mask  = 1 - even_mask

        encoding = tf.sin(angle) * even_mask + tf.cos(angle) * odd_mask
        return x + encoding[None, :, :]

# =========================================================
# LOAD MODELS (LAZY)
# =========================================================
_MODELS = None

def load_models():
    global _MODELS
    if _MODELS:
        return _MODELS

    custom_objects = {"PositionalEncoding": PositionalEncoding}

    model_files = {
        "resnet": "resnet_final.keras",
        "inception": "inception_final.keras",
        "transformer": "transformer_final.keras"
    }

    models = {}

    print("\n🔄 Loading models...")
    for name, file in model_files.items():
        path = os.path.join(MODEL_DIR, file)

        if not os.path.exists(path):
            raise FileNotFoundError(f"Model not found: {path}")

        print(f"  {name}...", end="")
        models[name] = tf.keras.models.load_model(path, custom_objects=custom_objects)
        print(" ✅")

    _MODELS = models
    return models

# =========================================================
# PREPROCESS
# =========================================================
def preprocess(signal):
    nyq = 0.5 * SAMPLING_RATE

    low  = 0.5 / nyq
    high = min(45.0 / nyq, 0.99)

    b, a = butter(3, [low, high], btype="bandpass")
    signal = filtfilt(b, a, signal)

    # Notch filter
    w0 = 50.0 / nyq
    if w0 < 1.0:
        bn, an = iirnotch(w0, 30)
        signal = filtfilt(bn, an, signal)

    # Normalize
    signal = (signal - np.mean(signal)) / (np.std(signal) + 1e-8)
    return signal.astype(np.float32)

# =========================================================
# LOAD SIGNAL
# =========================================================
def load_signal(path):
    try:
        df = pd.read_csv(path, comment="#", header=None)
    except Exception as e:
        raise ValueError(f"Cannot read CSV: {e}")

    col = df.iloc[:, -1]
    col = pd.to_numeric(col, errors="coerce").dropna()

    return col.values.astype(np.float32)

# =========================================================
# PREDICT SINGLE WINDOW
# =========================================================
def predict_window(window, models):
    x = window.reshape(1, WINDOW_SIZE, 1)

    preds = {}
    preds["resnet"]      = models["resnet"].predict(x, verbose=0)[0]
    preds["inception"]   = models["inception"].predict(x, verbose=0)[0]
    preds["transformer"] = models["transformer"].predict(x, verbose=0)[0]

    ensemble = sum(WEIGHTS[k] * preds[k] for k in preds)

    idx = int(np.argmax(ensemble))

    return idx, ensemble

# =========================================================
# FULL SIGNAL PREDICTION
# =========================================================
def predict_ecg(path):
    signal = load_signal(path)

    total_samples = len(signal)
    n_segments = total_samples // WINDOW_SIZE

    if n_segments == 0:
        raise ValueError("Signal too short")

    models = load_models()

    all_probs = np.zeros(len(CLASS_NAMES))
    results = []

    for i in range(n_segments):
        start = i * WINDOW_SIZE
        end   = start + WINDOW_SIZE

        window = preprocess(signal[start:end])

        idx, probs = predict_window(window, models)

        all_probs += probs

        results.append({
            "segment": i + 1,
            "prediction": CLASS_NAMES[idx],
            "confidence": float(probs[idx])
        })

    avg_probs = all_probs / n_segments
    top_idx = int(np.argmax(avg_probs))

    return {
        "final_prediction": CLASS_NAMES[top_idx],
        "confidence": float(avg_probs[top_idx]),
        "class_probabilities": {
            CLASS_NAMES[i]: float(avg_probs[i])
            for i in range(len(CLASS_NAMES))
        },
        "segments": results
    }

# =========================================================
# CLI ENTRY
# =========================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--json", action="store_true")

    args = parser.parse_args()

    if not os.path.exists(args.input):
        print("❌ File not found")
        sys.exit(1)

    result = predict_ecg(args.input)

    if args.json:
        print(json.dumps(result, indent=2))
    else:
        print("\n📊 FINAL RESULT")
        print("=" * 40)
        print(f"Prediction : {result['final_prediction']}")
        print(f"Confidence : {result['confidence']*100:.2f}%")