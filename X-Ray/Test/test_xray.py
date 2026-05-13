"""
test_xray.py
------------
Inference script for multi-label chest X-ray disease classification.
Loads three pretrained models (DenseNet121, ResNet50, ViT-Base) and
produces per-model and ensemble predictions for a given chest X-ray image.

Usage:
    # Batch mode — runs all images inside test_images_by_class/
    python test_xray.py

    # Single image mode
    python test_xray.py path/to/image.jpg

Project: HealthBridge Clinical AI Platform
Dataset: CheXpert (Stanford)
"""

import os
import sys
import torch
import torch.nn as nn
import numpy as np
from torchvision import models, transforms
from PIL import Image


# =============================================================================
# CONFIGURATION
# =============================================================================

# Derive root paths relative to this script's location
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))   # .../Test/
ROOT_DIR    = os.path.dirname(BASE_DIR)                    # .../X-Ray/
MODEL_DIR   = os.path.join(ROOT_DIR, 'Trained Model')
TEST_IMAGES = os.path.join(ROOT_DIR, 'test_images_by_class', 'test_images')

# Disease classes — must match the order used during training
TARGET_COLS = [
    'No Finding', 'Cardiomegaly', 'Edema',
    'Consolidation', 'Pneumonia', 'Atelectasis'
]
NUM_CLASSES = len(TARGET_COLS)

# Probability threshold above which a class is considered detected
THRESHOLD = 0.5


# =============================================================================
# MODEL DEFINITIONS
# Architecture and final layer must exactly match the training configuration.
# =============================================================================

def build_densenet121():
    """DenseNet-121 with a sigmoid multi-label classification head."""
    model = models.densenet121(weights=None)
    model.classifier = nn.Sequential(
        nn.Linear(model.classifier.in_features, NUM_CLASSES),
        nn.Sigmoid()
    )
    return model


def build_resnet50():
    """ResNet-50 with a sigmoid multi-label classification head."""
    model = models.resnet50(weights=None)
    model.fc = nn.Sequential(
        nn.Linear(model.fc.in_features, NUM_CLASSES),
        nn.Sigmoid()
    )
    return model


def build_vit_base():
    """Vision Transformer ViT-B/16 with a sigmoid multi-label classification head."""
    model = models.vit_b_16(weights=None)
    model.heads.head = nn.Sequential(
        nn.Linear(model.heads.head.in_features, NUM_CLASSES),
        nn.Sigmoid()
    )
    return model


# =============================================================================
# MODEL LOADER
# =============================================================================

# Registry maps display name to (weight filename, builder function)
MODEL_REGISTRY = [
    ('DenseNet121', 'densenet121_chest_xray.pth', build_densenet121),
    ('ResNet50',    'resnet50_chest_xray.pth',    build_resnet50),
    ('ViT-Base',    'vit_base_chest_xray.pth',    build_vit_base),
]


def load_models(device):
    """
    Load all registered models from MODEL_DIR onto the specified device.

    Returns a dict { display_name: model } for successfully loaded models.
    Models whose weight files are missing are skipped with a warning.
    """
    loaded = {}
    for display_name, filename, builder in MODEL_REGISTRY:
        weight_path = os.path.join(MODEL_DIR, filename)
        if not os.path.exists(weight_path):
            print(f"  [SKIP] {display_name} — weight file not found: {weight_path}")
            continue
        model = builder()
        model.load_state_dict(torch.load(weight_path, map_location=device))
        model.to(device)
        model.eval()
        loaded[display_name] = model
        print(f"  [OK]   {display_name} loaded from {filename}")
    return loaded


# =============================================================================
# IMAGE PREPROCESSING
# Normalization parameters match those used during training.
# =============================================================================

_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


def preprocess(image_path):
    """
    Load an image from disk, convert to RGB, apply standard transforms,
    and return a (1, 3, 224, 224) tensor ready for inference.
    """
    image = Image.open(image_path).convert('RGB')
    return _transform(image).unsqueeze(0)


# =============================================================================
# INFERENCE
# =============================================================================

def predict(models_dict, image_path, device):
    """
    Run inference on a single image using all loaded models.

    Parameters
    ----------
    models_dict : dict
        { model_name: model } as returned by load_models().
    image_path : str
        Absolute or relative path to the input image.
    device : torch.device

    Returns
    -------
    per_model : dict
        { model_name: np.ndarray of shape (NUM_CLASSES,) } raw probabilities.
    ensemble_probs : np.ndarray of shape (NUM_CLASSES,)
        Mean probability across all models (simple average ensemble).
    """
    tensor = preprocess(image_path).to(device)
    all_probs = []
    per_model = {}

    with torch.no_grad():
        for name, model in models_dict.items():
            probs = model(tensor).cpu().numpy()[0]
            per_model[name] = probs
            all_probs.append(probs)

    ensemble_probs = np.mean(all_probs, axis=0)
    return per_model, ensemble_probs


# =============================================================================
# OUTPUT FORMATTING
# =============================================================================

def print_result(image_path, per_model, ensemble_probs):
    """
    Print a formatted prediction report for a single image.

    Displays per-model probabilities, ensemble probabilities,
    and a final verdict based on THRESHOLD.
    """
    print("\n" + "=" * 65)
    print(f"  Image : {os.path.basename(image_path)}")
    print("=" * 65)

    # Header row
    print(f"\n  {'Disease':<26}", end="")
    for name in per_model:
        print(f"{name:>13}", end="")
    print(f"{'Ensemble':>13}")
    print("  " + "-" * 61)

    # Probability per class
    for i, label in enumerate(TARGET_COLS):
        print(f"  {label:<26}", end="")
        for name in per_model:
            print(f"{per_model[name][i] * 100:>12.1f}%", end="")
        print(f"{ensemble_probs[i] * 100:>12.1f}%")

    # Final verdict
    print("\n" + "-" * 65)
    print(f"  VERDICT  (ensemble probability >= {THRESHOLD * 100:.0f}%)")
    print("-" * 65)

    detected = [
        (TARGET_COLS[i], ensemble_probs[i])
        for i in range(NUM_CLASSES)
        if ensemble_probs[i] >= THRESHOLD
    ]

    if not detected:
        top_idx = int(np.argmax(ensemble_probs))
        print(f"  No class detected above threshold.")
        print(f"  Highest probability: {TARGET_COLS[top_idx]}"
              f" ({ensemble_probs[top_idx] * 100:.1f}%)")
    else:
        for label, prob in sorted(detected, key=lambda x: -x[1]):
            bar = '#' * int(prob * 20)
            print(f"  [DETECTED] {label:<22} {prob * 100:.1f}%  {bar}")

    print("=" * 65)


# =============================================================================
# BATCH TEST
# Iterates over every subfolder in TEST_IMAGES and runs inference on up to
# MAX_PER_CLASS images per folder, then prints a summary hit-rate table.
# =============================================================================

MAX_PER_CLASS = 5   # Maximum number of images to test per class folder


def test_all_images(models_dict, device):
    """
    Run batch inference on all class subfolders inside TEST_IMAGES.

    For each folder the script checks whether the correct class label
    appears in the ensemble predictions and reports a per-class hit rate.
    """
    print(f"\nBatch test directory: {TEST_IMAGES}\n")
    summary = []

    for class_folder in sorted(os.listdir(TEST_IMAGES)):
        folder_path = os.path.join(TEST_IMAGES, class_folder)
        if not os.path.isdir(folder_path):
            continue

        print(f"\n{'─' * 65}")
        print(f"  Class folder: {class_folder}")
        print(f"{'─' * 65}")

        image_files = [
            f for f in os.listdir(folder_path)
            if f.lower().endswith(('.jpg', '.jpeg', '.png'))
        ]

        if not image_files:
            print("  No image files found in this folder.")
            continue

        correct = 0
        for img_file in image_files[:MAX_PER_CLASS]:
            img_path = os.path.join(folder_path, img_file)
            try:
                per_model, ensemble = predict(models_dict, img_path, device)
                print_result(img_path, per_model, ensemble)

                # Determine whether the ground-truth class was detected
                predicted_labels = [
                    TARGET_COLS[i]
                    for i, p in enumerate(ensemble)
                    if p >= THRESHOLD
                ]
                if class_folder in predicted_labels:
                    correct += 1

            except Exception as exc:
                print(f"  [ERROR] Could not process {img_file}: {exc}")

        summary.append({
            'class':   class_folder,
            'tested':  min(MAX_PER_CLASS, len(image_files)),
            'correct': correct
        })

    # Summary table
    print("\n\n" + "=" * 65)
    print("  BATCH TEST SUMMARY")
    print("=" * 65)
    print(f"  {'Class':<28} {'Tested':>6}  {'Correct':>7}  {'Hit Rate':>9}")
    print("  " + "-" * 54)

    total_tested = total_correct = 0
    for row in summary:
        hit_rate = (row['correct'] / row['tested'] * 100) if row['tested'] > 0 else 0.0
        print(f"  {row['class']:<28} {row['tested']:>6}  {row['correct']:>7}  {hit_rate:>8.0f}%")
        total_tested  += row['tested']
        total_correct += row['correct']

    overall = (total_correct / total_tested * 100) if total_tested > 0 else 0.0
    print("  " + "-" * 54)
    print(f"  {'OVERALL':<28} {total_tested:>6}  {total_correct:>7}  {overall:>8.1f}%")
    print("=" * 65)


# =============================================================================
# SINGLE IMAGE MODE
# =============================================================================

def test_single_image(models_dict, image_path, device):
    """
    Run inference on a single user-specified image and print the result.
    """
    if not os.path.exists(image_path):
        print(f"[ERROR] File not found: {image_path}")
        return
    per_model, ensemble = predict(models_dict, image_path, device)
    print_result(image_path, per_model, ensemble)


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == '__main__':
    print("\n" + "=" * 65)
    print("  HealthBridge — Chest X-Ray Disease Classifier")
    print("  Ensemble: DenseNet121 | ResNet50 | ViT-Base")
    print("  Classes : " + " | ".join(TARGET_COLS))
    print("=" * 65)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n  Inference device : {device}")

    print("\nLoading model weights...")
    models_dict = load_models(device)

    if not models_dict:
        print("\n[ERROR] No models were loaded. "
              "Verify that .pth files exist in the 'Trained Model' folder.")
        sys.exit(1)

    print(f"\n  {len(models_dict)} model(s) ready for inference.")

    if len(sys.argv) > 1:
        # Single image mode: path provided as a command-line argument
        test_single_image(models_dict, sys.argv[1], device)
    else:
        # Batch mode: test all images in the test_images_by_class directory
        test_all_images(models_dict, device)