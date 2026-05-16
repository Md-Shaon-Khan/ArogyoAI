"""
Brain CT Disease Detection — Test Script
=========================================
Usage:
    python test_brain_ct.py --image scan.png --model ./models/brain_classifier_final.pth
    python test_brain_ct.py --image scan.png --model ./models/brain_classifier_final.pth --save result.png
    python test_brain_ct.py --folder ./test_images/ --model ./models/brain_classifier_final.pth

Only needs: brain_classifier_final.pth (or brain_classifier_best.pth)
Install:    pip install torch torchvision Pillow numpy matplotlib opencv-python-headless
"""

import argparse
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from torchvision import transforms
from torchvision.models import resnet50, ResNet50_Weights
from PIL import Image
from pathlib import Path

# ── Model (identical to training) ────────────────────────────────────────────
class BrainCTClassifier(nn.Module):
    def __init__(self, num_classes, dropout=0.4):
        super().__init__()
        backbone = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        backbone.fc = nn.Identity()
        backbone.eval()
        with torch.no_grad():
            feat_dim = backbone(torch.zeros(1, 3, 224, 224)).shape[1]
        self.backbone = backbone
        self.classifier = nn.Sequential(
            nn.Linear(feat_dim, 512), nn.BatchNorm1d(512),
            nn.GELU(), nn.Dropout(dropout),
            nn.Linear(512, 256), nn.BatchNorm1d(256),
            nn.GELU(), nn.Dropout(dropout / 2),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.backbone(x))


# ── Config ────────────────────────────────────────────────────────────────────
DEVICE   = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
IMG_SIZE = 224

TRANSFORM = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])

DEFAULT_CLASSES = [
    'Normal', 'Glioma Tumor', 'Meningioma Tumor',
    'Pituitary Tumor', 'Hemorrhage'
]
SEVERITY = {
    'Normal':           'None',
    'Glioma Tumor':     'CRITICAL',
    'Meningioma Tumor': 'HIGH',
    'Pituitary Tumor':  'MODERATE',
    'Hemorrhage':       'CRITICAL',
}
SEV_ICON = {'CRITICAL': '🔴', 'HIGH': '🟠', 'MODERATE': '🟡', 'None': '🟢'}


# ── Load model ────────────────────────────────────────────────────────────────
def load_model(model_path):
    print(f"\n📂 Loading model: {model_path}")
    ck          = torch.load(model_path, map_location=DEVICE)
    classes     = ck.get('classes', DEFAULT_CLASSES)
    num_classes = ck.get('num_classes', len(classes))

    model = BrainCTClassifier(num_classes=num_classes).to(DEVICE)
    model.load_state_dict(ck['model_state'])
    model.eval()

    print(f"   ✅ Loaded — {num_classes} classes")
    if ck.get('val_acc'):
        print(f"   Val  Acc : {ck['val_acc']*100:.2f}%")
    if ck.get('test_acc'):
        print(f"   Test Acc : {ck['test_acc']*100:.2f}%")
    print(f"   Classes  : {classes}")
    return model, classes, ck


# ── Predict ───────────────────────────────────────────────────────────────────
def predict_image(image_path, model, classes):
    img_pil    = Image.open(image_path).convert('RGB')
    img_tensor = TRANSFORM(img_pil).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        probs = F.softmax(model(img_tensor), dim=1).squeeze().cpu().numpy()

    pred_idx      = int(np.argmax(probs))
    confidence    = float(probs[pred_idx]) * 100
    is_anomaly    = pred_idx != 0
    anomaly_score = (1.0 - float(probs[0])) * 100

    return {
        'img_pil':       img_pil,
        'pred_idx':      pred_idx,
        'pred_class':    classes[pred_idx],
        'confidence':    confidence,
        'is_anomaly':    is_anomaly,
        'anomaly_score': anomaly_score,
        'severity':      SEVERITY.get(classes[pred_idx], '—'),
        'probs':         {classes[i]: float(probs[i]) * 100 for i in range(len(classes))},
    }


# ── Print report ──────────────────────────────────────────────────────────────
def print_report(result, image_path):
    sev  = result['severity']
    icon = SEV_ICON.get(sev, '⚪')
    line = '─' * 55

    print(f"\n{line}")
    print(f"  🧠  BRAIN CT SCAN ANALYSIS REPORT")
    print(f"{line}")
    print(f"  File          : {Path(image_path).name}")
    print(f"  Status        : {'⚠️  ANOMALY DETECTED' if result['is_anomaly'] else '✅  NORMAL'}")
    print(f"  Diagnosis     : {result['pred_class']}")
    print(f"  Confidence    : {result['confidence']:.1f}%")
    print(f"  Anomaly Score : {result['anomaly_score']:.1f}%")
    if result['is_anomaly']:
        print(f"  Severity      : {icon}  {sev}")
    print(f"{line}")
    print(f"  All probabilities:")
    for cls, pct in sorted(result['probs'].items(), key=lambda x: x[1], reverse=True):
        bar    = '█' * int(pct / 4)
        marker = '  ◄ PREDICTED' if cls == result['pred_class'] else ''
        print(f"    {cls:<25} {pct:5.1f}%  {bar}{marker}")
    print(f"{line}")
    if result['is_anomaly']:
        print("  ⚕️   Please consult a neurologist / radiologist.")
    print()


# ── Heatmap ───────────────────────────────────────────────────────────────────
def make_heatmap(img_pil, model):
    try:
        import cv2
    except ImportError:
        return None

    model.eval()
    tensor      = TRANSFORM(img_pil).unsqueeze(0).to(DEVICE)
    activations = []

    def hook(module, inp, out):
        activations.append(out.detach())

    h = model.backbone.layer4.register_forward_hook(hook)
    with torch.no_grad():
        model(tensor)
    h.remove()

    feat_map  = activations[0].squeeze(0).mean(0).cpu().numpy()
    feat_map  = (feat_map - feat_map.min()) / (feat_map.max() - feat_map.min() + 1e-8)
    heat_up   = cv2.resize(feat_map, (IMG_SIZE, IMG_SIZE))
    orig      = np.array(img_pil.convert('L').resize((IMG_SIZE, IMG_SIZE)))
    orig_rgb  = np.stack([orig] * 3, axis=-1).astype(np.float32) / 255.0
    heat_rgb  = cm.get_cmap('jet')(heat_up)[:, :, :3]
    overlay   = (0.55 * orig_rgb + 0.45 * heat_rgb).clip(0, 1)
    return Image.fromarray((overlay * 255).astype(np.uint8))


# ── Visualize ─────────────────────────────────────────────────────────────────
def visualize(result, image_path, save_path=None):
    heatmap   = make_heatmap(result['img_pil'], model_global)
    n_panels  = 3 if heatmap else 2
    fig, axes = plt.subplots(1, n_panels, figsize=(5 * n_panels, 5))
    fig.patch.set_facecolor('#080c10')

    title_color = '#f87171' if result['is_anomaly'] else '#4ade80'
    fig.suptitle(
        f"{'⚠ ANOMALY' if result['is_anomaly'] else '✓ NORMAL'}  —  "
        f"{result['pred_class']}  ({result['confidence']:.1f}%)",
        fontsize=13, fontweight='bold', color=title_color
    )

    # Original
    axes[0].imshow(np.array(result['img_pil'].resize((IMG_SIZE, IMG_SIZE)))[:, :, 0], cmap='bone')
    axes[0].set_title('Original CT Scan', color='#94a3b8', fontsize=10)
    axes[0].set_facecolor('#0d1117'); axes[0].axis('off')

    # Heatmap
    if heatmap:
        axes[1].imshow(heatmap)
        axes[1].set_title('Activation Heatmap', color='#94a3b8', fontsize=10)
        axes[1].set_facecolor('#0d1117'); axes[1].axis('off')
        prob_ax = axes[2]
    else:
        prob_ax = axes[1]

    # Probability bars
    prob_ax.set_facecolor('#0d1117')
    classes_sorted = sorted(result['probs'].items(), key=lambda x: x[1], reverse=True)
    bar_colors_map = {
        'Normal': '#4ade80', 'Glioma Tumor': '#f87171',
        'Meningioma Tumor': '#fb923c', 'Pituitary Tumor': '#facc15',
        'Hemorrhage': '#f43f5e',
    }
    names  = [c for c, _ in classes_sorted]
    values = [v for _, v in classes_sorted]
    colors = [bar_colors_map.get(n, '#64748b') for n in names]
    bars   = prob_ax.barh(names, values, color=colors, edgecolor='#1e2a38')
    prob_ax.set_xlim(0, 110)
    prob_ax.set_title('Class Probabilities', color='#94a3b8', fontsize=10)
    prob_ax.tick_params(colors='#64748b', labelsize=9)
    prob_ax.set_xlabel('Confidence (%)', color='#475569', fontsize=9)
    for spine in prob_ax.spines.values():
        spine.set_color('#1e2a38')
    prob_ax.axvline(50, color='#334155', linestyle='--', alpha=0.5)
    for bar, v in zip(bars, values):
        prob_ax.text(v + 1, bar.get_y() + bar.get_height() / 2,
                     f'{v:.1f}%', va='center', fontsize=9, color='#64748b')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight',
                    facecolor='#080c10')
        print(f"💾 Result saved → {save_path}")
    else:
        plt.show()
    plt.close()


# ── Batch test ────────────────────────────────────────────────────────────────
def test_folder(folder_path, model, classes, save_dir=None):
    IMG_EXTS = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff'}
    folder   = Path(folder_path)
    images   = [p for p in folder.rglob('*') if p.suffix.lower() in IMG_EXTS]

    if not images:
        print(f"❌ No images found in {folder_path}"); return

    print(f"\n🔍 Testing {len(images)} images in {folder_path}")
    print('─' * 70)

    results = []
    for img_path in sorted(images):
        result = predict_image(img_path, model, classes)
        results.append({**result, 'file': img_path.name})
        status = '⚠' if result['is_anomaly'] else '✓'
        print(f"  {status}  {img_path.name:<35} "
              f"{result['pred_class']:<25} {result['confidence']:5.1f}%")

        if save_dir:
            out = Path(save_dir) / f"result_{img_path.stem}.png"
            visualize(result, img_path, save_path=str(out))

    # Summary
    n_anomaly = sum(1 for r in results if r['is_anomaly'])
    n_normal  = len(results) - n_anomaly
    print('─' * 70)
    print(f"\n📊 Batch Summary:")
    print(f"   Total   : {len(results)}")
    print(f"   Normal  : {n_normal}")
    print(f"   Anomaly : {n_anomaly}")
    cls_counts = {}
    for r in results:
        cls_counts[r['pred_class']] = cls_counts.get(r['pred_class'], 0) + 1
    print(f"\n   By class:")
    for cls, cnt in sorted(cls_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"     {cls:<25} {cnt}")


# ── Global model reference for visualize() ────────────────────────────────────
model_global = None


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Brain CT Disease Detection — Test Script',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  Single image:
    python test_brain_ct.py --image scan.png --model ./models/brain_classifier_final.pth

  Save result image:
    python test_brain_ct.py --image scan.png --model ./models/brain_classifier_final.pth --save result.png

  Batch test a folder:
    python test_brain_ct.py --folder ./test_scans/ --model ./models/brain_classifier_final.pth

  Batch + save all results:
    python test_brain_ct.py --folder ./test_scans/ --model ./models/brain_classifier_final.pth --save ./results/
        """
    )
    parser.add_argument('--image',  type=str, help='Path to a single CT scan image')
    parser.add_argument('--folder', type=str, help='Path to folder of CT scan images (batch mode)')
    parser.add_argument('--model',  type=str, required=True, help='Path to .pth model file')
    parser.add_argument('--save',   type=str, default=None,  help='Save result image or folder')
    parser.add_argument('--no-viz', action='store_true',     help='Skip visualization (print only)')
    args = parser.parse_args()

    if not args.image and not args.folder:
        parser.error("Provide either --image or --folder")

    print(f"\n🖥️  Device : {DEVICE}")

    # Load model
    model, classes, ck = load_model(args.model)
    model_global = model

    # Single image
    if args.image:
        result = predict_image(args.image, model, classes)
        print_report(result, args.image)
        if not args.no_viz:
            visualize(result, args.image, save_path=args.save)

    # Batch folder
    if args.folder:
        save_dir = args.save if args.save else None
        if save_dir:
            Path(save_dir).mkdir(parents=True, exist_ok=True)
        test_folder(args.folder, model, classes, save_dir=save_dir if not args.no_viz else None)
