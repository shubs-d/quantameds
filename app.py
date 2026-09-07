"""
Hybrid Quantum-Classical Keratoconus Detection — Interactive Demo Dashboard
===========================================================================
Serves the trained HybridQuantumClassifier (CNN encoder → 8-qubit VQC)
for live inference on Orbscan IIz corneal topography maps.

Usage:
    cd /home/shubs/Projects/Keratoconus/Dataset
    source .venv/bin/activate
    python app.py
"""

from __future__ import annotations

import sys
import math
import logging
from pathlib import Path

# Ensure project root is importable
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import gradio as gr
import numpy as np
import torch
from PIL import Image

from quantum_kc.config import config as cfg, get_device
from quantum_kc.models.encoder import CornealEncoder
from quantum_kc.models.hybrid_classifier import HybridQuantumClassifier
from quantum_kc.data.image_transforms import get_eval_transforms

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════════════
# MODEL LOADING (done once at startup)
# ═══════════════════════════════════════════════════════════════════════
DEVICE = get_device()
CHECKPOINT = PROJECT_ROOT / "checkpoints" / "hybrid_classifier_best.pt"

encoder = CornealEncoder(in_channels=3, latent_dim=cfg.LATENT_DIM)
model = HybridQuantumClassifier(
    encoder=encoder,
    n_qubits=cfg.N_QUBITS,
    n_layers=cfg.N_LAYERS,
    n_classes=2,
    freeze_encoder=True,
    dev_name=cfg.QML_DEVICE,
    diff_method=cfg.DIFF_METHOD,
)

if CHECKPOINT.exists():
    model.load_state_dict(
        torch.load(str(CHECKPOINT), map_location=DEVICE, weights_only=True)
    )
    logger.info("✓ Loaded trained model from %s", CHECKPOINT)
else:
    logger.warning("⚠ Checkpoint not found at %s — using random weights!", CHECKPOINT)

model.to(DEVICE).eval()

# Image preprocessing (same as evaluation pipeline)
eval_transform = get_eval_transforms(cfg.IMG_SIZE)

# ═══════════════════════════════════════════════════════════════════════
# PRESET PATIENT CASES
# ═══════════════════════════════════════════════════════════════════════
DATA_ROOT = Path(cfg.LABELED_ROOT)

PRESETS = {
    "✅ Normal Eye — EGKE2 OD": {
        "patient_code": "EGKE2", "eye": "OD",
        "clinical": "33y | Kmax 43.7 D | Thinnest Pachy 571 µm | Asph_ant -0.44",
    },
    "⚠️ Keratoconus — A31ZS OD": {
        "patient_code": "A31ZS", "eye": "OD",
        "clinical": "41y | Kmax 67.5 D | Thinnest Pachy 275 µm | Asph_ant -1.74",
    },
    "❓ Borderline — SPFQX OD": {
        "patient_code": "SPFQX", "eye": "OD",
        "clinical": "25y | Kmax 42.7 D | Thinnest Pachy 589 µm | Asph_ant -0.38",
    },
}


def load_patient_maps(patient_code: str, eye: str) -> Image.Image | None:
    """Load and stack Axial+Anterior+Posterior maps into an RGB image."""
    map_types = ["Axial", "Anterior", "Posterior"]
    channels = []
    img_dir = DATA_ROOT / patient_code / eye

    for mt in map_types:
        path = img_dir / f"{patient_code}_{eye}_{mt}.png"
        if not path.exists():
            logger.error("Missing map: %s", path)
            return None
        img = Image.open(path).convert("L")
        channels.append(img)

    # Normalise sizes
    widths, heights = zip(*(c.size for c in channels))
    tw, th = max(widths), max(heights)
    if len(set(widths)) > 1 or len(set(heights)) > 1:
        channels = [c.resize((tw, th), Image.LANCZOS) for c in channels]

    return Image.merge("RGB", tuple(channels))


# ═══════════════════════════════════════════════════════════════════════
# INFERENCE
# ═══════════════════════════════════════════════════════════════════════
@torch.no_grad()
def predict_from_image(rgb_image: Image.Image) -> tuple[str, str, str]:
    """Run the full CNN → VQC pipeline on a single 3-channel image.

    Returns (diagnosis_html, confidence_html, details_text).
    """
    img_tensor = eval_transform(rgb_image).unsqueeze(0).to(DEVICE)
    logits = model(img_tensor)
    probs = torch.softmax(logits, dim=1).cpu().numpy()[0]

    pred_class = int(np.argmax(probs))
    normal_prob = float(probs[0])
    kc_prob = float(probs[1])

    # ── Diagnosis badge ──
    if pred_class == 0:
        badge_color = "#22c55e"   # green
        label = "Normal"
        icon = "✅"
    else:
        badge_color = "#ef4444"   # red
        label = "Keratoconus"
        icon = "⚠️"

    diagnosis_html = f"""
    <div style="text-align:center; padding:24px;">
        <div style="font-size:56px; margin-bottom:8px;">{icon}</div>
        <div style="font-size:32px; font-weight:bold; color:{badge_color};">
            {label}
        </div>
        <div style="font-size:16px; color:#94a3b8; margin-top:4px;">
            Hybrid Quantum-Classical Prediction
        </div>
    </div>
    """

    # ── Confidence gauge ──
    bar_pct_kc = kc_prob * 100
    bar_pct_normal = normal_prob * 100
    bar_color_kc = "#ef4444" if kc_prob > 0.5 else "#f97316"
    bar_color_normal = "#22c55e" if normal_prob > 0.5 else "#84cc16"

    confidence_html = f"""
    <div style="padding:16px; background:#1e293b; border-radius:12px;">
        <div style="margin-bottom:12px;">
            <div style="display:flex; justify-content:space-between; margin-bottom:4px;">
                <span style="color:#94a3b8; font-size:14px;">Normal</span>
                <span style="color:white; font-weight:bold; font-size:14px;">{normal_prob:.1%}</span>
            </div>
            <div style="background:#334155; border-radius:6px; height:24px; overflow:hidden;">
                <div style="width:{bar_pct_normal:.1f}%; height:100%; background:{bar_color_normal};
                            border-radius:6px; transition:width 0.4s;"></div>
            </div>
        </div>
        <div>
            <div style="display:flex; justify-content:space-between; margin-bottom:4px;">
                <span style="color:#94a3b8; font-size:14px;">Keratoconus</span>
                <span style="color:white; font-weight:bold; font-size:14px;">{kc_prob:.1%}</span>
            </div>
            <div style="background:#334155; border-radius:6px; height:24px; overflow:hidden;">
                <div style="width:{bar_pct_kc:.1f}%; height:100%; background:{bar_color_kc};
                            border-radius:6px; transition:width 0.4s;"></div>
            </div>
        </div>
    </div>
    """

    details = (
        f"Logits: [{logits[0][0]:.4f}, {logits[0][1]:.4f}]  |  "
        f"Softmax: [Normal={normal_prob:.4f}, KC={kc_prob:.4f}]  |  "
        f"Device: {DEVICE}"
    )

    return diagnosis_html, confidence_html, details


def predict_uploaded(axial_img, anterior_img, posterior_img):
    """Predict from 3 uploaded grayscale map images."""
    if axial_img is None or anterior_img is None or posterior_img is None:
        return (
            "<div style='text-align:center;color:#f59e0b;padding:20px;'>"
            "⚠️ Please upload all 3 maps (Axial, Anterior, Posterior)</div>",
            "", ""
        )

    # Convert numpy arrays from Gradio to PIL
    channels = []
    for arr in [axial_img, anterior_img, posterior_img]:
        if isinstance(arr, np.ndarray):
            pil = Image.fromarray(arr)
        else:
            pil = arr
        channels.append(pil.convert("L"))

    # Normalise sizes
    widths, heights = zip(*(c.size for c in channels))
    tw, th = max(widths), max(heights)
    if len(set(widths)) > 1 or len(set(heights)) > 1:
        channels = [c.resize((tw, th), Image.LANCZOS) for c in channels]

    rgb_image = Image.merge("RGB", tuple(channels))
    return predict_from_image(rgb_image)


def predict_preset(preset_name: str):
    """Load a preset patient and run inference."""
    if preset_name not in PRESETS:
        return (
            "<div style='text-align:center;color:#f59e0b;'>Select a preset case</div>",
            "", "", None
        )

    info = PRESETS[preset_name]
    rgb_image = load_patient_maps(info["patient_code"], info["eye"])

    if rgb_image is None:
        return (
            "<div style='text-align:center;color:#ef4444;'>❌ Maps not found on disk</div>",
            "", "", None
        )

    diagnosis, confidence, details = predict_from_image(rgb_image)
    clinical_note = f"**{preset_name}**\n\n{info['clinical']}"

    return diagnosis, confidence, details, rgb_image


# ═══════════════════════════════════════════════════════════════════════
# MODEL PERFORMANCE PANEL (static HTML)
# ═══════════════════════════════════════════════════════════════════════
METRICS_HTML = """
<div style="padding:16px; background:#0f172a; border-radius:12px; border:1px solid #1e293b;">
    <h3 style="margin:0 0 12px 0; color:#e2e8f0; font-size:16px;">
        📊 Model Performance on Held-Out Test Set (n=291)
    </h3>
    <table style="width:100%; border-collapse:collapse; font-size:14px;">
        <tr style="border-bottom:1px solid #334155;">
            <td style="padding:8px; color:#94a3b8;">Accuracy</td>
            <td style="padding:8px; color:white; font-weight:bold; text-align:right;">76.98%</td>
        </tr>
        <tr style="border-bottom:1px solid #334155;">
            <td style="padding:8px; color:#94a3b8;">AUC-ROC</td>
            <td style="padding:8px; color:#38bdf8; font-weight:bold; text-align:right;">0.879</td>
        </tr>
        <tr style="border-bottom:1px solid #334155;">
            <td style="padding:8px; color:#94a3b8;">KC Sensitivity (Recall)</td>
            <td style="padding:8px; color:#22c55e; font-weight:bold; text-align:right;">84.96%</td>
        </tr>
        <tr style="border-bottom:1px solid #334155;">
            <td style="padding:8px; color:#94a3b8;">Normal Specificity</td>
            <td style="padding:8px; color:#22c55e; font-weight:bold; text-align:right;">71.91%</td>
        </tr>
        <tr style="border-bottom:1px solid #334155;">
            <td style="padding:8px; color:#94a3b8;">KC Precision</td>
            <td style="padding:8px; color:white; font-weight:bold; text-align:right;">65.75%</td>
        </tr>
        <tr>
            <td style="padding:8px; color:#94a3b8;">Macro F1-Score</td>
            <td style="padding:8px; color:white; font-weight:bold; text-align:right;">76.69%</td>
        </tr>
    </table>
    <div style="margin-top:12px; padding:10px; background:#1e293b; border-radius:8px;">
        <div style="font-size:12px; color:#64748b; margin-bottom:4px;">Confusion Matrix</div>
        <div style="display:grid; grid-template-columns:1fr 1fr; gap:4px; max-width:240px;">
            <div style="background:#166534; color:white; padding:8px; text-align:center; border-radius:4px; font-size:13px;">
                TN: 128
            </div>
            <div style="background:#7f1d1d; color:white; padding:8px; text-align:center; border-radius:4px; font-size:13px;">
                FP: 50
            </div>
            <div style="background:#7f1d1d; color:white; padding:8px; text-align:center; border-radius:4px; font-size:13px;">
                FN: 17
            </div>
            <div style="background:#166534; color:white; padding:8px; text-align:center; border-radius:4px; font-size:13px;">
                TP: 96
            </div>
        </div>
    </div>
</div>
"""

ARCH_HTML = """
<div style="padding:16px; background:#0f172a; border-radius:12px; border:1px solid #1e293b;">
    <h3 style="margin:0 0 10px 0; color:#e2e8f0; font-size:16px;">
        🧬 Architecture
    </h3>
    <div style="font-size:13px; color:#94a3b8; line-height:1.7;">
        <b>Stage 1:</b> Convolutional Autoencoder pretrained on 2,633 unlabeled
        Orbscan IIz scans (unsupervised representation learning)<br/>
        <b>Stage 2:</b> 8-qubit Variational Quantum Classifier (VQC) fine-tuned
        on 1,163 labeled CornOrb records<br/><br/>
        <span style="color:#64748b;">Pipeline:</span><br/>
        <code style="color:#38bdf8; font-size:12px;">
        3 Topography Maps → CNN Encoder (frozen) → 8-dim latent<br/>
        → Sigmoid scaling × π → AngleEmbedding(R<sub>y</sub>)<br/>
        → StronglyEntanglingLayers(depth=2) → ⟨Z⟩ × 8<br/>
        → Linear(8→2) → Softmax → [Normal, KC]
        </code>
    </div>
</div>
"""


# ═══════════════════════════════════════════════════════════════════════
# GRADIO UI
# ═══════════════════════════════════════════════════════════════════════
THEME = gr.themes.Soft(
    primary_hue="blue",
    secondary_hue="slate",
    neutral_hue="slate",
    font=gr.themes.GoogleFont("Inter"),
).set(
    body_background_fill="#0f172a",
    block_background_fill="#1e293b",
    block_border_color="#334155",
    block_label_text_color="#e2e8f0",
    input_background_fill="#334155",
    button_primary_background_fill="#3b82f6",
    button_primary_background_fill_hover="#2563eb",
)

CUSTOM_CSS = """
.gradio-container { max-width: 1200px !important; }
h1, h2, h3 { color: #e2e8f0 !important; }
.prose p { color: #94a3b8 !important; }
"""

with gr.Blocks(title="Quantum KC Detector") as demo:

    gr.Markdown(
        """
        # 🔬 Hybrid Quantum-Classical Keratoconus Detector
        **Upload 3 Orbscan IIz corneal topography maps** (Axial Power, Anterior Elevation, Posterior Elevation)
        or **select a preset case** for instant inference through the 8-qubit variational quantum circuit.
        """,
    )

    with gr.Row(equal_height=False):
        # ── Left Column: Input ──
        with gr.Column(scale=3):
            with gr.Tabs() as tabs:
                # ── Tab 1: Preset Cases (for quick demo) ──
                with gr.Tab("⚡ Preset Cases", id="presets"):
                    gr.Markdown("*One-click demo with real patient data from the test set*")
                    preset_dropdown = gr.Dropdown(
                        choices=list(PRESETS.keys()),
                        label="Select Patient Case",
                        value=None,
                    )
                    preset_btn = gr.Button("🔍 Run Inference", variant="primary", size="lg")
                    preset_preview = gr.Image(
                        label="Stacked Maps (Axial / Anterior / Posterior → RGB)",
                        type="pil",
                        interactive=False,
                        height=300,
                    )

                # ── Tab 2: Upload Maps ──
                with gr.Tab("📂 Upload Maps", id="upload"):
                    gr.Markdown("*Upload 3 grayscale PNG maps from an Orbscan IIz scan*")
                    with gr.Row():
                        axial_input = gr.Image(label="Axial Power Map", type="numpy", height=180)
                        anterior_input = gr.Image(label="Anterior Elevation", type="numpy", height=180)
                        posterior_input = gr.Image(label="Posterior Elevation", type="numpy", height=180)
                    upload_btn = gr.Button("🔍 Run Inference", variant="primary", size="lg")

        # ── Right Column: Results ──
        with gr.Column(scale=2):
            diagnosis_out = gr.HTML(
                value="<div style='text-align:center;color:#64748b;padding:40px;'>"
                "Awaiting input…</div>",
                label="Diagnosis",
            )
            confidence_out = gr.HTML(label="Confidence")
            details_out = gr.Textbox(
                label="Raw Output",
                interactive=False,
                lines=2,
                max_lines=3,
            )

    gr.Markdown("---")

    with gr.Row(equal_height=True):
        with gr.Column(scale=1):
            gr.HTML(METRICS_HTML)
        with gr.Column(scale=1):
            gr.HTML(ARCH_HTML)

    gr.Markdown(
        "<div style='text-align:center; color:#475569; font-size:12px; padding:16px;'>"
        "Built with PennyLane + PyTorch • 8-qubit StronglyEntanglingLayers ansatz • "
        "Trained on CornOrb dataset (1,454 eyes)"
        "</div>"
    )

    # ── Event handlers ──
    preset_btn.click(
        fn=predict_preset,
        inputs=[preset_dropdown],
        outputs=[diagnosis_out, confidence_out, details_out, preset_preview],
    )

    upload_btn.click(
        fn=predict_uploaded,
        inputs=[axial_input, anterior_input, posterior_input],
        outputs=[diagnosis_out, confidence_out, details_out],
    )


# ═══════════════════════════════════════════════════════════════════════
# LAUNCH
# ═══════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("  🔬 Quantum Keratoconus Detector — Dashboard")
    print(f"  Model: {CHECKPOINT}")
    print(f"  Device: {DEVICE}")
    print("=" * 60 + "\n")

    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True,
        theme=THEME,
        css=CUSTOM_CSS,
    )
