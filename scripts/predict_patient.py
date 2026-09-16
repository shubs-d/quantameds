#!/usr/bin/env python3
"""QuantaMED Patient Keratoconus Predictor.

Evaluates single or paired patient eyes using the trained hybrid quantum-classical
RobustQuantaStudent model (Tier 1: ARK autokeratometer readings alone, Tier 2: with optional pachymetry).

Usage Examples:
    # 1. Direct ARK inputs (Kmax, Cylinder, Axis)
    python scripts/predict_patient.py --kmax 47.0 --cyl -2.25 --axis 86
    python scripts/predict_patient.py --kmax 47.5 --cyl -2.50 --axis 89 --pachy 520

    # 2. Dual-meridian Autokeratometry (K1, K2, Axes, Cylinder)
    python scripts/predict_patient.py --k1 44.75 --k1-axis 176 --k2 47.00 --k2-axis 86 --cyl -2.25

    # 3. Interactive prompt (asks for all values step-by-step)
    python scripts/predict_patient.py --interactive
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from quantum_kc.config import config
from quantum_kc.data.preprocessing import RobustStudentTabularPipeline
from quantum_kc.models.robust_student import RobustQuantaStudent


def clean_float_input(val: any, default: float | None = None) -> float | None:
    """Safely parse float inputs handling negative signs, strings, and whitespace."""
    if val is None:
        return default
    if isinstance(val, (int, float)):
        return float(val)
    s = str(val).strip().replace("−", "-").replace("—", "-")
    if not s:
        return default
    try:
        return float(s)
    except ValueError:
        raise ValueError(f"Could not parse numeric value from '{val}'")


def load_pipeline_and_model():
    """Load fitted tabular preprocessing pipeline and trained RobustQuantaStudent."""
    df = pd.read_csv(config.LABELED_CSV, dtype={"patient_code": str})
    df["patient_code"] = df["patient_code"].str.replace("3E+132", "3E132", regex=False)
    train_df = df[df["split_80_20"] == "train"].reset_index(drop=True)

    pipe = RobustStudentTabularPipeline()
    pipe.fit(train_df)

    ckpt_path = config.CHECKPOINT_DIR / "robust_student_best.pt"
    if not ckpt_path.exists():
        ckpt_path = REPO_ROOT / "qml" / "checkpoints" / "robust_student_best.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found at {ckpt_path}")

    ckpt = torch.load(str(ckpt_path), map_location="cpu")
    model = RobustQuantaStudent(n_layers=2)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    return pipe, model


def diagnose_eye(
    kmax: float,
    cyl: float,
    axis: float,
    pachy: float | None = None,
    pipe: RobustStudentTabularPipeline | None = None,
    model: RobustQuantaStudent | None = None,
) -> dict:
    """Run diagnosis for a single eye given Kmax, cylinder, axis, and optional pachymetry."""
    if pipe is None or model is None:
        pipe, model = load_pipeline_and_model()

    kmax_val = clean_float_input(kmax)
    raw_cyl = clean_float_input(cyl)
    axis_val = clean_float_input(axis)
    pachy_val = clean_float_input(pachy)

    if kmax_val is None:
        raise ValueError("Steep K / Kmax must be provided.")
    if raw_cyl is None:
        raise ValueError("Cylinder / Astigmatism must be provided.")
    if axis_val is None:
        raise ValueError("Cylinder axis must be provided.")

    # Convert cylinder to standard negative cylinder convention
    astig_val = -abs(raw_cyl)

    # Prepare DataFrame row
    dummy_pachy = pachy_val if pachy_val is not None else 500.0
    patient_df = pd.DataFrame([{
        "kmax_value_D": float(kmax_val),
        "astig_value_D": float(astig_val),
        "astig_axis_deg": float(axis_val),
        "pachy_central_um": float(dummy_pachy),
    }])

    base = pipe.transform_base(patient_df)

    # Tier 1 (Autokeratometer ARK only: Kmax, Cyl, Axis)
    t1_input = pipe.apply_dropout(base, mode="tier1")
    with torch.no_grad():
        z_t1, logit_t1 = model(torch.tensor(t1_input, dtype=torch.float32), return_latent=True)
        prob_t1 = torch.sigmoid(logit_t1).item()

    result = {
        "kmax_D": kmax_val,
        "cyl_D": astig_val,
        "raw_cyl_input": raw_cyl,
        "axis_deg": axis_val,
        "t1_logit": logit_t1.item(),
        "t1_prob_kc": prob_t1,
        "t1_prediction": "Keratoconus Suspicious" if prob_t1 >= 0.50 else "Normal / Unlikely KC",
        "t1_latent": z_t1.numpy()[0].tolist(),
    }

    # Tier 2 (ARK + Pachymetry)
    if pachy_val is not None:
        t2_input = pipe.apply_dropout(base, mode="tier2")
        with torch.no_grad():
            z_t2, logit_t2 = model(torch.tensor(t2_input, dtype=torch.float32), return_latent=True)
            prob_t2 = torch.sigmoid(logit_t2).item()
        result["pachy_um"] = pachy_val
        result["t2_logit"] = logit_t2.item()
        result["t2_prob_kc"] = prob_t2
        result["t2_prediction"] = "Keratoconus Suspicious" if prob_t2 >= 0.50 else "Normal / Unlikely KC"
        result["t2_latent"] = z_t2.numpy()[0].tolist()

    return result


def diagnose_keratometry(
    k1: float,
    k1_axis: float,
    k2: float,
    k2_axis: float,
    cyl: float | None = None,
    pachy: float | None = None,
    pipe: RobustStudentTabularPipeline | None = None,
    model: RobustQuantaStudent | None = None,
) -> dict:
    """Diagnose an eye from autokeratometer K1 and K2 readings.
    
    K2 is the steep meridian (Kmax proxy). Its orientation is the steep axis.
    """
    if pipe is None or model is None:
        pipe, model = load_pipeline_and_model()

    k1_val = clean_float_input(k1)
    k2_val = clean_float_input(k2)
    k1_ax = clean_float_input(k1_axis, default=180.0)
    k2_ax = clean_float_input(k2_axis, default=90.0)

    kmax = max(k1_val, k2_val)
    kflat = min(k1_val, k2_val)
    steep_axis = k2_ax if k2_val >= k1_val else k1_ax
    flat_axis = k1_ax if k2_val >= k1_val else k2_ax

    # Use explicitly provided cylinder or calculate from delta K
    if cyl is not None:
        corneal_cyl = -abs(clean_float_input(cyl))
    else:
        corneal_cyl = -abs(kmax - kflat)

    # Primary evaluation with steep axis (meridian of maximum curvature / cone)
    res_steep = diagnose_eye(kmax=kmax, cyl=corneal_cyl, axis=steep_axis, pachy=pachy, pipe=pipe, model=model)
    # Secondary evaluation with flat axis
    res_flat = diagnose_eye(kmax=kmax, cyl=corneal_cyl, axis=flat_axis, pachy=pachy, pipe=pipe, model=model)

    return {
        "kflat_D": kflat,
        "flat_axis": flat_axis,
        "ksteep_D": kmax,
        "steep_axis": steep_axis,
        "cyl_D": corneal_cyl,
        "pachy_um": clean_float_input(pachy),
        "steep_assessment": res_steep,
        "flat_assessment": res_flat,
    }


def print_result_card(eye_name: str, res: dict):
    print("\n" + "=" * 68)
    print(f"  QUANTA-MED CLINICAL REPORT: {eye_name.upper()}")
    print("=" * 68)
    print(f"  Steep K (Kmax)       : {res['kmax_D']:.2f} D")
    print(f"  Corneal Astigmatism  : {res['cyl_D']:.2f} D (Input: {res['raw_cyl_input']})")
    print(f"  Cylinder Axis        : {res['axis_deg']:.1f}°")
    if "pachy_um" in res:
        print(f"  Corneal Pachymetry   : {res['pachy_um']:.1f} µm")
    print("-" * 68)

    prob = res["t1_prob_kc"]
    pred = res["t1_prediction"]
    icon = "⚠️" if prob >= 0.50 else "✅"

    print(f"  [Tier 1: Autokeratometer Alone (Kmax, Cyl, Axis)]")
    print(f"    AI Classification  : {icon} {pred}")
    print(f"    KC Probability     : {prob:.1%} (Logit: {res['t1_logit']:.4f})")
    
    if prob >= 0.50:
        print(f"    Clinical Status    : ⚠️ SUSPICIOUS FOR KERATOCONUS")
        print(f"    Action Required    : REFER to corneal specialist for Scheimpflug / Orbscan.")
    elif prob >= 0.30:
        print(f"    Clinical Status    : ❓ BORDERLINE. Recommend pachymetry / 6-month review.")
    else:
        print(f"    Clinical Status    : ✅ LOW RISK. Normal regular astigmatism.")

    if "t2_prob_kc" in res:
        p2 = res["t2_prob_kc"]
        icon2 = "⚠️" if p2 >= 0.50 else "✅"
        print("-" * 68)
        print(f"  [Tier 2: Autokeratometer + Central Pachymetry]")
        print(f"    AI Classification  : {icon2} {res['t2_prediction']}")
        print(f"    KC Probability     : {p2:.1%} (Logit: {res['t2_logit']:.4f})")

    print("=" * 68 + "\n")


def print_keratometry_card(eye_name: str, res: dict):
    print("\n" + "=" * 70)
    print(f"  QUANTA-MED HYBRID AI REPORT: {eye_name.upper()}")
    print("=" * 70)
    print(f"  Flat K (K1)          : {res['kflat_D']:.2f} D @ {res['flat_axis']:.0f}°")
    print(f"  Steep K (K2 / Kmax)  : {res['ksteep_D']:.2f} D @ {res['steep_axis']:.0f}°")
    print(f"  Corneal Cylinder     : {res['cyl_D']:.2f} D")
    if res.get("pachy_um") is not None:
        print(f"  Pachymetry (CCT)     : {res['pachy_um']:.1f} µm")
    print("-" * 70)

    prob_steep = res["steep_assessment"]["t1_prob_kc"]
    pred_steep = res["steep_assessment"]["t1_prediction"]
    icon_steep = "⚠️" if prob_steep >= 0.50 else "✅"

    print(f"  [Primary Diagnosis: Steep Ectasia Meridian ({res['steep_axis']:.0f}°)]")
    print(f"    AI Classification  : {icon_steep} {pred_steep}")
    print(f"    KC Probability     : {prob_steep:.1%} ({prob_steep:.4f})")

    if prob_steep >= 0.50:
        print(f"    Clinical Status    : ⚠️ SUSPICIOUS FOR KERATOCONUS")
        print(f"    Action Required    : REFER to corneal specialist for Scheimpflug / Orbscan.")
    elif prob_steep >= 0.30:
        print(f"    Clinical Status    : ❓ BORDERLINE. Recommend pachymetry / 6-month review.")
    else:
        print(f"    Clinical Status    : ✅ LOW RISK. Normal regular astigmatism.")

    if "t2_prob_kc" in res["steep_assessment"]:
        p2 = res["steep_assessment"]["t2_prob_kc"]
        icon2 = "⚠️" if p2 >= 0.50 else "✅"
        print("-" * 70)
        print(f"  [Tier 2 Assessment: Combined with Pachymetry]")
        print(f"    KC Probability     : {icon2} {p2:.1%} ({p2:.4f})")
    print("=" * 70 + "\n")


def interactive_mode(pipe, model):
    print("\n" + "═" * 60)
    print("  QuantaMED Interactive Keratoconus Examination")
    print("═" * 60)
    print("Select input mode:")
    print("  [1] Standard ARK Readings (Kmax, Cylinder, Axis, optional Pachy) [Default]")
    print("  [2] Dual-Meridian Keratometry (K1, K1-axis, K2, K2-axis, Cylinder)")
    choice = input("Enter choice [1/2] (default 1): ").strip()

    if choice == "2":
        print("\n--- Dual-Meridian Keratometry Input ---")
        k1 = float(input("  Enter Flat K (K1) in D (e.g. 44.75): "))
        k1_ax = float(input("  Enter Flat K Axis in degrees (e.g. 176): "))
        k2 = float(input("  Enter Steep K (K2) in D (e.g. 47.00): "))
        k2_ax = float(input("  Enter Steep K Axis in degrees (e.g. 86): "))
        cyl_in = input("  Enter Measured Cylinder (CYL) in D [press Enter to auto-calculate]: ").strip()
        cyl = float(cyl_in) if cyl_in else None
        p_in = input("  Enter Central Pachymetry in µm [press Enter to skip]: ").strip()
        pachy = float(p_in) if p_in else None

        res = diagnose_keratometry(k1, k1_ax, k2, k2_ax, cyl=cyl, pachy=pachy, pipe=pipe, model=model)
        print_keratometry_card("Patient Examination", res)
    else:
        print("\n--- Standard ARK Readings Input ---")
        kmax = float(input("  Enter Steep K / Kmax in D (e.g. 47.00): "))
        cyl = float(input("  Enter Cylinder / Astigmatism in D (e.g. -2.25 or 2.25): "))
        axis = float(input("  Enter Cylinder Axis in degrees (e.g. 86): "))
        p_in = input("  Enter Central Pachymetry in µm [press Enter to skip]: ").strip()
        pachy = float(p_in) if p_in else None

        res = diagnose_eye(kmax=kmax, cyl=cyl, axis=axis, pachy=pachy, pipe=pipe, model=model)
        print_result_card("Patient Examination", res)


def main():
    parser = argparse.ArgumentParser(description="Predict Keratoconus for a patient from ARK readings.")
    # Standard ARK inputs
    parser.add_argument("--kmax", type=str, default=None, help="Steep K / Kmax (Diopters, e.g. 47.0)")
    parser.add_argument("--cyl", "--cylinder", "-c", type=str, default=None, help="Corneal cylinder (Diopters, e.g. -2.25 or 2.25)")
    parser.add_argument("--axis", "-a", type=str, default=None, help="Astigmatism axis (Degrees, e.g. 86)")
    parser.add_argument("--pachy", "-p", type=str, default=None, help="Optional central pachymetry (µm, e.g. 540)")
    
    # Dual-meridian inputs
    parser.add_argument("--k1", type=str, default=None, help="Flat K (Diopters, e.g. 44.75)")
    parser.add_argument("--k1-axis", type=str, default=None, help="Flat K axis (Degrees, e.g. 176)")
    parser.add_argument("--k2", type=str, default=None, help="Steep K (Diopters, e.g. 47.00)")
    parser.add_argument("--k2-axis", type=str, default=None, help="Steep K axis (Degrees, e.g. 86)")
    
    parser.add_argument("--eye", type=str, default="Patient Eye", help="Eye label (OD or OS)")
    parser.add_argument("--interactive", "-i", action="store_true", help="Launch interactive prompt")
    args = parser.parse_args()

    pipe, model = load_pipeline_and_model()

    # Dual-meridian mode
    if args.k1 is not None and args.k2 is not None:
        res = diagnose_keratometry(
            k1=args.k1,
            k1_axis=args.k1_axis,
            k2=args.k2,
            k2_axis=args.k2_axis,
            cyl=args.cyl,
            pachy=args.pachy,
            pipe=pipe,
            model=model,
        )
        print_keratometry_card(args.eye, res)

    # Standard ARK mode
    elif args.kmax is not None and args.cyl is not None and args.axis is not None:
        res = diagnose_eye(
            kmax=args.kmax,
            cyl=args.cyl,
            axis=args.axis,
            pachy=args.pachy,
            pipe=pipe,
            model=model,
        )
        print_result_card(args.eye, res)

    # Interactive mode fallback
    else:
        try:
            interactive_mode(pipe, model)
        except (ValueError, KeyboardInterrupt) as e:
            print(f"\nExiting: {e}")
            sys.exit(0)


if __name__ == "__main__":
    main()
