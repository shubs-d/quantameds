"""
QuantaMED - data-pairing verification & patient-level grouping helper.

Two prerequisites for Stage 2, both handled here:

1. Confirms every labeled eye actually has its full set of 4 Orbscan
   modality images (Axial / Anterior / Posterior / Pachymetry), so the
   "paired data" assumption behind the distillation pipeline is
   verified against the real files, not just assumed from the CSV.

2. Builds a patient_code -> eyes grouping and writes it to CSV, for use
   with sklearn's GroupKFold / StratifiedGroupKFold, so train/val/test
   splits are done at the PATIENT level and never split a patient's
   two eyes across folds (caveat 9: ~95% of patients have both eyes).

Adjusted for the repository path structure and handles the Excel '3E+132' artifact.
"""

from collections import defaultdict
from pathlib import Path
import pandas as pd

BASE_DIR = Path("/home/shubs/Projects/Keratoconus/Dataset")
CSV_PATH = BASE_DIR / "ORBSCAN_Dataset" / "clinical_data_and_labels.csv"
IMAGE_ROOT = BASE_DIR / "ORBSCAN_Dataset"
PATIENT_COL = "patient_code"
EYE_COL = "eye"  # expects values like "OS" / "OD"
MODALITIES = ["Axial", "Anterior", "Posterior", "Pachymetry"]

# Clinical hardware tier mapping
FEATURE_TIER = {
    "astig_value_D": "ark",  # standard ARK cylinder power
    "astig_axis_deg": "ark",  # standard ARK cylinder axis (orthogonal to Kmax axis)
    "kmax_value_D": "ark",  # standard ARK steep K (SimK Max)
    "kmax_axis_deg": "ark",  # standard ARK steep K axis
    "pachy_central_um": "pachymeter",  # ultrasound or optical pachymeter / topographer
    "pachy_thinnest_um": "topographer",  # requires spatial pachymetry map (Orbscan/Pentacam/AS-OCT)
    "pachy_thinnest_x": "topographer",  # spatial coordinate of thinnest point
    "pachy_thinnest_y": "topographer",  # spatial coordinate of thinnest point
    "asphericity_anterior": "topographer",  # anterior conic constant (Q-value)
    "asphericity_posterior": "topographer",  # posterior conic constant (Q-value)
    "age_years": "intake",  # patient demographic
    "gender": "intake",  # patient demographic
}


def verify_pairing(csv_path=CSV_PATH, image_root=IMAGE_ROOT):
    df = pd.read_csv(csv_path, dtype={PATIENT_COL: str})
    # Resolve Excel floating-point scientific notation artifact: '3E132' -> '3E+132'
    df[PATIENT_COL] = df[PATIENT_COL].str.replace("3E+132", "3E132", regex=False)

    missing_folder = []
    missing_files = defaultdict(list)
    ok_eyes = []

    for _, row in df.iterrows():
        patient, eye = str(row[PATIENT_COL]), str(row[EYE_COL])
        eye_dir = image_root / patient / eye

        if not eye_dir.exists():
            missing_folder.append((patient, eye))
            continue

        eye_ok = True
        for modality in MODALITIES:
            fname = f"{patient}_{eye}_{modality}.png"
            if not (eye_dir / fname).exists():
                missing_files[(patient, eye)].append(modality)
                eye_ok = False

        if eye_ok:
            ok_eyes.append((patient, eye))

    patient_eyes = defaultdict(list)
    for patient, eye in ok_eyes:
        patient_eyes[patient].append(eye)
    bilateral = {p: e for p, e in patient_eyes.items() if len(e) > 1}

    print(f"CSV rows:                {len(df)}")
    print(f"Fully paired eyes:       {len(ok_eyes)}")
    print(f"Missing folder entirely: {len(missing_folder)}")
    print(f"Eyes missing >=1 file:   {len(missing_files)}")
    print(f"Unique patients:         {len(patient_eyes)}")
    denom = max(len(patient_eyes), 1)
    print(f"Patients with both eyes: {len(bilateral)} ({len(bilateral) / denom:.1%})")

    if missing_folder:
        print("\nSample missing folders:", missing_folder[:5])
    if missing_files:
        print("Sample incomplete eyes:", list(missing_files.items())[:5])

    return {
        "ok_eyes": ok_eyes,
        "missing_folder": missing_folder,
        "missing_files": dict(missing_files),
        "patient_eyes": dict(patient_eyes),
        "bilateral": bilateral,
    }


if __name__ == "__main__":
    results = verify_pairing()

    groups = pd.DataFrame(
        [(p, e) for p, eyes in results["patient_eyes"].items() for e in eyes],
        columns=[PATIENT_COL, EYE_COL],
    )
    output_path = BASE_DIR / "patient_eye_groups.csv"
    groups.to_csv(output_path, index=False)
    print(
        f"\nWrote {output_path} ({len(groups)} rows) -- "
        f"use `{PATIENT_COL}` as the `groups` argument in "
        f"GroupKFold / StratifiedGroupKFold."
    )

    print("\nFull clinical column list by hardware tier:")
    for col, tier in FEATURE_TIER.items():
        print(f"  {col:<24} {tier}")
