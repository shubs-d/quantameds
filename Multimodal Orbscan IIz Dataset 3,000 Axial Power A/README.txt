===========================================================================
Orbscan IIz Multimodal Corneal Topography Dataset
===========================================================================
DOI        : 10.17632/78wt2nc387.2
License    : CC BY-NC 4.0
Version    : 3
Updated    : July 2026
Contact    : eng.ali78@tu.edu.iq

---------------------------------------------------------------------------
DATASET OVERVIEW
---------------------------------------------------------------------------
This dataset contains 3,000 anonymized anterior axial power map images
acquired from 3,000 unique patients using the Orbscan IIz slit-scanning
corneal topography device (Bausch & Lomb; Serial No. ZDW-08/02-0015) at a
single private ophthalmology clinic in Iraq between 2007 and 2025.

Each image is paired with 19 structured clinical parameter fields extracted
from the original device output using an automated OCR-based pipeline.

A quality control pipeline assessed image integrity, metadata completeness,
and anatomical plausibility. A total of 2,633 image-metadata pairs
satisfied all inclusion criteria (QC_status = PASS) and constitute the
quality-controlled subset used for descriptive characterization.

---------------------------------------------------------------------------
REPOSITORY CONTENTS
---------------------------------------------------------------------------

1. Images.zip
   - 3,000 PNG image files (1024 x 1024 pixels, lossless)
   - Naming convention: PXXXX_AxialPower_Anterior.png
     (XXXX = zero-padded patient identifier, 0001 to 3000)
   - Each image represents a single anterior axial power map acquired from
     one unique patient

2. metadata.csv
   - 3,000 rows x 20 columns
   - Contains the original OCR-extracted clinical parameter values in their
     raw composite text format (e.g., "47.1 D @ 93 deg")
   - One column per clinical parameter plus one filename identifier column
   - Missing values are represented as empty fields
   - Columns: filename, SimK_Astig, MaxK, MinK, Zone3mm_Irreg,
     Zone3mm_MeanPwr, Zone3mm_AstigPwr, Zone3mm_SteepAxis,
     Zone3mm_FlatAxis, Zone5mm_Irreg, Zone5mm_MeanPwr, Zone5mm_AstigPwr,
     Zone5mm_SteepAxis, Zone5mm_FlatAxis, WhiteToWhite_mm,
     PupilDiameter_mm, Thinnest, ACD_Ep_mm, Kappa, KappaIntercept

3. metadata_with_QC.csv
   - 3,000 rows x 21 columns
   - Same as metadata.csv with one additional column: QC_status
   - QC_status = PASS : record satisfies all quality control criteria (N=2,633)
   - QC_status = FAIL : record fails one or more criteria (N=367)
   - To reproduce the descriptive statistics in the article, filter for
     QC_status = PASS before analysis

4. metadata_numeric.csv
   - 3,000 rows x 30 columns
   - Analysis-ready version with composite text fields parsed into
     separate numeric columniini
   - Example: MaxK "47.1 D @ 93 deg" is split into MaxK_D (47.1) and
     MaxK_Axis_deg (93.0)
   - Thinnest is split into Thinnest_um, Thinnest_x_mm, Thinnest_y_mm
   - KappaIntercept is split into KappaIntercept_x and KappaIntercept_y
   - QC_status column is included
   - Anatomical plausibility filter has been applied; values outside
     clinical ranges are replaced with NaN

5. analysis_script.py
   - Python script that reproduces all tables, figures, and statistics
     reported in the associated Data in Brief article
   - Requires: pandas, numpy, scipy, matplotlib (see requirements.txt)
   - Input: metadata_with_QC.csv and metadata_numeric.csv
   - Output: Table 3 statistics (CSV), Figure 1 (PNG), Figure 2 (PNG)
   - Usage: python analysis_script.py

6. requirements.txt
   - Python package requirements for running analysis_script.py
   - Install with: pip install -r requirements.txt

7. README.txt
   - This file

---------------------------------------------------------------------------
IMAGE-METADATA LINKAGE
---------------------------------------------------------------------------
Images and metadata records are linked by the "filename" field in all CSV
files. The value in this field corresponds directly to the PNG filename
without the file extension.

Example:
  filename = P0001_AxialPower_Anterior
  corresponds to image file: P0001_AxialPower_Anterior.png

---------------------------------------------------------------------------
EXAMPLE RECORD (Row 1: P0001_AxialPower_Anterior)
---------------------------------------------------------------------------
filename       : P0001_AxialPower_Anterior
SimK_Astig     : -2.4 D @ 3 deg
MaxK           : 47.1 D @ 93 deg
MinK           : 44.7 D @ 3 deg
Zone3mm_Irreg  : +/- 1.1 D
Zone3mm_MeanPwr: 50.9 +/- 0.6 D
Zone3mm_AstigPwr: 2.2 +/- 0.9 D
Zone3mm_SteepAxis: 92 +/- 10 deg
Zone3mm_FlatAxis: 1 +/- 10 deg
Zone5mm_Irreg  : +/- 2.0 D
Zone5mm_MeanPwr: 50.4 +/- 0.9 D
Zone5mm_AstigPwr: 2.0 +/- 1.8 D
Zone5mm_SteepAxis: 91 +/- 24 deg
Zone5mm_FlatAxis: 1 +/- 23 deg
WhiteToWhite_mm: 11.4
PupilDiameter_mm: 3.8
Thinnest       : 531 um @ (0.8, -0.2)
ACD_Ep_mm      : 3.62
Kappa          : 6.86 deg @ 337.29 deg
KappaIntercept : 0.37, 0.01
QC_status      : PASS

---------------------------------------------------------------------------
REPRODUCING THE QUALITY-CONTROLLED SUBSET
---------------------------------------------------------------------------
To reproduce the quality-controlled subset (N=2,633) used in the article:

  import pandas as pd
  df = pd.read_csv('metadata_with_QC.csv')
  qc_pass = df[df['QC_status'] == 'PASS']
  print(f"QC PASS records: {len(qc_pass)}")

---------------------------------------------------------------------------
REPRODUCING ARTICLE RESULTS
---------------------------------------------------------------------------
To reproduce all tables and figures from the article:

  1. Ensure metadata_with_QC.csv and metadata_numeric.csv are in the
     same directory as analysis_script.py
  2. Install requirements: pip install -r requirements.txt
  3. Run: python analysis_script.py
  4. Outputs will be saved in the same directory

---------------------------------------------------------------------------
MISSING VALUES
---------------------------------------------------------------------------
Missing values in the released metadata files are represented as empty
fields (NaN in pandas). The columns with the highest missing value rates
in the full released dataset (N=3,000) are:

  MaxK           : 233 missing (7.8%)
  MinK           :  97 missing (3.2%)
  SimK_Astig     :  67 missing (2.2%)
  Zone3mm_MeanPwr:  89 missing (3.0%)
  Zone5mm_Irreg  :  69 missing (2.3%)

Records with missing values in key fields are assigned QC_status = FAIL.

---------------------------------------------------------------------------
DATA DICTIONARY (Key Fields)
---------------------------------------------------------------------------
Field              | Unit        | Type          | Valid Range
-------------------|-------------|---------------|------------------
SimK_Astig         | D, deg      | Composite text| -15 to +15 D
MaxK               | D, deg      | Composite text| 35 to 68 D
MinK               | D, deg      | Composite text| 35 to 68 D
Zone3mm_Irreg      | D           | Composite text| 0 to 15 D
Zone3mm_MeanPwr    | D           | Composite text| 35 to 65 D
Zone3mm_AstigPwr   | D           | Composite text| 0 to 15 D
Zone3mm_SteepAxis  | deg         | Composite text| 0 to 180 deg
Zone3mm_FlatAxis   | deg         | Composite text| 0 to 180 deg
Zone5mm_Irreg      | D           | Composite text| 0 to 15 D
Zone5mm_MeanPwr    | D           | Composite text| 35 to 65 D
Zone5mm_AstigPwr   | D           | Composite text| 0 to 15 D
Zone5mm_SteepAxis  | deg         | Composite text| 0 to 180 deg
Zone5mm_FlatAxis   | deg         | Composite text| 0 to 180 deg
WhiteToWhite_mm    | mm          | Numeric       | 9.5 to 13.5 mm
PupilDiameter_mm   | mm          | Numeric       | 1.5 to 8.5 mm
Thinnest           | um, mm, mm  | Composite text| 200 to 700 um
ACD_Ep_mm          | mm          | Numeric       | 2.0 to 6.0 mm
Kappa              | deg         | Composite text| 0 to 18 deg
KappaIntercept     | mm          | Composite text| -3 to +3 mm
QC_status          | -           | Text (PASS/FAIL)| PASS or FAIL

Note: Composite text fields contain the primary measurement plus
associated axis, coordinate, or uncertainty information as output
by the Orbscan IIz device. Use metadata_numeric.csv for analysis
requiring separate numeric columns.

---------------------------------------------------------------------------
ETHICS AND DATA USE
---------------------------------------------------------------------------
The dataset was compiled retrospectively from anonymized corneal
topography images. All patient-identifiable information was permanently
removed before release. Written authorization was obtained from the
original data custodian. Ethical approval is being sought in
accordance with institutional requirements.

This dataset is released under CC BY-NC 4.0. Non-commercial use only.
For questions about data use, contact: eng.ali78@tu.edu.iq

---------------------------------------------------------------------------
CITATION
---------------------------------------------------------------------------
If you use this dataset, please cite:

Alshaykha AMA, Mohammed QAH, Salih MBO (2026). Orbscan IIz Multimodal
Corneal Topography Dataset of Anterior Axial Power Maps with Structured
Clinical Parameters for Unsupervised and Self-Supervised Representation
Learning. Mendeley Data, V3.
DOI: 10.17632/78wt2nc387.2

===========================================================================
