"""
generate_dataset.py
-------------------
Generates a synthetic symptom-based disease prediction dataset with ~150,000 rows.

Each row represents one patient's symptom profile.
All symptom columns are binary (0 = absent, 1 = present).
The label column contains the predicted disease class.

Output: symptom_disease_dataset.csv
"""

import numpy as np
import pandas as pd
from collections import defaultdict

# =============================================================================
# CONFIGURATION
# =============================================================================

RANDOM_SEED  = 42
TARGET_ROWS  = 150_000
OUTPUT_FILE  = "symptom_disease_dataset.csv"

np.random.seed(RANDOM_SEED)

# =============================================================================
# DISEASE DEFINITIONS
# Each disease has:
#   - weight        : relative share of total rows
#   - age_range     : (min, max) typical age
#   - sex_bias      : probability of being Male (0.5 = equal)
#   - core          : symptoms almost always present (probability range)
#   - supporting    : symptoms often present
#   - risk_factors  : background conditions that correlate
#   - noise         : unrelated symptoms that may appear occasionally
# =============================================================================

DISEASES = {

    # ── ECG Classes ────────────────────────────────────────────────────────────

    "normal": {
        "weight": 0.10,
        "age_range": (18, 65),
        "sex_bias": 0.50,
        "core": {},
        "supporting": {
            "fatigue": (0.0, 0.10),
        },
        "risk_factors": {
            "hypertension_history": (0.0, 0.10),
            "diabetes_history":     (0.0, 0.10),
        },
        "noise_rate": 0.03,
    },

    "myocardial_infarction": {
        "weight": 0.07,
        "age_range": (45, 85),
        "sex_bias": 0.65,
        "core": {
            "chest_pain":           (0.90, 1.00),
            "chest_pain_radiates":  (0.75, 0.95),
            "sweating_with_pain":   (0.80, 0.95),
            "nausea":               (0.70, 0.90),
        },
        "supporting": {
            "shortness_of_breath":  (0.60, 0.85),
            "fatigue":              (0.50, 0.75),
            "dizziness":            (0.40, 0.65),
            "palpitation":          (0.30, 0.55),
        },
        "risk_factors": {
            "hypertension_history": (0.55, 0.80),
            "diabetes_history":     (0.35, 0.60),
            "smoking_history":      (0.40, 0.65),
        },
        "noise_rate": 0.04,
    },

    "atrial_fibrillation": {
        "weight": 0.07,
        "age_range": (50, 85),
        "sex_bias": 0.55,
        "core": {
            "palpitation":          (0.85, 1.00),
            "irregular_heartbeat":  (0.90, 1.00),
        },
        "supporting": {
            "dizziness":            (0.60, 0.85),
            "shortness_of_breath":  (0.50, 0.75),
            "fatigue":              (0.55, 0.80),
            "syncope":              (0.20, 0.40),
            "chest_pain":           (0.20, 0.40),
        },
        "risk_factors": {
            "hypertension_history": (0.50, 0.75),
            "diabetes_history":     (0.25, 0.45),
        },
        "noise_rate": 0.04,
    },

    "ventricular_arrhythmia": {
        "weight": 0.05,
        "age_range": (35, 80),
        "sex_bias": 0.60,
        "core": {
            "palpitation":          (0.80, 1.00),
            "irregular_heartbeat":  (0.75, 0.95),
            "syncope":              (0.65, 0.90),
        },
        "supporting": {
            "chest_pain":           (0.55, 0.80),
            "shortness_of_breath":  (0.50, 0.75),
            "dizziness":            (0.60, 0.85),
        },
        "risk_factors": {
            "hypertension_history": (0.40, 0.65),
            "diabetes_history":     (0.25, 0.45),
        },
        "noise_rate": 0.04,
    },

    "supraventricular_arrhythmia": {
        "weight": 0.05,
        "age_range": (25, 70),
        "sex_bias": 0.48,
        "core": {
            "palpitation":          (0.85, 1.00),
        },
        "supporting": {
            "dizziness":            (0.65, 0.85),
            "shortness_of_breath":  (0.40, 0.65),
            "fatigue":              (0.40, 0.65),
            "syncope":              (0.15, 0.35),
            "chest_pain":           (0.20, 0.40),
        },
        "risk_factors": {
            "hypertension_history": (0.20, 0.40),
        },
        "noise_rate": 0.04,
    },

    "conduction_disorder": {
        "weight": 0.05,
        "age_range": (50, 85),
        "sex_bias": 0.55,
        "core": {
            "slow_heartbeat":       (0.85, 1.00),
            "syncope":              (0.70, 0.90),
        },
        "supporting": {
            "fatigue":              (0.65, 0.85),
            "dizziness":            (0.60, 0.80),
            "shortness_of_breath":  (0.45, 0.70),
        },
        "risk_factors": {
            "hypertension_history": (0.35, 0.60),
            "diabetes_history":     (0.25, 0.45),
        },
        "noise_rate": 0.04,
    },

    "ischemia": {
        "weight": 0.06,
        "age_range": (45, 80),
        "sex_bias": 0.60,
        "core": {
            "chest_pain":           (0.85, 1.00),
        },
        "supporting": {
            "chest_pain_radiates":  (0.30, 0.55),
            "shortness_of_breath":  (0.50, 0.75),
            "fatigue":              (0.45, 0.70),
            "dizziness":            (0.30, 0.55),
            "sweating_with_pain":   (0.25, 0.50),
        },
        "risk_factors": {
            "hypertension_history": (0.55, 0.80),
            "diabetes_history":     (0.40, 0.65),
            "smoking_history":      (0.35, 0.60),
        },
        "noise_rate": 0.04,
    },

    "hypertrophy": {
        "weight": 0.05,
        "age_range": (40, 80),
        "sex_bias": 0.55,
        "core": {
            "fatigue":              (0.70, 0.90),
            "shortness_of_breath":  (0.65, 0.85),
        },
        "supporting": {
            "worse_lying_down":     (0.30, 0.55),
            "dizziness":            (0.25, 0.50),
            "palpitation":          (0.20, 0.40),
        },
        "risk_factors": {
            "hypertension_history": (0.60, 0.85),
            "diabetes_history":     (0.25, 0.45),
        },
        "noise_rate": 0.04,
    },

    # ── X-Ray Classes ──────────────────────────────────────────────────────────

    "cardiomegaly": {
        "weight": 0.06,
        "age_range": (45, 85),
        "sex_bias": 0.52,
        "core": {
            "leg_swelling":         (0.80, 0.95),
            "worse_lying_down":     (0.75, 0.95),
            "shortness_of_breath":  (0.80, 0.95),
        },
        "supporting": {
            "fatigue":              (0.65, 0.85),
            "palpitation":          (0.35, 0.60),
            "cough":                (0.30, 0.55),
        },
        "risk_factors": {
            "hypertension_history": (0.55, 0.80),
            "diabetes_history":     (0.30, 0.55),
        },
        "noise_rate": 0.04,
    },

    "pulmonary_edema": {
        "weight": 0.05,
        "age_range": (45, 85),
        "sex_bias": 0.52,
        "core": {
            "shortness_of_breath":  (0.90, 1.00),
            "worse_lying_down":     (0.85, 1.00),
            "cough":                (0.75, 0.95),
        },
        "supporting": {
            "leg_swelling":         (0.55, 0.80),
            "fatigue":              (0.60, 0.80),
            "fever":                (0.10, 0.25),
        },
        "risk_factors": {
            "hypertension_history": (0.50, 0.75),
        },
        "noise_rate": 0.04,
    },

    "pneumonia": {
        "weight": 0.07,
        "age_range": (5, 80),
        "sex_bias": 0.50,
        "core": {
            "fever":                (0.85, 1.00),
            "cough":                (0.85, 1.00),
            "chest_pain":           (0.60, 0.85),
        },
        "supporting": {
            "shortness_of_breath":  (0.55, 0.80),
            "fatigue":              (0.65, 0.85),
            "nausea":               (0.25, 0.50),
            "dizziness":            (0.20, 0.40),
        },
        "risk_factors": {
            "diabetes_history":     (0.15, 0.35),
            "smoking_history":      (0.20, 0.40),
        },
        "noise_rate": 0.04,
    },

    "consolidation": {
        "weight": 0.05,
        "age_range": (18, 80),
        "sex_bias": 0.50,
        "core": {
            "fever":                (0.80, 0.95),
            "cough":                (0.80, 0.95),
            "shortness_of_breath":  (0.65, 0.85),
        },
        "supporting": {
            "chest_pain":           (0.30, 0.55),
            "fatigue":              (0.60, 0.80),
        },
        "risk_factors": {
            "smoking_history":      (0.15, 0.35),
        },
        "noise_rate": 0.04,
    },

    "atelectasis": {
        "weight": 0.04,
        "age_range": (25, 80),
        "sex_bias": 0.50,
        "core": {
            "shortness_of_breath":  (0.85, 1.00),
        },
        "supporting": {
            "fatigue":              (0.55, 0.80),
            "cough":                (0.40, 0.65),
            "fever":                (0.20, 0.40),
        },
        "risk_factors": {},
        "noise_rate": 0.04,
    },

    # ── CT Scan ────────────────────────────────────────────────────────────────

    "ct_anomaly": {
        "weight": 0.04,
        "age_range": (40, 85),
        "sex_bias": 0.60,
        "core": {
            "cough_with_blood":     (0.65, 0.90),
            "sudden_weight_loss":   (0.70, 0.90),
            "cough":                (0.75, 0.95),
        },
        "supporting": {
            "chest_pain":           (0.35, 0.60),
            "shortness_of_breath":  (0.40, 0.65),
            "fatigue":              (0.55, 0.80),
        },
        "risk_factors": {
            "smoking_history":      (0.55, 0.80),
        },
        "noise_rate": 0.04,
    },

    # ── Skin Classes ───────────────────────────────────────────────────────────

    "melanoma": {
        "weight": 0.04,
        "age_range": (30, 80),
        "sex_bias": 0.50,
        "core": {
            "skin_lesion":              (1.00, 1.00),
            "lesion_changing":          (0.90, 1.00),
            "lesion_irregular_border":  (0.85, 1.00),
            "lesion_color_multiple":    (0.80, 1.00),
        },
        "supporting": {
            "lesion_bleeding_itching":  (0.40, 0.65),
            "sudden_weight_loss":       (0.20, 0.40),
        },
        "risk_factors": {
            "sun_exposure_history": (0.60, 0.85),
        },
        "noise_rate": 0.03,
    },

    "basal_cell_carcinoma": {
        "weight": 0.03,
        "age_range": (40, 80),
        "sex_bias": 0.55,
        "core": {
            "skin_lesion":              (1.00, 1.00),
            "lesion_bleeding_itching":  (0.75, 0.95),
        },
        "supporting": {
            "lesion_irregular_border":  (0.55, 0.80),
            "lesion_changing":          (0.40, 0.65),
        },
        "risk_factors": {
            "sun_exposure_history": (0.65, 0.90),
        },
        "noise_rate": 0.03,
    },

    "actinic_keratosis": {
        "weight": 0.03,
        "age_range": (45, 80),
        "sex_bias": 0.55,
        "core": {
            "skin_lesion":              (1.00, 1.00),
            "lesion_bleeding_itching":  (0.60, 0.85),
        },
        "supporting": {
            "lesion_color_multiple":    (0.20, 0.40),
            "lesion_changing":          (0.30, 0.55),
        },
        "risk_factors": {
            "sun_exposure_history": (0.75, 0.95),
        },
        "noise_rate": 0.03,
    },

    "benign_keratosis": {
        "weight": 0.03,
        "age_range": (40, 75),
        "sex_bias": 0.50,
        "core": {
            "skin_lesion":          (1.00, 1.00),
        },
        "supporting": {
            "lesion_changing":      (0.05, 0.15),
        },
        "risk_factors": {},
        "noise_rate": 0.02,
    },

    "dermatofibroma": {
        "weight": 0.02,
        "age_range": (20, 65),
        "sex_bias": 0.40,
        "core": {
            "skin_lesion":          (1.00, 1.00),
        },
        "supporting": {
            "lesion_bleeding_itching": (0.10, 0.25),
            "lesion_changing":         (0.05, 0.15),
        },
        "risk_factors": {},
        "noise_rate": 0.02,
    },

    "melanocytic_nevi": {
        "weight": 0.03,
        "age_range": (15, 60),
        "sex_bias": 0.48,
        "core": {
            "skin_lesion":          (1.00, 1.00),
        },
        "supporting": {
            "lesion_changing":      (0.10, 0.25),
            "lesion_bleeding_itching": (0.05, 0.15),
        },
        "risk_factors": {},
        "noise_rate": 0.02,
    },

    "vascular_lesion": {
        "weight": 0.02,
        "age_range": (10, 70),
        "sex_bias": 0.50,
        "core": {
            "skin_lesion":          (1.00, 1.00),
        },
        "supporting": {
            "lesion_bleeding_itching": (0.15, 0.30),
        },
        "risk_factors": {},
        "noise_rate": 0.02,
    },

    "others": {
        "weight": 0.04,
        "age_range": (18, 80),
        "sex_bias": 0.50,
        "core": {},
        "supporting": {
            "fatigue":              (0.20, 0.50),
            "dizziness":            (0.10, 0.30),
            "nausea":               (0.10, 0.30),
            "fever":                (0.15, 0.35),
        },
        "risk_factors": {},
        "noise_rate": 0.05,
    },
}

# All binary symptom columns (order matters — this is the column order in CSV)
SYMPTOM_COLS = [
    # Demographics
    "age", "sex", "smoking_history",
    # Chest / Cardiac
    "chest_pain", "chest_pain_radiates", "palpitation", "irregular_heartbeat",
    "slow_heartbeat", "shortness_of_breath", "worse_lying_down", "leg_swelling",
    "cough", "cough_with_blood", "fever",
    # Systemic
    "syncope", "fatigue", "sudden_weight_loss", "sweating_with_pain", "nausea", "dizziness",
    # Skin
    "skin_lesion", "lesion_changing", "lesion_irregular_border",
    "lesion_bleeding_itching", "lesion_color_multiple",
    # Risk factors
    "hypertension_history", "diabetes_history", "sun_exposure_history",
]

BINARY_SYMPTOM_COLS = [c for c in SYMPTOM_COLS if c not in ("age", "sex")]

# =============================================================================
# ROW GENERATION
# =============================================================================

def sample_binary(prob_range):
    """Return 1 with probability drawn uniformly from prob_range, else 0."""
    prob = np.random.uniform(*prob_range)
    return int(np.random.random() < prob)


def generate_row(disease_name, disease_cfg):
    """Generate one patient row for the given disease."""
    row = {}

    # Demographics
    row["age"] = int(np.random.randint(*disease_cfg["age_range"]))
    row["sex"] = "M" if np.random.random() < disease_cfg["sex_bias"] else "F"

    # Initialize all binary symptom columns to 0
    for col in BINARY_SYMPTOM_COLS:
        row[col] = 0

    # Core symptoms
    for col, prob_range in disease_cfg["core"].items():
        row[col] = sample_binary(prob_range)

    # Supporting symptoms
    for col, prob_range in disease_cfg["supporting"].items():
        row[col] = sample_binary(prob_range)

    # Risk factors
    for col, prob_range in disease_cfg["risk_factors"].items():
        row[col] = sample_binary(prob_range)

    # Random noise — occasional unrelated symptoms
    noise_rate = disease_cfg.get("noise_rate", 0.03)
    for col in BINARY_SYMPTOM_COLS:
        if row[col] == 0 and np.random.random() < noise_rate:
            row[col] = 1

    row["label"] = disease_name
    return row


# =============================================================================
# DATASET GENERATION
# =============================================================================

def build_dataset(target_rows, diseases):
    """
    Generate target_rows patient records distributed across diseases
    according to their weight values.
    """
    # Normalize weights
    names   = list(diseases.keys())
    weights = np.array([diseases[n]["weight"] for n in names], dtype=float)
    weights /= weights.sum()

    # Determine row count per disease
    counts = (weights * target_rows).astype(int)
    # Distribute rounding remainder to the first disease
    counts[0] += target_rows - counts.sum()

    print(f"Target rows : {target_rows:,}")
    print(f"Diseases    : {len(names)}")
    print()

    rows = []
    for name, count in zip(names, counts):
        print(f"  Generating {count:>6,} rows for  {name}")
        cfg = diseases[name]
        for _ in range(count):
            rows.append(generate_row(name, cfg))

    # Shuffle to avoid disease blocks
    np.random.shuffle(rows)
    return rows


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    print("=" * 55)
    print("  Symptom-Disease Synthetic Dataset Generator")
    print("=" * 55)
    print()

    rows = build_dataset(TARGET_ROWS, DISEASES)

    # Build DataFrame with consistent column order
    col_order = SYMPTOM_COLS + ["label"]
    df = pd.DataFrame(rows, columns=col_order)

    # Basic validation
    assert len(df) == TARGET_ROWS, "Row count mismatch."
    assert df["label"].nunique() == len(DISEASES), "Missing disease classes."
    assert df.isnull().sum().sum() == 0, "Null values found."

    # Save
    df.to_csv(OUTPUT_FILE, index=False)

    print()
    print("=" * 55)
    print(f"  Dataset saved : {OUTPUT_FILE}")
    print(f"  Total rows    : {len(df):,}")
    print(f"  Total columns : {len(df.columns)}  ({len(SYMPTOM_COLS)} features + label)")
    print()
    print("  Class distribution:")
    dist = df["label"].value_counts()
    for label, count in dist.items():
        pct = count / len(df) * 100
        print(f"    {label:<30}  {count:>6,}  ({pct:.1f}%)")
    print("=" * 55)