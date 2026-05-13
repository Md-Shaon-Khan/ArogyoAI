"""
test_symptom_model.py
─────────────────────
HealthBridge — Symptom Disease Predictor
Adaptive question engine: 10-12 questions → 28 column fill → Ensemble predict

Folder structure expected:
    All Disease Checkup/
    └── trained_models/
        ├── symptom_model.pkl
        ├── label_encoder.pkl
        ├── feature_columns.pkl
        └── question_engine.pkl

Usage:
    python test.py
    python test.py --models-dir ./trained_models
"""

import os
import sys
import pickle
import argparse
import numpy as np

# ─────────────────────────────────────────────
# Arguments
# ─────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument(
    "--models-dir",
    default="trained_models",
    help="Path to trained_models folder",
)
args = parser.parse_args()
MODELS_DIR = args.models_dir

# ─────────────────────────────────────────────
# Terminal colors
# ─────────────────────────────────────────────
CYAN   = "\033[96m"
GREEN  = "\033[92m"
YELLOW = "\033[93m"
RED    = "\033[91m"
BOLD   = "\033[1m"
RESET  = "\033[0m"
DIM    = "\033[2m"

def header(text):  print(f"\n{BOLD}{CYAN}{'─'*60}{RESET}\n{BOLD}{CYAN}  {text}{RESET}\n{BOLD}{CYAN}{'─'*60}{RESET}")
def info(text):    print(f"  {DIM}{text}{RESET}")
def ask(text):     return input(f"\n{YELLOW}{BOLD}  > {text}: {RESET}")
def success(text): print(f"\n{GREEN}{BOLD}  ✓ {text}{RESET}")
def warn(text):    print(f"\n{YELLOW}  ! {text}{RESET}")
def err(text):     print(f"\n{RED}{BOLD}  X {text}{RESET}")


# ─────────────────────────────────────────────
# Model load
# ─────────────────────────────────────────────
def load_artifacts(model_dir: str) -> dict:
    required = ["symptom_model", "label_encoder", "feature_columns", "question_engine"]
    arts = {}

    script_dir   = os.path.dirname(os.path.abspath(__file__))
    resolved_dir = (
        os.path.join(script_dir, model_dir)
        if not os.path.isabs(model_dir)
        else model_dir
    )

    if not os.path.isdir(resolved_dir):
        err(f"Folder not found: {resolved_dir}")
        err("Make sure 'trained_models/' is in the same directory as test.py")
        sys.exit(1)

    for name in required:
        path = os.path.join(resolved_dir, f"{name}.pkl")
        if not os.path.exists(path):
            err(f"{name}.pkl not found  →  {path}")
            err("Run the Kaggle notebook first and download the trained_models/ folder.")
            sys.exit(1)
        with open(path, "rb") as f:
            arts[name] = pickle.load(f)

    success(f"All models loaded  ({resolved_dir})")
    return arts


# ─────────────────────────────────────────────
# map_answers_to_features  (28 column fill)
# ─────────────────────────────────────────────
def map_answers_to_features(answers: dict, feature_columns: list) -> dict:
    features = {col: 0 for col in feature_columns}

    # Age
    if "age" in answers:
        features["age"] = int(answers["age"])

    # Sex: encode as numeric  (Male=1, Female=0)
    if "sex" in answers:
        features["sex"] = 1 if answers["sex"] == "M" else 0

    features["smoking_history"] = int(answers.get("smoking_history", 0))
    features["chest_pain"]      = int(answers.get("chest_pain", 0))

    radiates = int(answers.get("chest_pain_radiates", 0))
    features["chest_pain_radiates"] = radiates
    if radiates:
        features["sweating_with_pain"] = 1

    pal = answers.get("palpitation_type", "none")
    if pal != "none":      features["palpitation"]        = 1
    if pal == "irregular": features["irregular_heartbeat"] = 1
    if pal == "slow":      features["slow_heartbeat"]     = 1

    features["syncope"] = int(answers.get("syncope", 0))

    breath = answers.get("breath_pattern", "none")
    if breath != "none":       features["shortness_of_breath"] = 1
    if breath == "lying_down": features["worse_lying_down"]    = 1

    features["leg_swelling"] = int(answers.get("leg_swelling", 0))

    cough = answers.get("cough_type", "none")
    if cough != "none":  features["cough"]            = 1
    if cough == "blood": features["cough_with_blood"] = 1

    features["fever"]                = int(answers.get("fever", 0))
    features["skin_lesion"]          = int(answers.get("skin_lesion", 0))
    features["lesion_changing"]      = int(answers.get("lesion_changing", 0))
    features["sun_exposure_history"] = int(answers.get("sun_exposure_history", 0))

    lf = answers.get("lesion_features", "none")
    if lf == "irregular_border":   features["lesion_irregular_border"] = 1
    elif lf == "multiple_colors":  features["lesion_color_multiple"]   = 1
    elif lf == "bleeding_itching": features["lesion_bleeding_itching"] = 1

    sys_ans = answers.get("systemic", "none")
    if sys_ans in ("fatigue", "both"):     features["fatigue"]            = 1
    if sys_ans in ("weight_loss", "both"): features["sudden_weight_loss"] = 1

    if int(answers.get("other_systemic", 0)):
        features["nausea"] = features["dizziness"] = 1

    risk = answers.get("risk_factors", "none")
    if risk in ("hypertension", "both"): features["hypertension_history"] = 1
    if risk in ("diabetes", "both"):     features["diabetes_history"]     = 1

    return features


# ─────────────────────────────────────────────
# Predict
# ─────────────────────────────────────────────
def predict_from_answers(answers: dict, arts: dict) -> dict:
    model  = arts["symptom_model"]
    enc    = arts["label_encoder"]
    f_cols = arts["feature_columns"]

    fvec = map_answers_to_features(answers, f_cols)
    X    = np.array([[fvec[col] for col in f_cols]], dtype=np.float32)

    proba = model.predict_proba(X)[0]
    idx   = int(np.argmax(proba))

    top3 = [
        {
            "disease":     enc.inverse_transform([i])[0],
            "probability": round(float(proba[i]) * 100, 1),
        }
        for i in np.argsort(proba)[::-1][:3]
    ]

    return {
        "prediction":     enc.inverse_transform([idx])[0],
        "confidence":     round(float(proba[idx]) * 100, 1),
        "inconclusive":   float(proba[idx]) < 0.50,
        "top3":           top3,
        "filled_columns": sum(1 for v in fvec.values() if v != 0),
        "total_columns":  len(f_cols),
    }


# ─────────────────────────────────────────────
# Input helpers
# ─────────────────────────────────────────────
def get_binary(question: str, hint: str) -> int:
    """Yes (1) / No (0)"""
    info(f"Hint: {hint}")
    while True:
        val = ask(question).strip().lower()
        if val in ("1", "yes", "y"):
            return 1
        if val in ("0", "no", "n"):
            return 0
        warn("Please enter  1 (Yes)  or  0 (No)")


def get_choice(question: str, options: list, descriptions: list) -> str:
    """Multiple choice — returns the option key."""
    print(f"  {DIM}Options:{RESET}")
    for i, desc in enumerate(descriptions):
        print(f"    {CYAN}{i + 1}{RESET}. {desc}")
    while True:
        val = ask(f"{question}  (1-{len(options)})").strip()
        if val.isdigit() and 1 <= int(val) <= len(options):
            return options[int(val) - 1]
        if val in options:
            return val
        warn(f"Enter a number between 1 and {len(options)}")


def get_numeric(question: str, hint: str) -> int:
    """Numeric input (age)."""
    info(f"Hint: {hint}")
    while True:
        val = ask(question).strip()
        if val.isdigit() and 1 <= int(val) <= 120:
            return int(val)
        warn("Enter a valid age (1-120)")


# ─────────────────────────────────────────────
# Adaptive Question Engine
# ─────────────────────────────────────────────
def run_question_engine() -> dict:
    answers = {}
    q_count = 0

    header("Patient Information Collection")
    info("Approximately 10-15 questions will be asked.")

    # ── Stage 1: Personal info ───────────────────────────────
    print(f"\n  {BOLD}[Step 1/5]  Personal Information{RESET}")

    answers["age"] = get_numeric(
        "What is your age?",
        "Enter as a number, e.g. 35",
    )
    q_count += 1

    answers["sex"] = get_choice(
        "What is your biological sex?",
        ["M", "F"],
        ["Male", "Female"],
    )
    q_count += 1

    answers["smoking_history"] = get_binary(
        "Do you smoke or have you smoked in the past?",
        "Yes=1  /  No=0",
    )
    q_count += 1

    # ── Stage 2: Chief complaint ─────────────────────────────
    print(f"\n  {BOLD}[Step 2/5]  Main Concern{RESET}")

    answers["chief_complaint"] = get_choice(
        "Which area best describes your main concern?",
        ["chest_cardiac", "skin", "general_other"],
        [
            "Chest pain / Heart problem / Breathing difficulty",
            "Skin problem / Lesion / Unusual mole",
            "General illness / Fever / Fatigue",
        ],
    )
    q_count += 1

    # ── Stage 3a: Chest / Cardiac ────────────────────────────
    if answers["chief_complaint"] == "chest_cardiac":
        print(f"\n  {BOLD}[Step 3/5]  Chest & Cardiac Symptoms{RESET}")

        answers["chest_pain"] = get_binary(
            "Do you feel chest pain or pressure?",
            "Yes=1  /  No=0",
        )
        q_count += 1

        if answers["chest_pain"] == 1:
            answers["chest_pain_radiates"] = get_binary(
                "Does the pain spread to your left arm, jaw, or shoulder?",
                "Yes=1 (radiates)  /  No=0",
            )
            q_count += 1

        answers["palpitation_type"] = get_choice(
            "Do you feel palpitations (heart fluttering)?",
            ["none", "irregular", "rapid", "slow"],
            [
                "No palpitations",
                "Yes — Irregular (sometimes fast, sometimes slow)",
                "Yes — Very rapid / Racing heart",
                "Yes — Very slow heartbeat",
            ],
        )
        q_count += 1

        answers["syncope"] = get_binary(
            "Have you ever fainted or nearly fainted?",
            "Yes=1  /  No=0",
        )
        q_count += 1

        answers["breath_pattern"] = get_choice(
            "Do you have shortness of breath?",
            ["none", "exertion", "rest", "lying_down"],
            [
                "No shortness of breath",
                "Yes — During exertion (walking / climbing stairs)",
                "Yes — Even at rest",
                "Yes — Worse when lying down",
            ],
        )
        q_count += 1

        answers["leg_swelling"] = get_binary(
            "Do you have swelling in your legs or ankles?",
            "Yes=1  /  No=0",
        )
        q_count += 1

        answers["cough_type"] = get_choice(
            "Do you have a cough?",
            ["none", "dry", "sputum", "blood"],
            [
                "No cough",
                "Yes — Dry cough",
                "Yes — Cough with phlegm / sputum",
                "Yes — Cough with blood  (serious!)",
            ],
        )
        q_count += 1

        answers["fever"] = get_binary(
            "Do you have a fever?",
            "Yes=1  /  No=0",
        )
        q_count += 1

    # ── Stage 3b: Skin ───────────────────────────────────────
    elif answers["chief_complaint"] == "skin":
        print(f"\n  {BOLD}[Step 3/5]  Skin Symptoms{RESET}")

        answers["skin_lesion"] = get_binary(
            "Do you have a skin lesion, mole, or unusual mark?",
            "Yes=1  /  No=0",
        )
        q_count += 1

        if answers["skin_lesion"] == 1:
            answers["lesion_changing"] = get_binary(
                "Is the lesion growing or changing in color / shape?",
                "Yes=1  /  No=0",
            )
            q_count += 1

            answers["lesion_features"] = get_choice(
                "Which features best describe the lesion?",
                ["none", "irregular_border", "multiple_colors", "bleeding_itching"],
                [
                    "Normal, no special features",
                    "Irregular / uneven border",
                    "Multiple colors present",
                    "Bleeding or itching",
                ],
            )
            q_count += 1

        answers["sun_exposure_history"] = get_binary(
            "Do you have a history of prolonged sun exposure?",
            "Yes=1  /  No=0",
        )
        q_count += 1

    # ── Stage 3c: General / Other ────────────────────────────
    else:
        print(f"\n  {BOLD}[Step 3/5]  General Symptoms{RESET}")

        answers["fever"] = get_binary(
            "Do you have a fever?",
            "Yes=1  /  No=0",
        )
        q_count += 1

        answers["cough_type"] = get_choice(
            "Do you have a cough?",
            ["none", "dry", "sputum", "blood"],
            [
                "No cough",
                "Yes — Dry cough",
                "Yes — Cough with phlegm / sputum",
                "Yes — Cough with blood  (serious!)",
            ],
        )
        q_count += 1

    # ── Stage 4: Systemic (always) ───────────────────────────
    print(f"\n  {BOLD}[Step 4/5]  General Physical Condition{RESET}")

    answers["systemic"] = get_choice(
        "Do you have any of the following?",
        ["none", "fatigue", "weight_loss", "both"],
        [
            "None of these",
            "Excessive fatigue / weakness",
            "Sudden unexplained weight loss",
            "Both fatigue and weight loss",
        ],
    )
    q_count += 1

    answers["other_systemic"] = get_binary(
        "Do you have nausea, dizziness, or excessive sweating?",
        "Yes=1 (any one of them)  /  No=0",
    )
    q_count += 1

    # ── Stage 5: Risk factors (always) ──────────────────────
    print(f"\n  {BOLD}[Step 5/5]  Medical History{RESET}")

    answers["risk_factors"] = get_choice(
        "Do you have any of these pre-existing conditions?",
        ["none", "hypertension", "diabetes", "both"],
        [
            "None",
            "Hypertension / High blood pressure",
            "Diabetes / High blood sugar",
            "Both hypertension and diabetes",
        ],
    )
    q_count += 1

    info(f"Total {q_count} questions answered.")
    return answers


# ─────────────────────────────────────────────
# Result display
# ─────────────────────────────────────────────
def show_result(result: dict):
    header("Prediction Result")

    confidence = result["confidence"]
    disease    = result["prediction"]
    conf_color = GREEN if confidence >= 70 else (YELLOW if confidence >= 50 else RED)

    print(f"\n  {BOLD}Disease    :{RESET}  {BOLD}{CYAN}{disease}{RESET}")
    print(f"  {BOLD}Confidence :{RESET}  {conf_color}{BOLD}{confidence}%{RESET}")

    if result["inconclusive"]:
        warn("Confidence below 50% — Please consult a doctor immediately!")

    print(f"\n  {BOLD}Top 3 Possible Conditions:{RESET}")
    for i, item in enumerate(result["top3"]):
        bar_len = int(item["probability"] / 5)
        bar     = "█" * bar_len + "░" * (20 - bar_len)
        color   = CYAN if i == 0 else DIM
        print(f"    {i + 1}. {color}{item['disease']:<38}{RESET}  {bar}  {item['probability']}%")

    print(f"\n  {DIM}Feature columns filled: "
          f"{result['filled_columns']} / {result['total_columns']}{RESET}")

    header("WARNING")
    print(f"  {YELLOW}This prediction is for preliminary reference only.{RESET}")
    print(f"  {YELLOW}Always consult a qualified doctor for final diagnosis.{RESET}")


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────
def main():
    print(f"\n{BOLD}{CYAN}{'═'*60}{RESET}")
    print(f"{BOLD}{CYAN}    HealthBridge — Symptom Disease Predictor{RESET}")
    print(f"{BOLD}{CYAN}    Ensemble: XGBoost + LightGBM + Random Forest{RESET}")
    print(f"{BOLD}{CYAN}{'═'*60}{RESET}")

    arts = load_artifacts(MODELS_DIR)

    while True:
        answers = run_question_engine()

        info("Running prediction...")
        result = predict_from_answers(answers, arts)

        show_result(result)

        print()
        again = ask("Run again for another patient? (y/n)").strip().lower()
        if again not in ("y", "yes"):
            print(f"\n{CYAN}{BOLD}  Thank you! Stay healthy. :){RESET}\n")
            break


if __name__ == "__main__":
    main()