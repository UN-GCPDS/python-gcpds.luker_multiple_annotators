# train_fixed.py
# Run:  python train_fixed.py
# Adjust the paths and hyperparams in the CONFIG section below.

import os
os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")  # TF 2.16 + tf-keras compatibility

from pathlib import Path
import json
import sys
import joblib
import tensorflow as tf

# --- Make sure we can import your package if using src/ layout ---
# Project/
#   src/compogp/...
PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"
if SRC_DIR.exists():
    sys.path.insert(0, str(SRC_DIR))

# ---- import your pipeline (as defined in the structured layout) ----
from compogp.pipeline import run_pipeline  # uses your existing functions

# =========================
# CONFIG (edit these)
# =========================
SENSORY_CSV      = "data/data_sens.csv"        # path to your sensory CSV
INGREDIENTS_CSV  = "data/data_recipe.csv"    # path to your ingredients CSV
OUT_DIR          = "runs/exp1"               # where to save artifacts

# Data/options
ID_COL           = "idx"               # or "idx" if that’s your ID column
MIN_PRESENCE     = 1
ADD_OTHER        = True                      # add "Other" bucket

# Model & training
NUM_INDUCING     = 20
TAU              = 100.0
MAX_ITERS        = 10
SEED             = 0

# Cross-validation (choose exactly one mode)
USE_LOOCV        = False                      # Leave-One-Out CV
KFOLDS           = 5                      # e.g., 5 for 5-fold; None to disable

# =========================
# Helpers
# =========================
def save_artifacts_bundle(artifacts: dict, out_dir: Path) -> None:
    """
    Saves:
      - TensorFlow checkpoint for the GPflow model
      - A joblib bundle with non-model pieces (scaler, medians, config, etc.)
      - A small JSON summary
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) Save TF checkpoint for the GPflow model
    model = artifacts.get("model")
    ckpt_path = out_dir / "ckpt"
    if model is not None:
        ckpt = tf.train.Checkpoint(model=model)
        ckpt_save_path = ckpt.save(str(ckpt_path))
        print(f"[save] TensorFlow checkpoint saved at: {ckpt_save_path}")
    else:
        print("[save] WARNING: artifacts['model'] is None, skipping TF checkpoint.")

    # 2) Save the rest with joblib (exclude the live model object)
    artifacts_copy = dict(artifacts)
    artifacts_copy["model"] = None
    joblib.dump(artifacts_copy, out_dir / "state.joblib")
    print(f"[save] Non-model state saved to: {out_dir / 'state.joblib'}")

    # 3) Small JSON summary
    summary = {
        "id_col": artifacts.get("id_col"),
        "ingredient_count": len(artifacts.get("ingredient_names", [])),
        "ingredient_names_sample": artifacts.get("ingredient_names", [])[:10],
        "config": artifacts.get("config", {}),
        "has_cv": "cv" in artifacts,
        "cv": artifacts.get("cv", None),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"[save] Summary written to: {out_dir / 'summary.json'}")


def main():
    print("=== Training config ===")
    print(json.dumps({
        "sensory_csv": SENSORY_CSV,
        "ingredients_csv": INGREDIENTS_CSV,
        "out_dir": OUT_DIR,
        "id_col": ID_COL,
        "min_presence": MIN_PRESENCE,
        "add_other": ADD_OTHER,
        "num_inducing": NUM_INDUCING,
        "tau": TAU,
        "max_iters": MAX_ITERS,
        "seed": SEED,
        "cv": "LOOCV" if USE_LOOCV else (f"{KFOLDS}-fold" if KFOLDS else "none"),
    }, indent=2))

    # Pick CV mode
    if USE_LOOCV:
        do_loocv = True
    elif KFOLDS and KFOLDS >= 2:
        do_loocv = int(KFOLDS)
    else:
        do_loocv = False

    artifacts = run_pipeline(
        sensory_csv=SENSORY_CSV,
        ingredients_csv=INGREDIENTS_CSV,
        id_col=ID_COL,
        min_presence=MIN_PRESENCE,
        add_other=ADD_OTHER,
        num_inducing=NUM_INDUCING,
        tau=TAU,
        max_iters=MAX_ITERS,
        do_loocv=do_loocv,
        seed=SEED,
    )

    out_dir = PROJECT_ROOT / OUT_DIR
    save_artifacts_bundle(artifacts, out_dir)

    if "cv" in artifacts:
        print("\n[cv] Cross-validation summary")
        for k, v in artifacts["cv"].items():
            print(f"  {k}: {v}")

    print("\n[done] Training complete.")


if __name__ == "__main__":
    main()
