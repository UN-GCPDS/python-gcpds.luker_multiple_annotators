# sweep_inducing_points.py
# Run:  python sweep_inducing_points.py
# Requires your structured layout (src/compogp/...) and the pipeline functions.

import os
os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")  # TF 2.16 + tf-keras

from pathlib import Path
import sys
import time
import json
import pandas as pd
import matplotlib.pyplot as plt

# Make sure we can import compogp.* if you use the src/ layout
PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"
if SRC_DIR.exists():
    sys.path.insert(0, str(SRC_DIR))

from compogp.pipeline import run_pipeline  # your existing pipeline

# =========================
# CONFIG (edit as needed)
# =========================
SENSORY_CSV      = "data/data_sens.csv"        # path to your sensory CSV
INGREDIENTS_CSV  = "data/data_recipe.csv"    # path to your ingredients CSV
ID_COL           = "idx"   # or "idx"
MIN_PRESENCE     = 1
ADD_OTHER        = True

# "Standard" training hyperparams you’ve been using
TAU              = 100.0
MAX_ITERS        = 5000           # keep as your standard; lower if you want faster sweeps
SEED             = 0

# CV setup
FOLDS            = 10              # as requested
SHUFFLE_SPLITS   = True           # controlled inside pipeline via seed

# Output
OUT_DIR          = PROJECT_ROOT / "runs" / "m_sweep"
OUT_DIR.mkdir(parents=True, exist_ok=True)

def build_M_grid(n_rows: int) -> list[int]:
    if n_rows <= 10:
        return [n_rows]
    grid = list(range(10, n_rows + 1, 10))
    if grid[-1] != n_rows:
        grid.append(n_rows)
    return grid

def main():
    # Determine N (rows) from sensory CSV
    if not Path(SENSORY_CSV).exists():
        raise FileNotFoundError(f"{SENSORY_CSV} not found")
    if not Path(INGREDIENTS_CSV).exists():
        raise FileNotFoundError(f"{INGREDIENTS_CSV} not found")

    n_rows = pd.read_csv(SENSORY_CSV).shape[0]
    M_grid = build_M_grid(n_rows)

    print("=== Inducing points sweep ===")
    print(json.dumps({
        "rows": n_rows,
        "M_grid": M_grid,
        "folds": FOLDS,
        "tau": TAU,
        "max_iters": MAX_ITERS,
        "id_col": ID_COL,
        "min_presence": MIN_PRESENCE,
        "add_other": ADD_OTHER
    }, indent=2))

    records = []
    for M in M_grid:
        print(f"\n[M={M}] 5-fold CV running...")
        t0 = time.perf_counter()
        arts = run_pipeline(
            sensory_csv=SENSORY_CSV,
            ingredients_csv=INGREDIENTS_CSV,
            id_col=ID_COL,
            min_presence=MIN_PRESENCE,
            add_other=ADD_OTHER,
            num_inducing=M,
            tau=TAU,
            max_iters=MAX_ITERS,
            do_loocv=FOLDS,     # ← 5 folds
            seed=SEED,
        )
        secs = time.perf_counter() - t0
        if "cv" not in arts:
            raise RuntimeError("Pipeline did not return CV metrics. Ensure do_loocv is set correctly.")
        cv = arts["cv"]
        rec = {
            "M": M,
            "aitchison_mean": cv["aitchison_mean"],
            "aitchison_std":  cv["aitchison_std"],
            "l2_mean":        cv["l2_mean"],
            "l2_std":         cv["l2_std"],
            "time_seconds":   secs,
            "n_samples":      cv["n_samples"],
            "folds":          cv["folds"],
        }
        records.append(rec)
        print(f"  -> Aitchison {rec['aitchison_mean']:.6f} (±{rec['aitchison_std']:.6f}), "
              f"L2 {rec['l2_mean']:.6f} (±{rec['l2_std']:.6f}), time {secs/60:.1f} min")

    df = pd.DataFrame.from_records(records).sort_values("M")
    csv_path = OUT_DIR / "m_sweep_results.csv"
    df.to_csv(csv_path, index=False)
    print(f"\n[save] Results -> {csv_path}")

    # --------- Plots (two separate figures) ---------
    # Aitchison vs M
    plt.figure()
    plt.plot(df["M"], df["aitchison_mean"], marker="o")
    plt.xlabel("Number of inducing points (M)")
    plt.ylabel("Mean Aitchison distance (5-fold)")
    plt.title("Aitchison distance vs M")
    a_path = OUT_DIR / "aitchison_vs_M.png"
    plt.savefig(a_path, dpi=160, bbox_inches="tight")
    print(f"[save] {a_path}")

    # L2 vs M
    plt.figure()
    plt.plot(df["M"], df["l2_mean"], marker="o")
    plt.xlabel("Number of inducing points (M)")
    plt.ylabel("Mean L2 distance (5-fold)")
    plt.title("L2 distance vs M")
    l2_path = OUT_DIR / "l2_vs_M.png"
    plt.savefig(l2_path, dpi=160, bbox_inches="tight")
    print(f"[save] {l2_path}")

    print("\n[done] Sweep finished.")

if __name__ == "__main__":
    main()
