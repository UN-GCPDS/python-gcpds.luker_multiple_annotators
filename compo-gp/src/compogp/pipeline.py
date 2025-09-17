from __future__ import annotations
from typing import Dict
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold

from .data import DataSpec, load_and_align, build_composition
from .models.svgp_softmax import ModelConfig, build_svgp_softmax, train_svgp, predict_composition
from .utils import aitchison_distance, renorm_simplex, apply_threshold_and_renorm

def run_pipeline(
    sensory_csv: str,
    ingredients_csv: str,
    id_col: str = "sample_id",
    min_presence: int = 1,
    add_other: bool = True,
    num_inducing: int = 20,
    tau: float = 100.0,
    max_iters: int = 5000,
    do_loocv: bool = True,
    seed: int = 0
) -> Dict:
    spec = DataSpec(
        sensory_csv=sensory_csv,
        ingredients_csv=ingredients_csv,
        id_col=id_col,
        add_other=add_other,
        min_presence=min_presence,
        scale_X=True
    )

    # Load
    Xdf, Ydf = load_and_align(spec)
    meds = Xdf.iloc[:,1:].median(axis=0)
    Xdf = Xdf.fillna(meds)
    Ydf = Ydf.fillna(0)
    X_cols = [c for c in Xdf.columns if c != id_col]
    X_raw = Xdf[X_cols].to_numpy(dtype=float)
    Y_mat, ing_names = build_composition(Ydf, spec)  # (N, D)

    # Per-ingredient minimum strictly > 0 observed in training data
    nz = np.where(Y_mat > 0, Y_mat, np.nan)
    per_ing_min = np.nanmin(nz, axis=0)         # shape (D,)
    per_ing_min = np.where(np.isfinite(per_ing_min), per_ing_min, 0.0)  # if an ingredient is always 0

    # # Optional: global minimum > 0 (if you ever want a single scalar threshold)
    # valid = nz[nz > 1e-5]

    # if valid.size > 0:
    #     global_min = float(np.min(valid))
    # else:
    #     global_min = 0.0  # fallback if no values above threshold
    global_min = 0.0005
    # Standardize X
    xscaler = StandardScaler().fit(X_raw)
    X = xscaler.transform(X_raw)

    # Build + train model
    cfg = ModelConfig(
        num_inducing=num_inducing, tau=tau, max_iters=max_iters, seed=seed, lr=0.01, ard=True
    )
    model, lik, Z = build_svgp_softmax(X, Y_mat, cfg)

    train_svgp(model, X, Y_mat, cfg)

    artifacts = {
        "xscaler": xscaler,
        "x_imputer": meds,     
        "model": model,
        "ingredient_names": ing_names,
        "X_cols": X_cols,
        "id_col": id_col,
        "config": cfg.__dict__,
        "threshold_global": global_min
    }

    n = X.shape[0]
    run_cv = bool(do_loocv)
    n_splits = None
    scheme = None

    if do_loocv is True:
        n_splits = n
        scheme = "LOOCV"
    elif isinstance(do_loocv, int) and do_loocv >= 2:
        n_splits = min(do_loocv, n)  # cap at N
        scheme = f"{n_splits}-Fold CV"

    if run_cv and n_splits is not None and n_splits >= 2:
        # We will CV on *raw* X (before full-data scaling) to avoid leakage.
        # Recreate raw X and fixed Y from the same columns used to train the full model.
        X_raw_full = Xdf[X_cols].to_numpy(dtype=float)  # raw (post-imputation only for final model)
        Y_full = Y_mat                                  # (N, D) fixed composition columns

        kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
        ad_all, l2_all = [], []

        for fold_idx, (tr_idx, te_idx) in enumerate(kf.split(X_raw_full), 1):
            # --- Train split: imputer + scaler fit only on TRAIN ---
            Xtr_raw = X_raw_full[tr_idx]
            Xte_raw = X_raw_full[te_idx]

            # Impute with training medians (column-wise)
            meds_tr = np.nanmedian(Xtr_raw, axis=0)
            # If a column is entirely NaN, replace nanmedian with 0
            meds_tr = np.where(np.isfinite(meds_tr), meds_tr, 0.0)
            Xtr_imp = np.where(np.isnan(Xtr_raw), meds_tr[None, :], Xtr_raw)
            Xte_imp = np.where(np.isnan(Xte_raw), meds_tr[None, :], Xte_raw)

            # Scale with training scaler
            scaler_tr = StandardScaler().fit(Xtr_imp)
            # Guard zero-variance columns
            if hasattr(scaler_tr, "scale_"):
                zero_scale = (scaler_tr.scale_ == 0)
                if np.any(zero_scale):
                    scaler_tr.scale_[zero_scale] = 1.0
                    if hasattr(scaler_tr, "var_"):
                        scaler_tr.var_[zero_scale] = 1.0

            Xtr = scaler_tr.transform(Xtr_imp)
            Xte = scaler_tr.transform(Xte_imp)

            # Targets
            Ytr, Yte = Y_full[tr_idx], Y_full[te_idx]

            # Build & train fold model
            cfg_cv = ModelConfig(
                num_inducing=min(cfg.num_inducing, Xtr.shape[0]),
                tau=cfg.tau,
                max_iters=max(1, int(max_iters / 2)),   # faster per fold
                seed=seed,
                lr=cfg.lr,
                ard=True
            )
            m_cv, _, _ = build_svgp_softmax(Xtr, Ytr, cfg_cv)
            train_svgp(m_cv, Xtr, Ytr, cfg_cv)

            # Predict on test split
            Yhat, _ = predict_composition(m_cv, Xte, mc_samples=cfg.mc_pred)

            # Metrics
            ad = aitchison_distance(Yte, Yhat)
            l2 = np.sqrt(np.sum((Yte - Yhat) ** 2, axis=1))
            ad_all.extend(list(ad))
            l2_all.extend(list(l2))

        artifacts["cv"] = {
            "scheme": scheme,
            "folds": int(n_splits),
            "aitchison_mean": float(np.mean(ad_all)),
            "aitchison_std":  float(np.std(ad_all)),
            "l2_mean":        float(np.mean(l2_all)),
            "l2_std":         float(np.std(l2_all)),
            "n_samples":      int(n),
        }

    return artifacts

def predict_from_artifacts(artifacts: Dict, X_new: pd.DataFrame) -> pd.DataFrame:
    X_cols = artifacts["X_cols"]
    xscaler = artifacts["xscaler"]
    x_imputer = artifacts["x_imputer"]      # <-- use the stored medians
    model = artifacts["model"]
    ing_names = artifacts["ingredient_names"]

    per_ing_min = artifacts.get("thresholds_per_ingredient", None)
    global_min  = artifacts.get("threshold_global", None)
    # print("global min", global_min)
    # Make a copy, ensure order, impute
    Xarr_df = X_new[X_cols].astype(float).copy()
    Xarr_df = Xarr_df.fillna(x_imputer)

    # (Optional) if any constant columns at train time, skip second check here
    Xs = xscaler.transform(Xarr_df.values)

    # Simple finite check (see Patch C)
    assert np.isfinite(Xs).all(), "Xs has non-finite values after impute+scale."

    Yhat, vars = predict_composition(model, Xs, mc_samples=artifacts["config"].get("mc_pred", 64))
    Yhat_pct = 100 * Yhat    # ---- Apply thresholding ----
    if global_min is not None:
        # Per-ingredient thresholding (recommended)
        # Yhat = apply_threshold_and_renorm(Yhat, per_ing_min, mode="per_ingredient")
        # If you prefer a single global threshold, use:
        Yhat = apply_threshold_and_renorm(Yhat, global_min, mode="global")

    # Final sanity
    Yhat = renorm_simplex(Yhat)
    Yhat_pct = 100.0 * Yhat
    return pd.DataFrame(Yhat_pct, columns=ing_names, index=X_new.index), pd.DataFrame(vars, columns=ing_names, index=X_new.index)

