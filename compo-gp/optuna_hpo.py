# optuna_hpo.py
# Run:  python optuna_hpo.py
# Requires: optuna, numpy, pandas, scikit-learn, matplotlib (optional), tensorflow, gpflow
# Uses your compogp modules (src/ layout). Fixes M=60, does 10-fold CV, TPE + pruning.

import os
os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")  # TF 2.16 + tf-keras compat

from pathlib import Path
import sys
import time
import json
from typing import Tuple

import numpy as np
import pandas as pd

import optuna
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner

import tensorflow as tf
import gpflow
from gpflow.kernels import RBF, Matern32, Matern52, RationalQuadratic, Linear
from gpflow.inducing_variables import InducingPoints, SharedIndependentInducingVariables
from gpflow.kernels import SharedIndependent

from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

# Make sure we can import your package if you use src/ layout
PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"
if SRC_DIR.exists():
    sys.path.insert(0, str(SRC_DIR))

# ---- Import your own pieces ----
from compogp.data import DataSpec, load_and_align, build_composition
from compogp.likelihoods.softmax_compositional import SoftmaxCompositional
from compogp.utils import aitchison_distance, renorm_simplex
from compogp.pipeline import apply_threshold_and_renorm  # if not available, reimplement here

# =========================
# FIXED CONFIG
# =========================
SENSORY_CSV     = "data/data_sens.csv"
INGREDIENTS_CSV = "data/data_recipe.csv"
ID_COL          = "idx"

FOLDS        = 10
M_FIXED      = 60
SEED         = 0
MAX_ITERS    = 5000 
        # per-trial overall cap; folds will use MAX_ITERS//2
N_TRIALS     = 120          # adjust as you like
STORAGE      = str(PROJECT_ROOT / "runs" / "optuna_m60.db")  # sqlite storage for resume
STUDY_NAME   = "m60_tpe_hpo"

OUT_DIR = PROJECT_ROOT / "runs" / "hpo_m60"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Use float64 (matches your pipeline defaults)
gpflow.config.set_default_float(np.float64)
tf.random.set_seed(SEED)
np.random.seed(SEED)

# =========================
# Helpers
# =========================
def make_kernel(name: str, P: int, D: int, ard: bool,
                lengthscale_init: float, variance_init: float, rq_alpha: float | None) -> SharedIndependent:
    if ard:
        ls = np.ones(P) * float(lengthscale_init)
    else:
        ls = float(lengthscale_init)

    if name == "RBF":
        base = RBF(lengthscales=ls, variance=variance_init)
    elif name == "Matern32":
        base = Matern32(lengthscales=ls, variance=variance_init)
    elif name == "Matern52":
        base = Matern52(lengthscales=ls, variance=variance_init)
    elif name == "RationalQuadratic":
        alpha = float(rq_alpha if rq_alpha is not None else 1.0)
        base = RationalQuadratic(lengthscales=ls, variance=variance_init, alpha=alpha)
    elif name == "RBF+Linear":
        base = RBF(lengthscales=ls, variance=variance_init) + Linear(variance=1.0)
    else:
        raise ValueError(f"Unknown kernel: {name}")

    return SharedIndependent(base, output_dim=D)

def init_inducing(X: np.ndarray, M: int, method: str, seed: int) -> np.ndarray:
    if method == "subset":
        rng = np.random.RandomState(seed)
        idx = rng.permutation(X.shape[0])[:M]
        return X[idx].copy()
    elif method == "kmeans++":
        from sklearn.cluster import KMeans
        km = KMeans(n_clusters=M, init="k-means++", n_init="auto", random_state=seed)
        km.fit(X)
        return km.cluster_centers_.astype(np.float64)
    else:
        raise ValueError(method)

def build_model_for_fold(Xtr: np.ndarray, Ytr: np.ndarray, params: dict):
    N, P = Xtr.shape
    D    = Ytr.shape[1]
    Z    = init_inducing(Xtr, min(M_FIXED, N), params["z_init"], SEED)

    kern = make_kernel(
        name=params["kernel"], P=P, D=D, ard=params["ard"],
        lengthscale_init=params["lengthscale_init"],
        variance_init=params["variance_init"],
        rq_alpha=params.get("rq_alpha")
    )

    lik = SoftmaxCompositional(D=D, tau=params["tau"], eps=params["eps_lik"])

    inducing = SharedIndependentInducingVariables(InducingPoints(Z.astype(np.float64)))
    model = gpflow.models.SVGP(
        kernel=kern,
        likelihood=lik,
        inducing_variable=inducing,
        num_latent_gps=D,
        q_diag=True,
        whiten=True,
    )
    # init variational params
    model.q_mu.assign(tf.zeros_like(model.q_mu))
    model.q_sqrt.assign(1e-3 * tf.ones_like(model.q_sqrt))

    # optionally freeze Z
    gpflow.set_trainable(model.inducing_variable.inducing_variable.Z, params["z_trainable"])

    return model

def train_model(model, Xtr: np.ndarray, Ytr: np.ndarray, params: dict) -> float:
    # Optimizer
    if params["optimizer"] == "adam":
        opt = tf.optimizers.Adam(learning_rate=params["lr"], clipnorm=params["clipnorm"] if params["clipnorm"] else None)
    else:
        opt = tf.optimizers.RMSprop(learning_rate=params["lr"], clipnorm=params["clipnorm"] if params["clipnorm"] else None)

    # Natural gradient on (q_mu, q_sqrt)
    # use_natgrad = params["use_natgrad"]
    # if use_natgrad:
        # natgrad = gpflow.optimizers.NaturalGradient(gamma=params["natgrad_gamma"])
        variational_params = [(model.q_mu, model.q_sqrt)]

    Xtf = tf.convert_to_tensor(Xtr, dtype=gpflow.default_float())
    Ytf = tf.convert_to_tensor(Ytr, dtype=gpflow.default_float())

    @tf.function(autograph=False)
    def adam_step():
        with tf.GradientTape() as tape:
            loss = -model.elbo((Xtf, Ytf))
        grads = tape.gradient(loss, model.trainable_variables)
        opt.apply_gradients(zip(grads, model.trainable_variables))
        return -loss  # ELBO

    best_elbo = -np.inf
    wait = 0
    max_iters = params["max_iters_fold"]
    patience  = params["patience"]
    min_delta = params["min_delta"]

    for it in range(max_iters):
        # if use_natgrad:
            # one natgrad] step on variational parameters
            # natgrad.minimi]ze(lambda: -model.elbo((Xtf, Ytf)), var_list=variational_params)
        elbo = float(adam_step().numpy())

        if elbo > best_elbo + min_delta:
            best_elbo = elbo
            wait = 0
        else:
            wait += 1
        if wait >= patience:
            break
    return best_elbo

def predict_fold(model, Xte: np.ndarray, params: dict) -> Tuple[np.ndarray, np.ndarray]:
    Xtf = tf.convert_to_tensor(Xte, dtype=gpflow.default_float())
    Fmu, Fvar = model.predict_f(Xtf, full_cov=False, full_output_cov=False)
    Pmean, Pvar = model.likelihood.predictive_mean_from_moments(Fmu, Fvar, mc=params["mc_pred"])
    Pmean = renorm_simplex(Pmean)
    return Pmean, Pvar

# =========================
# Load data once
# =========================
spec_base = DataSpec(
    sensory_csv=SENSORY_CSV,
    ingredients_csv=INGREDIENTS_CSV,
    id_col=ID_COL,
    add_other=True,
    min_presence=1,
    scale_X=True
)
Xdf_all, Ydf_all = load_and_align(spec_base)
X_cols = [c for c in Xdf_all.columns if c != ID_COL]
X_raw_full = Xdf_all[X_cols].to_numpy(dtype=float)

def make_targets(min_presence: int, add_other: bool) -> Tuple[np.ndarray, list[str]]:
    spec = DataSpec(
        sensory_csv=SENSORY_CSV,
        ingredients_csv=INGREDIENTS_CSV,
        id_col=ID_COL,
        add_other=add_other,
        min_presence=min_presence,
        scale_X=True
    )
    # reuse already-loaded Ydf_all
    Y_mat, ing_names = build_composition(Ydf_all, spec)
    return Y_mat, ing_names

# =========================
# Optuna objective
# =========================
def objective(trial: optuna.trial.Trial) -> float:
    # ---- sample hyperparameters ----
    hp = {}
    hp["min_presence"]   = trial.suggest_categorical("min_presence", [1, 2, 3, 5])
    hp["add_other"]      = trial.suggest_categorical("add_other", [True, False])

    hp["tau"]            = trial.suggest_float("tau", 5.0, 1500.0, log=True)
    hp["eps_lik"]        = trial.suggest_categorical("eps_lik", [1e-8, 1e-6])

    hp["kernel"]         = trial.suggest_categorical("kernel", ["RBF", "Matern32", "Matern52", "RationalQuadratic", "RBF+Linear"])
    hp["ard"]            = trial.suggest_categorical("ard", [True, False])
    hp["lengthscale_init"]= trial.suggest_float("lengthscale_init", 0.1, 10.0, log=True)
    hp["variance_init"]  = trial.suggest_float("variance_init", 0.1, 10.0, log=True)
    if hp["kernel"] == "RationalQuadratic":
        hp["rq_alpha"]  = trial.suggest_float("rq_alpha", 0.2, 5.0, log=True)
    else:
        hp["rq_alpha"]  = None

    hp["z_init"]         = trial.suggest_categorical("z_init", ["subset", "kmeans++"])
    hp["z_trainable"]    = trial.suggest_categorical("z_trainable", [True, False])

    # hp["use_natgrad"]    = trial.suggest_categorical("use_natgrad", [True, False])
    # if hp["use_natgrad"]:
    #     hp["natgrad_gamma"] = trial.suggest_categorical("natgrad_gamma", [0.05, 0.1, 0.5])

    hp["optimizer"]      = trial.suggest_categorical("optimizer", ["adam", "rmsprop"])
    hp["lr"]             = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
    hp["clipnorm"]       = trial.suggest_categorical("clipnorm", [None, 1.0, 5.0])
    hp["patience"]       = trial.suggest_int("patience", 50, 300, step=25)
    hp["min_delta"]      = trial.suggest_float("min_delta", 1e-6, 1e-4, log=True)

    hp["threshold_mode"] = trial.suggest_categorical("threshold_mode", ["global", "per_ingredient"])
    if hp["threshold_mode"] == "global":
        hp["threshold_global"] = trial.suggest_float("threshold_global", 1e-5, 1e-2, log=True)
        hp["per_ing_scale"]    = None
    else:
        hp["per_ing_scale"]    = trial.suggest_float("per_ing_scale", 0.5, 2.0, log=True)
        hp["threshold_global"] = None

    # derived
    hp["mc_pred"]       = trial.suggest_categorical("mc_pred", [64, 128])
    hp["max_iters_fold"]= max(1, int(MAX_ITERS // 2))  # train fewer iters per fold

    # ---- make targets (depends on min_presence/add_other) ----
    Y_full, ing_names = make_targets(hp["min_presence"], hp["add_other"])
    n_rows = X_raw_full.shape[0]

    # ---- CV loop with pruning ----
    kf = KFold(n_splits=FOLDS, shuffle=True, random_state=SEED)
    fold_ad = []
    fold_l2 = []

    for fold_idx, (tr_idx, te_idx) in enumerate(kf.split(X_raw_full), 1):
        Xtr_raw = X_raw_full[tr_idx]
        Xte_raw = X_raw_full[te_idx]
        Ytr = Y_full[tr_idx]
        Yte = Y_full[te_idx]

        # Impute medians (train only)
        meds_tr = np.nanmedian(Xtr_raw, axis=0)
        meds_tr = np.where(np.isfinite(meds_tr), meds_tr, 0.0)
        Xtr_imp = np.where(np.isnan(Xtr_raw), meds_tr[None, :], Xtr_raw)
        Xte_imp = np.where(np.isnan(Xte_raw), meds_tr[None, :], Xte_raw)

        # Scale (train only), guard zero-variance
        scaler_tr = StandardScaler().fit(Xtr_imp)
        if hasattr(scaler_tr, "scale_"):
            zero_scale = (scaler_tr.scale_ == 0)
            if np.any(zero_scale):
                scaler_tr.scale_[zero_scale] = 1.0
                if hasattr(scaler_tr, "var_"):
                    scaler_tr.var_[zero_scale] = 1.0
        Xtr = scaler_tr.transform(Xtr_imp)
        Xte = scaler_tr.transform(Xte_imp)

        # Build & train
        model = build_model_for_fold(Xtr, Ytr, hp)
        train_model(model, Xtr, Ytr, hp)

        # Predict
        Yhat, _ = predict_fold(model, Xte, hp)

        # Thresholding post-process
        if hp["threshold_mode"] == "global" and hp["threshold_global"] is not None:
            Yhat = apply_threshold_and_renorm(Yhat, hp["threshold_global"], mode="global")
        elif hp["threshold_mode"] == "per_ingredient" and hp["per_ing_scale"] is not None:
            # observed per-ingredient >0 minimum on TRAIN
            nz = np.where(Ytr > 0, Ytr, np.nan)
            per_ing_min = np.nanmin(nz, axis=0)
            per_ing_min = np.where(np.isfinite(per_ing_min), per_ing_min, 0.0)
            thr = hp["per_ing_scale"] * per_ing_min
            Yhat = apply_threshold_and_renorm(Yhat, thr, mode="per_ingredient")

        Yhat = renorm_simplex(Yhat)

        # Metrics
        ad = float(np.mean(aitchison_distance(Yte, Yhat)))
        l2 = float(np.mean(np.sqrt(np.sum((Yte - Yhat) ** 2, axis=1))))
        fold_ad.append(ad)
        fold_l2.append(l2)

        # Report intermediate value for pruning (lower is better)
        trial.report(ad, step=fold_idx)
        if trial.should_prune():
            raise optuna.TrialPruned()

    # Aggregate across folds
    ad_mean = float(np.mean(fold_ad))
    l2_mean = float(np.mean(fold_l2))

    # Store extras
    trial.set_user_attr("ad_per_fold", fold_ad)
    trial.set_user_attr("l2_per_fold", fold_l2)
    trial.set_user_attr("l2_mean", l2_mean)

    return ad_mean

def main():
    storage_uri = f"sqlite:///{STORAGE}"
    try:
        optuna.delete_study(study_name=STUDY_NAME, storage=storage_uri)
        print(f"[HPO] Deleted existing study '{STUDY_NAME}' from {storage_uri}")
    except KeyError:
        pass  # study didn't exist, that's fine
    study = optuna.create_study(
        study_name=STUDY_NAME,
        direction="minimize",
        storage=storage_uri,
        load_if_exists=True,
        sampler=TPESampler(multivariate=True, consider_endpoints=True, n_startup_trials=20, seed=SEED),
        pruner=MedianPruner(n_startup_trials=15, n_warmup_steps=2),
    )
    print(f"[HPO] Study: {study.study_name}  Storage: {storage_uri}")
    t0 = time.perf_counter()
    study.optimize(objective, n_trials=N_TRIALS, gc_after_trial=True)
    secs = time.perf_counter() - t0

    best = study.best_trial
    print("\n=== Best trial ===")
    print("value (Aitchison mean):", best.value)
    print("params:", json.dumps(best.params, indent=2))
    print("l2_mean:", best.user_attrs.get("l2_mean"))

    # Save artifacts
    df = study.trials_dataframe(attrs=("number", "value", "state"))
    df.to_csv(OUT_DIR / "trials.csv", index=False)
    (OUT_DIR / "best_params.json").write_text(json.dumps(best.params, indent=2))
    (OUT_DIR / "best_attrs.json").write_text(json.dumps(best.user_attrs, indent=2))
    meta = {
        "folds": FOLDS,
        "M_fixed": M_FIXED,
        "seed": SEED,
        "n_trials": N_TRIALS,
        "total_seconds": secs,
        "paths": {"sensory": SENSORY_CSV, "ingredients": INGREDIENTS_CSV, "id_col": ID_COL},
    }
    (OUT_DIR / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"\n[HPO] Done in {secs/60:.1f} min. Results in {OUT_DIR}")

if __name__ == "__main__":
    main()
