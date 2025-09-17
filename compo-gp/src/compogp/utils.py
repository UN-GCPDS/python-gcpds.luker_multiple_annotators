from __future__ import annotations
import numpy as np
import tensorflow as tf

def to_simplex(rows: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    rows = np.clip(rows, eps, None)
    rows = rows / rows.sum(axis=1, keepdims=True)
    return rows

def clr(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    x = np.clip(x, eps, None)
    g = np.exp(np.mean(np.log(x), axis=1, keepdims=True))
    return np.log(x / g)

def aitchison_distance(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    C1 = clr(y_true, eps)
    C2 = clr(y_pred, eps)
    return np.sqrt(np.sum((C1 - C2) ** 2, axis=1))

def softmax_tf(z, axis=-1):
    zmax = tf.reduce_max(z, axis=axis, keepdims=True)
    ez = tf.exp(z - zmax)
    return ez / tf.reduce_sum(ez, axis=axis, keepdims=True)

def renorm_simplex(A: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """
    Force rows to be valid simplex:
      - replace NaN/Inf with 0
      - clamp negatives to 0
      - renormalize row-wise to sum exactly 1
    """
    A = np.nan_to_num(A, nan=0.0, posinf=0.0, neginf=0.0)
    A = np.maximum(A, 0.0)
    s = A.sum(axis=1, keepdims=True)
    s[s <= eps] = 1.0  # avoid div-by-zero: keep all-zero rows as zeros
    return A / s

def apply_threshold_and_renorm(P: np.ndarray,
                               thresholds: np.ndarray,
                               mode: str = "per_ingredient",
                               eps: float = 1e-12) -> np.ndarray:
    """
    P: (N, D) predicted fractions on simplex (row-sum ~1)
    thresholds: (D,) per-ingredient thresholds (>0) OR scalar (global)
    mode: "per_ingredient" (default) or "global"
    """
    P = np.nan_to_num(P, nan=0.0, posinf=0.0, neginf=0.0)
    P = np.maximum(P, 0.0)

    if mode == "per_ingredient":
        thr = thresholds.reshape(1, -1)  # (1, D)
    elif mode == "global":
        thr = float(thresholds)          # scalar
    else:
        raise ValueError("mode must be 'per_ingredient' or 'global'")

    # Zero out entries below threshold
    P_thr = P.copy()
    P_thr[P_thr < thr] = 0.0

    # Renormalize row-wise; if a row becomes all-zero, fallback to original P
    row_sums = P_thr.sum(axis=1, keepdims=True)
    fallback_mask = (row_sums <= eps).ravel()
    if np.any(fallback_mask):
        # keep original for those rows (or you could distribute uniformly)
        P_thr[fallback_mask] = P[fallback_mask]
        row_sums[fallback_mask] = P[fallback_mask].sum(axis=1, keepdims=True)

    P_thr = P_thr / np.maximum(row_sums, 1.0)  # safe divide
    return P_thr