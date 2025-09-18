from __future__ import annotations
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Tuple, List
from .utils import to_simplex

@dataclass
class DataSpec:
    sensory_csv: str
    ingredients_csv: str
    id_col: str = "idx"
    add_other: bool = True
    min_presence: int = 1     # keep an ingredient if present >= this many samples
    scale_X: bool = True

def load_and_align(spec: DataSpec) -> Tuple[pd.DataFrame, pd.DataFrame]:
    Xdf = pd.read_csv(spec.sensory_csv)
    Ydf = pd.read_csv(spec.ingredients_csv)
    df = Xdf.merge(Ydf, on=spec.id_col, how="inner")
    sensory_cols = [c for c in Xdf.columns if c != spec.id_col]
    ingredient_cols = [c for c in Ydf.columns if c != spec.id_col]
    X = df[[spec.id_col] + sensory_cols].copy()
    Y = df[[spec.id_col] + ingredient_cols].copy()
    Y = Y.fillna(0.0)
    return X, Y

def build_composition(Y: pd.DataFrame, spec: DataSpec) -> Tuple[np.ndarray, List[str]]:
    id_col = spec.id_col
    cols = [c for c in Y.columns if c != id_col]
    Yvals = Y[cols].to_numpy(dtype=float)

    # If given in 0..100, normalize to 0..1 by row sum
    row_sums = Yvals.sum(axis=1, keepdims=True)
    row_sums = np.where(row_sums == 0.0, 1.0, row_sums)
    Yvals = Yvals / row_sums

    # Presence filter
    present_counts = (Yvals > 0).sum(axis=0)
    keep_mask = present_counts >= spec.min_presence
    kept_cols_raw = [c for c, m in zip(cols, keep_mask) if m]
    Y_keep = Yvals[:, keep_mask]

    if spec.add_other:
        leftover = 1.0 - Y_keep.sum(axis=1)
        other = np.clip(leftover, 0.0, 1.0)
        Y_aug = np.concatenate([Y_keep, other[:, None]], axis=1)
        kept_cols = kept_cols_raw + ["Other"]
    else:
        # Renormalize kept subset
        s = Y_keep.sum(axis=1, keepdims=True)
        s = np.where(s == 0.0, 1.0, s)
        Y_aug = Y_keep / s
        kept_cols = kept_cols_raw

    Y_aug = to_simplex(Y_aug)
    return Y_aug, kept_cols