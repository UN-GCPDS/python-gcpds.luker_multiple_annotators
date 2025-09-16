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