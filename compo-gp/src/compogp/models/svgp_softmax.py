from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import tensorflow as tf
from gpflow.models import SVGP
from gpflow.kernels import RBF, SharedIndependent
from gpflow.inducing_variables import InducingPoints, SharedIndependentInducingVariables
from ..likelihoods.softmax_compositional import SoftmaxCompositional
from ..utils import to_simplex

@dataclass
class ModelConfig:
    num_inducing: int = 20
    ard: bool = True
    tau: float = 100.0
    mc_pred: int = 64
    max_iters: int = 5000
    lr: float = 0.01
    seed: int = 0

def build_svgp_softmax(X: np.ndarray, Y: np.ndarray, cfg: ModelConfig):
    tf.random.set_seed(cfg.seed)
    N, P = X.shape
    D = Y.shape[1]

    M = min(cfg.num_inducing, N)
    perm = np.random.RandomState(cfg.seed).permutation(N)[:M]
    Z = X[perm, :].copy()

    base_kern = RBF(lengthscales=np.ones(P), variance=1.0)  # ARD via vector lengthscales
    kern = SharedIndependent(base_kern, output_dim=D)

    lik = SoftmaxCompositional(D=D, tau=cfg.tau)  # <-- pass D here

    inducing = SharedIndependentInducingVariables(InducingPoints(Z.astype(np.float64)))
    model = SVGP(
        kernel=kern,
        likelihood=lik,
        inducing_variable=inducing,
        num_latent_gps=D,
        q_diag=True,
        whiten=True,
    )

    # small init for q_var improves stability
    # Zero mean
    model.q_mu.assign(tf.zeros_like(model.q_mu))

    # q_sqrt shape = (L, M) when q_diag=True
    model.q_sqrt.assign(1e-3 * tf.ones_like(model.q_sqrt))

    return model, lik, Z



def train_svgp(
    model,
    X,
    Y,
    cfg: ModelConfig,
    patience: int = 100,       # how many iterations to wait for improvement
    min_delta: float = 1e-5,  # minimal ELBO gain to count as improvement
    verbose: bool = True
):
    """
    Train SVGP with Adam, monitoring ELBO and stopping early if it stalls.

    Args:
        model: gpflow SVGP
        X, Y: training arrays
        cfg: ModelConfig
        patience: stop if no ELBO improvement for this many steps
        min_delta: threshold for "improvement"
        verbose: whether to print progress
    """
    Xtf = tf.convert_to_tensor(X, dtype=tf.float64)
    Ytf = tf.convert_to_tensor(Y, dtype=tf.float64)

    opt = tf.optimizers.Adam(learning_rate=cfg.lr)

    @tf.function(autograph=False)
    def step():
        with tf.GradientTape() as tape:
            elbo = model.elbo((Xtf, Ytf))
            loss = -elbo
        grads = tape.gradient(loss, model.trainable_variables)
        opt.apply_gradients(zip(grads, model.trainable_variables))
        return elbo

    best_elbo = -np.inf
    wait = 0

    for it in range(cfg.max_iters):
        elbo_val = step().numpy()

        # Check improvement
        if elbo_val > best_elbo + min_delta:
            best_elbo = elbo_val
            wait = 0
        else:
            wait += 1

        # Logging
        if verbose and (it + 1) % 500 == 0:
            print(f"[train] iter {it+1:5d}, ELBO={elbo_val:.4f}, best={best_elbo:.4f}, wait={wait}")

        # Early stop
        if wait >= patience:
            if verbose:
                print(f"[train] Early stopping at iter {it+1}, best ELBO={best_elbo:.4f}")
            break

    return best_elbo

def predict_composition(model, Xnew: np.ndarray, mc_samples: int = 64) -> np.ndarray:
    """
    Returns predictive mean composition (N*, D) by MC over q(f).
    """
    Xtf = tf.convert_to_tensor(Xnew, dtype=tf.float64)
    Fmu, Fvar = model.predict_f(Xtf, full_cov=False, full_output_cov=False)
    lik: SoftmaxCompositional = model.likelihood  # type: ignore
    Pmean, Pvar = lik.predictive_mean_from_moments(Fmu, Fvar, mc=mc_samples)
    # Ensure valid simplex numerically
    Pmean = to_simplex(Pmean)
    return Pmean, Pvar