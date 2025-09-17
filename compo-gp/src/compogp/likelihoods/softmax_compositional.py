from __future__ import annotations
import tensorflow as tf
import gpflow

class SoftmaxCompositional(gpflow.likelihoods.Likelihood):
    """
    Softmax-based compositional likelihood:
      log p(y | f) = tau * sum_k y_k * log softmax(f)_k
    Vector-valued: latent_dim = observation_dim = input_dim = D.
    """

    def __init__(self, D: int, tau: float = 100.0, eps: float = 1e-8):
        super().__init__(input_dim=D, latent_dim=D, observation_dim=D)
        self.D = int(D)
        self.tau = tf.convert_to_tensor(tau, dtype=gpflow.default_float())
        self.eps = tf.convert_to_tensor(eps, dtype=gpflow.default_float())

    # -------- helpers --------
    @staticmethod
    def _softmax(z, axis=-1):
        zmax = tf.reduce_max(z, axis=axis, keepdims=True)
        ez = tf.exp(z - zmax)
        return ez / tf.reduce_sum(ez, axis=axis, keepdims=True)

    def _loglik_point(self, F, Y):
        P = tf.clip_by_value(self._softmax(F), self.eps, 1.0)
        return self.tau * tf.reduce_sum(Y * tf.math.log(P), axis=-1)  # (...,)

    def _mc_sample_f(self, Fmu, Fvar, num_mc: int):
        # Build epsilon from Fvar to guarantee broadcast compatibility
        eps_shape = tf.concat([[num_mc], tf.shape(Fvar)], axis=0)  # (S, N, D)
        eps = tf.random.normal(shape=eps_shape, dtype=Fvar.dtype)
        Fstd = tf.sqrt(tf.maximum(Fvar, 0.0))
        return Fmu[None, ...] + Fstd[None, ...] * eps  # (S, N, D)
    def _ensure_nd_and_cast(self, X, Fmu, Fvar, Y):
        # Cast dtypes
        Fmu = tf.cast(Fmu, gpflow.default_float())
        Fvar = tf.cast(Fvar, gpflow.default_float())
        Y   = tf.cast(Y,   gpflow.default_float())

        # Match N and D
        n_fmu = tf.shape(Fmu)[0]; d_fmu = tf.shape(Fmu)[1]
        n_fvar= tf.shape(Fvar)[0]; d_fvar= tf.shape(Fvar)[1]
        n_y   = tf.shape(Y)[0];    d_y   = tf.shape(Y)[1]

        # One-time shape print (remove after debugging)
        tf.print("[LIK] shapes -> Fmu", tf.shape(Fmu), "Fvar", tf.shape(Fvar), "Y", tf.shape(Y), summarize=-1)
        return Fmu, Fvar, Y

    # -------- required by GPflow Likelihood (note the X arg) --------
    def _log_prob(self, X, F, Y):
        # X is unused, but required by the signature
        return self._loglik_point(F, Y)

    def _variational_expectations(self, X, Fmu, Fvar, Y):
        F_sample = self._mc_sample_f(Fmu, Fvar, num_mc=8)
        logp = self._loglik_point(F_sample, Y[None, ...])  # (S, N)
        return tf.reduce_mean(logp, axis=0)

    def _predict_mean_and_var(self, X, Fmu, Fvar):
        F_sample = self._mc_sample_f(Fmu, Fvar, num_mc=64)
        P = self._softmax(F_sample, axis=-1)              # (S, N, D)
        mean = tf.reduce_mean(P, axis=0)                  # (N, D)
        var  = tf.math.reduce_variance(P, axis=0)         # (N, D)
        mean = tf.clip_by_value(mean, self.eps, 1.0)
        mean = mean / tf.reduce_sum(mean, axis=-1, keepdims=True)
        return mean, var

    def _predict_log_density(self, X, Fmu, Fvar, Y):
        F_sample = self._mc_sample_f(Fmu, Fvar, num_mc=32)
        logp = self._loglik_point(F_sample, Y[None, ...])  # (S, N)
        return tf.reduce_mean(logp, axis=0)

    def predictive_mean_from_moments(self, Fmu, Fvar, mc: int = 64):
        F_sample = self._mc_sample_f(Fmu, Fvar, num_mc=mc)
        P = self._softmax(F_sample, axis=-1)
        mean = tf.reduce_mean(P, axis=0)
        var = tf.math.reduce_variance(P, axis=0)
        mean = tf.clip_by_value(mean, self.eps, 1.0)
        mean = mean / tf.reduce_sum(mean, axis=-1, keepdims=True)
        return mean, var
