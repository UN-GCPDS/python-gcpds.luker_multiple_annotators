import tensorflow as tf
import gpflow as gpf
from gpflow.ci_utils import reduce_in_tests
from gpflow.utilities import print_summary
from check_shapes import check_shapes, inherit_check_shapes
from gpflow.ci_utils import reduce_in_tests
import numpy as np
MAXITER = reduce_in_tests(500)
gpf.config.set_default_float(tf.float32)

def safe_exp(f):
    """
    Numerically stable exponential function:
    - Clips `f` to prevent overflow when computing `exp(f)`.
    - Uses an upper bound `_lim_val_exp = log(float_max)`.
    """
    _lim_val = np.finfo(np.float32).max  # Maximum finite float32 value
    _lim_val_exp = np.log(_lim_val)      # Log of max float32 to avoid overflow

    return tf.exp(tf.clip_by_value(f, -10, _lim_val_exp/10))

def safe_square(f):
    """
    Numerically stable exponential function:
    - Clips `f` to prevent overflow when computing `exp(f)`.
    - Uses an upper bound `_lim_val_exp = log(float_max)`.
    """
    _lim_val = np.finfo(np.float32).max  # Maximum finite float32 value
    _lim_val_square = np.sqrt(_lim_val)      # Log of max float32 to avoid overflow

    return tf.square(tf.clip_by_value(f, -_lim_val_square/10, _lim_val_square/10))


class multiAnnotator_Gaussian(gpf.likelihoods.Likelihood):
    def __init__(self, num_ann: int) -> None:
        """
        Likelihood function for CCGPMA (Regression) with multiple annotators.
        
        Args:
        - num_ann (int): Number of annotators providing labels.
        """
        super().__init__(input_dim=None, latent_dim=1 + num_ann, observation_dim=None)
        self.R = num_ann  # Number of annotators

    @inherit_check_shapes
    def _log_prob(self, X, F, Y):
        """
        Computes the log probability of the observed data under the model.
        Uses a Gaussian likelihood: Y ~ N(f(x), v(x)).
        """
        iAnn = tf.where(Y == -1e20, tf.zeros_like(Y), tf.ones_like(Y))
        
        F = tf.cast(F, tf.float32)
        Y = tf.cast(Y, tf.float32)
        
        f_mean = F[:, :1]  # Mean function f(x)
        f_var = tf.clip_by_value(safe_exp(F[:, 1:]), 1e-9, 1e9)  # Ensure variance is positive

        log_prob = -0.5 * tf.math.reduce_sum(
            ((- ((Y - f_mean) ** 2)) / f_var + tf.math.log(f_var) + tf.math.log(2 * tf.constant(3.14159265359)))*iAnn ,
            axis=1
        )
        return log_prob
        
    def _variational_expectations(self, X, Fmu, Fvar, Y, Y_metadata=None):
        """
        Computes the variational expectation E_q[log p(Y | F)] under a Gaussian likelihood.
        """
        # Ensure everything is float32
        X = tf.cast(X, tf.float32)
        Fmu = tf.cast(Fmu, tf.float32)
        Fvar = tf.cast(Fvar, tf.float32)
        Y = tf.cast(Y, tf.float32)
    
        N = tf.shape(Y)[0]  # Number of samples
        iAnn = tf.cast(Y_metadata, tf.float32) if Y_metadata is not None else tf.ones_like(Y)
    
        # Extract mean and variance
        m_fmean, m_fvar = Fmu[:, :1], Fmu[:, 1:]  # Mean function, variance function
        v_fmean, v_fvar = Fvar[:, :1], Fvar[:, 1:]  # Variance associated with GP model
    
        # Compute precision: exp(-variance + 0.5 * variance) [Ensures positive precision]
        precision = safe_exp(-m_fvar + 0.5 * v_fvar)
        precision = tf.clip_by_value(precision, -1e9, 1e9)  # Numerical stability
    
        # Compute squared error term: (Y^2 + mean^2 + variance - 2 * mean * Y)
        squares = (safe_square(Y) + safe_square(m_fmean) + v_fmean - 2 * m_fmean * Y)
        squares = tf.clip_by_value(squares, -1e9, 1e9)  # Numerical stability
    
        # Compute variational expectation
        var_exp = tf.reduce_sum(
            (-0.5 * tf.math.log(2 * np.pi) - 0.5 * m_fvar - 0.5 * precision * squares) * iAnn, 
            axis=1
        )
    
        return tf.reshape(var_exp, (N, 1))
 
    def _predict_mean_and_var(self, X, Fmu, Fvar):
        """
        Computes predictive mean and variance for multi-annotator regression.
    
        Args:
            X (tensor): Input test points (N, D).
            Fmu (tensor): Mean of latent functions (N, 1 + R).
            Fvar (tensor): Variance of latent functions (N, 1 + R).
    
        Returns:
            mean (tensor): Predictive mean of shape (N, 1 + R).
            variance (tensor): Predictive variance of shape (N, 1 + R).
        """
        Fmu = tf.cast(Fmu, tf.float32)
        Fvar = tf.cast(Fvar, tf.float32)
    
        # Mean prediction (First function)
        mean_regression = Fmu[:, :1]  # Regression function mean (N, 1)
        variance_regression = Fvar[:, :1]  # Variance of the regression function (N, 1)
    
        if Fmu.shape[1] > 1:
            # Annotator variance modeling (for R annotators)
            annotator_means = safe_exp(Fmu[:, 1:] + 0.5 * Fvar[:, 1:])  # (N, R)
            annotator_vars = (safe_exp(Fvar[:, 1:]) - 1) * safe_exp(2 * Fmu[:, 1:] + Fvar[:, 1:])  # (N, R)
        else:
            annotator_means = tf.zeros_like(Fmu[:, 1:])  # Placeholder if no annotators
            annotator_vars = tf.zeros_like(Fvar[:, 1:])
    
        # Concatenate regression mean with annotator means
        mean = tf.concat([mean_regression, annotator_means], axis=1)  # (N, 1 + R)
    
        # Concatenate regression variance with annotator variances
        variance = tf.concat([variance_regression, annotator_vars], axis=1)  # (N, 1 + R)
    
        return mean, variance

    def _predict_log_density(self, F):
        raise NotImplementedError  # Not required for regression tasks

def run_adam(model, train_dataset, minibatch_size, iterations, lr):
    """
    Runs Adam optimizer for GPflow model.
    
    :param model: GPflow model
    :param train_dataset: Training dataset
    :param minibatch_size: Batch size
    :param iterations: Number of training iterations
    :param lr: Learning rate
    :return: List of ELBO values over iterations
    """
    logf = []
    train_iter = iter(train_dataset.batch(minibatch_size))
    training_loss = model.training_loss_closure(train_iter, compile=True)
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
    
    for step in range(iterations):
        with tf.GradientTape() as tape:
            loss = training_loss()
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    
        if step % 10 == 0:
            elbo = -loss.numpy()
            logf.append(elbo)
    
    return logf

def create_compiled_predict_y(model, n_features):
    return tf.function(
        lambda Xnew: model.predict_y(Xnew, full_cov=False),
        input_signature=[tf.TensorSpec(shape=[None, n_features], dtype=tf.float32)]
    )