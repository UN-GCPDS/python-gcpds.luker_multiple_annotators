import copy
from typing import Optional
import tensorflow as tf
import gpflow as gpf
from gpflow.ci_utils import reduce_in_tests
from gpflow.utilities import print_summary
from check_shapes import check_shapes, inherit_check_shapes
import numpy as np

# Set the maximum number of iterations for training, reducing it in tests for efficiency
MAXITER = reduce_in_tests(500)
gpf.config.set_default_float(tf.float32)

def safe_exp(f: tf.Tensor) -> tf.Tensor:
    """Computes a numerically stable exponential function.

    Clips the input `f` before computing `exp(f)`, ensuring values remain within a safe range.

    Args:
        f (tf.Tensor): Input tensor.

    Returns:
        tf.Tensor: Exponentiated tensor with stability checks.
    """
    _lim_val = np.finfo(np.float32).max  # Maximum finite float32 value
    _lim_val_exp = np.log(_lim_val) / 10  # Prevents overflow in exponentiation

    return tf.exp(tf.clip_by_value(f, -10, _lim_val_exp))

def safe_square(f: tf.Tensor) -> tf.Tensor:
    """Computes a numerically stable squared function.

    Clips the input `f` before computing `square(f)`, ensuring values remain within a safe range.

    Args:
        f (tf.Tensor): Input tensor.

    Returns:
        tf.Tensor: Squared tensor with stability checks.
    """
    _lim_val = np.finfo(np.float32).max  # Maximum finite float32 value
    _lim_val_square = np.sqrt(_lim_val) / 10  # Prevents overflow when squaring

    return tf.square(tf.clip_by_value(f, -_lim_val_square, _lim_val_square))

class MultiAnnotatorGaussian(gpf.likelihoods.Likelihood):
    """Custom Gaussian likelihood for regression with multiple annotators.

    This model handles the uncertainty associated with multiple annotations.

    Attributes:
        R (int): Number of annotators.
    """

    def __init__(self, num_ann: int) -> None:
        """Initializes the multi-annotator Gaussian likelihood model.

        Args:
            num_ann (int): Number of annotators providing labels.
        """
        super().__init__(input_dim=None, latent_dim=1 + num_ann, observation_dim=None)
        self.R = num_ann  # Store the number of annotators

    @inherit_check_shapes
    def _log_prob(self, X: tf.Tensor, F: tf.Tensor, Y: tf.Tensor) -> tf.Tensor:
        """Computes the log probability of observed data under the Gaussian likelihood.

        Args:
            X (tf.Tensor): Input data (ignored in the likelihood function).
            F (tf.Tensor): Latent function values (N, 1 + R).
            Y (tf.Tensor): Observed labels (N, R).

        Returns:
            tf.Tensor: Log probability of Y given F.
        """
        # Create a mask to filter out missing annotations (-1e20 used as a missing label indicator)
        iAnn = tf.where(Y == -1e20, tf.zeros_like(Y), tf.ones_like(Y))

        # Convert inputs to float32 for numerical stability
        F = tf.cast(F, tf.float32)
        Y = tf.cast(Y, tf.float32)

        # Extract mean and variance components
        f_mean = F[:, :1]  # Regression function mean
        f_var = tf.clip_by_value(safe_exp(F[:, 1:]), 1e-9, 1e9)  # Ensure variance is positive

        # Compute log probability using Gaussian distribution formula
        log_prob = -0.5 * tf.reduce_sum(
            ((- ((Y - f_mean) ** 2)) / f_var + tf.math.log(f_var) + tf.math.log(2 * np.pi)) * iAnn,
            axis=1
        )
        return log_prob

    def _variational_expectations(
        self, X: tf.Tensor, Fmu: tf.Tensor, Fvar: tf.Tensor, Y: tf.Tensor, Y_metadata=None
    ) -> tf.Tensor:
        """Computes the variational expectation E_q[log p(Y | F)] under a Gaussian likelihood.

        Args:
            X (tf.Tensor): Input data.
            Fmu (tf.Tensor): Mean of latent functions (N, 1 + R).
            Fvar (tf.Tensor): Variance of latent functions (N, 1 + R).
            Y (tf.Tensor): Observed labels (N, R).
            Y_metadata (tf.Tensor, optional): Additional metadata (e.g., annotation masks).

        Returns:
            tf.Tensor: Variational expectation values (N, 1).
        """
        # Convert all inputs to float32 for consistency
        X, Fmu, Fvar, Y = map(tf.cast, [X, Fmu, Fvar, Y], [tf.float32] * 4)

        N = tf.shape(Y)[0]  # Number of samples
        iAnn = tf.cast(Y_metadata, tf.float32) if Y_metadata is not None else tf.ones_like(Y)

        # Extract mean and variance components
        m_fmean, m_fvar = Fmu[:, :1], Fmu[:, 1:]  # Mean function, variance function
        v_fmean, v_fvar = Fvar[:, :1], Fvar[:, 1:]  # Variance associated with GP model

        # Compute precision (inverse variance) ensuring numerical stability
        precision = safe_exp(-m_fvar + 0.5 * v_fvar)
        precision = tf.clip_by_value(precision, -1e9, 1e9)

        # Compute squared error term
        squares = safe_square(Y) + safe_square(m_fmean) + v_fmean - 2 * m_fmean * Y
        squares = tf.clip_by_value(squares, -1e9, 1e9)

        # Compute variational expectation
        var_exp = tf.reduce_sum(
            (-0.5 * tf.math.log(2 * np.pi) - 0.5 * m_fvar - 0.5 * precision * squares) * iAnn,
            axis=1
        )

        return tf.reshape(var_exp, (N, 1))

    def _predict_mean_and_var(self, X: tf.Tensor, Fmu: tf.Tensor, Fvar: tf.Tensor) -> tuple:
        """Computes predictive mean and variance for multi-annotator regression.

        Args:
            X (tf.Tensor): Input test points (N, D).
            Fmu (tf.Tensor): Mean of latent functions (N, 1 + R).
            Fvar (tf.Tensor): Variance of latent functions (N, 1 + R).

        Returns:
            tuple: Predictive mean (tf.Tensor) and variance (tf.Tensor), both of shape (N, 1 + R).
        """
        Fmu, Fvar = tf.cast(Fmu, tf.float32), tf.cast(Fvar, tf.float32)

        mean_regression = Fmu[:, :1]
        variance_regression = Fvar[:, :1]

        if Fmu.shape[1] > 1:
            annotator_means = safe_exp(Fmu[:, 1:] + 0.5 * Fvar[:, 1:])
            annotator_vars = (safe_exp(Fvar[:, 1:]) - 1) * safe_exp(2 * Fmu[:, 1:] + Fvar[:, 1:])
        else:
            annotator_means = tf.zeros_like(Fmu[:, 1:])
            annotator_vars = tf.zeros_like(Fvar[:, 1:])

        mean = tf.concat([mean_regression, annotator_means], axis=1)
        variance = tf.concat([variance_regression, annotator_vars], axis=1)

        return mean, variance

    def _predict_log_density(self, F: tf.Tensor):
        """Raises a NotImplementedError since log density prediction is not required for regression tasks.

        Args:
            F (tf.Tensor): Latent function values.

        Raises:
            NotImplementedError: This method is not implemented.
        """
        raise NotImplementedError("Log density prediction is not required for regression tasks.")

def run_adam(
    model: gpf.models.GPModel,
    train_dataset: tf.data.Dataset,
    minibatch_size: int,
    iterations: int,
    lr: float,
    callbacks: Optional[dict] = None,
) -> list:
    """Runs the Adam optimizer to train a GPflow model with built-in callbacks.

    Args:
        model (gpf.models.GPModel): GPflow model instance.
        train_dataset (tf.data.Dataset): Training dataset.
        minibatch_size (int): Batch size for training.
        iterations (int): Number of iterations.
        lr (float): Initial learning rate.
        callbacks (dict): Configuration for callbacks:
            - early_stopping_patience (int)
            - lr_patience (int)
            - lr_factor (float)
            - min_lr (float)

    Returns:
        list: ELBO values over training iterations.
    """
    callbacks = callbacks or {}
    early_stopping_patience = callbacks.get("early_stopping_patience", 500)
    lr_patience = callbacks.get("lr_patience", 200)
    lr_factor = callbacks.get("lr_factor", 0.5)
    min_lr = callbacks.get("min_lr", 1e-6)

    logf = []
    train_iter = iter(train_dataset.batch(minibatch_size))
    training_loss = model.training_loss_closure(train_iter, compile=True)
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)

    best_loss = float("inf")
    best_weights = None
    wait_early = 0
    wait_lr = 0

    for step in range(iterations):
        with tf.GradientTape() as tape:
            loss = training_loss()
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        loss_val = loss.numpy()
        if step % 10 == 0:
            logf.append(-loss_val)

        # Track best model
        if loss_val < best_loss - 1e-6:
            best_loss = loss_val
            best_weights = copy.deepcopy(model.trainable_variables)
            wait_early = 0
            wait_lr = 0
        else:
            wait_early += 1
            wait_lr += 1

        # Early stopping
        if wait_early >= early_stopping_patience:
            print(f"Early stopping at step {step}, best loss: {best_loss:.4f}")
            break

        # Reduce LR on plateau
        if wait_lr >= lr_patience:
            old_lr = float(optimizer.learning_rate.numpy())
            new_lr = max(old_lr * lr_factor, min_lr)
            if new_lr < old_lr:
                print(f"Reducing LR from {old_lr:.6f} to {new_lr:.6f} at step {step}")
                optimizer.learning_rate.assign(new_lr)
            wait_lr = 0

    # Restore best model
    if best_weights is not None:
        for var, best_var in zip(model.trainable_variables, best_weights):
            var.assign(best_var)

    return logf


def create_compiled_predict_y(model: gpf.models.GPModel, n_features: int) -> tf.function:
    """Creates a compiled TensorFlow function for efficient prediction.

    This function compiles `model.predict_y` using `tf.function` to optimize
    inference speed. It ensures that the prediction function can be efficiently
    executed on TensorFlow’s computation graph.

    Args:
        model (gpf.models.GPModel): The Gaussian Process model from GPflow.
        n_features (int): The number of features in the input data.

    Returns:
        tf.function: A compiled function that takes a tensor of shape (None, n_features)
        and returns the model's predictive mean and variance.
    """
    return tf.function(
        lambda Xnew: model.predict_y(Xnew, full_cov=False),
        input_signature=[tf.TensorSpec(shape=[None, n_features], dtype=tf.float32)]
    )