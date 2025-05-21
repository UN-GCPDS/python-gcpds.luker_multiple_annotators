import pickle
import numpy as np
# from tensorflow.keras import mixed_precision
import gpflow
from gpflow.likelihoods import Gaussian
from gpflow.models import SVGP
import tensorflow as tf
from sklearn.cluster import MiniBatchKMeans
from tensorflow.keras import mixed_precision
gpflow.config.set_default_float(tf.float32)
gpflow.config.set_default_jitter(1e-4)


class AnnotatorGPRTrainer:
    def __init__(self, threshold_samples=2000, inducing_points=500):
        """
        Parameters:
        -----------
        threshold_samples: int
            Maximum number of samples to use full GPR. Above this, use sparse GP.
        inducing_points: int
            Number of inducing points for Sparse GP.
        """
        self.threshold_samples = threshold_samples
        self.inducing_points = inducing_points
        self.models = []  # one model per annotator (sklearn or GPflow)
        self.model_types = []  # 'full' or 'sparse' per annotator

    def train_gprs(self, X, Y_ann):
        n_samples, n_features = X.shape
        n_annotators = Y_ann.shape[1]

        self.models = []
        self.model_types = []

        for ann in range(n_annotators):
            print(f"Training GPR for Annotator {ann} (Samples: {n_samples})")
            y = Y_ann[:, ann]

            gpr_model = SimpleGPR(
                threshold_samples=self.threshold_samples,
                inducing_points=self.inducing_points
            )
            gpr_model.fit(X, y)

            self.models.append(gpr_model)
            self.model_types.append(gpr_model.model_type)

        print(f"Trained {n_annotators} GPR models.")

    def predict(self, X_new):
        preds = []
        vars_ = []

        for model in self.models:
            mean, std = model.predict(X_new)
            preds.append(mean)
            vars_.append(std)

        return np.stack(preds, axis=1), np.stack(vars_, axis=1)


# class SimpleGPR:
#     def __init__(self, threshold_samples=2000, inducing_points=100, max_iter=1200, batch_size=512):
#         self.threshold_samples = threshold_samples
#         self.inducing_points = inducing_points
#         self.max_iter = max_iter
#         self.batch_size = batch_size
#         self.model = None
#         self.model_type = None  # 'full' or 'sparse'

#     def fit(self, X, y):
#         n_samples, _ = X.shape

#         if n_samples < self.threshold_samples:
#             # Full GPR using sklearn
#             kernel = C(1.0) * RBF(length_scale=1.0)
#             gpr = GaussianProcessRegressor(kernel=kernel, normalize_y=True, n_restarts_optimizer=5)
#             gpr.fit(X, y)
#             self.model = gpr
#             self.model_type = 'full'
#         else:
#             # Sparse GP using SVGP (GPflow)
#             Z_init = X[np.random.choice(n_samples, self.inducing_points, replace=False)]
#             kernel = gpflow.kernels.SquaredExponential()
#             likelihood = Gaussian()
#             model = SVGP(kernel=kernel,
#                          likelihood=likelihood,
#                          inducing_variable=Z_init,
#                          num_latent_gps=1)

#             # Prepare mini-batch dataset
#             dataset = tf.data.Dataset.from_tensor_slices((X.astype(np.float64), y.reshape(-1, 1).astype(np.float64)))
#             dataset = dataset.shuffle(buffer_size=10000).batch(self.batch_size)

#             opt = tf.optimizers.Adam(learning_rate=0.01)

#             # Custom training loop
#             for step, batch in enumerate(dataset.repeat()):
#                 with tf.GradientTape() as tape:
#                     loss = -model.elbo(batch)
#                 grads = tape.gradient(loss, model.trainable_variables)
#                 opt.apply_gradients(zip(grads, model.trainable_variables))

#                 if step >= self.max_iter:
#                     break

#             self.model = model
#             self.model_type = 'sparse'

#         print(f"✅ Trained {self.model_type.upper()} GPR model.")

#     def predict(self, X_new):
#         if self.model_type == 'full':
#             mean, std = self.model.predict(X_new, return_std=True)
#         else:
#             mean, var = self.model.predict_f(X_new.astype(np.float64))
#             mean = mean.numpy().flatten()
#             std = np.sqrt(var.numpy().flatten())
#         return mean, std
#     def save(self, path):
#         """
#         Save the SimpleGPR model to a single .pkl file.

#         Parameters:
#         -----------
#         path : str
#             Path without extension.
#         """
#         metadata = {
#             'threshold_samples': self.threshold_samples,
#             'inducing_points': self.inducing_points,
#             'model_type': self.model_type,
#         }

#         if self.model_type == 'full':
#             data = {
#                 'metadata': metadata,
#                 'model': self.model,
#             }
#         else:
#             params = gpflow.utilities.parameter_dict(self.model)
#             data = {
#                 'metadata': metadata,
#                 'params': params,
#             }

#         with open(path + '.pkl', 'wb') as f:
#             pickle.dump(data, f)

#     def load(self, path):
#         """
#         Load the SimpleGPR model from a .pkl file.

#         Parameters:
#         -----------
#         path : str
#             Path without extension.
#         """
#         with open(path + '.pkl', 'rb') as f:
#             data = pickle.load(f)

#         metadata = data['metadata']
#         self.threshold_samples = metadata['threshold_samples']
#         self.inducing_points = metadata['inducing_points']
#         self.model_type = metadata['model_type']

#         if self.model_type == 'full':
#             self.model = data['model']
#         else:
#             # Create a dummy SGPR model with same structure
#             kernel = gpflow.kernels.SquaredExponential()
#             self.model = gpflow.models.SGPR(
#                 data=(np.zeros((1, 1)), np.zeros((1, 1))),
#                 kernel=kernel,
#                 inducing_variable=np.zeros((1, 1))
#             )
#             gpflow.utilities.restore_model(self.model, data['params'])

# -----------------------------------------------------------------------------
# Global numerical settings
# -----------------------------------------------------------------------------
# Use single‑precision everywhere to lower memory consumption and improve speed
# on compatible hardware (e.g. consumer GPUs with larger FP32 throughput).
# ----------------------------------------------------------------------------


class SimpleGPR:
    """Sparse variational Gaussian‑process regressor (SVGP) in float32.

    *   **Single precision** (`tf.float32`) greatly reduces memory footprint and
        bandwidth, which is crucial for very large datasets.
    *   **Mixed‑precision** (policy `mixed_float16`) leverages tensor‑core
        hardware where available; most computations execute in float16 while
        model parameters remain in float32 for numerical stability.
    *   Early stopping monitors the *full* ELBO every `monitor_every` steps.
    """

    def __init__(
        self,
        inducing_points: int = 200,
        max_iter: int = 1000,
        batch_size: int = 128,
        early_stop_patience: int = 20,
        early_stop_min_delta: float = 1e-3,
        learning_rate: float = 1e-2,
        monitor_every: int = 5,
        seed: int | None = None,
    ) -> None:
        self.inducing_points = inducing_points
        self.max_iter = max_iter
        self.batch_size = batch_size
        self.early_stop_patience = early_stop_patience
        self.early_stop_min_delta = early_stop_min_delta
        self.learning_rate = learning_rate
        self.monitor_every = monitor_every
        self.rng = np.random.default_rng(seed)

        self.model: SVGP | None = None

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Fit the sparse GP to training data (expects `float32`)."""
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32).reshape(-1, 1)
        n_samples, _ = X.shape

        kmeans = MiniBatchKMeans(n_clusters=self.inducing_points, batch_size=1024, random_state=42).fit(X)
        Z_init = kmeans.cluster_centers_


        kernel = gpflow.kernels.SquaredExponential()
        likelihood = Gaussian()
        self.model = SVGP(
            kernel=kernel,
            likelihood=likelihood,
            inducing_variable=Z_init,
            num_latent_gps=1,
        )

        dataset = (
            tf.data.Dataset.from_tensor_slices((X, y))
            .shuffle(buffer_size=min(10_000, n_samples), seed=self.rng.integers(1 << 31))
            .batch(self.batch_size)
            .repeat()
        )
        data_iter = iter(dataset)

        opt = tf.optimizers.Adam(self.learning_rate)

        best_elbo = -np.inf
        patience = 0
        self.elbo_history = []

        for step in range(1, self.max_iter + 1):
            batch = next(data_iter)
            with tf.GradientTape() as tape:
                elbo = self.model.elbo(batch)
                loss = -elbo  # maximise ELBO == minimise -ELBO
            grads = tape.gradient(loss, self.model.trainable_variables)
            opt.apply_gradients(zip(grads, self.model.trainable_variables))

            # ---------- early‑stopping monitor ----------
            if step % self.monitor_every == 0:
                current_elbo = self.model.elbo((X, y)).numpy()
                self.elbo_history.append(current_elbo)
                improvement = current_elbo - best_elbo
                if improvement > self.early_stop_min_delta:
                    best_elbo = current_elbo
                    patience = 0
                else:
                    patience += 1
                if patience >= self.early_stop_patience:
                    print(f"⏹️  Early stopping at step {step}. Best ELBO = {best_elbo:.3f}")
                    break
        else:
            print("⚠️  Reached max_iter without early stopping.")

        print("✅ Trained SPARSE GPR model (float32 / mixed precision).")

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------
    def predict(self, X_new: np.ndarray):
        """Posterior mean & standard deviation at `X_new` (returned as float32)."""
        if self.model is None:
            raise ValueError("Model has not been trained. Call `fit` first.")

        mean, var = self.model.predict_f(np.asarray(X_new, dtype=np.float32))
        return mean.numpy().flatten().astype(np.float32), np.sqrt(var.numpy().flatten()).astype(np.float32)

    # ------------------------------------------------------------------
    # Persistence helpers
    # ------------------------------------------------------------------
    def save(self, path: str) -> None:
        """Serialise the model parameters to `<path>.pkl`."""
        if self.model is None:
            raise ValueError("Model has not been trained. Nothing to save.")

        params = gpflow.utilities.parameter_dict(self.model)
        metadata = {
            "inducing_points": self.inducing_points,
            "dtype": "float32",
            "policy": mixed_precision.global_policy().name,
        }
        with open(path + ".pkl", "wb") as f:
            pickle.dump({"metadata": metadata, "params": params}, f)

    def load(self, path: str) -> None:
        """Load model parameters from `<path>.pkl` (re‑creates float32 model)."""
        with open(path + ".pkl", "rb") as f:
            data = pickle.load(f)

        kernel = gpflow.kernels.SquaredExponential()
        dummy_X = np.zeros((1, 1), dtype=np.float32)
        inducing_Z = np.zeros((1, 1), dtype=np.float32)
        self.model = SVGP(
            kernel=kernel,
            likelihood=Gaussian(),
            inducing_variable=inducing_Z,
            num_latent_gps=1,
        )
        gpflow.utilities.restore_model(self.model, data["params"])
        print("✅ Model parameters restored (float32 / mixed precision).")