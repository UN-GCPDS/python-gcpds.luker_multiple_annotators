import pickle
import numpy as np
# from tensorflow.keras import mixed_precision
import gpflow
from gpflow.likelihoods import Gaussian
from gpflow.models import SVGP
import tensorflow as tf
from typing import List, Optional
from sklearn.cluster import MiniBatchKMeans
from tensorflow.keras import mixed_precision
gpflow.config.set_default_float(tf.float32)
gpflow.config.set_default_jitter(1e-4)


class AnnotatorGPRTrainer:
    """Train a separate *sparse* Gaussian‑process regressor (``SimpleGPR``) for
    each human annotator.

    The original implementation toggled between *full* and *sparse* GPs based on
    the dataset size. Since ``SimpleGPR`` now implements **only** the sparse
    variational formulation, this trainer no longer needs to handle the
    threshold logic.
    """

    def __init__(
        self,
        inducing_points: int = 200,
        max_iter: int = 1_000,
        batch_size: int = 128,
        early_stop_patience: int = 20,
        early_stop_min_delta: float = 1e-3,
        learning_rate: float = 1e-2,
        monitor_every: int = 5,
        seed: Optional[int] = None,
    ) -> None:
        # self.inducing_points = inducing_points
        self.max_iter = max_iter
        self.batch_size = batch_size
        self.early_stop_patience = early_stop_patience
        self.early_stop_min_delta = early_stop_min_delta
        self.learning_rate = learning_rate
        self.monitor_every = monitor_every
        self.seed = seed

        self.models: List["SimpleGPR"] = []

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------
    def train_gprs(self, X: np.ndarray, Y_ann: np.ndarray) -> None:
        """Fit one :class:`SimpleGPR` per annotator.

        Parameters
        ----------
        X : array‑like, shape (n_samples, n_features)
            Shared input space for *all* annotators.

        Y_ann : array‑like, shape (n_samples, n_annotators)
            Column *j* contains the labels provided by annotator *j*.
        """
        n_samples, _ = X.shape
        n_annotators = Y_ann.shape[1]

        self.models = []

        for ann in range(n_annotators):
            print(f"Training SPARSE GPR for Annotator {ann} (Samples: {n_samples})")
            y = Y_ann[:, ann]

            gpr_model = SimpleGPR(
                inducing_points=self.inducing_points,
                max_iter=self.max_iter,
                batch_size=self.batch_size,
                early_stop_patience=self.early_stop_patience,
                early_stop_min_delta=self.early_stop_min_delta,
                learning_rate=self.learning_rate,
                monitor_every=self.monitor_every,
                seed=self.seed,
            )
            gpr_model.fit(X, y)
            self.models.append(gpr_model)

        print(f"\u2705 Trained {n_annotators} sparse GPR models.")

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------
    def predict(self, X_new: np.ndarray):
        """Posterior mean and standard deviation for every annotator.

        Returns
        -------
        means : ndarray, shape (n_samples, n_annotators)
        stds  : ndarray, shape (n_samples, n_annotators)
        """
        if not self.models:
            raise ValueError("No models present. Call `train_gprs` or `load` first.")

        preds, vars_ = zip(*(model.predict(X_new) for model in self.models))
        return np.stack(preds, axis=1), np.stack(vars_, axis=1)

    # ------------------------------------------------------------------
    # Persistence helpers
    # ------------------------------------------------------------------
    def save(self, directory: str) -> None:
        """Serialise the trainer and all underlying models.

        * Each :class:`SimpleGPR` is saved to ``<directory>/annotator_{i}.pkl``.
        * Trainer‑level metadata is written to ``trainer_meta.pkl``.
        """
        os.makedirs(directory, exist_ok=True)

        meta = {
            "n_annotators": len(self.models),
            "inducing_points": self.inducing_points,
        }

        # save each annotator model
        for idx, model in enumerate(self.models):
            model_path = os.path.join(directory, f"annotator_{idx}")
            model.save(model_path)

        # save metadata
        with open(os.path.join(directory, "trainer_meta.pkl"), "wb") as f:
            pickle.dump(meta, f)

        print(f"\u2705 Saved {meta['n_annotators']} SimpleGPR models to '{directory}'")

    def load(self, directory: str) -> None:
        """Restore the trainer and all :class:`SimpleGPR` models persisted by
        :meth:`save`.
        """
        meta_path = os.path.join(directory, "trainer_meta.pkl")
        if not os.path.exists(meta_path):
            raise FileNotFoundError(f"Could not find metadata file at '{meta_path}'")

        with open(meta_path, "rb") as f:
            meta = pickle.load(f)

        self.inducing_points = meta["inducing_points"]
        n_annotators = meta["n_annotators"]

        self.models = []
        for idx in range(n_annotators):
            model_path = os.path.join(directory, f"annotator_{idx}")
            gpr = SimpleGPR()  # default hyper‑params are fine when loading
            gpr.load(model_path)
            self.models.append(gpr)

        print(f"\u2705 Loaded {n_annotators} SimpleGPR models from '{directory}'")


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