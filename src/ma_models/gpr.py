import os
import pickle
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
import gpflow

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


class SimpleGPR:
    def __init__(self, threshold_samples=2000, inducing_points=500):
        """
        GPR wrapper that uses full GPR for small datasets, and Sparse GP for large ones.

        Parameters:
        -----------
        threshold_samples: int
            Use full GPR if number of samples is below this threshold.
        inducing_points: int
            Number of inducing points for Sparse GP.
        """
        self.threshold_samples = threshold_samples
        self.inducing_points = inducing_points
        self.model = None
        self.model_type = None  # 'full' or 'sparse'

    def fit(self, X, y):
        n_samples, _ = X.shape

        if n_samples < self.threshold_samples:
            # Full GPR (scikit-learn)
            kernel = C(1.0) * RBF(length_scale=1.0)
            gpr = GaussianProcessRegressor(kernel=kernel, normalize_y=True, n_restarts_optimizer=5)
            gpr.fit(X, y)
            self.model = gpr
            self.model_type = 'full'
        else:
            # Sparse GP (GPflow)
            Z_init = X[np.random.choice(n_samples, self.inducing_points, replace=False)]
            kernel = gpflow.kernels.SquaredExponential()
            data = (X, y.reshape(-1, 1))
            model = gpflow.models.SGPR(data, kernel=kernel, inducing_variable=Z_init)
            opt = gpflow.optimizers.Scipy()
            opt.minimize(model.training_loss, variables=model.trainable_variables)
            self.model = model
            self.model_type = 'sparse'

        print(f"✅ Trained {self.model_type.upper()} GPR model.")

    def predict(self, X_new):
        if self.model_type == 'full':
            mean, std = self.model.predict(X_new, return_std=True)
        else:
            mean, var = self.model.predict_f(X_new)
            mean = mean.numpy().flatten()
            std = np.sqrt(var.numpy().flatten())
        return mean, std
    def save(self, path):
        """
        Save the SimpleGPR model to a single .pkl file.

        Parameters:
        -----------
        path : str
            Path without extension.
        """
        metadata = {
            'threshold_samples': self.threshold_samples,
            'inducing_points': self.inducing_points,
            'model_type': self.model_type,
        }

        if self.model_type == 'full':
            data = {
                'metadata': metadata,
                'model': self.model,
            }
        else:
            params = gpflow.utilities.parameter_dict(self.model)
            data = {
                'metadata': metadata,
                'params': params,
            }

        with open(path + '.pkl', 'wb') as f:
            pickle.dump(data, f)

    def load(self, path):
        """
        Load the SimpleGPR model from a .pkl file.

        Parameters:
        -----------
        path : str
            Path without extension.
        """
        with open(path + '.pkl', 'rb') as f:
            data = pickle.load(f)

        metadata = data['metadata']
        self.threshold_samples = metadata['threshold_samples']
        self.inducing_points = metadata['inducing_points']
        self.model_type = metadata['model_type']

        if self.model_type == 'full':
            self.model = data['model']
        else:
            # Create a dummy SGPR model with same structure
            kernel = gpflow.kernels.SquaredExponential()
            self.model = gpflow.models.SGPR(
                data=(np.zeros((1, 1)), np.zeros((1, 1))),
                kernel=kernel,
                inducing_variable=np.zeros((1, 1))
            )
            gpflow.utilities.restore_model(self.model, data['params'])