# Standard libraries
import pickle

# Numerical and scientific computing
import numpy as np
from scipy.special import softmax
from scipy.spatial.distance import cdist

# Machine learning models and preprocessing
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.neighbors import NearestNeighbors

# Deep learning and GPflow
import tensorflow as tf
import tensorflow_probability as tfp

# Visualization
import matplotlib.pyplot as plt
import matplotlib
# from umap import UMAP

# GPR
from gpr import AnnotatorGPRTrainer

from typing import Dict, Optional
from pathlib import Path
from utils import get_iAnn

DTYPE = tf.float64 
DTYPEN = np.float64

class CKA(BaseEstimator, TransformerMixin):
    """Class for computing the Centered Kernel Alignment (CKA).

    This class provides methods to compute the CKA loss, fit the model,
    transform the data, and plot the history and kernels.

    Parameters:
    -----------
    epochs: int, default=50
        Number of epochs for training.
    batch_size: int, default=100
        Batch size for training.
    ls_X: float, default=1e-2
        Length scale for the input features kernel.
    ls_Y: float, default=1e-13
        Length scale for the label kernel.
    iAnn: array-like of shape (n_samples, n_annotators), default=ones(2)
        Annotation matrix.
    lr: float, default=1e-2
        Learning rate for optimization.
    l1: float, default=1e-3
        L1 regularization parameter.
    l2: float, default=1e-3
        L2 regularization parameter.

    Methods:
    --------
    __init__(self, epochs=50, batch_size=100, ls_X=1e-2, ls_Y=1e-13, iAnn=np.ones(2), lr=1e-2, l1=1e-3, l2=1e-3)
    Constructor method for the CKA class.

    fit(self, X, Y)
    Fits the model using the provided data.


    transform(self, X, *_)
    Transforms the data using the trained model.


    fit_transform(self, X, y)
    Fits the model to the data and transforms it.


    plot_history(self)
    Plots the training loss history.

    plot_kernels(self, X, Y, iAnn)
    Plots the kernels computed during training.


    ComputeKernel_X(self, X)
    Computes the exponential quadratic kernel over the input features.


    ComputeKernel_Y(self, Y, iAnn)
    Computes the linear kernel over the labels.


    CKA_loss(self, X, Y, iAnn)
    Defines the loss function based on the centered kernel alignment framework.


    """
    def __init__(self,epochs=50,batch_size=100,ls_X=1e-2,
                 ls_Y=1e-13,iAnn=2,lr=1e-2,l1=1e-3,l2=1e-3):
        self.epochs = epochs
        self.batch_size = batch_size
        if isinstance(iAnn, int):
            iAnn = np.ones((iAnn, iAnn))
        self.iAnn = iAnn
        self.R = self.iAnn.shape[1] #annotators
        self.mu = tf.Variable(tf.random.uniform(shape=(1, self.R), dtype=tf.float64))
        self.ls_X = tf.Variable(ls_X, dtype=tf.float64)
        self.ls_Y = tf.Variable(ls_Y, dtype=tf.float64)
        self.lr = lr
        self.l1 = l1
        self.l2 = l2

    def ComputeKernel_X(self,X):
        """
        Computes an exponential quadratic kernel over the input features
        Parameters:
        -----------
        X: array-like of shape (n_samples, n_features)
            Input features.

        Returns:
        --------
        Kernel matrix.
        """
        kernel = tfp.math.psd_kernels.ExponentiatedQuadratic(length_scale=self.ls_X,
                                                  name='ExponentiatedQuadratic')
        return kernel.matrix(X,X)


    def ComputeKernel_Y(self,Y,iAnn):
        """
        Computes a linear kernel over the labels

        Parameters:
        -----------
        Y: array-like of shape (n_samples, n_annotators)
            Labels.
        iAnn: array-like of shape (n_samples, n_annotators)
            Annotation matrix.

        Returns:
        --------
        Kernel matrix.
        """
        #kernel = tfp.math.psd_kernels.Linear(bias_amplitude=None,
        #                                     slope_amplitude=None, shift=None,
        #                                     feature_ndims=1,validate_args=False,
        #                                     name='Linear')
        self.kernelY = tfp.math.psd_kernels.ExponentiatedQuadratic(length_scale=self.ls_Y,
                                                  name='ExponentiatedQuadratic')
        # pdb.set_trace()
        N, R = Y.shape
        Y = tf.multiply(Y,iAnn)
        K_mu = tf.zeros([N,N], dtype=tf.dtypes.float64)
        for r in range(R):
            iann = tf.tensordot(iAnn[:,r], iAnn[:,r], 0)
            # ker = tf.linalg.set_diag(tf.multiply(self.kernelY.matrix(Y[:,r:r+1],Y[:,r:r+1]), iann), tf.ones_like(iAnn[:,r]))
            ker = tf.multiply(self.kernelY.matrix(Y[:,r:r+1],Y[:,r:r+1]), iann)
            # K_mu = tf.math.add(K_mu,self.mu[0,r]*self.kernelY.matrix(Y[:,r:r+1],Y[:,r:r+1]))
            K_mu = tf.math.add(K_mu,self.mu[0,r]*ker)
        return K_mu


    def CKA_loss(self,X,Y,iAnn):
        """
        Custom loss functino based on the centered kernel alignment framework

        Parameters:
        -----------
        X: array-like of shape (n_samples, n_features)
            Input features.
        Y: array-like of shape (n_samples, n_annotators)
            Labels.
        iAnn: array-like of shape (n_samples, n_annotators)
            Annotation matrix.

        Returns:
        --------
        CKA loss.
        """
        N = X.shape[0]
        KXX = self.ComputeKernel_X(X)
        Kmu = self.ComputeKernel_Y(Y,iAnn)
        I = tf.eye(N, dtype=tf.dtypes.float64)
        ones = tf.ones([N,1], dtype=tf.dtypes.float64)
        H = I - tf.linalg.matmul(ones, ones, transpose_b=True)/N
        KXX_c = tf.linalg.matmul(H,tf.linalg.matmul(KXX,H))
        Kmu_c = tf.linalg.matmul(H,tf.linalg.matmul(Kmu,H))
        num = tf.linalg.trace(tf.matmul(Kmu_c,KXX_c,transpose_a=True))
        den1 = tf.sqrt(tf.linalg.trace(tf.matmul(Kmu_c,Kmu_c,transpose_a=True)))
        den2 = tf.sqrt(tf.linalg.trace(tf.matmul(KXX_c,KXX_c,transpose_a=True)))


        return -num/(den1*den2) + self.l2*tf.norm(self.mu,ord=2)+self.l1*tf.norm(self.mu,ord=1)

    def fit(self,X,Y):
        """
        Parameters:
        -----------
        X: array-like of shape (n_samples, n_features)
            Input features.
        Y: array-like of shape (n_samples, n_annotators)
            Labels.

        Returns:
        --------
        None
        """
        batch_size = self.batch_size
        train_data=tf.data.Dataset.from_tensor_slices((X,Y,self.iAnn))
        train_data=train_data.shuffle(buffer_size=100).batch(batch_size).repeat(5)
        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(initial_learning_rate=self.lr,
                                decay_steps=10,decay_rate=0.9)
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
        self.loss_ = np.zeros(self.epochs)
        for epoch in range(self.epochs):
            # if epoch % 10 == 0:
            #     print("Start of epoch %d" % (epoch,))
            print("Start of epoch %d" % (epoch,))
            # Iterate over the batches of the dataset.
            for step, (x_batch_train,y_batch_train,iAnn_batch) in enumerate(train_data):
                with tf.GradientTape() as tape:
                    loss =  self.CKA_loss(x_batch_train, y_batch_train,iAnn_batch)
                grads = tape.gradient(loss, [self.mu])
                self.optimizer.apply_gradients(zip(grads, [self.mu]))
                #grads = tape.gradient(loss, [self.mu, self.ls_X])
                #self.optimizer.apply_gradients(zip(grads, [self.mu, self.ls_X]))


                if step % 50 == 0:
                    mu_ = self.mu.numpy()[0]
                    mu_[mu_<0] = 0
                    mu_ = mu_/np.sum(mu_)
                    print(f"step {step}: mean loss {loss.numpy().round(4)} ls {self.ls_X.numpy().round(2)} lr {(self.optimizer.learning_rate.numpy()).round(5)} mu {mu_.round(2)}" )
            self.loss_[epoch] = loss
        return

    def transform(self, X, *_):
        """
        Parameters:
        -----------
        X: array-like of shape (n_samples, n_features)
            Input features.

        Returns:
        --------
        Transformed data.
        """
        mu = self.mu.numpy()[0]
        mu[mu<0] = 0
        mu = mu/np.sum(mu)

        return mu

    def fit_transform(self,X,y):
        """
        Parameters:
        -----------
        X: array-like of shape (n_samples, n_features)
            Input features.
        Y: array-like of shape (n_samples, n_annotators)
            Labels.

        Returns:
        --------
        Transformed data.
        """
        self.fit(X,y)
        return  self.transform(X)

    def plot_history(self):
        fig,ax = plt.subplots(1,figsize=(3,3))
        ax.plot(np.arange(self.epochs),self.loss_)
        ax.set_xlabel('epochs')
        ax.set_ylabel('loss')
        plt.show()
        return

    def plot_kernels(self,X,Y,iAnn,annotators=None):
        """
        Parameters:
        -----------
        X: array-like of shape (n_samples, n_features)
            Input features.
        Y: array-like of shape (n_samples, n_annotators)
            Labels.
        iAnn: array-like of shape (n_samples, n_annotators)
            Annotation matrix.

        Returns:
        --------
        None
        """

        N, R = Y.shape
        fig,ax = plt.subplots(nrows = 1,ncols=R+2,figsize=(4*(R+2),3))
        #K features
        KXX = self.ComputeKernel_X(X)
        ax[0].imshow(KXX,vmin=0,vmax=1)
        ax[0].set_title(f"Target")
        ax[0].set_xticks([])
        ax[0].set_yticks([])

        #K Y annotators and Kmu
        Y = tf.multiply(Y,iAnn)
        K_mu = tf.zeros([N,N], dtype=tf.dtypes.float64)
        mu_ = self.mu.numpy()[0]
        mu_[mu_<0] = 0
        mu_ = mu_/np.sum(mu_)
        for r in range(R):
            K_mu = tf.math.add(K_mu,mu_[r]*self.kernelY.matrix(Y[:,r:r+1],Y[:,r:r+1]))

        ax[1].imshow(K_mu,vmin=0,vmax=1)
        ax[1].set_title(f"$K_\mu$")
        ax[1].set_xticks([])
        ax[1].set_yticks([])

        if annotators is None:
            for r in range(R):
                ax[r+2].imshow(mu_[r]*self.kernelY.matrix(Y[:,r:r+1],Y[:,r:r+1]),vmin=0,vmax=1)
                ax[r+2].set_title(f"A{r+1} - $K\mu_{{r+1}}={mu_[r].round(2)}$")
                ax[r+2].set_xticks([])
                ax[r+2].set_yticks([])
        else:
            for r, ann in enumerate(annotators):
                ax[r+2].imshow(mu_[r]*self.kernelY.matrix(Y[:,r:r+1],Y[:,r:r+1]),vmin=0,vmax=1)
                ax[r+2].set_title(f"A{ann} - $K\mu_{{{ann}}}={mu_[r].round(2)}$")
                ax[r+2].set_xticks([])
                ax[r+2].set_yticks([])
        cax = fig.add_axes([0.925, 0.15, 0.01, 0.7])
        norm = matplotlib.colors.Normalize(vmin=0,vmax=1)
        sm = plt.cm.ScalarMappable(cmap=None, norm=norm)
        sm.set_array([])
        cbar = plt.colorbar(sm,cax=cax)
        plt.show()

        return


class LCKA(BaseEstimator, TransformerMixin):
    """Linear Centered‑Kernel Alignment with per‑annotator weights (float32)."""

    # ------------------------ constructor ------------------------
    def __init__(
        self,
        *,
        epochs: int = 50,
        batch_size: int = 100,
        ls_X: float = 1e-1,
        ls_Y: float = 1e-13,
        l1: float = 1e-3,
        l2: float = 1e-3,
        lr: float = 1e-3,
        patience: int = 30,
        reduce_patience: int = 5,
        lr_reduce_factor: float = 0.5,
        min_delta: float = 1e-5,
        min_lr: float = 1e-6,
        seed: Optional[int] = 42,
    ) -> None:
        super().__init__()
        # Hyper‑parameters -------------------------------------------------
        self.epochs = int(epochs)
        self.batch_size = int(batch_size)
        self.l1 = float(l1)
        self.l2 = float(l2)
        self.lr = float(lr)
        self.patience = int(patience)
        self.reduce_patience = int(reduce_patience)
        self.lr_reduce_factor = float(lr_reduce_factor)
        self.min_delta = float(min_delta)
        self.min_lr = float(min_lr)
        self.seed = seed

        # Trainable length‑scales -----------------------------------------
        self._raw_ls_X = tf.Variable(np.log(ls_X), dtype=DTYPE, name="raw_ls_X")
        self._raw_ls_Y = tf.Variable(np.log(ls_Y), dtype=DTYPE, name="raw_ls_Y")
        # Runtime state ----------------------------------------------------
        self.beta: Optional[tf.Variable] = None  # shape (N+1, R)
        self.X_full: Optional[tf.Tensor] = None
        self.Y_full: Optional[tf.Tensor] = None
        self.iAnn_full: Optional[tf.Tensor] = None
        self.idx_full: Optional[tf.Tensor] = None
        self.q: Optional[np.ndarray] = None
        self.loss_: list[float] = []
        self.optimizer: Optional[tf.keras.optimizers.Optimizer] = None

    # ------------------------------------------------------------------
    # Kernel utilities
    # ------------------------------------------------------------------
    @property
    def ls_X(self):
        # softplus(raw) ≥ 0 ; add a floor so it never collapses
        return tf.nn.softplus(self._raw_ls_X) + tf.constant(1e-6, DTYPE)
    
    @property
    def ls_Y(self):
        return tf.nn.softplus(self._raw_ls_Y) + tf.constant(1e-6, DTYPE)
    def _kernel_X(self, X: tf.Tensor) -> tf.Tensor:
        k = tfp.math.psd_kernels.ExponentiatedQuadratic(length_scale=self.ls_X)
        return k.matrix(X, X)

    def _compute_Q(self, KXX: tf.Tensor, iAnn: tf.Tensor, idx: tf.Tensor) -> tf.Tensor:
        ones = tf.ones([KXX.shape[0], 1], dtype=DTYPE)
        K_aug = tf.concat([ones, KXX], axis=1)
        beta_slice = tf.concat([self.beta[0:1, :], tf.gather(self.beta, idx)], axis=0)
        q = tf.multiply(tf.linalg.matmul(K_aug, beta_slice), iAnn)
        self.q[idx.numpy() - 1, :] = q.numpy()
        return q

    def _kernel_Y(self, Y: tf.Tensor, iAnn: tf.Tensor, KXX: tf.Tensor, idx: tf.Tensor) -> tf.Tensor:
        k_y = tfp.math.psd_kernels.ExponentiatedQuadratic(length_scale=self.ls_Y)
        N, R = Y.shape
        Y_mask = tf.multiply(Y, iAnn)
        K_mu = tf.zeros([N, N], dtype=DTYPE)
        q = self._compute_Q(KXX, iAnn, idx)
        H = tf.eye(N, dtype=DTYPE) - tf.ones([N, N], dtype=DTYPE) / N
        for r in range(R):
            Q = tf.linalg.diag(q[:, r])
            mask_vec = tf.tensordot(iAnn[:, r], iAnn[:, r], 0)
            Ky = k_y.matrix(Y_mask[:, r : r + 1], Y_mask[:, r : r + 1])
            Ky_c = tf.linalg.matmul(H, tf.linalg.matmul(tf.multiply(Ky, H), mask_vec))
            K_mu += tf.linalg.matmul(Q, tf.linalg.matmul(Ky_c, Q))
        return K_mu

    def _loss(self, X, Y, iAnn, idx):
        N = tf.cast(tf.shape(X)[0], DTYPE)
        KXX = self._kernel_X(X)
        Kmu_c = self._kernel_Y(Y, iAnn, KXX, idx)
        H = tf.eye(N, dtype=DTYPE) - tf.ones([N, N], dtype=DTYPE) / N
        KXX_c = tf.linalg.matmul(H, tf.linalg.matmul(KXX, H))
        num = tf.linalg.trace(tf.matmul(Kmu_c, KXX_c, transpose_a=True))
        eps = tf.constant(1e-8, DTYPE) 
        den = tf.sqrt(
            tf.maximum(eps, tf.linalg.trace(tf.matmul(Kmu_c, Kmu_c, transpose_a=True)))
            * tf.maximum(eps, tf.linalg.trace(tf.matmul(KXX_c, KXX_c, transpose_a=True)))
        )
        return -num / den + self.l2 * tf.norm(self.beta, ord=2) + self.l1 * tf.norm(self.beta, ord=1)

    # ------------------------------------------------------------------
    # Fitting
    # ------------------------------------------------------------------
    def fit(self, X: np.ndarray, Y: np.ndarray):
        X = np.asarray(X, dtype=DTYPEN)
        Y = np.asarray(Y, dtype=DTYPEN)
        iAnn = get_iAnn(Y).astype(DTYPEN)
        N, R = Y.shape

        # create / resize β
        if self.beta is None or self.beta.shape != (N + 1, R):
            rng = np.random.default_rng(self.seed)
            self.beta = tf.Variable(rng.uniform(size=(N + 1, R)).astype(np.float32), dtype=DTYPE)

        # cache full data for inference / save
        self.X_full = tf.convert_to_tensor(X, dtype=DTYPE)
        self.Y_full = tf.convert_to_tensor(Y, dtype=DTYPE)
        self.iAnn_full = tf.convert_to_tensor(iAnn, dtype=DTYPE)
        self.idx_full = tf.range(1, N + 1, dtype=tf.int32)
        self.q = np.zeros((N, R), dtype=np.float32)

        ds = (
            tf.data.Dataset.from_tensor_slices((X, Y, iAnn, self.idx_full))
            .shuffle(buffer_size=min(10_000, N), seed=self.seed)
            .batch(self.batch_size)
            .repeat()
        )
        itr = iter(ds)
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.lr)

        best_loss = np.inf
        best_beta = self.beta.numpy()
        wait = wait_red = 0
        self.loss_.clear()

        for _ in range(self.epochs):
            batch_losses = []
            for _ in range(N // self.batch_size + 1):
                xb, yb, ib, idxb = next(itr)
                with tf.GradientTape() as tape:
                    loss = self._loss(xb, yb, ib, idxb)
                grads = tape.gradient(loss, [self.beta, self._raw_ls_X])
                self.optimizer.apply_gradients(zip(grads, [self.beta, self._raw_ls_X]))
                batch_losses.append(float(loss))
            m_loss = float(np.mean(batch_losses))
            self.loss_.append(m_loss)

            if m_loss < best_loss - self.min_delta:
                best_loss = m_loss
                best_beta = self.beta.numpy()
                wait = wait_red = 0
            else:
                wait += 1
                wait_red += 1
            if wait_red >= self.reduce_patience:
                new_lr = max(float(self.optimizer.learning_rate.numpy()) * self.lr_reduce_factor, self.min_lr)
                self.optimizer.learning_rate = new_lr
                wait_red = 0
            if wait >= self.patience:
                break
        self.beta.assign(best_beta)
        return self

    # ------------------------------------------------------------------
    # Transform
    # ------------------------------------------------------------------
    def transform(self, X, *_):
        if self.q is None:
            raise RuntimeError("Model must be fitted before calling transform().")
        return softmax(self.q**2, axis=1)

    # ------------------------------------------------------------------
    # Fast q² extrapolation
    # ------------------------------------------------------------------
    def get_new_q2(
        self,
        X_new: np.ndarray,
        *,
        k: Optional[int] = 200,
        rbf_gamma: Optional[float] = None,
        reuse_train_values: bool = False,
    ) -> np.ndarray:
        if self.X_full is None or self.iAnn_full is None or self.q is None:
            raise RuntimeError("Model not ready — fit or load first.")

        X_train = self.X_full.numpy()
        iAnn = self.iAnn_full.numpy()
        q_sq = self.q**2
        gamma = rbf_gamma or float(1.0 / (self.ls_X.numpy() ** 2))

        N, R = iAnn.shape
        M = X_new.shape[0]
        q2_new = np.zeros((M, R), dtype=np.float32)

        # reuse exact duplicates ------------------------------------------------
        if reuse_train_values:
            lookup = {tuple(row): idx for idx, row in enumerate(X_train)}
            in_train = np.array([tuple(r) in lookup for r in X_new])
            present = np.where(in_train)[0]
            if present.size:
                idx_train = np.fromiter((lookup[tuple(r)] for r in X_new[present]), int)
                q2_new[present] = q_sq[idx_train]
            X_query = X_new[~in_train]
            idx_query = np.where(~in_train)[0]
        else:
            X_query = X_new
            idx_query = np.arange(M)

        # compute RBF‑weighted averages ----------------------------------------
        for ann in range(R):
            mask = iAnn[:, ann].astype(bool)
            X_ann = X_train[mask]
            if X_ann.size == 0:
                continue
            q_ann_sq = q_sq[mask, ann]

            if k is not None and k < X_ann.shape[0]:
                nn = NearestNeighbors(n_neighbors=k).fit(X_ann)
                dist, ind = nn.kneighbors(X_query, return_distance=True)
                w = np.exp(-gamma * dist**2)
                num = (w * q_ann_sq[ind]).sum(axis=1)
            else:
                dist = cdist(X_query, X_ann, metric="sqeuclidean")
                w = np.exp(-gamma * dist)
                num = w @ q_ann_sq
            q2_new[idx_query, ann] = num
        return softmax(q2_new, axis=1)
    def plot_history(self):
      fig,ax = plt.subplots(1,figsize=(3,3))
      ax.plot(np.arange(len(self.loss_)),self.loss_)
      ax.set_xlabel('epochs')
      ax.set_ylabel('loss')
      plt.show()
      return
    # ------------------------------------------------------------------
    # Persistence helpers
    # ------------------------------------------------------------------
    def _pack_state(self) -> dict:
        return {
            "beta": None if self.beta is None else self.beta.numpy(),
            "_raw_ls_X": float(self._raw_ls_X.numpy()),
            "ls_Y": float(self.ls_Y.numpy()),
            "q": self.q,
            "X": None if self.X_full is None else self.X_full.numpy(),
            "Y": None if self.Y_full is None else self.Y_full.numpy(),
            "hyper": {
                "epochs": self.epochs,
                "batch_size": self.batch_size,
                "l1": self.l1,
                "l2": self.l2,
                "lr": self.lr,
                "patience": self.patience,
                "reduce_patience": self.reduce_patience,
                "lr_reduce_factor": self.lr_reduce_factor,
                "min_delta": self.min_delta,
                "min_lr": self.min_lr,
                "seed": self.seed,
            },
        }

    def save(self, path: str | Path) -> None:
        """Pickle the model (weights + data + hyper‑params) to *path*."""
        path = Path(path)
        with path.open("wb") as f:
            pickle.dump(self._pack_state(), f)
        print(f"✅ LCKA saved to '{path}'.")

    @classmethod
    def load(cls, path: str | Path) -> "LCKA":
        """Restore a model saved with :py:meth:`save`."""
        path = Path(path)
        with path.open("rb") as f:
            data = pickle.load(f)

        # Re-instantiate with the saved hyper-parameters
        model = cls(**data["hyper"])  # type: ignore[arg-type]

        # Restore trainable variables
        model._raw_ls_X.assign(data["_raw_ls_X"])
        model.ls_Y.assign(data["ls_Y"])
        if data["beta"] is not None:
            model.beta = tf.Variable(data["beta"], dtype=DTYPE)

        # Restore cached training data & derived tensors
        model.q = data["q"]
        if data["X"] is not None and data["Y"] is not None:
            model.X_full   = tf.convert_to_tensor(data["X"], dtype=DTYPE)
            model.Y_full   = tf.convert_to_tensor(data["Y"], dtype=DTYPE)
            model.iAnn_full = tf.convert_to_tensor(get_iAnn(data["Y"]), dtype=DTYPE)
            model.idx_full  = tf.range(1, data["X"].shape[0] + 1, dtype=tf.int32)

        return model

    def plot_kernels(self,X,Y,iAnn,annotators=None):
        """
        Parameters:
        -----------
        X: array-like of shape (n_samples, n_features)
            Input features.
        Y: array-like of shape (n_samples, n_annotators)
            Labels.
        iAnn: array-like of shape (n_samples, n_annotators)
            Annotation matrix.

        Returns:
        --------
        None
        """

        N, R = Y.shape
        idx = tf.range(1,N+1,dtype=tf.int64)
        KXX = self.ComputeKernel_X(X)
        K_mu = self.ComputeKernel_Y(Y,iAnn,KXX,idx)

        fig,ax = plt.subplots(nrows = 1,ncols=R+2,figsize=(4*(R+2),3))

        ax[0].imshow(KXX,vmin=0,vmax=1)
        ax[0].set_title(f"Target")
        ax[0].set_xticks([])
        ax[0].set_yticks([])

        ax[1].imshow(K_mu,vmin=0,vmax=1)
        ax[1].set_title(f"$K_\mu$")
        ax[1].set_xticks([])
        ax[1].set_yticks([])

        q = (self.q)**2
        q = 100*softmax(q,axis=1)
        print(q.shape)
        I = tf.eye(N, dtype=tf.dtypes.float64)
        ones = tf.ones([N,1], dtype=tf.dtypes.float64)
        H = I - tf.linalg.matmul(ones, ones, transpose_b=True)/N
        vmin_=0
        vmax_=KXX.numpy().max()
        if annotators is None:
            for r in range(R):
                Q = tf.linalg.diag(q[:,r])
                KYY_c = tf.linalg.matmul(H,tf.linalg.matmul(self.kernelY.matrix(Y[:,r:r+1],Y[:,r:r+1]),H))
                Kq = tf.linalg.matmul(Q,tf.linalg.matmul(KYY_c,Q))
                ax[r+2].imshow(Kq,vmin=0,vmax=1)
                ax[r+2].set_title(f"A{r+1}")
                ax[r+2].set_xticks([])
                ax[r+2].set_yticks([])
        else:
            for r, ann in enumerate(annotators):
                Q = tf.linalg.diag(q[:,r])
                KYY_c = tf.linalg.matmul(H,tf.linalg.matmul(self.kernelY.matrix(Y[:,r:r+1],Y[:,r:r+1]),H))
                Kq = tf.linalg.matmul(Q,tf.linalg.matmul(KYY_c,Q))
                ax[r+2].imshow(Kq,vmin=0,vmax=1)
                ax[r+2].set_title(f"A{ann}")
                ax[r+2].set_xticks([])
                ax[r+2].set_yticks([])

        cax = fig.add_axes([0.925, 0.15, 0.01, 0.7])
        norm = matplotlib.colors.Normalize(vmin=0,vmax=vmax_)
        sm = plt.cm.ScalarMappable(cmap=None, norm=norm)
        sm.set_array([])
        cbar = plt.colorbar(sm,cax=cax)
        plt.show()
        return q

class LCKAGPR:
    """Meta‑model: LCKA (weights) + per‑annotator sparse GPRs."""

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------
    def __init__(
        self,
        *,
        lcka_params: Optional[Dict] = None,
        gpr_params: Optional[Dict] = None,
    ) -> None:
        self.lcka = LCKA(**(lcka_params or {}))
        self.gpr = AnnotatorGPRTrainer(**(gpr_params or {}))

    # ------------------------------------------------------------------
    # Fit
    # ------------------------------------------------------------------
    def fit(self, X: np.ndarray, Y: np.ndarray) -> "LCKAGPR":
        """1. Fit LCKA → q²; 2. Fit one GPR per annotator."""
        print("📐 Fitting LCKA …")
        self.lcka.fit(X, Y)

        print("📈 Fitting per‑annotator GPRs …")
        self.gpr.train_gprs(X, Y)
        return self

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _weights(self, X_new: np.ndarray) -> np.ndarray:
        """Row‑normalised weights = softmax(q²(x))."""
        q2 = self.lcka.get_new_q2(X_new)
        return softmax(q2, axis=1)

    # ------------------------------------------------------------------
    # Predictions
    # ------------------------------------------------------------------
    def predict_annotators(self, X_new: np.ndarray):
        """Per‑annotator (mean, std) from the GPRs."""
        return self.gpr.predict(X_new)

    def predict(self, X_new: np.ndarray) -> np.ndarray:
        """Ground‑truth estimate as convex combo of annotator means."""
        means, _ = self.predict_annotators(X_new)
        w = self._weights(X_new)
        return np.sum(means * w, axis=1)

    # ------------------------------------------------------------------
    # Persistence (no version file)
    # ------------------------------------------------------------------
    def save(self, dir_path: str | Path) -> None:
        """Save LCKA and GPR trainer under *dir_path*."""
        dir_path = Path(dir_path)
        dir_path.mkdir(parents=True, exist_ok=True)
        self.lcka.save(dir_path / "lcka.pkl")
        self.gpr.save(dir_path / "gpr")
        print(f"✅ LCKAGPR saved to '{dir_path}'.")

    @classmethod
    def load(cls, dir_path: str | Path) -> "LCKAGPR":
        """Restore from :py:meth:`save`."""
        dir_path = Path(dir_path)
        instance = cls()
        instance.lcka = LCKA.load(dir_path / "lcka.pkl")
        instance.gpr = AnnotatorGPRTrainer()
        instance.gpr.load(dir_path / "gpr")
        return instance


