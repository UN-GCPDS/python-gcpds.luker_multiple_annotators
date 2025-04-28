from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
import matplotlib.pyplot as plt
import matplotlib
from umap import UMAP
from scipy.special import softmax
from scipy.spatial.distance import cdist
from sklearn.manifold import TSNE
import keras

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
    """Class for computing the LCKA (Linear Centered Kernel Alignment).

    This class provides methods to compute the LCKA loss, fit the model,
    transform the data, and plot the history and kernels.

    Parameters:
    -----------
    epochs: int, default=50
        Number of epochs for training.
    batch_size: int, default=100
        Batch size for training.
    ls_X: float, default=1e-1
        Length scale for the input features kernel.
    ls_Y: float, default=1e-13
        Length scale for the label kernel.
    iAnn: array-like of shape (n_samples, n_annotators), default=ones(2)
        Annotation matrix.
    l1: float, default=1e-3
        L1 regularization parameter.
    l2: float, default=1e-3
        L2 regularization parameter.
    lr: float, default=1e-3
        Learning rate for optimization.

    Methods:
    --------
    __init__(self, epochs=50, batch_size=100, ls_X=1e-1, ls_Y=1e-13, iAnn=np.ones(2), l1=1e-3, l2=1e-3, lr=1e-3)
    Constructor method for the LCKA class.

    ComputeKernel_X(self, X)
    Compute an exponetial quadratic kernel over the input features


    ComputeKernel_Y(self, Y, iAnn, KXX, idx)
    Compute a linear kernel over the labels.


    Compute_Q(self, KXX, iAnn, idx)
    Compute the Q matrix used for kernel.


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

    LCKA_loss(self,X,Y,iAnn,idx)
    LCKA_loss calculates the loss function for the LCKA model.


    plot_lckaQ(self, X, q, redlcka='umap', random_state=123, n_neighbors=10, cmap='Reds')
    Plots the transformed data using either UMAP or t-SNE and color-coded by q values.

    """
    def __init__(self,epochs=50,batch_size=100,ls_X=1e-1,ls_Y=1e-13,iAnn=2,l1=1e-3,l2=1e-3,lr=1e-3):
        self.epochs = epochs
        self.batch_size = batch_size
        if isinstance(iAnn, int):
            iAnn = np.ones((iAnn, iAnn))
        self.iAnn = iAnn
        self.N, self.R = self.iAnn.shape #annotators
        self.beta = tf.Variable(tf.random.uniform(shape=(self.N+1, self.R),
                                                  dtype=tf.float64))
        self.ls_X = tf.Variable(ls_X, dtype=tf.float64)
        self.ls_Y = tf.Variable(ls_Y, dtype=tf.float64)
        self.idx = tf.range(1,self.N+1,dtype=tf.int64)
        self.q = np.zeros((self.N, self.R))
        self.l1 = l1
        self.l2 = l2
        self.lr = lr


    def ComputeKernel_X(self,X):
        """
        Computes an exponetial quadratic kernel over the input features

        Parameters:
        -----------
        X: array-like de forma (n_samples, n_features)
            Características de entrada.
        Returns:
        --------
        Matriz de kernel.
        """
        self.kernelX = tfp.math.psd_kernels.ExponentiatedQuadratic(length_scale=self.ls_X,
                                                  name='ExponentiatedQuadratic')
        return self.kernelX.matrix(X,X)


    def ComputeKernel_Y(self,Y,iAnn,KXX,idx):
        """
        Computes a linear kernel over the labels

        Parameters:
        -----------
        Y: array-like de forma (n_samples, n_annotators)
            Etiquetas.
        iAnn: array-like de forma (n_samples, n_annotators)
            Matriz de anotación.
        KXX: array-like de forma (n_samples, n_samples)
            Kernel sobre las características de entrada.
        idx: array-like de forma (n_samples,)
            Índices de las muestras.
        Returns:
        --------
        Matriz de kernel.
        """
        #kernel = tfp.math.psd_kernels.Linear(bias_variance=None,
         #                                    slope_variance=None, shift=None,
         #                                    feature_ndims=1,validate_args=False,
         #                                    name='Linear')
        self.kernelY = tfp.math.psd_kernels.ExponentiatedQuadratic(length_scale=self.ls_Y,
                                                  name='ExponentiatedQuadratic')
        N, R = Y.shape
        Y = tf.multiply(Y,iAnn)
        K_mu = tf.zeros([N,N], dtype=tf.dtypes.float64)
        q = self.Compute_Q(KXX,iAnn,idx)
        N = KXX.shape[0]
        I = tf.eye(N, dtype=tf.dtypes.float64)
        ones = tf.ones([N,1], dtype=tf.dtypes.float64)
        H = I - tf.linalg.matmul(ones, ones, transpose_b=True)/N
        for r in range(R):
            Q = tf.linalg.diag(q[:,r])
            iann = tf.tensordot(iAnn[:,r], iAnn[:,r], 0)
            # KYY_c = tf.linalg.matmul(H,tf.linalg.matmul(self.kernelY.matrix(Y[:,r:r+1],Y[:,r:r+1]),H))
            KYY_c = tf.linalg.matmul(H,tf.linalg.matmul(tf.multiply(self.kernelY.matrix(Y[:,r:r+1],Y[:,r:r+1]),H), iann))
            Kq = tf.linalg.matmul(Q,tf.linalg.matmul(KYY_c,Q))
            K_mu = tf.math.add(K_mu,Kq)
        return K_mu

    def Compute_Q(self,KXX,iAnn,idx):
        """
        Parameters:
        -----------
        KXX: array-like de forma (n_samples, n_samples)
            Kernel sobre las características de entrada.
        iAnn: array-like de forma (n_samples, n_annotators)
            Matriz de anotación.
        idx: array-like de forma (n_samples,)
            Índices de las muestras.
        Returns:
        --------
        Matriz Q.
        """
        ones = tf.ones([KXX.shape[0],1], dtype=tf.dtypes.float64)
        KXX = tf.concat([ones, KXX], 1)
        beta = tf.gather(self.beta,idx)
        beta = tf.concat([self.beta[0:1,:], beta], 0)
        q = tf.multiply(tf.linalg.matmul(KXX, beta), iAnn)
        self.q[idx.numpy()-1,:] = q.numpy()
        return q


    def LCKA_loss(self,X,Y,iAnn,idx):
        """
        Custom loss function based on the centered kernel alignment framework

        Parameters:
        -----------
        X: array-like, shape (n_samples, n_features)
            Input data.
        Y: array-like, shape (n_samples, n_annotators)
            Annotation data.
        iAnn: array-like, shape (n_samples, n_annotators)
            Annotation matrix.
        idx: array-like, shape (n_samples,)
            Indices.

        Returns:
        --------
        Loss value.
        """
        # N = X.shape[0]
        N = tf.cast(tf.shape(X)[0], tf.float64)
        KXX = self.ComputeKernel_X(X)
        Kmu_c = self.ComputeKernel_Y(Y,iAnn,KXX,idx)
        I = tf.eye(N, dtype=tf.dtypes.float64)
        ones = tf.ones([N,1], dtype=tf.dtypes.float64)
        H = I - tf.linalg.matmul(ones, ones, transpose_b=True)/N
        KXX_c = tf.linalg.matmul(H,tf.linalg.matmul(KXX,H))
        num = tf.linalg.trace(tf.matmul(Kmu_c,KXX_c,transpose_a=True))
        den1 = tf.sqrt(tf.linalg.trace(tf.matmul(Kmu_c,Kmu_c,transpose_a=True)))
        den2 = tf.sqrt(tf.linalg.trace(tf.matmul(KXX_c,KXX_c,transpose_a=True)))
        return -num/(den1*den2) + self.l2*tf.norm(self.beta,ord=2)+self.l1*tf.norm(self.beta,ord=1)

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
        N_total = X.shape[0]

        self.X_full = tf.convert_to_tensor(X, dtype=tf.float64)
        self.Y_full = tf.convert_to_tensor(Y, dtype=tf.float64)
        self.iAnn_full = tf.convert_to_tensor(self.iAnn, dtype=tf.float64)
        self.idx_full = tf.range(1, N_total+1, dtype=tf.int64)

        train_data = tf.data.Dataset.from_tensor_slices((X, Y, self.iAnn, self.idx))
        train_data = train_data.shuffle(buffer_size=100).batch(batch_size).repeat(5)

        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(initial_learning_rate=self.lr,
                                decay_steps=50,decay_rate=0.9)
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
        self.loss_ = np.zeros(self.epochs)

        # -------------------------------
        # Early stopping variables
        best_loss = np.inf
        patience = 30         # you can tune this
        wait = 0
        best_beta = None
        # -------------------------------        
        for epoch in range(self.epochs):
            if epoch % 10 == 0:
                print(f"Start of epoch {epoch}")

            epoch_losses = []
            # Iterate over the batches of the dataset.
            for step, (x_batch_train, y_batch_train, iAnn_batch, idx_batch) in enumerate(train_data):
                current_batch_size = tf.shape(x_batch_train)[0]

                if current_batch_size < batch_size:
                    num_missing = batch_size - current_batch_size

                    random_idx = tf.random.uniform(shape=(num_missing,), minval=0, maxval=N_total, dtype=tf.int32)
                    x_extra = tf.gather(self.X_full, random_idx)
                    y_extra = tf.gather(self.Y_full, random_idx)
                    iAnn_extra = tf.gather(self.iAnn_full, random_idx)
                    idx_extra = tf.gather(self.idx_full, random_idx)

                    x_batch_train = tf.concat([x_batch_train, x_extra], axis=0)
                    y_batch_train = tf.concat([y_batch_train, y_extra], axis=0)
                    iAnn_batch = tf.concat([iAnn_batch, iAnn_extra], axis=0)
                    idx_batch = tf.concat([idx_batch, idx_extra], axis=0)

                with tf.GradientTape() as tape:
                    loss = self.LCKA_loss(x_batch_train, y_batch_train,
                                         iAnn_batch,idx_batch)
                grads = tape.gradient(loss, [self.beta, self.ls_X])
                self.optimizer.apply_gradients(zip(grads, [self.beta, self.ls_X]))

                epoch_losses.append(loss.numpy())

                if step % 50 == 0:
                     print(f"step {step}: mean loss {loss.numpy().round(4)} ls {self.ls_X.numpy().round(2)} lr {(self.optimizer.learning_rate.numpy()).round(5)}" )
            
            mean_loss = np.mean(epoch_losses)
            self.loss_[epoch] = mean_loss

            # Early stopping check
            if mean_loss < best_loss - 1e-4:  # small improvement threshold
                best_loss = mean_loss
                best_beta = self.beta.numpy()  # save best weights
                wait = 0
            else:
                wait += 1
                if wait >= patience:
                    print(f"Early stopping at epoch {epoch}")
                    break

        # Restore best weights if early stopping triggered
        if best_beta is not None:
            self.beta.assign(best_beta)

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
        q = self.q**2
        #N = q.shape[0]
        q = softmax(q,axis=1)#np.sum(q,1).reshape(N,1)

        return q

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


    def plot_lckaQ(self,X,q, redlcka='umap',random_state=123,n_neighbors=10,cmap='Reds',annotators=None):
        """
        Parameters:
        -----------
        X: array-like of shape (n_samples, n_features)
            Input features.
        q: array-like of shape (n_samples, n_annotators)
            Transformed data.
        redlcka: {'umap', 'tsne'}, default='umap'
            Reduction method to use.
        random_state: int, default=123
            Random state for reproducibility.
        n_neighbors: int, default=10
            Number of neighbors to consider for UMAP or t-SNE.
        cmap: str, default='Reds'
            Colormap to use.

        Returns:
        --------
        None
        """
        if redlcka == 'umap':
            red_ = UMAP(n_components = 2, n_neighbors = n_neighbors,min_dist =0.9,random_state=random_state)
        else:
            red_ = TSNE(n_components = 2, perplexity = n_neighbors, random_state=random_state)
        X_ = MinMaxScaler().fit_transform(X)
        Z = red_.fit_transform(X_)
        R = q.shape[1]

        fig,ax = plt.subplots(nrows = 1,ncols=R,figsize=(4*(R),3))
        if annotators is None:
            for r in range(R):
                ax[r].scatter(Z[:,0],Z[:,1],c=q[:,r],vmin=q.ravel().min(),vmax=q.ravel().max(),cmap=cmap)
                ax[r].set_title(f"A{r+1} - qm = {q[:,r].ravel().mean().round(2)}")
                ax[r].set_xticks([])
                ax[r].set_yticks([])
        else:
            for r, ann in enumerate(annotators):
                ax[r].scatter(Z[:,0],Z[:,1],c=q[:,r],vmin=q.ravel().min(),vmax=q.ravel().max(),cmap=cmap)
                ax[r].set_title(f"A{ann} - qmean = {q[:,r].ravel().mean().round(2)}\n qmed = {np.median(q[:,r].ravel()).round(2)}")
                ax[r].set_xticks([])
                ax[r].set_yticks([])

        cax = fig.add_axes([0.925, 0.15, 0.01, 0.7])
        norm = matplotlib.colors.Normalize(vmin=q.ravel().min(),vmax=q.ravel().max())
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = plt.colorbar(sm,cax=cax)
        plt.show()
        return

    def get_new_q2(self, X, iAnn, X_new):
        R = iAnn.shape[1]
        q2_new = np.zeros((len(X_new), R))
        for ann in range(R):
            # subset of X with labels from annotator 1
            mask = iAnn[:,ann].astype('bool')
            X_ann = X[mask]
            # samples of X_new which are in X used for training the model
            idx = np.isin(X_new, X_ann).all(axis=1)
            X_1 = X_new[idx]
            X_2 = X_new[~idx]
            # for each sample in X_1 find its index in the original X, to find
            # its q later
            idx_in_q = np.array([], dtype=np.int32)
            for x in X_1:
                idx_in_q = np.concatenate((idx_in_q, np.where((X == x).all(axis=1))[0]))
            q2_new[idx, ann] = self.q[idx_in_q, ann]**2
            calculated_q2 = np.array([])
            for x in X_2:
                coefs = np.exp(-cdist(x.reshape(1, -1), X_ann))
                calculated_q2 = np.append(calculated_q2, (coefs*self.q[mask, ann]**2).sum()/coefs.sum())
            q2_new[~idx, ann] = calculated_q2
        return q2_new


class MA_GCCE():
 #Constructor __init__. Special method: identified by a double underscore at either side of their name
 #work in the background
 # initialize data members to the object. lets the class initialize the object’s attributes and serves no other purpose.
    def __init__(self,epochs=100,batch_size=30,R=5, K=2, dropout=0.5, learning_rate=1e-3,optimizer='Adam',
                  l1_param=0, validation_split=0.3, verbose=1, q = 0.1):
        self.epochs=epochs
        self.dropout=dropout
        self.batch_size = batch_size
        self.learning_rate=learning_rate
        self.l1_param=l1_param
        self.l2_param=l1_param
        self.validation_split = validation_split
        self.verbose = verbose
        self.optimizer = optimizer
        self.R=R
        self.K=K
        self.q=q
    @keras.saving.register_keras_serializable()
    def GCCE_MA_loss(self, y_true, y_pred):
        # print(y_true,y_pred)
       # q = 0.1
        pred = y_pred[:, self.R:]
        pred = tf.clip_by_value(pred, clip_value_min=1e-9, clip_value_max=1)
        ann_ = y_pred[:, :self.R]
        # ann_ = tf.clip_by_value(ann_, clip_value_min=1e-9, clip_value_max=1-1e-9)
        Y_true = tf.one_hot(tf.cast(y_true, dtype=tf.int32), depth=self.K, axis=1)
        Y_hat = tf.repeat(tf.expand_dims(pred,-1), self.R, axis = -1)

        p_gcce = Y_true*(1 - Y_hat**self.q)/self.q
        temp1 = ann_*tf.math.reduce_sum(p_gcce, axis=1)
        temp2 = (1 - ann_)*(1-(1/self.K)**self.q)/self.q*tf.reduce_sum(Y_true,axis=1)
        return tf.math.reduce_sum((temp1 + temp2))

    def fit(self, X, Y):
        #input X numpy array first dimension samples (N)x features (P)
        #input Y numpy array vector len = samples (N) x  annotators (R)
        if self.optimizer == "Adam":
            opt = tf.keras.optimizers.Adam(learning_rate=self.learning_rate, clipnorm=1.0)
        elif self.optimizer == "SGD":
            opt = tf.keras.optimizers.SGD(learning_rate=self.learning_rate, clipnorm=1.0)
        else:
            opt=self.optimizer

        inputs = tf.keras.layers.Input(shape=(5,), name='entrada')
        x = tf.keras.layers.Dense(64, activation='relu')(inputs)
        x = tf.keras.layers.Dense(64, activation='relu')(x)
        x = tf.keras.layers.Dense(32, activation='relu')(x)

        output_R = tf.keras.layers.Dense(self.R,activation="sigmoid",
                                    kernel_regularizer=tf.keras.regularizers.l1_l2(l1=self.l1_param,l2=self.l2_param), name= 'out_R_GCCE' )(x)

        output_K = tf.keras.layers.Dense(self.K,activation="softmax",
                                    kernel_regularizer=tf.keras.regularizers.l1_l2(l1=self.l1_param,l2=self.l2_param), name= 'out_K_GCCE')(x)

        output = tf.keras.layers.concatenate([output_R, output_K])
        self.model = tf.keras.Model(inputs=inputs, outputs=output)
        self.model.compile(loss=self.GCCE_MA_loss, optimizer=opt)

        self.history = self.model.fit(X, Y, epochs=self.epochs, validation_split=self.validation_split,   #
                                      batch_size=self.batch_size,verbose=self.verbose)

        return self

    def my_model(self):
        return self.model

    def predict(self, X, *_):
       #input X numpy array first dimension samples (N)x features (P)
      return  self.model.predict(X)

    def fit_predict(self,X,y):
        #input X numpy array first dimension samples (N)x features (P)
        #input Y numpy array vector len = samples (N) x  annotators (R)
        self.fit(X,y)
        return self.predict(X)

     #graphics
    def plot_history(self):
        pd.DataFrame(self.history.history).plot(figsize=(8, 5))
        plt.grid(True)
        #plt.gca().set_ylim(0, 1)
        #save_fig("keras_learning_curves_plot")
        plt.show()
        return

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)

        return self

    def get_config(self, deep=True):
        return { 'l1_param':self.l1_param, 'dropout':self.dropout, 'optimizer':self.optimizer,
                'learning_rate':self.learning_rate, 'batch_size':self.batch_size,
                'epochs':self.epochs, 'verbose':self.verbose, 'validation_split':self.validation_split,
                'R':self.R, 'K':self.K, 'q':self.q
                }

