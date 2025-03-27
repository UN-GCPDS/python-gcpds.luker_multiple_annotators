import torch
import torch.nn as nn
from torch.nn import Linear
from pytorch_tabnet.tab_network import TabNetEncoder, FeatTransformer, initialize_non_glu

class TabNetDecoderSingleInput(nn.Module):
    def __init__(
        self,
        input_dim,
        n_d=8,
        n_steps=3,
        n_independent=1,
        n_shared=1,
        virtual_batch_size=128,
        momentum=0.02,
    ):
        """
        TabNet-based decoder used to reconstruct input features from latent representation.

        Parameters
        ----------
        input_dim : int
            Number of original input features.
        n_d : int
            Dimension of the decision step output (used as input here).
        n_steps : int
            Number of sequential attention steps.
        n_independent : int
            Number of independent GLU layers in each transformer block.
        n_shared : int
            Number of shared GLU layers across steps.
        virtual_batch_size : int
            Size for Ghost Batch Normalization.
        momentum : float
            Momentum used in batch normalization.
        """
        super().__init__()
        self.input_dim = input_dim
        self.n_d = n_d
        self.n_steps = n_steps
        self.feat_transformers = nn.ModuleList()

        if n_shared > 0:
            shared_feat_transform = nn.ModuleList([Linear(n_d, 2 * n_d, bias=False) for _ in range(n_shared)])
        else:
            shared_feat_transform = None

        for _ in range(n_steps):
            transformer = FeatTransformer(
                n_d,
                n_d,
                shared_feat_transform,
                n_glu_independent=n_independent,
                virtual_batch_size=virtual_batch_size,
                momentum=momentum,
            )
            self.feat_transformers.append(transformer)

        self.reconstruction_layer = Linear(n_d, input_dim, bias=False)
        initialize_non_glu(self.reconstruction_layer, n_d, input_dim)

    def forward(self, steps_output):
        """
        Forward pass through the decoder.

        Parameters
        ----------
        steps_output : Tensor
            Input tensor from latent space (N, n_d).

        Returns
        -------
        Tensor
            Reconstructed input features (N, input_dim).
        """
        res = 0
        for step_nb in range(self.n_steps):
            x = self.feat_transformers[step_nb](steps_output)
            res = torch.add(res, x)
        res = self.reconstruction_layer(res)
        return res


class TabNetVAE(nn.Module):
    def __init__(self, input_dim, latent_dim, output_dim, n_d=16, n_a=16, n_steps=3, gamma=1.3, hidden_sizes=[8]):
        """
        Variational Autoencoder based on TabNet for tabular regression tasks.

        Parameters
        ----------
        input_dim : int
            Number of input features.
        latent_dim : int
            Dimension of the latent space.
        output_dim : int
            Dimension of the regression output.
        n_d, n_a, n_steps, gamma : TabNet parameters.
        hidden_sizes : list of int
            Sizes of the hidden layers in the regression head.
        """
        super().__init__()

        self.encoder = TabNetEncoder(
            input_dim=input_dim, gamma=gamma,
            output_dim=n_d,  # Not used internally by TabNet
            n_d=n_d, n_a=n_a, n_steps=n_steps
        )

        self.latent_projection = nn.Linear(n_d, latent_dim * 2)
        self.decoder = TabNetDecoderSingleInput(input_dim=input_dim, n_d=latent_dim, n_steps=n_steps)
        self.latent_dim = latent_dim

        layers = []
        in_dim = latent_dim
        for h in hidden_sizes:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.ReLU())
            in_dim = h
        layers.append(nn.Linear(in_dim, output_dim))
        self.regression_head = nn.Sequential(*layers)

    def reparameterize(self, mu, logvar):
        """
        Apply the reparameterization trick: z = mu + sigma * epsilon
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        """
        Forward pass through the VAE model.

        Parameters
        ----------
        x : Tensor
            Input features (N, input_dim).

        Returns
        -------
        reconstructed_x : Tensor
            Reconstructed input.
        regression_output : Tensor
            Predicted output.
        mu : Tensor
            Mean of latent distribution.
        logvar : Tensor
            Log-variance of latent distribution.
        M_loss : float
            Sparsity loss from TabNet encoder.
        """
        encoded_output, M_loss = self.encoder(x)
        latent_params = self.latent_projection(torch.sum(torch.stack(encoded_output), dim=0))
        mu, logvar = latent_params[:, :self.latent_dim], latent_params[:, self.latent_dim:]
        z = self.reparameterize(mu, logvar)
        reconstructed_x = self.decoder(z)
        regression_output = self.regression_head(z)
        return reconstructed_x, regression_output, mu, logvar, M_loss

    def loss_function(self, x, reconstructed_x, target_regression, regression_output, mu, logvar, M_loss, **weights):
        """
        Compute the total loss of the VAE: 
        reconstruction + KL divergence + regression + sparsity loss.

        Parameters
        ----------
        x : Tensor
            Original input.
        reconstructed_x : Tensor
            Reconstructed input.
        target_regression : Tensor
            Ground truth for regression.
        regression_output : Tensor
            Model regression prediction.
        mu, logvar : Tensor
            Latent parameters.
        M_loss : float
            Attention sparsity loss.
        weights : dict
            Loss weights: 'recon', 'kl', 'reg', 'sparse'.

        Returns
        -------
        Tuple of total loss and its components.
        """
        recon_loss = nn.MSELoss()(reconstructed_x, x)
        kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        reg_loss = nn.MSELoss()(regression_output, target_regression)
        w1, w2, w3, w4 = weights['recon'], weights['kl'], weights['reg'], weights['sparse']
        total_loss = w1*recon_loss + w2*kl_div + w3*reg_loss + w4*M_loss
        return total_loss, recon_loss, kl_div, reg_loss, M_loss
