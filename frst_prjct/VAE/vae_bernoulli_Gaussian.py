# Code for DTU course 02460 (Advanced Machine Learning Spring) by Jes Frellsen, 2024
# Version 1.2 (2024-02-06)
# Inspiration is taken from:
# - https://github.com/jmtomczak/intro_dgm/blob/main/vaes/vae_example.ipynb
# - https://github.com/kampta/pytorch-distributions/blob/master/gaussian_vae.py

import torch
import torch.nn as nn
import torch.distributions as td
import torch.utils.data
from torch.nn import functional as F
from tqdm import tqdm
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
#VampPrior
class VampPrior(nn.Module):
    """
    A VampPrior (Variational Mixture of Posteriors) using K pseudo-inputs each of shape (28,28).
    """
    def __init__(self, encoder, latent_dim, K=450):
        super().__init__()
        self.encoder = encoder
        self.latent_dim = latent_dim
        self.K = K
        self.pseudo_inputs = nn.Parameter(torch.randn(K, 28, 28))  # Shape (K,28,28)

    def forward(self):
        qz_x = self.encoder(self.pseudo_inputs)  # Get a batch of K distributions output mean and std
        mean, std = qz_x.base_dist.loc, qz_x.base_dist.scale  # (K, latent_dim)

        mix_weight = torch.zeros(self.K, device=mean.device)
        mix_cat = td.Categorical(logits=mix_weight) # every distribution has the same weight
        comps = td.Independent(td.Normal(loc=mean, scale=std), 1) 
        return td.MixtureSameFamily(mix_cat, comps)

class GaussianPrior(nn.Module):#p(z)
    def __init__(self, M):
        """
        Define a Gaussian prior distribution with zero mean and unit variance.

                Parameters:
        M: [int] 
           Dimension of the latent space.
        """
        super(GaussianPrior, self).__init__()
        self.M = M
        self.mean = nn.Parameter(torch.zeros(self.M), requires_grad=False)
        self.std = nn.Parameter(torch.ones(self.M), requires_grad=False)

    def forward(self):
        """
        Return the prior distribution.

        Returns:
        prior: [torch.distributions.Distribution]
        """
        #td.normal(loc=self.mean, scale=self.std) is an M-dimensional normal distribution.
        # means each latent dimension is independent, so the final distribution has M dimensions instead of M independent 1D distributions.
        return td.Independent(td.Normal(loc=self.mean, scale=self.std), 1) #1D dimension is independent

class MixtureGaussianPrior(nn.Module):
    def __init__(self, M, num_components=450):
        """
        A Mixture of Gaussians prior with 'num_components' components,
        each of dimension 'latent_dim'.
        """
        super().__init__()
        self.M = M
        self.num_components = num_components

        # weight
        self.weight = nn.Parameter(torch.zeros(self.num_components))
        # Means
        self.mean = nn.Parameter(torch.randn(self.num_components, self.M))
        # std
        self.std = nn.Parameter(torch.ones(self.num_components, self.M))

    def forward(self):
        mixing_weight = td.Categorical(logits=self.weight)  # pi_k
        # Each component is an Independent Normal distribution
        component_dist = td.Independent(td.Normal(loc=self.mean, scale=self.std),1)
        mixture = td.MixtureSameFamily(mixing_weight, component_dist)
        return mixture


class GaussianEncoder(nn.Module):
    def __init__(self, encoder_net):
        """
        Define a Gaussian encoder distribution based on a given encoder network.

        Parameters:
        encoder_net: [torch.nn.Module]             
           The encoder network that takes as a tensor of dim `(batch_size,
           feature_dim1, feature_dim2)` and output a tensor of dimension
           `(batch_size, 2M)`, where M is the dimension of the latent space.
        """
        super(GaussianEncoder, self).__init__()
        self.encoder_net = encoder_net 

    def forward(self, x):
        """
        Given a batch of data, return a Gaussian distribution over the latent space.

        Parameters:
        x: [torch.Tensor] 
           A tensor of dimension `(batch_size, feature_dim1, feature_dim2)`
        """
        # reparameterization trick  P(x|z) = P(x|u(x)+sigma(x)*emma)
        # td.normal(loc=self.mean, scale=self.std) is an M-dimensional normal distribution.

        mean, std = torch.chunk(self.encoder_net(x), 2, dim=-1) #split the tensor into two parts one mean one std self.encoder_net(x) is (batch size x 2M)
        # means each latent dimension is independent, so the final distribution has M dimensions instead of M independent 1D distributions
        return td.Independent(td.Normal(loc=mean, scale=torch.exp(std)), 1) #1D dimension is independent


class BernoulliDecoder(nn.Module):
    def __init__(self, decoder_net):
        """
        Define a Bernoulli decoder distribution based on a given decoder network.

        Parameters: 
        encoder_net: [torch.nn.Module]             
           The decoder network that takes as a tensor of dim `(batch_size, M) as
           input, where M is the dimension of the latent space, and outputs a
           tensor of dimension (batch_size, feature_dim1, feature_dim2).
        """
        super(BernoulliDecoder, self).__init__()
        self.decoder_net = decoder_net
        self.std = nn.Parameter(torch.ones(28, 28)*0.5, requires_grad=True)

    def forward(self, z):
        """
        Given a batch of latent variables, return a Bernoulli distribution over the data space.

        Parameters:
        z: [torch.Tensor] 
           A tensor of dimension `(batch_size, M)`, where M is the dimension of the latent space.
        """
        logits = self.decoder_net(z) #input(batch size x M) output(batch size x 28 x 28)
        #every pixel is sample from bernoulli distribution, Let the last 2 dimensions (H, W) be considered as a whole, instead as 28*28 pixels
        return td.Independent(td.Bernoulli(logits=logits), 2)  # change to categorical distribution
    
class VAE(nn.Module):
    """
    Define a Variational Autoencoder (VAE) model.
    """
    def __init__(self, prior, decoder, encoder):
        """
        Parameters:
        prior: [torch.nn.Module] 
           The prior distribution over the latent space.
        decoder: [torch.nn.Module]
              The decoder distribution over the data space.
        encoder: [torch.nn.Module]
                The encoder distribution over the latent space.
        """
            
        super(VAE, self).__init__()
        self.prior = prior
        self.decoder = decoder
        self.encoder = encoder

    def elbo(self, x):
        """
        Compute the ELBO for the given batch of data.

        Parameters:
        x: [torch.Tensor] 
           A tensor of dimension `(batch_size, feature_dim1, feature_dim2, ...)`
           n_samples: [int]
           Number of samples to use for the Monte Carlo estimate of the ELBO.
        """
        q = self.encoder(x)
        z = q.rsample()
        #self.decoder(z).log_prob(x) is the log likelihood z = P(x|z) .log_prob(x) make it a scalar
        #td.kl_divergence(q, self.prior()) is the KL divergence
        #self.decoder(z) is (batch size x 28 x 28),self.decoder(z).log_prob(x) is (batch size,)
        #td.kl_divergence(q, self.prior()) is (batch size,)
        # elbo = torch.mean(self.decoder(z).log_prob(x) - td.kl_divergence(q, self.prior()), dim=0) 
        q_z_x = q.log_prob(z)
        entropy = torch.mean(q_z_x)
        p_z = self.prior().log_prob(z)
        cross_entropy = -torch.mean(p_z) 
        kl_div =  entropy + cross_entropy
        elbo = torch.mean(self.decoder(z).log_prob(x)) - kl_div
        return elbo

    def sample(self, n_samples=1):
        """
        Sample from the model.
        
        Parameters:
        n_samples: [int]
           Number of samples to generate.
        """
        z = self.prior().sample(torch.Size([n_samples]))
        return self.decoder(z).sample()
    
    def forward(self, x):
        """
        Compute the negative ELBO for the given batch of data.

        Parameters:
        x: [torch.Tensor] 
           A tensor of dimension `(batch_size, feature_dim1, feature_dim2)`
        """
        return -self.elbo(x)


def train(model, optimizer, data_loader, epochs, device):
    """
    Train a VAE model.

    Parameters:
    model: [VAE]
       The VAE model to train.
    optimizer: [torch.optim.Optimizer]
         The optimizer to use for training.
    data_loader: [torch.utils.data.DataLoader]
            The data loader to use for training.
    epochs: [int]
        Number of epochs to train for.
    device: [torch.device]
        The device to use for training.
    """
    model.train()

    total_steps = len(data_loader)*epochs
    progress_bar = tqdm(range(total_steps), desc="Training")

    for epoch in range(epochs):
        data_iter = iter(data_loader)
        for x in data_iter:
            x = x[0].to(device)
            optimizer.zero_grad()
            loss = model(x)
            loss.backward()
            optimizer.step()

            # Update progress bar
            progress_bar.set_postfix(loss=f"⠀{loss.item():12.4f}", epoch=f"{epoch+1}/{epochs}")
            progress_bar.update()


def test(model, test_loader, device, latent_dim, save_figure=True):
    
    model.eval()
    total_elbo = 0.0
    total_count = 0

    posterior_samples = []
    labels = []

    # Compute ELBO and gather posterior samples ----
    with torch.no_grad():
        for x_batch, y_batch in test_loader:
            x_batch = x_batch.to(device)
            elbo_batch = model.elbo(x_batch) * x_batch.size(0)
            total_elbo += elbo_batch.item()
            total_count += x_batch.size(0)

            qz_x = model.encoder(x_batch)
            z_sample = qz_x.rsample()  # (batch_size, latent_dim)
            posterior_samples.append(z_sample.cpu())
            labels.append(y_batch)

    avg_elbo = total_elbo / total_count
    print(f"Average ELBO: {avg_elbo:.4f}")

    # Concatenate and randomly keep 500 posterior points ----
    posterior_samples = torch.cat(posterior_samples, dim=0).numpy()
    labels = torch.cat(labels, dim=0).numpy()

    n_show = 700  # Limit to 500 points for visualization
    total_points = posterior_samples.shape[0]
    if total_points > n_show:
        idx = np.random.choice(total_points, size=n_show, replace=False)
        posterior_samples = posterior_samples[idx]
        labels = labels[idx]

    # If latent_dim > 2, use first 2 dimensions instead of PCA ----
    if latent_dim > 2:
        posterior_2d = posterior_samples[:, :2]  # Use first two dimensions directly
    else:
        posterior_2d = posterior_samples

    # Compute the standard Gaussian prior density (N(0, I)) ----
    z1 = np.linspace(-3, 3, 200)  # X-axis range
    z2 = np.linspace(-3, 3, 200)  # Y-axis range
    Z1, Z2 = np.meshgrid(z1, z2)  # Create a 2D grid

    # Compute probability density function for standard normal N(0, I)
    P = (1 / (2 * np.pi)) * np.exp(-0.5 * (Z1**2 + Z2**2))

    # Plot prior as true Gaussian density + posterior as scatter ----
    plt.figure(figsize=(8, 6))

    # Standard normal density background
    plt.contourf(Z1, Z2, P, levels=30, cmap="hot_r", alpha=0.7)

    # Scatter plot for posterior samples
    scatter_plot = plt.scatter(
        posterior_2d[:, 0],
        posterior_2d[:, 1],
        c=labels,
        cmap="tab10",
        alpha=0.7,
        s=6,
        edgecolor="k",  # Black edge to make points visible
        linewidth=0.4,
        label="Posterior Samples"
    )

    # Colorbar for posterior labels
    cbar = plt.colorbar(scatter_plot)
    cbar.set_label("Class Label")

    # Equal axis scaling for a circular Gaussian
    plt.axis("equal")
    plt.xlabel("z1")
    plt.ylabel("z2")
    plt.title("Prior (Standard Normal) + Posterior (Scatter)")
    plt.grid(False)
    plt.legend()

    # Save the figure
    if save_figure:
        plt.savefig("prior_posterior_Gaussian.png", bbox_inches="tight", dpi=150)

    plt.show()

    return avg_elbo


if __name__ == "__main__":
    from torchvision import datasets, transforms
    from torchvision.utils import save_image, make_grid
    import glob
    import sys
    import argparse    # Arguments
    sys.argv = [
        "vae_bernoulli_Gaussian.py", 
        "--mode", "test", 
        "--model", "VAE/model_Gaussian_FNN_20.pt", 
        "--samples", "model_Gaussian.png", 
        "--device", "cuda", 
        "--batch-size", "128", 
        "--epochs", "50", 
        "--latent-dim", "20",
        "--network","FNN"
    ]
    # Parse arguments
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'sample', 'test'], help='what to do when running the script (default: %(default)s)')
    parser.add_argument('--model', type=str, default='model.pt', help='file to save model to or load model from (default: %(default)s)')
    parser.add_argument('--samples', type=str, default='samples.png', help='file to save samples in (default: %(default)s)')
    parser.add_argument('--device', type=str, default='cpu', choices=['cpu', 'cuda', 'mps'], help='torch device (default: %(default)s)')
    parser.add_argument('--batch-size', type=int, default=32, metavar='N', help='batch size for training (default: %(default)s)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N', help='number of epochs to train (default: %(default)s)')
    parser.add_argument('--latent-dim', type=int, default=32, metavar='N', help='dimension of latent variable (default: %(default)s)')
    parser.add_argument('--network', type=str, default='FNN', choices=['FNN', 'CNN'], help='dimension of latent variable (default: %(default)s)')

    args = parser.parse_args()
    print('# Options')
    for key, value in sorted(vars(args).items()):
        print(key, '=', value)

    device = args.device

    # Load MNIST as binarized at 'thresshold' and create data loaders
    thresshold = 0.5
    mnist_train_loader = torch.utils.data.DataLoader(datasets.MNIST('data/', train=True, download=True,
                                                                    transform=transforms.Compose([transforms.ToTensor(), transforms.Lambda(lambda x: (thresshold < x).float().squeeze())])),
                                                    batch_size=args.batch_size, shuffle=True)
    mnist_test_loader = torch.utils.data.DataLoader(datasets.MNIST('data/', train=False, download=True,
                                                                transform=transforms.Compose([transforms.ToTensor(), transforms.Lambda(lambda x: (thresshold < x).float().squeeze())])),
                                                    batch_size=args.batch_size, shuffle=True)

    # Define prior distribution
    M = args.latent_dim
    # Define encoder and decoder networks
    if args.network == 'FNN':
        encoder_net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(784, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, M*2),
        )

        decoder_net = nn.Sequential(
            nn.Linear(M, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 784),
            nn.Unflatten(-1, (28, 28))
        )
    elif args.network == 'CNN':
        encoder_net = nn.Sequential(
        # (batch, 28, 28)，
        nn.Conv1d(in_channels=28, out_channels=32, kernel_size=3, stride=2, padding=1),  # (batch, 32, 14)
        nn.ReLU(),
        nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=1),  # (batch, 64, 7)
        nn.ReLU(),
        nn.Flatten(),   # (batch, 64*7=448)
        nn.Linear(64 * 7, 512),
        nn.ReLU(),
        nn.Linear(512, args.latent_dim * 2)
        )  # (batch, latent_dim*2)

        decoder_net = nn.Sequential(
        nn.Linear(args.latent_dim, 64 * 7),
        nn.ReLU(),
        nn.Unflatten(1, (64, 7)),  # (batch, 64, 7)
        nn.ConvTranspose1d(in_channels=64, out_channels=32, kernel_size=3, stride=2, padding=1, output_padding=1),  # (batch, 32, 14)
        nn.ReLU(),
        nn.ConvTranspose1d(in_channels=32, out_channels=28, kernel_size=3, stride=2, padding=1, output_padding=1)   # (batch, 28, 28)
        )

   
    # Define VAE model
    decoder = BernoulliDecoder(decoder_net)
    encoder = GaussianEncoder(encoder_net)
    prior = GaussianPrior(M)
    model = VAE(prior, decoder, encoder).to(device)

    # Choose mode to run
    if args.mode == 'train':
        # Define optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        # Train model
        train(model, optimizer, mnist_train_loader, args.epochs, args.device)

        # Save model
        torch.save(model.state_dict(), args.model)

    elif args.mode == 'sample':
        model.load_state_dict(torch.load(args.model, map_location=torch.device(args.device)))

        # Generate samples
        model.eval()
        with torch.no_grad():
            samples = (model.sample(64)).cpu() 
            save_image(samples.view(64, 1, 28, 28), args.samples)

    elif args.mode == 'test':
        model.load_state_dict(torch.load(args.model, map_location=torch.device(args.device)))
        elbo = test(model, mnist_test_loader, args.device, latent_dim=M)
        print(f"ELBO: {elbo:.4f}")
