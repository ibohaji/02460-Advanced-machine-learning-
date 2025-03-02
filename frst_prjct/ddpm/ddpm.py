 # Code for DTU course 02460 (Advanced Machine Learning Spring) by Jes Frellsen, 2024
# Version 1.0 (2024-02-11)

import torch
import torch.nn as nn
import torch.distributions as td
import torch.nn.functional as F
from tqdm import tqdm


class DDPM(nn.Module):
    def __init__(self, network, beta_1=1e-4, beta_T=2e-2, T=100):
        """
        Initialize a DDPM model.

        Parameters:
        network: [nn.Module]
            The network to use for the diffusion process.
        beta_1: [float]
            The noise at the first step of the diffusion process.
        beta_T: [float]
            The noise at the last step of the diffusion process.
        T: [int]
            The number of steps in the diffusion process.
        """
        super(DDPM, self).__init__()
        self.network = network
        self.beta_1 = beta_1
        self.beta_T = beta_T
        self.T = T

        self.beta = nn.Parameter(torch.linspace(beta_1, beta_T, T), requires_grad=False)
        self.alpha = nn.Parameter(1 - self.beta, requires_grad=False)
        self.alpha_cumprod = nn.Parameter(self.alpha.cumprod(dim=0), requires_grad=False) # Cumulative product of alpha
    
    def negative_elbo(self, x):
        """
        Evaluate the DDPM negative ELBO on a batch of data.

        Parameters:
        x: [torch.Tensor]
            A batch of data (x) of dimension `(batch_size, *)`.
        Returns:
        [torch.Tensor]
            The negative ELBO of the batch of dimension `(batch_size,)`.
        """
        # x is (batch_size, C, H, W)
        bsz = x.size(0) # batch size
        device = x.device

        # Sample t uniformly from {1, ..., T} for each sample in the batch
        t = torch.randint(1, self.T + 1, (bsz,), device=device)
        
        # Sample noise ε ~ N(0, I)
        eps = torch.randn_like(x)

        # Compute x_t
        #    alpha_cumprod[t-1] is ᾱₜ (since t is 1-based).
        alpha_bar_t = self.alpha_cumprod[t - 1].view(bsz, *([1] * (x.dim() - 1))) #x.dim() is 4, so we need to broadcast, alpha_bar_t to the same shape as x(batch_size, 1, 1, 1)
        x_t = torch.sqrt(alpha_bar_t) * x + torch.sqrt(1. - alpha_bar_t) * eps

        # Predict the noise using the network. It expects (x_t, t).
        #    We need to pass t as shape (bsz,1) if the network just appends it.
        # t_input = t.float().view(-1, 1)
        t_input= (t.float() / self.T).view(-1, 1) # normalizating t to [0, 1]
        eps_theta = self.network(x_t, t_input)

        # Compute the MSE loss between eps and eps_theta
        loss = F.mse_loss(eps_theta, eps, reduction='mean')
        return loss

    def sample(self, shape):
        """
        Sample from the model.

        Parameters:
        shape: [tuple]
            The shape of the samples to generate.
        Returns:
        [torch.Tensor]
            The generated samples.
        """
        device = self.alpha.device

        # Gaussian noise x_T ~ N(0, I)
        x_t = torch.randn(shape, device=device)

        # Iterate in reverse from T to 1
        for t in reversed(range(self.T)):
            # In the paper, t is 1-based, but in the code, it is 0-based. use t+1 to match the paper
            t_input = ((t + 1).float()/self.T) * torch.ones(shape[0], 1, device=device) #(t+1)*(bach_size, 1) normalization
            
            # Predict the noise epsilon_theta
            eps_theta = self.network(x_t, t_input)

            # Retrieve alpha_t, alpha_bar_t, and beta_t
            alpha_t = self.alpha[t]         
            alpha_bar_t = self.alpha_cumprod[t]  
            beta_t = self.beta[t]           

            # Sample random noise z ~ N(0, I) if t > 0, otherwise set z = 0
            if t > 0:
                z = torch.randn_like(x_t)
            else:
                z = 0.0

            # Reshape scalars for broadcasting
            alpha_t = alpha_t.view(1, *([1] * (x_t.dim() - 1))) 
            alpha_bar_t = alpha_bar_t.view(1, *([1] * (x_t.dim() - 1))) 
            beta_t = beta_t.view(1, *([1] * (x_t.dim() - 1)))

            # mean = (1/sqrt(alpha_t)) * [ x_t - (1-alpha_t)/sqrt(1-alpha_bar_t) * eps_theta ]
            mean = (1. / alpha_t.sqrt()) * (x_t - (1. - alpha_t) / torch.sqrt(1. - alpha_bar_t) * eps_theta)
            

            # Add the variance term sqrt(beta_t) * z
            x_t = mean + beta_t.sqrt() * z

        # After T iterations, x_t is the final sampled x_0
        return x_t

    def loss(self, x):
        """
        Evaluate the DDPM loss on a batch of data.

        Parameters:
        x: [torch.Tensor]
            A batch of data (x) of dimension `(batch_size, *)`.
        Returns:
        [torch.Tensor]
            The loss for the batch.
        """
        return self.negative_elbo(x).mean()


def train(model, optimizer, data_loader, epochs, device):
    """
    Train a Flow model.

    Parameters:
    model: [Flow]
       The model to train.
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
            if isinstance(x, (list, tuple)):
                x = x[0]
            x = x.to(device)
            optimizer.zero_grad()
            loss = model.loss(x)
            loss.backward()
            optimizer.step()

            # Update progress bar
            progress_bar.set_postfix(loss=f"⠀{loss.item():12.4f}", epoch=f"{epoch+1}/{epochs}")
            progress_bar.update()


class FcNetwork(nn.Module):
    def __init__(self, input_dim, num_hidden):
        """
        Initialize a fully connected network for the DDPM, where the forward function also take time as an argument.
        
        parameters:
        input_dim: [int]
            The dimension of the input data.
        num_hidden: [int]
            The number of hidden units in the network.
        """
        super(FcNetwork, self).__init__()
        self.network = nn.Sequential(nn.Linear(input_dim+1, num_hidden), nn.ReLU(),  #x of shape (batch_size, input_dim) t of shape (batch_size, 1)
                                     nn.Linear(num_hidden, num_hidden), nn.ReLU(), 
                                     nn.Linear(num_hidden, input_dim))

    def forward(self, x, t):
        """"
        Forward function for the network.
        
        parameters:
        x: [torch.Tensor]
            The input data of dimension `(batch_size, input_dim)`
        t: [torch.Tensor]
            The time steps to use for the forward pass of dimension `(batch_size, 1)`
        """
        x_t_cat = torch.cat([x, t], dim=1)
        return self.network(x_t_cat) #(batch_size, input_dim + 1)


if __name__ == "__main__":
    import torch.utils.data
    from torchvision import datasets, transforms
    from torchvision.utils import save_image
    import ToyData
    import sys
        # Arguments
    sys.argv = [
        "ddpm.py",
        "--mode", "train",
        "--data", "tg",
        "--model", "model.pt", 
        "--samples", "samples.png", 
        "--device", "cuda", 
        "--batch-size", "128", 
        "--epochs", "5", 
        "--lr", "1e-3"
    ]
    # Parse arguments
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'sample', 'test'], help='what to do when running the script (default: %(default)s)')
    parser.add_argument('--data', type=str, default='tg', choices=['tg', 'cb', 'mnist'], help='dataset to use {tg: two Gaussians, cb: chequerboard} (default: %(default)s)')
    parser.add_argument('--model', type=str, default='model.pt', help='file to save model to or load model from (default: %(default)s)')
    parser.add_argument('--samples', type=str, default='samples.png', help='file to save samples in (default: %(default)s)')
    parser.add_argument('--device', type=str, default='cpu', choices=['cpu', 'cuda', 'mps'], help='torch device (default: %(default)s)')
    parser.add_argument('--batch-size', type=int, default=10000, metavar='N', help='batch size for training (default: %(default)s)')
    parser.add_argument('--epochs', type=int, default=1, metavar='N', help='number of epochs to train (default: %(default)s)')
    parser.add_argument('--lr', type=float, default=1e-3, metavar='V', help='learning rate for training (default: %(default)s)')

    args = parser.parse_args()
    print('# Options')
    for key, value in sorted(vars(args).items()):
        print(key, '=', value)

    # Generate the data
    n_data = 10000000
    toy = {'tg': ToyData.TwoGaussians, 'cb': ToyData.Chequerboard}[args.data]()
    transform = lambda x: (x-0.5)*2.0
    train_loader = torch.utils.data.DataLoader(transform(toy().sample((n_data,))), batch_size=args.batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(transform(toy().sample((n_data,))), batch_size=args.batch_size, shuffle=True)

    # Get the dimension of the dataset
    D = next(iter(train_loader)).shape[1]

    # Define the network
    num_hidden = 64
    network = FcNetwork(D, num_hidden)

    # Set the number of steps in the diffusion process
    T = 1000

    # Define model
    model = DDPM(network, T=T).to(args.device)

    # Choose mode to run
    if args.mode == 'train':
        # Define optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

        # Train model
        train(model, optimizer, train_loader, args.epochs, args.device)

        # Save model
        torch.save(model.state_dict(), args.model)

    elif args.mode == 'sample':
        import matplotlib.pyplot as plt
        import numpy as np

        # Load the model
        model.load_state_dict(torch.load(args.model, map_location=torch.device(args.device)))

        # Generate samples
        model.eval()
        with torch.no_grad():
            samples = (model.sample((10000,D))).cpu() 

        # Transform the samples back to the original space
        samples = samples /2 + 0.5

        # Plot the density of the toy data and the model samples
        coordinates = [[[x,y] for x in np.linspace(*toy.xlim, 1000)] for y in np.linspace(*toy.ylim, 1000)]
        prob = torch.exp(toy().log_prob(torch.tensor(coordinates)))

        fig, ax = plt.subplots(1, 1, figsize=(7, 5))
        im = ax.imshow(prob, extent=[toy.xlim[0], toy.xlim[1], toy.ylim[0], toy.ylim[1]], origin='lower', cmap='YlOrRd')
        ax.scatter(samples[:, 0], samples[:, 1], s=1, c='black', alpha=0.5)
        ax.set_xlim(toy.xlim)
        ax.set_ylim(toy.ylim)
        ax.set_aspect('equal')
        fig.colorbar(im)
        plt.savefig(args.samples)
        plt.close()
