# Code for DTU course 02460 (Advanced Machine Learning Spring) by Jes Frellsen, 2024
# Version 1.0 (2024-02-11)

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as td
from tqdm import tqdm
from unet import Unet

class DDPM(nn.Module):
    def __init__(self, network, beta_1=1e-4, beta_T=2e-2, T=100):
        super(DDPM, self).__init__()
        self.network = network
        self.beta_1 = beta_1
        self.beta_T = beta_T
        self.T = T

        self.beta = nn.Parameter(torch.linspace(beta_1, beta_T, T), requires_grad=False)
        self.alpha = nn.Parameter(1 - self.beta, requires_grad=False)
        self.alpha_cumprod = nn.Parameter(self.alpha.cumprod(dim=0), requires_grad=False)

    def negative_elbo(self, x):
        bsz = x.size(0)
        device = x.device

        # Sample t ~ {1,...,T}
        t = torch.randint(1, self.T + 1, (bsz,), device=device)

        # Sample noise eps ~ N(0,I)
        eps = torch.randn_like(x)

        # Forward diffusion: x_t
        alpha_bar_t = self.alpha_cumprod[t - 1].view(bsz, *([1] * (x.dim() - 1)))
        x_t = torch.sqrt(alpha_bar_t) * x + torch.sqrt(1 - alpha_bar_t) * eps

        # Predict noise, passing t in [0,1]
        t_input = (t.float() / self.T).view(-1, 1)
        eps_theta = self.network(x_t, t_input)

        # 5) MSE loss: || eps - eps_theta ||
        return F.mse_loss(eps_theta, eps, reduction='mean')

    def sample(self, shape):
        device = self.alpha.device
        x_t = torch.randn(shape, device=device)

        for t in reversed(range(self.T)):
            t_input = ((t + 1) / self.T) * torch.ones(shape[0], 1, device=device)

            eps_theta = self.network(x_t, t_input)

            alpha_t     = self.alpha[t].view(1, *([1] * (x_t.dim() - 1)))
            alpha_bar_t = self.alpha_cumprod[t].view(1, *([1] * (x_t.dim() - 1)))
            beta_t      = self.beta[t].view(1, *([1] * (x_t.dim() - 1)))

            # reverse diffusion formula
            mean = (1. / alpha_t.sqrt()) * (
                x_t - (1. - alpha_t) / torch.sqrt(1. - alpha_bar_t) * eps_theta
            )
            if t > 0:
                z = torch.randn_like(x_t)
            else:
                z = 0.0

            x_t = mean + beta_t.sqrt() * z

        return x_t

    def loss(self, x):
        return self.negative_elbo(x).mean()


def train(model, optimizer, data_loader, epochs, device):
    model.train()
    total_steps = len(data_loader) * epochs
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

            progress_bar.set_postfix(
                loss=f"{loss.item():.4f}",
                epoch=f"{epoch+1}/{epochs}"
            )
            progress_bar.update()


class FcNetwork(nn.Module):
    def __init__(self, input_dim, num_hidden):
        super(FcNetwork, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim + 1, num_hidden), nn.ReLU(),
            nn.Linear(num_hidden, num_hidden),   nn.ReLU(),
            nn.Linear(num_hidden, input_dim)
        )

    def forward(self, x, t):
        x_t_cat = torch.cat([x, t], dim=1)
        return self.network(x_t_cat)


if __name__ == "__main__":
    import argparse
    import torch.utils.data
    from torchvision import datasets, transforms
    from torchvision.utils import save_image

    parser = argparse.ArgumentParser()
    parser.add_argument('mode', type=str, default='train',choices=['train', 'sample', 'test'],help='Mode: train or sample (test is optional).')
    parser.add_argument('--model', type=str, default='model.pt',help='File to save/load the model.')
    parser.add_argument('--samples', type=str, default='samples.png',help='File to save samples in.')
    parser.add_argument('--device', type=str, default='cpu',choices=['cpu','cuda','mps'],help='Torch device.')
    parser.add_argument('--batch-size', type=int, default=64,help='Batch size.')
    parser.add_argument('--epochs', type=int, default=10,help='Number of epochs.')
    parser.add_argument('--lr', type=float, default=1e-3,help='Learning rate.')
    parser.add_argument('--use-unet', action='store_true',help='Use a U-Net instead of FcNetwork.')
    args = parser.parse_args()

    print("# Options")
    for k, v in sorted(vars(args).items()):
        print(f"{k} = {v}")

    device = torch.device(args.device)
    # Load MNIST (dequantize + scale to [-1,1])
    # Flatten to (784,) if using FcNetwork
 
    transform_mnist = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x + torch.rand_like(x)/255.0),
        transforms.Lambda(lambda x: (x - 0.5)*2.0),
        transforms.Lambda(lambda x: x.flatten()),  # Flatten to 784
    ])
    train_data = datasets.MNIST(
        root='data',
        train=True,
        download=True,
        transform=transform_mnist
    )
    train_loader = torch.utils.data.DataLoader(
        train_data, batch_size=args.batch_size, shuffle=True
    )

    # The dimension is 784 for flattened MNIST
    D = 784

    # Define the network (U-Net or FcNetwork)
    if args.use_unet:
        print("Using U-Net.")
        network = Unet()
    else:
        print("Using FcNetwork.")
        num_hidden = 512
        network = FcNetwork(D, num_hidden)

    # Create DDPM model
    T = 1000
    model = DDPM(network, T=T).to(device)

    #Train or Sample
    if args.mode == 'train':
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        train(model, optimizer, train_loader, args.epochs, device)

        torch.save(model.state_dict(), args.model)
        print(f"Model saved to {args.model}")

    elif args.mode == 'sample':
        model.load_state_dict(torch.load(args.model, map_location=device))
        model.eval()

        with torch.no_grad():
            # Sample e.g. 64 images
            samples = model.sample((64, D)).cpu()

        # Rescale from [-1,1] to [0,1]
        samples = (samples + 1) / 2.0

        # Reshape to (64,1,28,28) for saving as an image grid
        samples = samples.view(64,1,28,28)
        save_image(samples, args.samples, nrow=8)
        print(f"MNIST samples saved to {args.samples}")
