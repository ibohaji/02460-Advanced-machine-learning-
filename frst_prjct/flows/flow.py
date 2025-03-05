# Code for DTU course 02460 (Advanced Machine Learning Spring) by Jes Frellsen, 2024
# Version 1.3 (2024-02-11)
# Inspiration is taken from:
# - https://github.com/jmtomczak/intro_dgm/blob/main/flows/realnvp_example.ipynb
# - https://github.com/VincentStimper/normalizing-flows/tree/master

import torch
import torch.nn as nn
import torch.distributions as td
from tqdm import tqdm
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np


class LearnableGaussianBase(nn.Module):
    def __init__(self, D):
        super(LearnableGaussianBase, self).__init__()
        self.D = D
        self.mean = nn.Parameter(torch.zeros(self.D))
        self.log_std = nn.Parameter(torch.zeros(self.D))

    def forward(self):
        return td.Independent(
            td.Normal(loc=self.mean, scale=torch.exp(self.log_std)), 1
        )

class MaskedCouplingLayer(nn.Module):
    def __init__(self, scale_net, translation_net, feature_dim, mask_type="chequerboard"):
        super(MaskedCouplingLayer, self).__init__()
        self.scale_net = scale_net
        self.translation_net = translation_net

        # ReZero parameters initialized at 0
        self.alpha = nn.Parameter(torch.zeros(1))
        self.beta = nn.Parameter(torch.zeros(1))

        # Initialize mask 
        mask = torch.zeros((feature_dim,))
        if mask_type == "chequerboard":
            mask[::2] = 1

        elif mask_type == "chequerboard_inverse":
            mask[1::2] = 1

        # Space so we could implement other masks (before we had random mask and that is why we keep if-elif structure)
        
        self.mask = nn.Parameter(mask, requires_grad=False)

    def forward(self, z):
        b = self.mask  # Shape: (batch_size, dim), with 0s and 1s per feature

        # z_keep contains the features we "keep" (identity)
        z_keep = b * z

        # z_transform contains the features we will transform
        z_transform = (1 - b) * z

        # Compute scale and translation from the kept part (masked input)
        s = self.scale_net(z_keep)      
        t = self.translation_net(z_keep)  

        # x_a holds the unchanged part (just z_keep)
        x_a = z_keep 

        # x_b is the transformed part, applying scale (with ReZero) and translation
        x_b = torch.exp(self.alpha * s) * z_transform + self.beta * t 

        # Final output combines the two parts: x_a already has zeros where (1 - b) is 1, (1 - b) * x_b zeroes out the parts we don't want from x_b
        x = x_a + (1 - b) * x_b  # Shape: (batch_size, dim)

        # Log-determinant of the Jacobian (only from the transformed part)
        log_det_J = torch.sum((1 - b) * self.alpha * s, dim=1)  # Shape: (batch_size,)
        return x, log_det_J

    def inverse(self, x):
        b = self.mask  # Shape: (batch_size, dim)

        # x_keep: the part of x we don't change (corresponds to z_keep)
        x_keep = b * x

        # x_transform: the part of x we will invert (corresponds to x_b)
        x_transform = (1 - b) * x

        # Compute scale and translation from x_keep
        s = self.scale_net(x_keep)      
        t = self.translation_net(x_keep)  

        # z_keep is the same as x_keep (unchanged part)
        z_keep = x_keep  

        # z_transform reverses the scaling and translation applied in forward
        z_transform = (x_transform - self.beta * t) * torch.exp(-self.alpha * s)  

        # Recombine the parts into the full z: z_keep already has zeros where (1 - b) is 1, (1 - b) * z_transform ensures we only add the transformed parts where needed
        z = z_keep + (1 - b) * z_transform  # Shape: (batch_size, dim)

        # Log-determinant (negative of forward)
        log_det_J = -torch.sum((1 - b) * self.alpha * s, dim=1)  # Shape: (batch_size,)
        return z, log_det_J


class Flow(nn.Module):
    def __init__(self, base, transformations):
        """
        Define a normalizing flow model.
        
        Parameters:
        base: [torch.distributions.Distribution]
            The base distribution.
        transformations: [list of torch.nn.Module]
            A list of transformations to apply to the base distribution.
        """
        super(Flow, self).__init__()
        self.base = base
        self.transformations = nn.ModuleList(transformations)
        
    def forward(self, z):
        """
        Transform a batch of data through the flow (from the base to data).
        
        Parameters:
        x: [torch.Tensor]
            The input to the transformation of dimension `(batch_size, feature_dim)`
        Returns:
        z: [torch.Tensor]
            The output of the transformation of dimension `(batch_size, feature_dim)`
        sum_log_det_J: [torch.Tensor]
            The sum of the log determinants of the Jacobian matrices of the forward transformations.            
        """
        sum_log_det_J = 0
        for i, T in enumerate(self.transformations):
            z, log_det_J = T(z)
            sum_log_det_J += log_det_J
        return z, sum_log_det_J

    def inverse(self, x):
        """
        Transform a batch of data through the flow (from data to the base).

        Parameters:
        x: [torch.Tensor]
            The input to the inverse transformation of dimension `(batch_size, feature_dim)`
        Returns:
        z: [torch.Tensor]
            The output of the inverse transformation of dimension `(batch_size, feature_dim)`
        sum_log_det_J: [torch.Tensor]
            The sum of the log determinants of the Jacobian matrices of the inverse transformations.
        """
        sum_log_det_J = 0
        for i, T in enumerate(reversed(self.transformations)):
            x, log_det_J = T.inverse(x)
            sum_log_det_J += log_det_J
        return x, sum_log_det_J
    
    def log_prob(self, x):
        """
        Compute the log probability of a batch of data under the flow.

        Parameters:
        x: [torch.Tensor]
            The data of dimension `(batch_size, feature_dim)`
        Returns:
        log_prob: [torch.Tensor]
            The log probability of the data under the flow.
        """
        z, log_det_J = self.inverse(x)

        return self.base().log_prob(z) + log_det_J 
    
    def sample(self, sample_shape=(1,)):
        """
        Sample from the flow.

        Parameters:
        n_samples: [int]
            Number of samples to generate.
        Returns:
        z: [torch.Tensor]
            The samples of dimension `(n_samples, feature_dim)`
        """
        z = self.base().sample(sample_shape)
        return self.forward(z)[0]
    
    def loss(self, x):
        """
        Compute the negative mean log likelihood for the given data bath.

        Parameters:
        x: [torch.Tensor] 
            A tensor of dimension `(batch_size, feature_dim)`
        Returns:
        loss: [torch.Tensor]
            The negative mean log likelihood for the given data batch.
        """
        return -torch.mean(self.log_prob(x))

def plot_loss_evolution(loss_history, epoch_steps, filename="loss_plot.png"):
    plt.figure(figsize=(10, 5))
    plt.plot(loss_history, label="Loss")
    for step in epoch_steps:
        plt.axvline(step, color='r', linestyle='--', alpha=0.5)
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.title("Training Loss Evolution")
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
    print(f"Loss plot saved as '{filename}'")


def train(model, data_loader, epochs, device, initial_lr=1e-3, decay_rate=0.98, patience=500, delta=1e-4):
    optimizer = torch.optim.Adam(model.parameters(), lr=initial_lr)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=decay_rate)

    model.train()
    total_steps = len(data_loader) * epochs
    progress_bar = tqdm(range(total_steps), desc="Training")

    best_loss = float('inf')
    steps_without_improvement = 0
    loss_history = []
    epoch_steps = []

    for epoch in range(epochs):
        data_iter = iter(data_loader)
        for x, _ in data_iter:
            x = x.to(device)
            optimizer.zero_grad()
            loss = model.loss(x)    # Here we compute the loss from the model
            loss.backward()
            optimizer.step()

            current_loss = loss.item()
            loss_history.append(current_loss)

            if best_loss - current_loss > delta:
                best_loss = current_loss
                steps_without_improvement = 0
            else:
                steps_without_improvement += 1

            if steps_without_improvement >= patience:
                print(f"\nEarly stopping at epoch {epoch+1}, step {progress_bar.n+1} (best loss: {best_loss:.6f})")
                progress_bar.close()
                plot_loss_evolution(loss_history, epoch_steps)
                return

            lr = optimizer.param_groups[0]['lr']
            progress_bar.set_postfix(
                loss=f"{current_loss:12.4f}",
                best_loss=f"{best_loss:12.4f}",
                epoch=f"{epoch+1}/{epochs}",
                lr=f"{lr:.6f}",
                patience=f"{steps_without_improvement}/{patience}"
            )
            progress_bar.update()

        scheduler.step()
        epoch_steps.append(len(loss_history))

    progress_bar.close()
    plot_loss_evolution(loss_history, epoch_steps, filename="new-tuning.png")


if __name__ == "__main__":
    import torch.utils.data
    from torchvision import datasets, transforms
    from torchvision.utils import save_image
    import sys

    # Arguments
    sys.argv = [
        "flow.py", "train", 
        "--model", "new-tuning.pt", 
        "--samples", "new-tuning.png", 
        "--device", "cuda", 
        "--batch-size", "512", 
        "--epochs", "50", 
        "--initial-lr", "1.5e-3"
    ]

    # Parse arguments
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', type=str, default='train', choices=['train', 'sample'], help='what to do when running the script (default: %(default)s)')
    parser.add_argument('--model', type=str, default='model-1.pt', help='file to save model to or load model from (default: %(default)s)')
    parser.add_argument('--samples', type=str, default='samples-1.png', help='file to save samples in (default: %(default)s)')
    parser.add_argument('--device', type=str, default='cpu', choices=['cpu', 'cuda', 'mps'], help='torch device (default: %(default)s)')
    parser.add_argument('--batch-size', type=int, default=10000, metavar='N', help='batch size for training (default: %(default)s)')
    parser.add_argument('--epochs', type=int, default=1, metavar='N', help='number of epochs to train (default: %(default)s)')
    parser.add_argument('--initial-lr', type=float, default=1e-5, metavar='V', help='learning rate for training (default: %(default)s)')

    args = parser.parse_args()
    print('# Options')
    for key, value in sorted(vars(args).items()):
        print(key, '=', value)

    # Load data from mnist
    
    train_loader = torch.utils.data.DataLoader(datasets.MNIST('data/', train=True, download=True,
                                                            transform=transforms.Compose([
                                                                transforms.ToTensor(),  # (1) Converts a PIL Image or numpy.ndarray (H x W x C) in the range [0, 255] to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0]
                                                                transforms.Lambda(lambda x: x + torch.rand(x.shape)/255),   # (2) Dequantization (on pixel space (not value)) Since MNIST pixels are discrete values (0, 1, 2, ..., 255), we add random noise to make them continuous. Flows learn continuous density functions, so adding small random noise makes training smoother.
                                                                transforms.Lambda(lambda x: x.flatten())    # (3) Flatten the image to a vector reshaped from (1, 28, 28) to (784,) to use fully connected layers in the flow model.
                                                                ])
                                                                ),
                                                batch_size=args.batch_size, shuffle=True)

    test_loader = torch.utils.data.DataLoader(datasets.MNIST('data/', train=False, download=True,
                                                            transform=transforms.Compose([
                                                                transforms.ToTensor(), 
                                                                transforms.Lambda(lambda x: x + torch.rand(x.shape)/255),
                                                                transforms.Lambda(lambda x: x.flatten())
                                                                ])
                                                                ),
                                                batch_size=args.batch_size, shuffle=True)

    # Define prior distribution
    D = next(iter(train_loader))[0].shape[1]  # Extract images, then get feature dim=784  # Retrieves first batch from train_loader (PyTorch DataLoader), extracts the shape of the batch's second dimension (feature_dim). So D is number of features in Dataset
    base = LearnableGaussianBase(D)

    # Define transformations
    transformations =[]

    # Original 
    num_transformations = 40
    num_hidden = 64
    
    for i in range(num_transformations):
        # Define scale and translation nets                        
        scale_net = nn.Sequential(nn.Linear(D, num_hidden),nn.ReLU(),nn.Linear(num_hidden, num_hidden), nn.ReLU(),nn.Linear(num_hidden, D), nn.Tanh())
        translation_net = nn.Sequential(nn.Linear(D, num_hidden),nn.ReLU(),nn.Linear(num_hidden, num_hidden), nn.ReLU(),nn.Linear(num_hidden, D))
        
        # Alternate the mask every layer
        layer_mask_type = "chequerboard" if i % 2 == 0 else "chequerboard_inverse"
        
        # Create transformation
        transformations.append(
            MaskedCouplingLayer(scale_net, translation_net, D, layer_mask_type)
        )

    # Define flow model
    model = Flow(base, transformations).to(args.device)

    # Choose mode to run
    if args.mode == 'train':

        # Train model
        train(model, train_loader, args.epochs, args.device, args.initial_lr)

        # Save model
        torch.save(model.state_dict(), args.model)

    elif args.mode == 'sample':
        # Load trained model
        model.load_state_dict(torch.load(args.model, map_location=torch.device(args.device)))
        model.eval()

        with torch.no_grad():
            # Generate samples from the flow model
            samples = model.sample((100,)).cpu()  # 100 samples

        # Reshape the samples to 28x28 images for visualization
        samples = samples.view(-1, 28, 28)

        # Plot generated samples
        fig, axes = plt.subplots(10, 10, figsize=(10, 10))
        for i, ax in enumerate(axes.flatten()):
            ax.imshow(samples[i], cmap='gray')
            ax.axis('off')

        plt.tight_layout()
        plt.savefig(args.samples)  # Save generated samples
        plt.show()