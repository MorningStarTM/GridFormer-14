import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class VAE(nn.Module):
    def __init__(self, config) -> None:
        super(VAE, self).__init__()
        self.config = config
        
        obs_dim = self.config['input_dim']
        input_dim = self.config['feature_dim']
        output_dim = self.config['latent_dim']

        self.feature_layer = nn.Linear(obs_dim, self.config['feature_dim'])
        self.encoder = nn.Sequential(
                        nn.Linear(input_dim, input_dim//2),
                        nn.ReLU(),
                        nn.Linear(input_dim//2, input_dim//4),
                        nn.ReLU(),
                        nn.Linear(input_dim//4, input_dim//8),
                        nn.ReLU(),
                        nn.Linear(input_dim//8, output_dim*2)
        )

        self.decoder = nn.Sequential(
                        nn.Linear(output_dim, output_dim*2),
                        nn.ReLU(),
                        nn.Linear(output_dim*2, output_dim*4),
                        nn.ReLU(),
                        nn.Linear(output_dim*4, output_dim*8),
                        nn.ReLU(),
                        nn.Linear(output_dim*8, input_dim)
        )


        self.optimizer = optim.Adam(self.parameters(), lr=self.config['lr'])
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)


    
    def forward(self, x):
        feature = self.feature_layer(x)
        latent_param = self.encoder(feature)
        mean, log_var = torch.chunk(latent_param, 2, dim=-1)

        log_var = torch.clamp(log_var, min=-10, max=10) # for avoid nan value return and numerical stability
        std = torch.exp(0.5 * log_var)
        dist = torch.distributions.Normal(mean, std)

        sample = dist.rsample()

        obs = self.decoder(sample)
        return feature, obs, mean, log_var
    

    def get_latent_space(self, x):
        feature = self.feature_layer(x)
        latent_param = self.encoder(feature)
        mean, log_var = torch.chunk(latent_param, 2, dim=-1)

        log_var = torch.clamp(log_var, min=-10, max=10) # for avoid nan value return and numerical stability
        std = torch.exp(0.5 * log_var)
        dist = torch.distributions.Normal(mean, std)

        sample = dist.rsample()

        return sample
    


    def discretize_continuous_values(self, x, num_categories):
        """
        Convert continuous values into discrete categorical values using fixed bins.

        Args:
            z_continuous (Tensor): Continuous latent variables of shape (batch_size, latent_dim).
            num_categories (int): Number of categories (bins) for each latent dimension.

        Returns:
            Tensor: Categorical latent variables of shape (batch_size, latent_dim), with discrete integer indices.
        """
        z_continuous = self.get_latent_space(x)
        # Get the min and max range for the continuous values (assuming normalized range [-1, 1])
        z_min, z_max = -1.0, 1.0  # You can customize this range based on your encoder's output

        # Scale the continuous values to the range [0, num_categories)
        z_scaled = (z_continuous - z_min) / (z_max - z_min) * num_categories

        # Discretize by assigning to the nearest integer bucket
        z_discrete = torch.clamp(z_scaled.long(), 0, num_categories - 1)  # Ensure values are within valid category range

        return z_discrete, z_continuous
    

    def convert_discrete_to_continuous(self, z_discrete, num_categories):
        """
        Convert discrete categorical values back into continuous values.

        Args:
            z_discrete (Tensor): Discrete latent variables of shape (batch_size, latent_dim),
                                with discrete integer indices.
            num_categories (int): Number of categories (bins) for each latent dimension.

        Returns:
            Tensor: Continuous latent variables of shape (batch_size, latent_dim).
        """
        z_min, z_max = -1.0, 1.0  # Use the same range as used during discretization

        # Compute bin width
        bin_width = (z_max - z_min) / num_categories

        # Map discrete values to continuous by computing the bin center
        z_continuous = z_min + (z_discrete.float() + 0.5) * bin_width

        return z_continuous

    
    

    def save(self, model_path:str, model_name:str = "VAE.pt"):
        torch.save(self.state_dict(), os.path.join(model_path, model_name))
        print(f"model saved at {model_path}")


    
    def load(self, model_path:str, model_name:str="VAE.pt"):
        self.load_state_dict(torch.load(os.path.join(model_path, model_name)))
        print(f"model loaded from {model_path}")


    
    def vae_loss(self, x, reconstructed_x, mean, log_var, beta=1.0):
        """
        Compute the VAE loss, which includes reconstruction loss and KL divergence.

        Args:
            x (Tensor): Original input data of shape (batch_size, input_dim).
            reconstructed_x (Tensor): Reconstructed data of shape (batch_size, input_dim).
            mean (Tensor): Mean of the latent space distribution of shape (batch_size, latent_dim).
            log_var (Tensor): Log variance of the latent space distribution of shape (batch_size, latent_dim).
            beta (float): Weight for the KL divergence term (default: 1.0).

        Returns:
            loss (Tensor): Total VAE loss.
            recon_loss (Tensor): Reconstruction loss.
            kl_div (Tensor): KL divergence.
        """
        # Reconstruction loss (using Mean Squared Error)
        recon_loss = F.mse_loss(reconstructed_x, x, reduction='mean')

        # KL Divergence: KL(N(mean, std) || N(0, 1))
        kl_div = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp(), dim=-1).mean()
        loss = recon_loss + beta * kl_div

        return loss, recon_loss, kl_div
    

    
    def train(self, data_loader, epochs):
        pass

    