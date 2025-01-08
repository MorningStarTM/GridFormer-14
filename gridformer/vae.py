import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class VAE(nn.Module):
    def __init__(self, config) -> None:
        super(VAE, self).__init__()
        self.config = config
        
        input_dim = self.config['input_dim']
        output_dim = self.config['latent_dim']
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
        latent_param = self.encoder(x)
        mean, log_var = torch.chunk(latent_param, 2, dim=-1)

        log_var = torch.clamp(log_var, min=-10, max=10) # for avoid nan value return and numerical stability
        std = torch.exp(0.5 * log_var)
        dist = torch.distributions.Normal(mean, std)

        sample = dist.rsample()

        obs = self.decoder(sample)
        return obs
    

    def get_latent_space(self, x):
        latent_param = self.encoder(x)
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

        return z_discrete
    


    def save(self, model_path:str, model_name:str = "VAE.pt"):
        torch.save(self.state_dict(), os.path.join(model_path, model_name))
        print(f"model saved at {model_path}")


    
    def load(self, model_path:str, model_name:str="VAE.pt"):
        self.load_state_dict(torch.load(os.path.join(model_path, model_name)))
        print(f"model loaded from {model_path}")


    