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

        self.policy_decoder = nn.Sequential(
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


        #convert latent space 
        continuous_latent_space_ = self.continuous_latent_space(sample, self.config['num_categories'])
        policy_obs = self.policy_decoder(continuous_latent_space_)

        return feature, obs, policy_obs, mean, log_var
    

    def get_latent_space(self, x):
        feature = self.feature_layer(x)
        latent_param = self.encoder(feature)
        mean, log_var = torch.chunk(latent_param, 2, dim=-1)

        log_var = torch.clamp(log_var, min=-10, max=10) # for avoid nan value return and numerical stability
        std = torch.exp(0.5 * log_var)
        dist = torch.distributions.Normal(mean, std)

        sample = dist.rsample()

        return sample
    

    def continuous_latent_space(self, sample, num_categories, noise_std=0.01):
        """
        Convert latent space to continuous space after discretization and add noise.

        Args:
            sample (Tensor): Latent space sample from the encoder.
            num_categories (int): Number of categories (bins).
            noise_std (float): Standard deviation of the Gaussian noise to add.

        Returns:
            Tensor: Continuous latent space with noise added.
        """ 
        # Get the min and max range for the continuous values (assuming normalized range [-1, 1])
        z_min, z_max = -1.0, 1.0  # Customize based on encoder's output range

        # Scale the continuous values to the range [0, num_categories)
        z_scaled = (sample - z_min) / (z_max - z_min) * num_categories

        # Discretize by assigning to the nearest integer bucket
        z_discrete = torch.clamp(z_scaled.long(), 0, num_categories - 1)  # Ensure values are within valid category range

        # Convert back to continuous values
        z_continuous = self.convert_discrete_to_continuous(z_discrete, num_categories)

        # Inject Gaussian noise into the continuous latent space
        noise = torch.randn_like(z_continuous) * noise_std
        z_continuous_with_noise = z_continuous + noise

        return z_continuous_with_noise



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


    
    def vae_loss(self, feature, reconstructed_x, policy_obs, mean, log_var, beta=1.0):
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
        recon_loss = F.mse_loss(reconstructed_x, feature, reduction='mean')
        policy_recon_loss = F.mse_loss(policy_obs, feature, reduction='mean')

        # KL Divergence: KL(N(mean, std) || N(0, 1))
        kl_div = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp(), dim=-1).mean()

        align_loss = F.kl_div(
                    policy_obs.log_softmax(dim=-1),
                    reconstructed_x.detach().softmax(dim=-1),  # Detach reconstructed_x
                    reduction='batchmean',
                )
        
        loss = recon_loss + policy_recon_loss + (beta * kl_div) + (beta * align_loss)

        return loss, recon_loss, kl_div, align_loss
    

    
    def train(self, data_loader, epochs):
        best_loss = float('inf')
        for i in range(epochs):
            total_loss = 0.0
            loop_count = 0

            for loop_count, (obs, _, _, _, _) in enumerate(data_loader, start=1): 

                obs = obs.to(self.device)
                feature, reconstructed_obs, policy_obs, mean, log_var = self.forward(obs)

                loss, _, _, _ = self.vae_loss(feature, reconstructed_obs, policy_obs, mean, log_var)

                total_loss += loss.item()

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            
            avg_total_loss = total_loss / loop_count

            print(
                f"Epochs : {loop_count+1}"
                f"Loss = {avg_total_loss:.3f}")
            

            if avg_total_loss < best_loss:
                best_loss = avg_total_loss
                self.save()
                print(f"New best model saved with Avg Total Loss = {avg_total_loss:.3f}")   


    