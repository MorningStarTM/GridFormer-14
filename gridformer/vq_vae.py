import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from gridformer.Utils.logger import logging



class Encoder(nn.Module):
    def __init__(self, config) -> None:
        super(Encoder, self).__init__()

        self.config = config
        input_dim = self.config.input_dim

        self.encoder = nn.Sequential(
                        nn.Linear(input_dim, input_dim//2),
                        nn.ReLU(),
                        nn.Linear(input_dim//2, input_dim//4),
                        nn.ReLU(),
                        nn.Linear(input_dim//4, input_dim//8),
                        nn.ReLU(),
                        nn.Linear(input_dim//8, self.config.latent_dim)
        )


    
    def forward(self, x):
        x = self.encoder(x)
        return x
    

    def save(self, path=None, model_name="encoder"):
        if path:
            model_path = os.path.join(path, model_name)
        else:
            model_path = os.path.join(self.config.path, model_name)
        torch.save(self.state_dict(), model_path)


    def load(self, model_name="encoder"):
        model_path = os.path.join(self.config.path, model_name)
    
        try:
            self.load_state_dict(torch.load(model_path))
            print(f"Model loaded successfully from {model_path}")
        except FileNotFoundError as e:
            print(f"Error: {e}. Model file not found at {model_path}")
        except Exception as e:
            print(f"An error occurred while loading the model: {e}")
        

class VectorQuantizer(nn.Module):
    def __init__(self, config):
        """
        Vector Quantizer for 1D data with latent space compression.

        Args:
            config (object): Configuration object containing:
                - num_embeddings (int): Number of embedding vectors.
                - embedding_dim (int): Dimensionality of each embedding vector.
                - commitment_cost (float): Weight for commitment loss.
        """
        super(VectorQuantizer, self).__init__()
        self.config = config
        self._embedding_dim = self.config.embedding_dim
        self._num_embeddings = self.config.num_embeddings
        self._commitment_cost = self.config.commitment_cost

        # Embedding layer
        self._embedding = nn.Embedding(self._num_embeddings, self._embedding_dim)
        self._embedding.weight.data.uniform_(-1 / self._num_embeddings, 1 / self._num_embeddings)

    def forward(self, z):
        """
        Forward pass of the Vector Quantizer.

        Args:
            z (torch.Tensor): Latent representation from the encoder. Shape: (batch_size, embedding_dim)

        Returns:
            z_q (torch.Tensor): Quantized representation. Shape: (batch_size, embedding_dim)
            loss (torch.Tensor): VQ loss (reconstruction + commitment).
            indices (torch.Tensor): Indices of the nearest embeddings. Shape: (batch_size,)
        """
        # Compute distances to each embedding
        z_flattened = z.unsqueeze(1)  # Shape: (batch_size, 1, embedding_dim)
        embeddings = self._embedding.weight.unsqueeze(0)  # Shape: (1, num_embeddings, embedding_dim)
        distances = torch.sum((z_flattened - embeddings) ** 2, dim=2)  # Shape: (batch_size, num_embeddings)

        # Find nearest embeddings
        indices = torch.argmin(distances, dim=1)  # Shape: (batch_size,)

        # Retrieve quantized vectors
        z_q = self._embedding(indices)  # Shape: (batch_size, embedding_dim)

        # Compute VQ loss
        commitment_loss = self._commitment_cost * F.mse_loss(z.detach(), z_q)
        vq_loss = F.mse_loss(z_q, z.detach()) + commitment_loss

        # Use straight-through estimator for backpropagation
        z_q = z + (z_q - z).detach()

        return z_q, vq_loss, indices

    

    def save(self, path=None, model_name="vq"):
        if path:
            model_path = os.path.join(path, model_name)
        else:
            model_path = os.path.join(self.config.path, model_name)
        torch.save(self.state_dict(), model_path)


    def load(self, model_name="vq"):
        model_path = os.path.join(self.config.path, model_name)
    
        try:
            self.load_state_dict(torch.load(model_path))
            print(f"Model loaded successfully from {model_path}")
        except FileNotFoundError as e:
            print(f"Error: {e}. Model file not found at {model_path}")
        except Exception as e:
            print(f"An error occurred while loading the model: {e}")





class Decoder(nn.Module):
    def __init__(self, config):
        super(Decoder, self).__init__()
        self.config = config

        input_dim = self.config.latent_dim

        self.decoder = nn.Sequential(
                        nn.Linear(input_dim, input_dim*2),
                        nn.ReLU(),
                        nn.Linear(input_dim*2, input_dim*4),
                        nn.ReLU(),
                        nn.Linear(input_dim*4, input_dim*8),
                        nn.ReLU(),
                        nn.Linear(input_dim*8, self.config.input_dim)
        )


    def forward(self, x):
        x = self.decoder(x)
        return x
    

    def save(self, path=None, model_name="decoder"):
        if path:
            model_path = os.path.join(path, model_name)
        else:
            model_path = os.path.join(self.config.path, model_name)
        torch.save(self.state_dict(), model_path)



    def load(self, model_name="decoder"):
        model_path = os.path.join(self.config.path, model_name)
    
        try:
            self.load_state_dict(torch.load(model_path))
            print(f"Model loaded successfully from {model_path}")
        except FileNotFoundError as e:
            print(f"Error: {e}. Model file not found at {model_path}")
        except Exception as e:
            print(f"An error occurred while loading the model: {e}")




class VQVAE(nn.Module):
    def __init__(self, config) -> None:
        super(VQVAE, self).__init__()

        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.encoder = Encoder(config=self.config).to(device=self.device)
        self.vq = VectorQuantizer(config=self.config).to(device=self.device)
        self.decoder = Decoder(config=self.config).to(device=self.device)



    def forward(self, x):
        z = self.encoder(x)
        z_q, vq_loss, indices = self.vq(z)
        pred_x = self.decoder(z_q)

        return pred_x, vq_loss, indices
    

    def save_model(self, custom_path=None):
        if custom_path:
            self.encoder.save(path=custom_path)
            self.decoder.save(path=custom_path)
            self.vq.save(path=custom_path)
            logging.info("model saved")
        else:
            self.encoder.save()
            self.decoder.save()
            self.vq.save()
            logging.info("model saved")

    def load_model(self):
        self.encoder.load()
        self.decoder.load()
        self.vq.load()
    

    
    

