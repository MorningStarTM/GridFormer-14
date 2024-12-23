import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim



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
    

    def save(self, model_name="encoder"):
        torch.save(self.state_dict(), os.path.join(self.config.path, model_name))

    def load(self, model_name="encoder"):
        model_path = os.path.join(self.config.path, model_name)
    
        try:
            self.load_state_dict(torch.load(model_path))
            print(f"Model loaded successfully from {model_path}")
        except FileNotFoundError as e:
            print(f"Error: {e}. Model file not found at {model_path}")
        except Exception as e:
            print(f"An error occurred while loading the model: {e}")
        

class VQEmbedding(nn.Module):
    def __init__(self, config, commitment_cost=0.25):
        super().__init__()
        self.config = config
        self.embedding_dim = self.config.embedding_dim
        self.num_embeddings = self.config.num_embeddings
        self.commitment_cost = commitment_cost

        # Embedding codebook
        self.embedding = nn.Embedding(self.num_embeddings, self.embedding_dim)
        self.embedding.weight.data.uniform_(-1 / self.num_embeddings, 1 / self.num_embeddings)

    def forward(self, z):
        # Reshape input for vector quantization
        batch_size, input_dim = z.shape
        assert input_dim == self.embedding_dim, "Input dimensionality must match embedding_dim"

        # Calculate distances between z and codebook embeddings |a-b|²
        z_flattened = z.view(-1, self.embedding_dim)  # Shape: (batch_size, embedding_dim)
        distances = (
            torch.sum(z_flattened ** 2, dim=1, keepdim=True)  # a²
            + torch.sum(self.embedding.weight.t() ** 2, dim=0, keepdim=True)  # b²
            - 2 * torch.matmul(z_flattened, self.embedding.weight.t())  # -2ab
        )

        # index with the smallest distance
        encoding_indices = torch.argmin(distances, dim=-1)

        # quantized vector
        z_q = self.embedding(encoding_indices)
        z_q = z_q.view(batch_size, self.embedding_dim)  # Reshape to original dimensions

        # Calculate the commitment loss
        commitment_loss = self.commitment_cost * F.mse_loss(z_q.detach(), z)
        embedding_loss = F.mse_loss(z_q, z.detach())
        loss = embedding_loss + commitment_loss

        # Straight-through estimator trick for gradient backpropagation
        z_q = z + (z_q - z).detach()

        return z_q, loss, encoding_indices





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
    

    def save(self, model_name="decoder"):
        torch.save(self.state_dict(), os.path.join(self.config.path, model_name))

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
        self.vq = VQEmbedding(config=self.config).to(device=self.device)
        self.decoder = Decoder(config=self.config).to(device=self.device)



    def forward(self, x):
        z = self.encoder(x)
        z_q, loss, encoding_indices = self.vq(z)
        pred_x = self.decoder(z_q)

        return z, z_q, pred_x, loss, encoding_indices
    

    
