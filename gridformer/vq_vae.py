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
        

class VectorQuantizer(nn.Module):
    def __init__(self, config):
        """
        Vector Quantizer for 1D data with latent space compression.

        Args:
            num_embeddings (int): Number of embedding vectors.
            embedding_dim (int): Dimensionality of each embedding vector.
            commitment_cost (float): Weight for commitment loss.
            input_dim (int): Dimensionality of the input data.
            latent_dim (int): Dimensionality of the latent space.
        """
        super(VectorQuantizer, self).__init__()
        self.config = config
        self._embedding_dim = self.config.embedding_dim
        self._num_embeddings = self.config.num_embeddings
        self._commitment_cost = self.config.commitment_cost

        # Embedding layer
        self._embedding = nn.Embedding(self._num_embeddings, self._embedding_dim)
        self._embedding.weight.data.uniform_(-1 / self._num_embeddings, 1 / self._num_embeddings)

        # Linear layer to compress input_dim to latent_dim
        #self.linear = nn.Linear(self.config.input_dim, self.config.latent_dim)

    def forward(self, inputs):
        # Compress input to latent_dim
        input_shape = inputs.shape  

        # Flatten input: (batch_size, latent_dim) -> (batch_size * latent_dim, embedding_dim)
        flat_input = inputs.view(-1, input_shape[1])

        # Calculate distances
        distances = (torch.sum(flat_input**2, dim=1, keepdim=True) +
                     torch.sum(self._embedding.weight**2, dim=1) -
                     2 * torch.matmul(flat_input, self._embedding.weight.t()))

        # Encoding
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        encodings = torch.zeros(encoding_indices.shape[0], self._num_embeddings, device=inputs.device)
        encodings.scatter_(1, encoding_indices, 1)

        # Quantize
        quantized = torch.matmul(encodings, self._embedding.weight).view(input_shape)

        # Loss
        e_latent_loss = torch.nn.functional.mse_loss(quantized.detach(), inputs)
        q_latent_loss = torch.nn.functional.mse_loss(quantized, inputs.detach())
        loss = q_latent_loss + self._commitment_cost * e_latent_loss

        # Gradient flow trick
        quantized = inputs + (quantized - inputs).detach()
        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))

        # Return quantized latent space
        return loss, quantized, perplexity, encodings.view(input_shape[0], -1)
    

    def save(self, model_name="vq"):
        torch.save(self.state_dict(), os.path.join(self.config.path, model_name))

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
        self.vq = VectorQuantizer(config=self.config).to(device=self.device)
        self.decoder = Decoder(config=self.config).to(device=self.device)



    def forward(self, x):
        z = self.encoder(x)
        loss, quantized, perplexity, encodings = self.vq(z)
        pred_x = self.decoder(quantized)

        return pred_x, loss, quantized, perplexity, encodings
    

    def save_model(self):
        self.encoder.save()
        self.decoder.save()
        self.vq.save()
        logging.info("model saved")

    def load_model(self):
        self.encoder.load()
        self.decoder.load()
        self.vq.load()
    

    
    

