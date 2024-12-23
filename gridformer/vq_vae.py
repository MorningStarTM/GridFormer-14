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
        