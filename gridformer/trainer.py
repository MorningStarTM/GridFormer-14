import os
import torch
from gridformer.vq_vae import VQVAE
import torch.optim as optim
import torch.nn.functional as F
from gridformer.Utils.logger import logging

class VQTrainer:
    def __init__(self, config, model:VQVAE) -> None:
        self.config = config
        self.model = model
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.config.learning_rate)

    def train(self, epochs, dataloader, data_variance, save_path=None):
        for epoch in range(epochs):
            epoch_recon_loss = 0.0
            epoch_vq_loss = 0.0
            epoch_total_loss = 0.0
            num_batches = 0

            best_score = float('inf')

            for observation in dataloader:
                num_batches += 1

                pred_x, vq_loss, encodings = self.model(observation)
                recon_error = F.mse_loss(pred_x, observation) / data_variance

                self.optimizer.zero_grad()
                loss = recon_error + vq_loss
                loss.backward()

                self.optimizer.step()
                
                epoch_recon_loss += recon_error.item()
                epoch_vq_loss += vq_loss.item()
                epoch_total_loss += loss.item()

            # Calculate average loss for the epoch
            avg_recon_loss = epoch_recon_loss / num_batches
            avg_vq_loss = epoch_vq_loss / num_batches
            avg_total_loss = epoch_total_loss / num_batches
 

            # Print loss after every epoch
            logging.info(
                f"Epoch {epoch + 1}/{epochs} - "
                f"Reconstruction Loss: {avg_recon_loss:.4f}, "
                f"VQ Loss: {avg_vq_loss:.4f}, "
                f"Total Loss: {avg_total_loss:.4f}"
            )

            print(
                f"Epoch {epoch + 1}/{epochs} - "
                f"Reconstruction Loss: {avg_recon_loss:.4f}, "
            ) 


            if avg_recon_loss < best_score:
                if save_path:
                    self.model.save_model(custom_path=save_path)
                else:
                    self.model.save_model()
                print(f"Model saved at {avg_recon_loss:.3f}")
                best_score = avg_recon_loss
                