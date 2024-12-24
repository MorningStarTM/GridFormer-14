import torch
import os
import numpy as np
from torch.utils.data import Dataset, DataLoader
from gridformer.Utils.utils import Config
from gridformer.Utils.logger import logging


config = Config.from_yaml('config.yml')



def one_hot_encode(actions, num_actions):
    """
    Convert an array of actions into one-hot encoding.
    
    Args:
    - actions (np.array): Array of integer actions.
    - num_actions (int): Total number of possible actions (size of action space).
    
    Returns:
    - np.array: One-hot encoded actions, shape (num_samples, num_actions).
    """
    actions = np.array(actions)  
    one_hot_actions = np.zeros((len(actions), num_actions), dtype=np.float32)
    one_hot_actions[np.arange(len(actions)), actions] = 1
    return np.array(one_hot_actions)




def load_npz_files_from_folder(folder_path, start=0, end=100):
    all_observations = []
    all_rewards = []
    all_actions = []
    all_dones = []
    all_next_observations = []

    
    folder = os.listdir(folder_path)
    # Iterate through all .npz files in the folder
    for filename in folder[start:end]:
        if filename.endswith(".npz"):
            file_path = os.path.join(folder_path, filename)
            npz_data = np.load(file_path, allow_pickle=True)
            
            all_observations.append(npz_data['obs'])
            all_rewards.append(npz_data['reward'])
            all_actions.append(npz_data['action'])
            all_dones.append(npz_data['done'])
            all_next_observations.append(npz_data['obs_next'])
    

    # Concatenate all arrays along the first axis (stacking the data)
    observations = np.concatenate(all_observations, axis=0)
    rewards = np.concatenate(all_rewards, axis=0)
    actions = np.concatenate(all_actions, axis=0)
    dones = np.concatenate(all_dones, axis=0)
    next_observations = np.concatenate(all_next_observations, axis=0)


    reward_min = rewards.min()
    reward_max = rewards.max()
    rewards = (rewards - reward_min) / (reward_max - reward_min)
    logging.info(f"reward min : {reward_min} - reward max : {reward_max}")

    one_hot_actions = one_hot_encode(actions, config.action_dim)
    
    return observations, rewards, one_hot_actions, dones, next_observations




class GrdiDataset(Dataset):
    def __init__(self, observations, rewards, actions, dones, next_observations, device):
        self.observations = torch.tensor(np.array(observations, np.float32), dtype=torch.float32, device=device)
        self.rewards = torch.tensor(np.array(rewards, np.float32), dtype=torch.float32, device=device)
        self.actions = torch.tensor(actions, dtype=torch.int32, device=device)  # Assuming discrete actions
        self.dones = torch.tensor(np.array(dones, np.float32), dtype=torch.float32, device=device)  # Done flags as float
        self.next_observations = torch.tensor(np.array(next_observations, np.float32), dtype=torch.float32, device=device)
        
    def __len__(self):
        return len(self.observations)
    
    def __getitem__(self, idx):
        return (self.observations[idx], self.rewards[idx], self.actions[idx], self.dones[idx], self.next_observations[idx])



def load_observation(folder_path, start=0, end=100):
    all_observations = []

    
    folder = os.listdir(folder_path)
    # Iterate through all .npz files in the folder
    for filename in folder[start:end]:
        if filename.endswith(".npz"):
            file_path = os.path.join(folder_path, filename)
            npz_data = np.load(file_path, allow_pickle=True)
            
            all_observations.append(npz_data['obs'])
    

    # Concatenate all arrays along the first axis (stacking the data)
    observations = np.concatenate(all_observations, axis=0)
    
    return observations



class ObsDataset(Dataset):
    def __init__(self, observations, device):
        self.observations = torch.tensor(np.array(observations, np.float32), dtype=torch.float32, device=device)
        
    def __len__(self):
        return len(self.observations)
    
    def __getitem__(self, idx):
        return (self.observations[idx])
