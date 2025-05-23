import torch
import os
import dill
import numpy as np
from torch.utils.data import Dataset, DataLoader
from gridformer.Utils.utils import Config
from gridformer.Utils.logger import logger


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




def extract_essential_obs(obs_dict: dict):
    """
    Extract important features from the observation dictionary and flatten into a list.
    """
    flat = []

    # Flatten nested components
    for comp in ['loads', 'gens', 'prods', 'lines_or', 'lines_ex']:
        if comp in obs_dict:
            for subkey, values in obs_dict[comp].items():
                flat.extend(values.tolist())

    # Flatten flat components
    for key in ['rho', 'line_status', 'topo_vect']:
        if key in obs_dict:
            flat.extend(obs_dict[key].astype(float).tolist())

    return flat




def load_from_multiple_pkl_as_numpy(file_paths):
    """
    Load multiple .pkl files and convert obs/next_obs to essential vectors.
    Return everything as concatenated np arrays.

    Args:
        file_paths (list): list of paths to .pkl files

    Returns:
        tuple of np.ndarray: (observations, rewards, actions, dones, next_observations)
    """
    all_obs = []
    all_next_obs = []
    all_rewards = []
    all_actions = []
    all_dones = []

    for file_path in file_paths:
        print(file_path)
        with open(file_path, 'rb') as f:
            data = dill.load(f)

        for ob, nob in zip(data['obs'], data['next_obs']):
            ob_dict = ob.to_dict() if hasattr(ob, 'to_dict') else ob
            nob_dict = nob.to_dict() if hasattr(nob, 'to_dict') else nob

            all_obs.append(extract_essential_obs(ob_dict))
            all_next_obs.append(extract_essential_obs(nob_dict))

        all_rewards.extend(data['rewards'])
        all_actions.extend(data['actions'])
        all_dones.extend(data['done'])

    # Convert to NumPy arrays
    obs = np.array(all_obs, dtype=np.float32)
    next_obs = np.array(all_next_obs, dtype=np.float32)
    rewards = np.array(all_rewards, dtype=np.float32)
    actions = np.array(all_actions, dtype=np.int32)
    dones = np.array(all_dones, dtype=np.float32)

    return obs, rewards, actions, dones, next_obs


def load_from_pkl_as_numpy(file_path):
    """
    Load .pkl file and convert obs/next_obs to essential vectors. Return everything as np arrays.
    
    Args:
        file_path (str): path to .pkl file
    
    Returns:
        tuple of np.ndarray: (observations, rewards, actions, dones, next_observations)
    """
    with open(file_path, 'rb') as f:
        data = dill.load(f)

    obs = []
    next_obs = []
    rewards = data['rewards']
    actions = data['actions']
    dones = data['done']

    for ob, nob in zip(data['obs'], data['next_obs']):
        ob_dict = ob.to_dict() if hasattr(ob, 'to_dict') else ob
        nob_dict = nob.to_dict() if hasattr(nob, 'to_dict') else nob
        

        obs.append(extract_essential_obs(ob_dict))
        next_obs.append(extract_essential_obs(nob_dict))

    # Convert everything to NumPy arrays
    obs = np.array(obs, dtype=np.float32)
    next_obs = np.array(next_obs, dtype=np.float32)
    rewards = np.array(rewards, dtype=np.float32)
    actions = np.array(actions, dtype=np.int32)
    dones = np.array(dones, dtype=np.float32)

    return obs, rewards, actions, dones, next_obs



class GridSequenceDataset(Dataset):
    def __init__(self, observations, rewards, actions, dones, next_observations, seq_len, device):
        self.seq_len = seq_len
        self.device = device

        # Convert to tensors first
        self.observations = torch.tensor(np.array(observations, np.float32), dtype=torch.float32)
        self.rewards = torch.tensor(np.array(rewards, np.float32), dtype=torch.float32)
        self.actions = torch.tensor(np.array(actions, np.float32), dtype=torch.float32)  # One-hot or continuous
        self.dones = torch.tensor(np.array(dones, np.float32), dtype=torch.float32)
        self.next_observations = torch.tensor(np.array(next_observations, np.float32), dtype=torch.float32)

    def __len__(self):
        return len(self.observations) - self.seq_len

    def __getitem__(self, idx):
        obs_seq = self.observations[idx:idx+self.seq_len]
        act_seq = self.actions[idx:idx+self.seq_len].unsqueeze(-1).to(self.device)  # Make it [seq_len, 1]

        # Make target also 3D: [seq_len=1, dim]
        reward = self.rewards[idx + self.seq_len - 1].unsqueeze(0)
        done = self.dones[idx + self.seq_len - 1].unsqueeze(0)
        next_obs = self.next_observations[idx + self.seq_len - 1].unsqueeze(0)

        return obs_seq.to(self.device), act_seq.to(self.device), reward.to(self.device), done.to(self.device), next_obs.to(self.device)



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
