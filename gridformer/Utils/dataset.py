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


import numpy as np

def filter_relevant_features(obs_dict):
    """
    Extract and filter only the relevant features from the observation dictionary.

    Args:
        obs_dict (dict): The observation dictionary.

    Returns:
        list: A flattened list of relevant features.
    """
    relevant_features = []

    # Line features
    if "lines_or" in obs_dict:
        lines_or = obs_dict["lines_or"]
        relevant_features.extend(lines_or["p"].tolist())  # Active power
        relevant_features.extend(lines_or["q"].tolist())  # Reactive power
        relevant_features.extend(lines_or["v"].tolist())  # Voltage magnitude
        relevant_features.extend(lines_or["a"].tolist())  # Current magnitude

    if "rho" in obs_dict:
        relevant_features.extend(obs_dict["rho"].tolist())  # Line loading ratio

    # Load features
    if "loads" in obs_dict:
        loads = obs_dict["loads"]
        relevant_features.extend(loads["p"].tolist())  # Active power
        relevant_features.extend(loads["q"].tolist())  # Reactive power
        relevant_features.extend(loads["v"].tolist())  # Voltage magnitude

    # Generator features
    if "gens" in obs_dict:
        gens = obs_dict["gens"]
        relevant_features.extend(gens["p"].tolist())  # Active power
        relevant_features.extend(gens["q"].tolist())  # Reactive power
        relevant_features.extend(gens["v"].tolist())  # Voltage magnitude

    # Topology features
    if "topo_vect" in obs_dict:
        relevant_features.extend(obs_dict["topo_vect"].tolist())  # Topology vector

    if "line_status" in obs_dict:
        relevant_features.extend(obs_dict["line_status"].astype(float).tolist())  # Line status (binary)

    # Maintenance and cooldown features
    if "maintenance" in obs_dict and "time_next_maintenance" in obs_dict["maintenance"]:
        relevant_features.extend(obs_dict["maintenance"]["time_next_maintenance"].tolist())  # Time to next maintenance

    if "cooldown" in obs_dict and "line" in obs_dict["cooldown"]:
        relevant_features.extend(obs_dict["cooldown"]["line"].tolist())  # Line cooldown times

    # Redispatching features
    if "redispatching" in obs_dict:
        relevant_features.extend(obs_dict["redispatching"]["target_redispatch"].tolist())  # Target redispatch
        relevant_features.extend(obs_dict["redispatching"]["actual_dispatch"].tolist())  # Actual redispatch

    return relevant_features



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




def load_from_multiple_pkl_as_numpy(folder_path, start=0, end=None):
    """
    Load multiple .pkl files from a folder and convert obs/next_obs to essential vectors.
    Process files from the given start to end indices.
    Return everything as concatenated np arrays.

    Args:
        folder_path (str): Path to the folder containing .pkl files.
        start (int): Start index of the files to process.
        end (int): End index of the files to process.

    Returns:
        tuple of np.ndarray: (observations, rewards, actions, dones, next_observations)
    """
    all_obs = []
    all_next_obs = []
    all_rewards = []
    all_actions = []
    all_dones = []

    # List all .pkl files in the folder and sort them
    file_paths = sorted([os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.pkl')])

    # Slice the file_paths list based on start and end
    file_paths = file_paths[start:end]

    for file_path in file_paths:
        print(f"Processing file: {file_path}")
        with open(file_path, 'rb') as f:
            data = dill.load(f)

        # Extract data using the correct keys
        obs_data = data.get('obs_data', [])
        obs_next_data = data.get('obs_next_data', [])
        reward_data = data.get('reward_data', [])
        action_data = data.get('action_data', [])
        done_data = data.get('done_data', [])

        for ob, nob in zip(obs_data, obs_next_data):
            ob_dict = ob.to_dict() if hasattr(ob, 'to_dict') else ob
            nob_dict = nob.to_dict() if hasattr(nob, 'to_dict') else nob

            all_obs.append(filter_relevant_features(ob_dict))
            all_next_obs.append(filter_relevant_features(nob_dict))

        all_rewards.extend(reward_data)
        all_actions.extend(action_data)
        all_dones.extend(done_data)

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
        

        obs.append(filter_relevant_features(ob_dict))
        next_obs.append(filter_relevant_features(nob_dict))

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




def load_from_multiple_pkl_as_numpy_with_steps(
    folder_path,
    filter_fn=None,          # e.g., filter_relevant_features
    start=0,
    end=None,
    step_cycle=8064,         # steps 0..step_cycle-1
    keys_map=None,           # optional: map your key names
    verbose=True,
):
    """
    Reads multiple episode_*.pkl files and returns:
      (obs, rewards, actions, dones, next_obs, steps) as numpy arrays.

    This version is compatible with your DataGenerator that saves keys:
      "obs", "actions", "rewards", "next_obs", "done"

    Args:
        folder_path: directory containing .pkl files
        filter_fn: function(dict)->vector. If None, tries best-effort flatten.
        start/end: slice on sorted .pkl files
        step_cycle: steps wrap-around period
        keys_map: override default keys if your pkl uses different names, e.g.
            {"obs":"obs_data","next_obs":"obs_next_data","rewards":"reward_data",
             "actions":"action_data","done":"done_data"}
        verbose: prints progress

    Returns:
        obs:      (N, D) float32
        rewards:  (N,) float32
        actions:  (N,) int32
        dones:    (N,) float32
        next_obs: (N, D) float32
        steps:    (N,) int32
    """
    # Default keys for YOUR DataGenerator
    km = {
        "obs": "obs",
        "next_obs": "next_obs",
        "rewards": "rewards",
        "actions": "actions",
        "done": "done",
    }
    if keys_map:
        km.update(keys_map)

    def _default_filter(ob_dict):
        """
        Best-effort: make a numeric 1D vector from a dict by concatenating
        any numpy-like / list-like numeric values found.
        (Recommended: pass your own filter_relevant_features instead.)
        """
        vals = []
        for v in ob_dict.values():
            if isinstance(v, (int, float, bool, np.number)):
                vals.append(np.array([v], dtype=np.float32))
            elif isinstance(v, (list, tuple, np.ndarray)):
                arr = np.asarray(v)
                if arr.dtype.kind in "iufb":  # numeric/bool
                    vals.append(arr.astype(np.float32).ravel())
        if not vals:
            raise ValueError("No numeric features found in observation dict. Provide filter_fn.")
        return np.concatenate(vals, axis=0)

    if filter_fn is None:
        filter_fn = _default_filter

    # collect
    all_obs, all_next_obs = [], []
    all_rewards, all_actions, all_dones = [], [], []

    file_paths = sorted(
        os.path.join(folder_path, f)
        for f in os.listdir(folder_path)
        if f.endswith(".pkl")
    )[start:end]

    if not file_paths:
        raise FileNotFoundError(f"No .pkl files found in: {folder_path}")

    for fp in file_paths:
        if verbose:
            print(f"Processing file: {fp}")

        with open(fp, "rb") as f:
            data = dill.load(f)

        obs_seq = data.get(km["obs"], [])
        next_obs_seq = data.get(km["next_obs"], [])
        rewards_seq = data.get(km["rewards"], [])
        actions_seq = data.get(km["actions"], [])
        dones_seq = data.get(km["done"], [])

        # sanity: sequence lengths should match (at least for obs/next_obs)
        m = min(len(obs_seq), len(next_obs_seq), len(rewards_seq), len(actions_seq), len(dones_seq))
        if m == 0:
            continue

        # per-step
        for i in range(m):
            ob = obs_seq[i]
            nob = next_obs_seq[i]

            ob_dict = ob.to_dict() if hasattr(ob, "to_dict") else ob
            nob_dict = nob.to_dict() if hasattr(nob, "to_dict") else nob

            all_obs.append(filter_fn(ob_dict))
            all_next_obs.append(filter_fn(nob_dict))

        all_rewards.extend(rewards_seq[:m])
        all_actions.extend(actions_seq[:m])
        all_dones.extend(dones_seq[:m])

    obs = np.asarray(all_obs, dtype=np.float32)
    next_obs = np.asarray(all_next_obs, dtype=np.float32)
    rewards = np.asarray(all_rewards, dtype=np.float32)
    actions = np.asarray(all_actions, dtype=np.int32)
    dones = np.asarray(all_dones, dtype=np.float32)

    n = len(obs)
    steps = (np.arange(n, dtype=np.int32) % step_cycle)

    if not (len(rewards) == len(actions) == len(dones) == len(next_obs) == len(steps) == n):
        raise ValueError(
            f"Length mismatch: obs={len(obs)}, next_obs={len(next_obs)}, "
            f"rewards={len(rewards)}, actions={len(actions)}, dones={len(dones)}, steps={len(steps)}"
        )

    return obs, rewards, actions, dones, next_obs, steps






import numpy as np
import torch
from torch.utils.data import Dataset


class RunningStandardScaler:
    """
    Streaming mean/std scaler (Welford-style merge) for feature-wise normalization.

    Usage patterns:
    1) Offline fit (current dataset):
        scaler = RunningStandardScaler()
        scaler.update(obs)
        scaler.update(next_obs)

    2) Streaming:
        scaler.update(obs_batch)
        scaler.update(next_obs_batch)
        # later freeze and reuse same scaler

    Save/load:
        state = scaler.state_dict()
        scaler.load_state_dict(state)
    """
    def __init__(self, eps=1e-8):
        self.eps = eps
        self.mean = None   # shape: (D,)
        self.var = None    # shape: (D,)
        self.count = 0

    def update(self, x):
        # x: (N, D) numpy or torch
        if isinstance(x, torch.Tensor):
            x_np = x.detach().cpu().numpy()
        else:
            x_np = np.asarray(x)

        if x_np.ndim != 2:
            raise ValueError(f"Expected x to have shape (N, D). Got {x_np.shape}.")

        batch_count = x_np.shape[0]
        if batch_count == 0:
            return

        batch_mean = x_np.mean(axis=0)
        batch_var = x_np.var(axis=0)

        if self.mean is None:
            self.mean = batch_mean.astype(np.float64)
            self.var = batch_var.astype(np.float64)
            self.count = int(batch_count)
            return

        # Merge two sets of mean/var
        old_count = self.count
        new_count = old_count + batch_count

        delta = batch_mean - self.mean
        new_mean = self.mean + delta * (batch_count / new_count)

        m_a = self.var * old_count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + (delta * delta) * (old_count * batch_count / new_count)

        new_var = M2 / new_count

        self.mean = new_mean
        self.var = new_var
        self.count = int(new_count)

    def normalize(self, x, device=None, dtype=torch.float32):
        """
        Normalize x using current mean/std.
        x can be numpy or torch. Returns torch.Tensor.
        """
        if self.mean is None or self.var is None or self.count == 0:
            raise RuntimeError("Scaler has no stats. Call update() first or load_state_dict().")

        if isinstance(x, torch.Tensor):
            t = x
        else:
            t = torch.tensor(np.asarray(x, np.float32), dtype=dtype)

        if device is not None:
            t = t.to(device)

        mean_t = torch.tensor(self.mean, dtype=dtype, device=t.device)
        std_t = torch.sqrt(torch.tensor(self.var, dtype=dtype, device=t.device)) + self.eps
        return (t - mean_t) / std_t

    def state_dict(self):
        return {
            "eps": float(self.eps),
            "mean": None if self.mean is None else self.mean.astype(np.float64),
            "var": None if self.var is None else self.var.astype(np.float64),
            "count": int(self.count),
        }

    def load_state_dict(self, state):
        self.eps = float(state.get("eps", 1e-8))
        mean = state.get("mean", None)
        var = state.get("var", None)
        self.mean = None if mean is None else np.asarray(mean, dtype=np.float64)
        self.var = None if var is None else np.asarray(var, dtype=np.float64)
        self.count = int(state.get("count", 0))


class WMDataset(Dataset):
    def __init__(
        self,
        observations,
        rewards,
        actions,
        dones,
        next_observations,
        steps,
        seq_len,
        device,
        scaler=None,
        fit_scaler=False,
        normalize_obs=True,
    ):
        """
        scaler: RunningStandardScaler (shared for obs and next_obs)
        fit_scaler: if True and scaler provided, updates scaler using observations + next_observations
        normalize_obs: if True, applies scaler normalization to obs and next_obs tensors
        """
        self.seq_len = seq_len
        self.device = device
        self.scaler = scaler

        # Convert raw arrays to tensors (still on CPU initially)
        obs_np = np.asarray(observations, np.float32)
        next_obs_np = np.asarray(next_observations, np.float32)

        self.rewards = torch.tensor(np.asarray(rewards, np.float32), dtype=torch.float32)
        self.actions = torch.tensor(np.asarray(actions, np.int64), dtype=torch.long)
        self.dones = torch.tensor(np.asarray(dones, np.float32), dtype=torch.float32)
        self.steps = torch.tensor(np.asarray(steps, np.int32), dtype=torch.long)

        n = len(obs_np)
        if not (len(self.rewards) == len(self.actions) == len(self.dones) == len(next_obs_np) == len(self.steps) == n):
            raise ValueError("Length mismatch in dataset inputs.")

        # Optionally fit/update scaler using available data
        if self.scaler is not None and fit_scaler:
            self.scaler.update(obs_np)
            self.scaler.update(next_obs_np)

        # Apply normalization (IMPORTANT: same scaler for obs and next_obs)
        if normalize_obs:
            if self.scaler is None:
                raise ValueError("normalize_obs=True but scaler=None. Provide a scaler or set normalize_obs=False.")

            self.observations = self.scaler.normalize(obs_np, device=None)         # keep CPU for now
            self.next_observations = self.scaler.normalize(next_obs_np, device=None)
        else:
            self.observations = torch.tensor(obs_np, dtype=torch.float32)
            self.next_observations = torch.tensor(next_obs_np, dtype=torch.float32)

    def __len__(self):
        return len(self.observations) - self.seq_len

    def __getitem__(self, idx):
        obs_seq      = self.observations[idx:idx+self.seq_len]                 # (T, state_dim)
        act_seq      = self.actions[idx:idx+self.seq_len]                      # (T,)
        next_obs_seq = self.next_observations[idx:idx+self.seq_len]            # (T, state_dim)
        reward_seq   = self.rewards[idx:idx+self.seq_len].unsqueeze(-1)        # (T, 1)
        done_seq     = self.dones[idx:idx+self.seq_len].unsqueeze(-1)          # (T, 1)
        steps_seq    = self.steps[idx:idx+self.seq_len]                        # (T,) in 0..8063

        return (
            obs_seq.to(self.device),
            act_seq.to(self.device),
            reward_seq.to(self.device),
            done_seq.to(self.device),
            next_obs_seq.to(self.device),
            steps_seq.to(self.device),
        )



#Usage example:
# scaler = RunningStandardScaler()
# scaler.update(observations)
# scaler.update(next_observations)

# train_ds = WMDataset(
#     observations, rewards, actions, dones, next_observations, steps,
#     seq_len=32, device=device,
#     scaler=scaler, fit_scaler=False, normalize_obs=True
# )


# scaler.update(new_observations)
# scaler.update(new_next_observations)

# IMPORTANT:
# Load model checkpoint (from 50)

# Load scaler v1 (from 50)

# Do NOT update scaler

# Train on new data normalized with scaler v1

# state = scaler.state_dict()
# np.save("wm_scaler.npy", state, allow_pickle=True)

# Later:
# state = np.load("wm_scaler.npy", allow_pickle=True).item()
# scaler.load_state_dict(state)
