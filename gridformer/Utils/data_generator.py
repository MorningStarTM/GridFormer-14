import os
import torch
import dill
from tqdm import tqdm
from grid2op.Exceptions import *
from gridformer.Utils.converter import ActionConverter
from grid2op.Agent import TopologyGreedy

class DataGenerator:
    def __init__(self, env, agent:TopologyGreedy, action_converter:ActionConverter, save_path="data", use_agent=False):
        """
        Args:
            env: The Grid2Op environment.
            agent: The agent to use (random or trained).
            action_converter: Object that converts actions to Grid2Op format.
            save_path: Directory to save the generated data.
            use_agent: Whether to use the trained agent or random actions.
        """
        self.env = env
        self.agent = agent
        self.converter = action_converter
        self.save_path = save_path
        self.use_agent = use_agent

        if not os.path.exists(save_path):
            os.makedirs(save_path)

    def generate(self, start_episode=0, end_episode=1):
        """
        Generate data for episodes in the given range.

        Args:
            start_episode: Starting episode index.
            end_episode: Ending episode index (non-inclusive).
        """
        for episode_id in range(start_episode, end_episode):
            print(f"Generating data for Episode: {episode_id}")
            self.env.set_id(episode_id)
            obs = self.env.reset()
            reward = self.env.reward_range[0]
            done = False

            obs_data, action_data, reward_data, next_obs_data, done_data = [], [], [], [], []

            for t in tqdm(range(self.env.max_episode_duration()), desc=f"Episode {episode_id}", leave=True):
                try:
                    if self.use_agent:
                        act = self.agent.act(obs, reward, done)
                        action = self.converter.action_idx(act)
                    else:
                        action = self.env.action_space.sample()

                    next_obs, reward, done, _ = self.env.step(act)

                    obs_data.append(obs)
                    action_data.append(action)
                    reward_data.append(reward)
                    next_obs_data.append(next_obs)
                    done_data.append(done)

                    obs = next_obs

                    if done:
                        self.env.set_id(episode_id)
                        
                        obs = self.env.reset()
                        done = False
                        reward = self.env.reward_range[0]

                        self.env.fast_forward_chronics(t - 1)
                        act = self.agent.act(obs, reward, done) 
                        next_obs, reward, done, _ = self.env.step(act)
                        action = self.converter.action_idx(act)


                        obs_data.append(obs)
                        action_data.append(action)
                        reward_data.append(reward)
                        next_obs_data.append(next_obs)
                        done_data.append(done)

                        obs = next_obs

                except NoForecastAvailable as e:
                    print(f"Grid2OpException encountered at step {t} in episode {episode_id}: {e}")
                    self.env.set_id(episode_id)
                    obs = self.env.reset()
                    self.env.fast_forward_chronics(t-1)
                    continue

                except Grid2OpException as e:
                    print(f"Grid2OpException encountered at step {t} in episode {episode_id}: {e}")
                    self.env.set_id(episode_id)
                    obs = self.env.reset()
                    self.env.fast_forward_chronics(t-1)
                    continue 

            data = {
                "obs": obs_data,
                "actions": action_data,
                "rewards": reward_data,
                "next_obs": next_obs_data,
                "done": done_data
            }

            self._save_data(data, episode_id)

    def _save_data(self, data, episode_id):
        """
        Save data to a .pkl file.

        Args:
            data: Dictionary containing episode data.
            episode_id: ID of the episode.
        """
        file_path = os.path.join(self.save_path, f"episode_{episode_id}.pkl")
        with open(file_path, "wb") as f:
            dill.dump(data, f)
        print(f"Saved episode {episode_id} data to {file_path}")
