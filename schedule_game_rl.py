import gymnasium as gym
from tqdm import tqdm
import numpy as np
from environment import PrioritySelectionPairedKidneyDonationEnv
from scipy.stats import gmean
import random
import matplotlib.pyplot as plt
import os
from stable_baselines3 import PPO
from omegaconf import OmegaConf
from fire import Fire
from stable_baselines3.common.vec_env import DummyVecEnv

class RLPairedKidneyDonationEnv(PrioritySelectionPairedKidneyDonationEnv):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.action_space = gym.spaces.Discrete(2 ** 6)
        self.observation_space = gym.spaces.Box(
            low=0,
            high=float('inf'),
            space=(13 + self.n_timesteps,),
            dtype=np.float32
        )

    def get_time_categorizations(self):
        obs = super().get_observation()
        time_remaining = obs["departures"] - obs["timestep"]
        is_urgent = time_remaining <= 1
        is_soon = (time_remaining > 1) & (time_remaining <= 3)
        is_early = time_remaining > 3
        
        times = np.zeros(self.n_agents, dtype=int)
        times[is_early] = 0
        times[is_soon] = 1
        times[is_urgent] = 2

        return times

    def get_observation(self):
        obs = super().get_observation()
        times = self.get_time_categorizations()
        is_hard = obs["is_hard"]

        agent_counts = np.zeros(6)
        for t in range(3):
            for h in range(2):
                agent_counts[t + h*3] = np.sum((times == t) & (is_hard == h))
        
        match_counts = np.zeros(6)
        for t in range(3):
            for h in range(2):
                category_mask = (times == t) & (is_hard == h)
                if np.any(category_mask):
                    match_counts[t + h*3] = np.sum(obs["adjacency"][category_mask])
        
        timesteps_discrete = np.zeros(self.n_timesteps + 1, dtype=int)
        timesteps_discrete[obs["timestep"]] = 1

        time_progress = np.array([obs["timestep"] / self.n_timesteps], dtype=np.float32)
        flat_obs = np.concatenate([match_counts, agent_counts, timesteps_discrete, time_progress])
        return flat_obs

    def step(self, action):
        obs = super().get_observation()

        bits = f"{action:06b}"
        selection = np.array([[int(bits[0]), int(bits[1]), int(bits[2])],
                        [int(bits[3]), int(bits[4]), int(bits[5])]])
        
        timestep = obs["timestep"]
        is_hard = obs["is_hard"]
        times = self.get_time_categorizations()

        important_nodes = np.zeros(self.n_agents, dtype=bool)

        for htm_index in range(2): # easy to match, hard to match
            for time_index in range(3): # early, middle, late
                if selection[htm_index, time_index] == 1: # if you should not select this, then zero out the action
                    relevant_nodes = np.where(np.logical_and(is_hard == htm_index, times == time_index))[0]
                    if len(relevant_nodes) > 0:
                        important_nodes[relevant_nodes] = True
        
        return super().step(important_nodes)
        
        
    def reset(self, **kwargs):
        obs, info = super().reset(**kwargs)
        return self.get_observation(), info
    
def make_env(seed, **kwargs):
    def inner_make_env():
        env = RLPairedKidneyDonationEnv(
            seed=seed,
            **kwargs
        )
        return env
    return inner_make_env

def evaluate_solution(model, envs):
    stats = {
        "greedy_ratios": [],
        "patient_ratios": [],
        "model_reward": [],
        "greedy_reward": [],
        "patient_reward": [],
    }

    for env in tqdm(envs, desc="Final evaluation"):
        obs, _ = env.reset()
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, _, done, _, _ = env.step(action)
        model_reward = np.sum(env.matched_agents) / env.n_agents
        stats["model_reward"].append(model_reward)
        greedy_reward = env.get_greedy_percentage()
        stats["greedy_reward"].append(greedy_reward)
        stats["greedy_ratios"].append(model_reward / greedy_reward)
        patient_reward = env.get_patient_percentage()
        stats["patient_reward"].append(patient_reward)
        stats["patient_ratios"].append(model_reward / patient_reward)
    
    return stats

def plot_evaluation_stats(stats, save_path):
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    plt.figure(figsize=(10, 5))
    plt.boxplot([stats["greedy_ratios"], stats["patient_ratios"]], 
                labels=['vs Greedy', 'vs Patient'])
    plt.title('Performance Ratios of RL Model')
    plt.ylabel('Ratio')
    plt.grid(True)
    plt.savefig(os.path.join(save_path, 'ratios.png'))

    plt.close()

    plt.figure(figsize=(10, 5))
    rewards_data = [stats["model_reward"], stats["greedy_reward"], stats["patient_reward"]]
    plt.boxplot(rewards_data, labels=['RL Model', 'Greedy', 'Patient'])
    plt.title('Matching Rewards Comparison')
    plt.ylabel('Percentage of Matched Agents')
    plt.grid(True)
    plt.savefig(os.path.join(save_path, 'rewards.png'))

    plt.close()


def main(config_file: str):
    config = OmegaConf.load(config_file)
    train_envs = DummyVecEnv([lambda: make_env(seed) for seed in range(config.num_train_envs)])
    test_envs = DummyVecEnv([lambda: make_env(seed) for seed in range(config.num_train_envs, config.num_train_envs + config.num_test_envs)])

    model = PPO("MlpPolicy", train_envs, verbose=1, **config.training)
    model.learn(total_timesteps=config.train_timesteps)

    if config.save_model:
        model.save(config.model_save_path)

    stats = evaluate_solution(model, test_envs)
    plot_evaluation_stats(stats, config.save_path)

if __name__ == "__main__":
    Fire(main)