# vibe coded based on the original schedule_game.py code

import gymnasium as gym
from tqdm import tqdm
import numpy as np
from environment import PrioritySelectionPairedKidneyDonationEnv
from scipy.stats import gmean
import random
import matplotlib.pyplot as plt
import os
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor

seed = 42
np.random.seed(seed)
random.seed(seed)

# Environment parameters
n_agents = 100
n_timesteps = 64
death_time = 32
p = 0.3
q = 0.15
pct_hard = 0.8

# Training and evaluation parameters
num_envs = 16
num_eval_envs = 128
eval_freq = 2048
total_timesteps = 500000

# Create environment seeds
env_seeds = np.random.randint(0, 2**32 - 1, size=num_envs).tolist()
eval_env_seeds = np.random.randint(0, 2**32 - 1, size=num_eval_envs).tolist()

class FlatKidneyDonationEnvWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        # Action space: 0 to 2^6 - 1 to represent different selection strategies
        self.action_space = gym.spaces.Discrete(2**6)
        
        # Flattened observation space - 13 values total
        N = self.env.n_agents
        
        # Create a flat observation space (all features combined into a single vector)
        self.observation_space = gym.spaces.Box(
            low=0,
            high=float('inf'),
            shape=(13,),  # 6 match counts + 1 time progress + 6 agent counts
            dtype=np.float32
        )
        
        # Keep full observation for action translation
        self._full_obs = None
    
    def _simplify_and_flatten_observation(self, obs):
        """Convert full observation to simplified form and flatten it"""
        N = self.env.n_agents
        timestep = obs["timestep"]
        if isinstance(timestep, np.ndarray):
            timestep = timestep[0]
            
        # Calculate time categories
        time_remaining = obs["departures"] - timestep
        is_urgent = time_remaining <= 1
        is_soon = (time_remaining > 1) & (time_remaining <= 3)
        is_early = time_remaining > 3
        
        # Create category arrays
        times = np.zeros(N, dtype=int)
        times[is_early] = 0
        times[is_soon] = 1
        times[is_urgent] = 2
        
        is_hard = obs["is_hard"]
        
        # Count agents in each category (3 time categories x 2 hardness categories)
        agent_counts = np.zeros(6, dtype=np.float32)
        for t in range(3):
            for h in range(2):
                agent_counts[t + h*3] = np.sum((times == t) & (is_hard == h))
        
        # Count potential matches for each category
        adjacency = obs["adjacency"]
        match_counts = np.zeros(6, dtype=np.float32)
        
        for t in range(3):
            for h in range(2):
                category_mask = (times == t) & (is_hard == h)
                if np.any(category_mask):
                    # Count total potential matches for this category
                    match_counts[t + h*3] = np.sum(adjacency[category_mask])
        
        # Time progress as a single value
        time_progress = np.array([timestep / n_timesteps], dtype=np.float32)
        
        # Flatten everything into a single vector
        flat_obs = np.concatenate([match_counts, time_progress, agent_counts])
        
        return flat_obs
    
    def step(self, action):
        # Convert the discrete action (0-63) into the selection matrix
        action_matrix = translate_number_to_action(action, self._full_obs)
        
        # Step the environment
        obs, reward, done, truncated, info = self.env.step(action_matrix)
        
        # Store full observation for action translation
        self._full_obs = obs.copy()
        self._full_obs["timestep"] = np.array([obs["timestep"]], dtype=np.int32)
        
        # Return flattened observation
        flat_obs = self._simplify_and_flatten_observation(self._full_obs)
        
        return flat_obs, reward, done, truncated, info
    
    def reset(self, **kwargs):
        obs, info = self.env.start_over()
        
        # Store full observation
        self._full_obs = obs.copy()
        self._full_obs["timestep"] = np.array([obs["timestep"]], dtype=np.int32)
        
        # Return flattened observation
        flat_obs = self._simplify_and_flatten_observation(self._full_obs)
        
        return flat_obs, info

def translate_number_to_action(number, obs): # number should be from 0 to 2^6 - 1
    bits = f"{number:06b}"
    selection = np.array([[int(bits[0]), int(bits[1]), int(bits[2])],
                       [int(bits[3]), int(bits[4]), int(bits[5])]])
    
    timestep = obs["timestep"]
    if isinstance(timestep, np.ndarray):
        timestep = timestep[0]  # Extract scalar value from array

    N, _ = obs["adjacency"].shape
    is_hard = obs["is_hard"]

    time_remaining = obs["departures"] - timestep
    is_urgent = time_remaining <= 1
    is_soon = (time_remaining > 1) & (time_remaining <= 3)
    is_early = time_remaining > 3
    
    times = np.zeros(N, dtype=int)
    times[is_early] = 0
    times[is_soon] = 1
    times[is_urgent] = 2

    action = obs["adjacency"].copy()

    for htm_index in range(2): # easy to match, hard to match
        for time_index in range(3): # early, middle, late
            if selection[htm_index, time_index] == 0: # if you should not select this, then zero out the action
                relevant_nodes = np.where(np.logical_and(is_hard == htm_index, times == time_index))[0]
                action[relevant_nodes, :] = 0
                action[:, relevant_nodes] = 0

    return action

# Function to make a vectorized environment
def make_env(seed, rank=0):
    def _init():
        env = PrioritySelectionPairedKidneyDonationEnv(
            n_agents=n_agents,
            n_timesteps=n_timesteps,
            death_time=death_time,
            seed=seed,
            p=p,
            q=q,
            pct_hard=pct_hard
        )
        env = FlatKidneyDonationEnvWrapper(env)
        env = Monitor(env)
        return env
    return _init

# Custom evaluation callback
class KidneyMatchingEvalCallback(BaseCallback):
    def __init__(self, eval_envs, greedy_rewards, eval_freq=1000, verbose=1):
        super().__init__(verbose)
        self.eval_envs = eval_envs
        self.eval_freq = eval_freq
        self.best_mean_reward = -np.inf
        self.greedy_rewards = greedy_rewards
        
    def _on_step(self):
        if self.n_calls % self.eval_freq != 0:
            return True
            
        model_rewards = []
        eval_sample = self.eval_envs[:16]  # Use subset for faster evaluation during training
        for env in tqdm(eval_sample, desc="Evaluating", leave=False):
            obs, _ = env.reset()
            done = False
            while not done:
                action, _ = self.model.predict(obs, deterministic=True)
                obs, _, done, _, _ = env.step(action)
            total_reward = np.sum(env.env.matched_agents) / env.env.n_agents
            model_rewards.append(total_reward)
        
        model_rewards = np.array(model_rewards)
        ratios = model_rewards / self.greedy_rewards[:16]
        
        mean_ratio = gmean(ratios)
        
        if mean_ratio > self.best_mean_reward:
            self.best_mean_reward = mean_ratio
            if self.verbose > 0:
                print(f"New best mean reward ratio: {mean_ratio:.4f}")
            # Save best model
            path = os.path.join("results", "best_kidney_model_flat")
            self.model.save(path)
        
        if self.verbose > 0:
            print(f"Eval episode mean ratio: {mean_ratio:.4f}")
            print(f"Min ratio: {np.min(ratios):.4f}")
            print(f"Max ratio: {np.max(ratios):.4f}")
            print(f"Median ratio: {np.median(ratios):.4f}")
        
        return True

def evaluate_solution(model, envs, greedy_rewards):
    model_rewards = []
    for env in tqdm(envs, desc="Final evaluation"):
        obs, _ = env.reset()
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, _, done, _, _ = env.step(action)
        total_reward = np.sum(env.env.matched_agents) / env.env.n_agents
        model_rewards.append(total_reward)
    
    model_rewards = np.array(model_rewards)
    ratios = model_rewards / greedy_rewards
    return ratios

def describe_performance(performance):
    print(f"Mean: {np.mean(performance)}")
    print(f"Std: {np.std(performance)}")
    print(f"Min: {np.min(performance)}")
    print(f"Max: {np.max(performance)}")
    print(f"Geometric mean: {gmean(performance)}")
    print(f"Median: {np.median(performance)}")
    print(f"10th percentile: {np.percentile(performance, 10)}")
    print(f"25th percentile: {np.percentile(performance, 25)}")
    print(f"75th percentile: {np.percentile(performance, 75)}")
    print(f"90th percentile: {np.percentile(performance, 90)}")

if __name__ == "__main__":
    # Create results directory if it doesn't exist
    os.makedirs("results", exist_ok=True)
    
    # Create vectorized environments for training
    # Using DummyVecEnv instead of SubprocVecEnv for better error reporting
    vec_env = DummyVecEnv([make_env(env_seeds[i], i) for i in range(num_envs)])
    
    # Create evaluation environments
    eval_envs = [
        FlatKidneyDonationEnvWrapper(
            PrioritySelectionPairedKidneyDonationEnv(
                n_agents=n_agents,
                n_timesteps=n_timesteps,
                death_time=death_time,
                seed=i,
                p=p,
                q=q,
                pct_hard=pct_hard
            )
        )
        for i in eval_env_seeds
    ]
    
    # Get greedy rewards for evaluation comparison
    eval_greedy_rewards = []
    for env in tqdm(eval_envs, desc="Computing greedy baselines"):
        eval_greedy_rewards.append(env.env.get_greedy_percentage())
    eval_greedy_rewards = np.array(eval_greedy_rewards)
    
    # Define model
    model = PPO(
        policy="MlpPolicy",  # For flat vectors
        env=vec_env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        clip_range_vf=None,
        normalize_advantage=True,
        ent_coef=0.01,  # Add some exploration
        vf_coef=0.5,
        max_grad_norm=0.5,
        verbose=1,
        seed=seed
    )
    
    # Setup evaluation callback
    eval_callback = KidneyMatchingEvalCallback(
        eval_envs=eval_envs,
        greedy_rewards=eval_greedy_rewards,
        eval_freq=eval_freq,
        verbose=1
    )
    
    # Train the model
    model.learn(
        total_timesteps=total_timesteps,
        callback=eval_callback
    )
    
    # Save the final model
    model.save("results/final_kidney_model_flat")
    
    # Load the best model for final evaluation
    best_model = PPO.load("results/best_kidney_model_flat")
    
    # Perform final evaluation
    eval_ratios = evaluate_solution(best_model, eval_envs, eval_greedy_rewards)
    
    print("Final evaluation stats:")
    describe_performance(eval_ratios)
    
    print(f"Final eval ratios: {eval_ratios}")
    plt.title("Flattened Model Performance Ratios")
    plt.boxplot(eval_ratios)
    plt.savefig("results/rl_schedule_ratios_flat.png")
    plt.show()