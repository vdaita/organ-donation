import torch
import numpy as np
import gymnasium as gym
from tqdm import tqdm
import os
import matplotlib.pyplot as plt
from model import DecisionTransformer
from gym_env import PairedOrganDonationEnv
import random

torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

def get_device():
    if torch.mps.is_available():
        return 'mps'
    elif torch.cuda.is_available():
        return 'cuda'
    else:
        return 'cpu'
    
class REINFORCE:
    """REINFORCE algorithm, from Gymnasium Documentation"""

    def __init__(self):
        """Initializes an agent that learns a policy via REINFORCE algorithm [1]
        to solve the task at hand.

        Args:
            obs_space_dims: Dimension of the observation space
            action_space_dims: Dimension of the action space
        """

        # Hyperparameters
        self.learning_rate = 1e-4  # Learning rate for policy optimization
        self.gamma = 0.99  # Discount factor
        self.eps = 1e-6  # small number for mathematical stability

        self.probs = []  # stores the probability values of the sampled actions
        self.rewards = []  # stores the rewards

        self.device = get_device()
        self.net = DecisionTransformer(n_attr=25, hidden_dim=128, n_heads=8, n_layers=2).to(self.device)
        self.optimizer = torch.optim.AdamW(self.net.parameters(), lr=self.learning_rate)

    def sample_action(self, obs: dict) -> float:
        """Returns an action, conditioned on the policy and observation.

        Args:
            state: Observation from the environment

        Returns:
            action: Action to be performed
        """

        patients = torch.tensor(obs["patients"], dtype=torch.float32, device=self.device)
        matched_patients = torch.tensor(obs["matched_patients"], dtype=torch.bool, device=self.device)
        current_selection = torch.tensor(obs["current_selection"], dtype=torch.bool, device=self.device)

        action_probs = self.net(matched_patients, current_selection, patients) # already softmaxed
        print("Probs: ", action_probs)

        # Sample an action from the action probabilities
        m = torch.distributions.Categorical(action_probs)
        action = m.sample()
        prob = m.log_prob(action)

        self.probs.append(prob)
        return action

    def update(self):
        """Updates the policy network's weights."""
        running_g = 0
        gs = []

        # Discounted return (backwards) - [::-1] will return an array in reverse
        for R in self.rewards[::-1]:
            running_g = R + self.gamma * running_g
            gs.insert(0, running_g)

        deltas = torch.tensor(gs)

        log_probs = torch.stack(self.probs)

        # Calculate the mean of log probabilities for all actions in the episode
        log_prob_mean = log_probs.mean()

        # Update the loss with the mean log probability and deltas
        # Now, we compute the correct total loss by taking the sum of the element-wise products.
        loss = -torch.sum(log_prob_mean * deltas)

        # Update the policy network
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Empty / zero out all episode-centric/related variables
        self.probs = []
        self.rewards = []

if __name__ == "__main__":
    # Initialize the environment
    env = PairedOrganDonationEnv(num_pairs=8, max_steps=16)

    # Initialize the agent
    agent = REINFORCE()

    # Training loop
    num_episodes = 500
    rewards = []
    for episode in tqdm(range(num_episodes)):
        obs, _ = env.reset()
        done = False
        total_reward = 0

        while not done:
            action = agent.sample_action(obs)
            obs, reward, done, _, _ = env.step(action.item())
            agent.rewards.append(reward)
            total_reward += reward

        rewards.append(total_reward)
        agent.update()

    # Plot the rewards
    plt.plot(rewards)
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title("REINFORCE Algorithm")
    plt.show()