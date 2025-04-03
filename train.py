from environment import PairedKidneyDonationEnv
from model import PairedKidneyModel
import numpy as np
from torch.distributions.normal import Normal
import torch
import gymnasium as gym
from tqdm import tqdm
from baselines import get_greedy_percentage, get_periodic_percentage

env = PairedKidneyDonationEnv(
    n_agents=250,
    n_timesteps=36,
    criticality_rate=18
)

class REINFORCE:
    def __init__(self, device="cpu", model=None):
        self.learning_rate = 1e-5
        self.gamma = 0.99
        
        if not model:
            self.model = PairedKidneyModel(
                hidden_dim=32,
                num_layers=6
            )
        else:
            self.model = model
        print("Number of parameters: ", sum(p.numel() for p in self.model.parameters()))
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.device = torch.device(device)

        self.model.to(self.device)

        self.probs = []

    def sample_action(self, state: dict) -> float:
        adj_matrix = torch.Tensor(state["adjacency_matrix"])
        timestep = torch.Tensor([state["timestep"]])
        adj_matrix = adj_matrix.to(self.device)
        timestep = timestep.to(self.device)

        item_priority, match_priority, match_global = self.model(adj_matrix, timestep)

        item_priority_distrib = Normal(item_priority, 0.1)
        match_priority_distrib = Normal(match_priority, 0.1)
        match_global_distrib = Normal(match_global, 0.1)

        item_priority = item_priority_distrib.sample()
        match_priority = match_priority_distrib.sample()
        match_global = match_global_distrib.sample()

        self.probs.append(torch.cat([
            item_priority_distrib.log_prob(item_priority).flatten(),
            match_priority_distrib.log_prob(match_priority).flatten(),
            match_global_distrib.log_prob(match_global).flatten()
        ]))

        item_priority = item_priority.cpu().detach().numpy()
        match_priority = match_priority.cpu().detach().numpy()
        match_global = match_global.cpu().detach().numpy()

        # after you've sampled values from the distribution, you have to align them to 0 or 1 for each of them
        item_priority = (item_priority > 0.5).astype(np.int32)
        match_priority = (match_priority > 0.5).astype(np.int32)
        match_global = (match_global > 0.5).astype(np.int32)

        return {
            "selection": item_priority,
            "match_selection": match_priority,
            "match_regular": match_global
        }

    def update(self, reward) -> None:
        reward_tensor = torch.tensor(reward, dtype=torch.float32, device=self.device)
        log_probs = torch.stack(self.probs)
        log_probs_mean = log_probs.mean(dim=0)

        # print(log_probs.shape, log_probs_mean.shape, reward)
        
        loss = -torch.mean(log_probs_mean) * reward_tensor

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.probs = []


# MPS has some issues when trying to compute the loss

# training constants
total_episodes = 2000
imitation_episodes = 10
greedy_eval_lookback = 4
eval_period = 20
batch_size = 16

# Imitation learning


basic_model = PairedKidneyModel(
    hidden_dim=16,
    num_layers=8
)
basic_optimizer = torch.optim.Adam(basic_model.parameters(), lr=1e-4)

for i in tqdm(range(imitation_episodes)):
    obs, info = env.reset()
    reward, done = 0, False

    while not done:
        action = {
            "selection": np.zeros(env.n_agents),
            "match_selection": 0,
            "match_regular": 1
        }
        observation, reward, done, _, info = env.step(action)
        
        adj_matrix = torch.Tensor(observation["adjacency_matrix"])
        timestep = torch.Tensor([observation["timestep"]])
        
        item_priority, match_priority, match_global = basic_model(adj_matrix, timestep)
        loss = torch.mean((item_priority) ** 2 + (match_priority) ** 2 + (match_global - 1) ** 2)
        
        basic_optimizer.zero_grad()
        loss.backward()
        basic_optimizer.step()
        
print(f"Greedy imitation loss = {loss.item()}")
    
agent = REINFORCE(
    device="cpu", # faster on CPU - probably because less transfers
    model=basic_model
)

# Reinforcement learning 
reward_over_episodes = []
recent_greedy_rewards = []

recent_rewards = []

for episode in tqdm(range(total_episodes)):
    obs, info = env.reset()
    reward, done = 0, False

    while not done:
        action = agent.sample_action(obs)
        observation, reward, done, _, info = env.step(action)
    
    reward_over_episodes.append(reward)
    recent_rewards.append(reward)

    if len(recent_rewards) > batch_size:
        agent.update(np.mean(recent_rewards))
        recent_rewards = []
    
    if ((episode + 1) % eval_period) in list(range(eval_period - greedy_eval_lookback, eval_period)):
        recent_greedy_rewards.append(get_greedy_percentage(env))

    if (episode + 1) % eval_period == 0:        
        mean_greedy_reward = np.mean(recent_greedy_rewards)
        recent_greedy_rewards = []
        print(f"Episode {episode}: average reward = {np.mean(reward_over_episodes[-greedy_eval_lookback:])}, greedy reward = {mean_greedy_reward}")
        

env.close()

torch.save(agent.model.state_dict(), "model.pth")