from environment import PairedKidneyDonationEnv
from model import PairedKidneyModel
import numpy as np
from torch.distributions.normal import Normal
import torch
import gymnasium as gym
from tqdm import tqdm

env = PairedKidneyDonationEnv(
    n_agents=2000,
    n_timesteps=360 // 10,
    criticality_rate=180 // 10
)

class REINFORCE:
    def __init__(self, device="mps"):
        self.learning_rate = 1e-4
        self.gamma = 0.99
        self.eps = 1e-6
        
        self.model = PairedKidneyModel(
            hidden_dim=32,
            num_layers=6
        )
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

    def update(self, reward: torch.Tensor) -> None:
        log_probs = torch.stack(self.probs)
        log_probs_mean = log_probs.mean(dim=0)

        print(log_probs.shape, log_probs_mean.shape, reward)
        
        loss = -torch.mean(log_probs_mean) * reward

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.probs = []

wrapped_env = gym.wrappers.RecordEpisodeStatistics(
    env,
    50
)

total_episodes = 2000
seed = 42

agent = REINFORCE(
    device="cpu" # faster on CPU - probably because less transfers
)
reward_over_episodes = []
for episode in tqdm(range(total_episodes)):
    obs, info = wrapped_env.reset(seed=seed)
    reward, done = 0, False

    while not done:
        action = agent.sample_action(obs)
        observation, reward, done, _, info = wrapped_env.step(action)
    
    agent.update(reward)
    reward_over_episodes.append(reward)

    if (episode + 1) % 100 == 0:
        print(f"Episode {episode}: average reward = {np.mean(reward_over_episodes[-100:])}")

wrapped_env.close()
agent.model.save("reinforce_model.pth") # model is pretty small - doesn't need to be gitignored