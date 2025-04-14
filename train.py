from environment import PairedKidneyDonationEnv
from model import PairedKidneyModel
import numpy as np
from torch.distributions.normal import Normal
import torch
import gymnasium as gym
from tqdm import tqdm
from baselines import get_greedy_percentage, get_periodic_percentage, get_patient_percentage


class REINFORCE:
    def __init__(self, device="cpu", model=None):
        self.learning_rate = 1e-5
        self.gamma = 1
        
        if not model:
            self.model = PairedKidneyModel(
                hidden_dim=32,
                num_layers=8
            )
            self.model.reset_parameters()
        else:
            self.model = model
        print("Number of parameters: ", sum(p.numel() for p in self.model.parameters()))
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.device = torch.device(device)

        self.model.to(self.device)

        self.probs = []

    def sample_action(self, state: dict) -> float:
        tensor_state = {}
        for key in state:
            if not (isinstance(state[key], int) or isinstance(state[key], float)):
                tensor_state[key] = torch.Tensor(state[key]).to(self.device)
        
        if sum(state["active_agents"]) == 0: # there are no elements to consider - will break model
            return {
                "selection": np.zeros(env.n_agents),
                "match_selection": 0,
                "match_regular": 0
            }
        

        item_priority = self.model(tensor_state["adjacency_matrix"], 
                                   state["timestep"], 
                                   tensor_state["arrivals"], 
                                   tensor_state["departures"],
                                   tensor_state["is_hard_to_match"], 
                                   state["total_timesteps"], 
                                   tensor_state["active_agents"])
        item_priority_distrib = Normal(item_priority, 0.1)
        item_priority = item_priority_distrib.sample()

        self.probs.append(item_priority_distrib.log_prob(item_priority).flatten())

        item_priority = item_priority.cpu().detach().numpy()

        # after you've sampled values from the distribution, you have to align them to 0 or 1 for each of them
        item_priority = (item_priority > 0.5).astype(np.int32)
        return {
            "selection": item_priority,
            "match_selection": 1,
            "match_regular": 0
        }

    def update(self, rewards) -> None:
        reward_tensor = torch.tensor(rewards, dtype=torch.float32, device=self.device)
        
        if len(rewards) > 1:
            reward_tensor = (reward_tensor - reward_tensor.mean()) / (reward_tensor.std() + 1e-8)
        
        log_probs = torch.stack(self.probs)

        policy_loss = 0
        for i, r in enumerate(reward_tensor):
            episode_log_probs = log_probs[i]
            policy_loss += -torch.mean(episode_log_probs) * r
        
        policy_loss /= len(rewards)        
        self.optimizer.zero_grad()
        policy_loss.backward()

        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()
        
        self.probs = []


# MPS has some issues when trying to compute the loss

# training constants
episodes_per_env = 128
eval_lookback = 16
eval_period = 16
batch_size = 16
    
agent = REINFORCE(
    device="cpu", # faster on CPU - probably because less data transferring back and forth w environment
)

envs = [
    PairedKidneyDonationEnv(
        n_agents=100,
        n_timesteps=36,
        criticality_rate=18
    ),
    PairedKidneyDonationEnv(
        n_agents=250,
        n_timesteps=36,
        criticality_rate=18
    ),
    PairedKidneyDonationEnv(
        n_agents=500,
        n_timesteps=36,
        criticality_rate=18
    ),
    PairedKidneyDonationEnv(
        n_agents=500,
        n_timesteps=180,
        criticality_rate=90
    )
]



for env in envs:
    reward_over_episodes = []
    recent_greedy_rewards = []
    recent_patient_rewards = []
    recent_rewards = []
    for episode in tqdm(range(episodes_per_env)):
        obs, info = env.reset()
        done = False

        while not done:
            action = agent.sample_action(obs)
            obs, reward, done, _, info = env.step(action)
        greedy_reward = get_greedy_percentage(env)
        advantage = reward - greedy_reward
        shaped_reward = reward + 0.25 * advantage # trying to adjust for the greedy reward

        reward_over_episodes.append(reward)
        recent_rewards.append(shaped_reward)

        if len(recent_rewards) >= batch_size:
            agent.update(recent_rewards)
            recent_rewards = []

        if ((episode + 1) % eval_period) >= (eval_period - eval_lookback):
            recent_greedy_rewards.append(greedy_reward)
            recent_patient_rewards.append(get_patient_percentage(env))

        if (episode + 1) % eval_period == 0:
            mean_reward = np.mean(reward_over_episodes[-eval_lookback:])
            mean_greedy_reward = np.mean(recent_greedy_rewards)
            mean_patient_reward = np.mean(recent_patient_rewards)

            print(f"Episode {episode + 1}: "
                f"avg reward = {mean_reward:.2f}, "
                f"greedy = {mean_greedy_reward:.2f}")

            recent_greedy_rewards.clear()
            recent_patient_rewards.clear()

    env.close()

torch.save(agent.model.state_dict(), "model.pth")